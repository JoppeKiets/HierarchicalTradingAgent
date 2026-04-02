"""Autonomous Analyst feature engineering via a local LLM (Ollama).

How it works
------------
1. **Proposal** — we send the list of available feature column names to a locally
   running Ollama model (default: llama3.1:8b-instruct-q8_0).  The LLM reasons
   about which combinations might carry new predictive signal and returns a JSON
   list of candidates, each with:
       {"name": "feat_...", "formula": "<pandas expression>", "rationale": "..."}

2. **Safe eval** — the formula string is executed in a restricted namespace that
   contains only the base feature DataFrame (called `df`) plus numpy (`np`).
   Any formula that raises an exception, produces NaN/Inf > 30 %, or is
   near-constant is silently dropped.

3. **IC ablation** — for each surviving candidate we run a fast ridge-regression
   information-coefficient (IC) test.  A candidate is *promoted* only when its
   out-of-sample IC exceeds the baseline by at least `min_ic_improvement`.

4. **Persistence** — promoted features are appended to
       data/feature_feedback/accepted_generated_features.json
   and the corresponding Python formulas are written to
       src/features/generated_features.py
   so that the next `--force-preprocess` training run picks them up.

LLM backend
-----------
We talk to Ollama's REST API directly (no third-party Python client needed):

    POST http://localhost:11434/api/generate
    {
        "model": "llama3.1:8b-instruct-q8_0",
        "prompt": "...",
        "stream": false
    }

This means the *only* runtime requirement is that `ollama serve` is running
(it starts automatically as a systemd service on most Linux installs).
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b-instruct-q8_0"
REQUEST_TIMEOUT = 120  # seconds – LLM inference can be slow on CPU

# Prompt template sent to the LLM.
# We inject the feature names at {feature_names} and the count at {n_candidates}.
_PROPOSAL_PROMPT = textwrap.dedent("""
You are a quantitative analyst expert in financial time-series features.

Below is a list of feature column names available in a daily stock-feature matrix.
Each row represents one trading day for one stock.  The target is the next-day
normalised return (positive = up, negative = down).

Available features:
{feature_names}

Your task: propose {n_candidates} NEW feature columns that could improve
next-day return prediction by combining the existing ones in non-obvious ways.
Think about momentum/volume divergences, regime indicators, cross-asset ratios, etc.

Rules:
- Each formula must be valid Python/pandas.  The base DataFrame is called `df`.
- Use only arithmetic operators, numpy (`np`), and existing column names.
- Column names with spaces must be quoted: df["col name"].
- The formula must evaluate to a pandas Series of the same length as `df`.
- Feature names must start with "feat_" and contain only letters, digits, underscores.

Respond ONLY with a valid JSON array – nothing else, no markdown fences.
Example format:
[
  {{
    "name": "feat_rsi_volume_divergence",
    "formula": "df['rsi'] - df['vol_change_5d']",
    "rationale": "RSI high while volume trend weak can indicate fragile momentum."
  }}
]
""").strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureCandidate:
    name: str
    formula: str
    rationale: str


# ---------------------------------------------------------------------------
# LLM proposal engine
# ---------------------------------------------------------------------------

class OllamaFeatureProposer:
    """Calls a local Ollama model to propose feature candidates."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = OLLAMA_URL,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the raw response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Keep it focused: low temperature for structured output
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }
        try:
            resp = requests.post(
                self.ollama_url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.ConnectionError:
            logger.warning(
                "OllamaFeatureProposer | Could not connect to %s – is `ollama serve` running?",
                self.ollama_url,
            )
            return ""
        except Exception as exc:
            logger.warning("OllamaFeatureProposer | LLM call failed: %s", exc)
            return ""

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Pull out the first JSON array from the raw LLM response."""
        # Strip potential markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Find the first [...] block
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return m.group(0)
        return "[]"

    def _validate_name(self, name: str) -> bool:
        return bool(re.match(r"^feat_[a-zA-Z0-9_]+$", name))

    def propose(
        self,
        feature_names: List[str],
        n_candidates: int = 5,
    ) -> List[FeatureCandidate]:
        """Ask the LLM to propose `n_candidates` new features."""
        names_str = "\n".join(f"  - {n}" for n in feature_names[:120])  # cap for prompt length
        prompt = _PROPOSAL_PROMPT.format(
            feature_names=names_str,
            n_candidates=n_candidates,
        )

        logger.info(
            "OllamaFeatureProposer | Requesting %d candidates from model=%s",
            n_candidates,
            self.model,
        )
        raw = self._call_llm(prompt)
        if not raw:
            logger.warning("OllamaFeatureProposer | LLM returned empty response; returning empty candidate list")
            return []

        json_str = self._extract_json(raw)
        try:
            items = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("OllamaFeatureProposer | JSON parse error: %s\nRaw text: %.300s", exc, raw)
            return []

        candidates: List[FeatureCandidate] = []
        seen_names: set = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            formula = str(item.get("formula", "")).strip()
            rationale = str(item.get("rationale", "")).strip()

            if not name or not formula:
                continue
            if not self._validate_name(name):
                logger.debug("OllamaFeatureProposer | Invalid name '%s', skipping", name)
                continue
            if name in seen_names:
                continue
            seen_names.add(name)

            candidates.append(FeatureCandidate(name=name, formula=formula, rationale=rationale))

        logger.info(
            "OllamaFeatureProposer | Parsed %d valid candidates from LLM response",
            len(candidates),
        )
        return candidates


# ---------------------------------------------------------------------------
# Safe formula evaluator
# ---------------------------------------------------------------------------

class SafeFormulaEvaluator:
    """Evaluate a pandas formula string in a restricted namespace."""

    # Maximum fraction of NaN/Inf allowed before we reject the result
    MAX_BAD_FRACTION = 0.30

    def evaluate(
        self,
        formula: str,
        df: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """
        Execute `formula` in a namespace where only `df` and `np` exist.

        Returns a float32 numpy array of shape (len(df),), or None on failure.
        """
        # Very basic safety: block common dangerous builtins
        forbidden = ["import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess"]
        for kw in forbidden:
            if kw in formula:
                logger.debug("SafeFormulaEvaluator | Blocked formula containing '%s'", kw)
                return None

        namespace = {"df": df, "np": np}
        try:
            result = eval(formula, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            logger.debug("SafeFormulaEvaluator | Formula eval error: %s | formula: %s", exc, formula)
            return None

        # Coerce to numpy
        if isinstance(result, pd.Series):
            arr = result.to_numpy(dtype=np.float32)
        elif isinstance(result, np.ndarray):
            arr = result.astype(np.float32)
        else:
            try:
                arr = np.array(result, dtype=np.float32)
            except Exception:
                return None

        if arr.shape != (len(df),):
            logger.debug("SafeFormulaEvaluator | Shape mismatch: %s vs expected (%d,)", arr.shape, len(df))
            return None

        bad_fraction = float(np.mean(~np.isfinite(arr)))
        if bad_fraction > self.MAX_BAD_FRACTION:
            logger.debug(
                "SafeFormulaEvaluator | Too many NaN/Inf (%.1f%%) in candidate output",
                bad_fraction * 100,
            )
            return None

        # Replace remaining NaN/Inf with 0 and check it's not constant
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(arr) < 1e-10:
            logger.debug("SafeFormulaEvaluator | Near-constant candidate output (std < 1e-10)")
            return None

        return arr


# ---------------------------------------------------------------------------
# Main feature engineering orchestrator
# ---------------------------------------------------------------------------

class AnalystAutoFeatureEngineer:
    """Propose → ablate → promote generated features using a local LLM."""

    def __init__(
        self,
        cache_dir: str = "data/feature_cache",
        feedback_dir: str = "data/feature_feedback",
        generated_feature_code_path: str = "src/features/generated_features.py",
        llm_model: str = DEFAULT_MODEL,
        ollama_url: str = OLLAMA_URL,
    ):
        self.cache_dir = Path(cache_dir)
        self.feedback_dir = Path(feedback_dir)
        self.generated_feature_code_path = Path(generated_feature_code_path)

        self.accepted_path = self.feedback_dir / "accepted_generated_features.json"
        self.reports_dir = self.feedback_dir / "generated_feature_reports"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.proposer = OllamaFeatureProposer(model=llm_model, ollama_url=ollama_url)
        self.evaluator = SafeFormulaEvaluator()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_samples(
        self,
        max_tickers: int = 200,
        max_rows_per_ticker: int = 250,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        daily_dir = self.cache_dir / "daily"
        meta = self.cache_dir / "metadata.json"
        if not daily_dir.exists() or not meta.exists():
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

        with open(meta) as f:
            metadata = json.load(f)
        daily_meta = metadata.get("daily", {})
        if not daily_meta:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

        tickers = sorted(daily_meta.keys())[:max_tickers]
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        feature_names: List[str] = []

        for ticker in tickers:
            fx = daily_dir / f"{ticker}_features.npy"
            ty = daily_dir / f"{ticker}_targets.npy"
            if not fx.exists() or not ty.exists():
                continue
            try:
                x = np.load(fx, mmap_mode="r")
                y = np.load(ty, mmap_mode="r")
            except Exception:
                continue
            if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
                continue
            if not feature_names:
                feature_names = list(daily_meta.get(ticker, {}).get("feature_names", []))

            valid = np.isfinite(y)
            if np.sum(valid) < 50:
                continue
            x_v = np.asarray(x[valid], dtype=np.float32)
            y_v = np.asarray(y[valid], dtype=np.float32)
            if len(y_v) > max_rows_per_ticker:
                x_v = x_v[-max_rows_per_ticker:]
                y_v = y_v[-max_rows_per_ticker:]

            all_x.append(x_v)
            all_y.append(y_v)

        if not all_x:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), feature_names

        x_all = np.vstack(all_x).astype(np.float32)
        y_all = np.concatenate(all_y).astype(np.float32)
        finite_rows = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
        x_all = x_all[finite_rows]
        y_all = y_all[finite_rows]
        return x_all, y_all, feature_names

    # ------------------------------------------------------------------
    # IC evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _ridge_ic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> float:
        """Pearson IC of a ridge-regression predictor on a held-out 20% slice."""
        n = len(y)
        if n < 50 or x.ndim != 2:
            return 0.0

        split = int(n * 0.8)
        if split <= 20 or split >= n:
            return 0.0

        x_tr, x_va = x[:split], x[split:]
        y_tr, y_va = y[:split], y[split:]

        mu = x_tr.mean(axis=0, keepdims=True)
        sigma = x_tr.std(axis=0, keepdims=True) + 1e-6
        x_tr = (x_tr - mu) / sigma
        x_va = (x_va - mu) / sigma

        x_tr = np.nan_to_num(x_tr)
        x_va = np.nan_to_num(x_va)

        xtx = x_tr.T @ x_tr
        reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
        try:
            w = np.linalg.solve(xtx + reg, x_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xtx + reg) @ (x_tr.T @ y_tr)

        pred = x_va @ w
        if np.std(pred) < 1e-12 or np.std(y_va) < 1e-12:
            return 0.0
        ic = float(np.corrcoef(pred, y_va)[0, 1])
        return 0.0 if not np.isfinite(ic) else ic

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_accepted(self) -> List[Dict[str, Any]]:
        if not self.accepted_path.exists():
            return []
        try:
            with open(self.accepted_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_accepted(self, accepted: List[Dict[str, Any]]) -> None:
        with open(self.accepted_path, "w") as f:
            json.dump(accepted, f, indent=2)

    def _render_generated_feature_module(self, accepted: List[Dict[str, Any]]) -> str:
        """
        Render `src/features/generated_features.py` from the accepted list.

        Each entry in `accepted` carries a `formula` that was already validated
        by SafeFormulaEvaluator (uses `df` as the DataFrame variable and `np`).
        We alias `base_features` → `df` so the generated module works with the
        caller's naming convention.
        """
        names = [a["name"] for a in accepted]

        lines = [
            '"""Auto-generated feature formulas promoted by Analyst LLM ablations.',
            "",
            "This file is OVERWRITTEN on every feature-engineering cycle.",
            "Do not edit manually.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "from typing import List",
            "import numpy as np",
            "import pandas as pd",
            "",
            f"GENERATED_FEATURE_NAMES: List[str] = {names!r}",
            "",
            "def compute_generated_features(base_features: pd.DataFrame) -> pd.DataFrame:",
            '    """Compute all promoted generated features from the base feature DataFrame."""',
            "    out = pd.DataFrame(index=base_features.index)",
            "    if base_features is None or len(base_features) == 0:",
            "        return out",
            "    df = base_features  # alias used by generated formulas",
        ]

        for a in accepted:
            name = a["name"]
            formula = a["formula"]
            rationale = a.get("rationale", "").replace("\n", " ")
            lines.append(f"    # {name}: {formula}")
            lines.append(f"    # rationale: {rationale}")
            lines.append(f"    try:")
            lines.append(f"        out[{name!r}] = {formula}")
            lines.append(f"    except Exception:")
            lines.append(f"        out[{name!r}] = 0.0")

        lines.extend([
            "    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)",
            "    return out.astype('float32')",
            "",
        ])
        return "\n".join(lines)

    def _write_generated_feature_code(self, accepted: List[Dict[str, Any]]) -> None:
        self.generated_feature_code_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_generated_feature_module(accepted)
        with open(self.generated_feature_code_path, "w") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        max_candidates: int = 5,
        max_tickers_sample: int = 200,
        max_rows_per_ticker: int = 250,
        min_ic_improvement: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Full propose → eval → ablate → promote cycle.

        Returns a summary dict written to:
            data/feature_feedback/generated_feature_reports/latest_feature_engineering.json
        """
        # 1. Load training samples
        x, y, feature_names = self._load_samples(
            max_tickers=max_tickers_sample,
            max_rows_per_ticker=max_rows_per_ticker,
        )
        if x.size == 0 or len(feature_names) == 0:
            return {
                "available": False,
                "reason": "No cached daily samples available",
                "baseline_ic": 0.0,
                "promoted": [],
                "promoted_count": 0,
            }

        base_df = pd.DataFrame(x, columns=feature_names)

        # 2. Ask LLM to propose candidates
        candidates = self.proposer.propose(
            feature_names=feature_names,
            n_candidates=max_candidates,
        )

        if not candidates:
            return {
                "available": True,
                "reason": "LLM returned no valid candidates (may be offline)",
                "baseline_ic": 0.0,
                "promoted": [],
                "promoted_count": 0,
            }

        # 3. Compute baseline IC
        baseline_ic = self._ridge_ic(x, y)
        logger.info("Analyst feature engineering | baseline_ic=%.4f", baseline_ic)

        # 4. Ablation loop
        results: List[Dict[str, Any]] = []
        promoted: List[Dict[str, Any]] = []

        for c in candidates:
            # Safe-eval the formula
            vec = self.evaluator.evaluate(c.formula, base_df)
            if vec is None:
                results.append({
                    "name": c.name,
                    "formula": c.formula,
                    "rationale": c.rationale,
                    "status": "rejected",
                    "reason": "formula eval failed or near-constant",
                    "ic": baseline_ic,
                    "delta_ic": 0.0,
                })
                continue

            # Append candidate column and re-measure IC
            x_aug = np.concatenate([x, vec.reshape(-1, 1)], axis=1)
            ic_aug = self._ridge_ic(x_aug, y)
            delta = float(ic_aug - baseline_ic)
            keep = bool(np.isfinite(delta) and delta >= min_ic_improvement)

            item = {
                "name": c.name,
                "formula": c.formula,
                "rationale": c.rationale,
                "status": "promoted" if keep else "rejected",
                "ic": ic_aug,
                "delta_ic": delta,
            }
            results.append(item)
            if keep:
                promoted.append(item)
                logger.info(
                    "Analyst feature engineering | PROMOTED %s | delta_ic=+%.4f",
                    c.name,
                    delta,
                )
            else:
                logger.info(
                    "Analyst feature engineering | REJECTED %s | delta_ic=%.4f (threshold %.4f)",
                    c.name,
                    delta,
                    min_ic_improvement,
                )

        # 5. Persist accepted features
        accepted = self._load_accepted()
        accepted_names = {a.get("name") for a in accepted}
        now = datetime.now(timezone.utc).isoformat()
        for p in promoted:
            if p["name"] not in accepted_names:
                accepted.append({
                    "name": p["name"],
                    "formula": p["formula"],
                    "rationale": p["rationale"],
                    "promoted_at": now,
                    "last_delta_ic": p["delta_ic"],
                })

        self._save_accepted(accepted)
        self._write_generated_feature_code(accepted)

        # 6. Write summary report
        summary: Dict[str, Any] = {
            "available": True,
            "timestamp": now,
            "llm_model": self.proposer.model,
            "baseline_ic": baseline_ic,
            "candidates_tested": len(candidates),
            "results": results,
            "promoted": promoted,
            "promoted_count": len(promoted),
            "accepted_total": len(accepted),
            "accepted_feature_names": [a.get("name") for a in accepted],
            "min_ic_improvement": min_ic_improvement,
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"feature_engineering_{ts}.json"
        latest_path = self.reports_dir / "latest_feature_engineering.json"
        for path in (report_path, latest_path):
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)

        logger.info(
            "Analyst feature engineering | baseline_ic=%.4f tested=%d promoted=%d accepted_total=%d",
            baseline_ic,
            len(candidates),
            len(promoted),
            len(accepted),
        )
        return summary
