"""Tools that wrap evaluate_hierarchical.py metrics computation.

Reuses:
  - compute_metrics() from scripts/evaluate_hierarchical.py
  - Saved evaluation_results.json from previous runs
  - Per-regime metrics from evaluation results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base import BaseTool
from agents.state import ModelConfidence


class RecentModelPerformanceTool(BaseTool):
    """Read model performance metrics from evaluation_results.json.

    Looks for the latest saved evaluation results (from a prior
    ``python scripts/evaluate_hierarchical.py`` run).
    """

    name = "recent_model_performance"
    description = "Get recent IC and directional accuracy metrics for the model."

    def __init__(
        self,
        results_dir: str = "results/hierarchical_eval",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.results_dir = results_dir

    def lazy_init(self):
        self._initialized = True

    def execute(self, **kwargs) -> Dict[str, Any]:
        results_path = Path(self.results_dir) / "evaluation_results.json"
        if not results_path.exists():
            # Also check model subdirectories for eval results
            for candidate in [
                Path("results/hierarchical_eval/evaluation_results.json"),
                Path("models/hierarchical_v8/evaluation_results.json"),
                Path("models/hierarchical_v7/evaluation_results.json"),
            ]:
                if candidate.exists():
                    results_path = candidate
                    break
            else:
                return {
                    "available": False,
                    "ic": 0.0,
                    "rank_ic": 0.0,
                    "directional_accuracy": 0.5,
                }

        with open(results_path) as f:
            data = json.load(f)

        agg = data.get("aggregate_metrics", {})
        return {
            "available": True,
            "ic": agg.get("ic", 0.0),
            "rank_ic": agg.get("rank_ic", 0.0),
            "directional_accuracy": agg.get("directional_accuracy", 0.5),
            "n_samples": agg.get("n_samples", 0),
            "source": str(results_path),
        }


class ClassifyModelConfidenceTool(BaseTool):
    """Classify overall model confidence as hot/warm/cold."""

    name = "classify_model_confidence"
    description = "Classify model confidence based on recent IC and directional accuracy."

    def lazy_init(self):
        self._initialized = True

    def execute(
        self,
        ic: float,
        directional_accuracy: float,
        ic_hot: float = 0.05,
        ic_cold: float = 0.02,
        dir_acc_threshold: float = 0.52,
        available: bool = True,
        **kwargs,
    ) -> ModelConfidence:
        # No historical eval results yet → assume WARM (not COLD).
        # COLD should only be assigned when we have evidence the model
        # is performing poorly, not when we simply haven't run eval yet.
        if not available:
            return ModelConfidence.WARM
        if ic >= ic_hot and directional_accuracy >= dir_acc_threshold:
            return ModelConfidence.HOT
        elif ic < ic_cold or directional_accuracy < 0.50:
            return ModelConfidence.COLD
        else:
            return ModelConfidence.WARM


class PerRegimePerformanceTool(BaseTool):
    """Read per-regime model performance metrics.

    Looks for per_regime_metrics in evaluation_results.json, or computes
    approximate metrics from saved predictions if available.

    Returns: {regime_label: {"ic": float, "directional_accuracy": float, "n_samples": int}}
    """

    name = "per_regime_performance"
    description = "Get model performance broken down by market regime."

    def __init__(self, results_dir: str = "results/hierarchical_eval", **kwargs):
        super().__init__(**kwargs)
        self.results_dir = results_dir

    def lazy_init(self):
        self._initialized = True

    def execute(self, **kwargs) -> Dict[str, Dict[str, float]]:
        # Look for evaluation results with per-regime breakdown
        results_path = Path(self.results_dir) / "evaluation_results.json"
        if not results_path.exists():
            for candidate in [
                Path("results/hierarchical_eval/evaluation_results.json"),
                Path("models/hierarchical_v8/evaluation_results.json"),
                Path("models/hierarchical_v7/evaluation_results.json"),
            ]:
                if candidate.exists():
                    results_path = candidate
                    break

        if not results_path.exists():
            # Fall back to per-sub-model regime IC files from training
            return self._read_training_regime_ic() or self._default_regime_metrics()

        with open(results_path) as f:
            data = json.load(f)

        # Check for pre-computed per-regime metrics
        per_regime = data.get("per_regime_metrics")
        if per_regime:
            return per_regime

        # Try per-sub-model IC files from training (regime_curriculum output)
        training_regime = self._read_training_regime_ic()
        if training_regime:
            return training_regime

        # Fallback: try to compute from per-ticker results + regime labels
        return self._compute_from_predictions(data)

    def _read_training_regime_ic(self) -> Dict[str, Dict[str, float]]:
        """Read per-regime IC files written by _save_per_model_regime_ic().

        Looks in common model output directories for regime_ic_*.json files.
        Aggregates across sub-models by averaging IC/DA, keeping n_samples sum.

        Returns empty dict if no files found.
        """
        search_dirs = [
            Path(self.results_dir),
            Path("models/hierarchical"),
            Path("models/hierarchical_v10"),
            Path("models/hierarchical_v9"),
            Path("models/hierarchical_v8"),
        ]
        # Find the most recent regime_ic_*.json from any dir
        regime_ic_files: list = []
        for d in search_dirs:
            if d.exists():
                regime_ic_files.extend(sorted(d.glob("regime_ic_*.json")))

        if not regime_ic_files:
            return {}

        # Aggregate across all sub-model files
        combined: Dict[str, Dict[str, list]] = {}  # regime → {metric → [values]}
        for f in regime_ic_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for regime, metrics in data.items():
                    if regime not in combined:
                        combined[regime] = {
                            "ic": [],
                            "directional_accuracy": [],
                            "rank_ic": [],
                            "n_samples": [],
                        }
                    for k in ["ic", "directional_accuracy", "rank_ic"]:
                        v = metrics.get(k, 0.0)
                        if np.isfinite(v):
                            combined[regime][k].append(v)
                    combined[regime]["n_samples"].append(metrics.get("n_samples", 0))
            except Exception:
                continue

        result: Dict[str, Dict[str, float]] = {}
        for regime, vals in combined.items():
            if not vals["ic"]:
                continue
            result[regime] = {
                "ic": float(np.mean(vals["ic"])),
                "directional_accuracy": float(np.mean(vals["directional_accuracy"])) if vals["directional_accuracy"] else 0.5,
                "rank_ic": float(np.mean(vals["rank_ic"])) if vals["rank_ic"] else 0.0,
                "n_samples": int(np.sum(vals["n_samples"])),
            }
        return result

    def _default_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """Sensible defaults when no evaluation data is available."""
        return {
            "bull_low_vol": {"ic": 0.05, "directional_accuracy": 0.55, "n_samples": 0},
            "bull_high_vol": {"ic": 0.03, "directional_accuracy": 0.52, "n_samples": 0},
            "sideways": {"ic": 0.02, "directional_accuracy": 0.50, "n_samples": 0},
            "bear_low_vol": {"ic": 0.01, "directional_accuracy": 0.48, "n_samples": 0},
            "bear_high_vol": {"ic": 0.00, "directional_accuracy": 0.45, "n_samples": 0},
        }

    def _compute_from_predictions(
        self, data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Try to extract regime-level metrics from stored prediction data."""
        # Look for per-ticker results and aggregate
        per_ticker = data.get("per_ticker_metrics", {})
        if not per_ticker:
            return self._default_regime_metrics()

        # If per-ticker data has regime labels, group and compute
        regime_preds: Dict[str, list] = {}
        regime_tgts: Dict[str, list] = {}

        for ticker, metrics in per_ticker.items():
            regime = metrics.get("regime", "unknown")
            if regime not in regime_preds:
                regime_preds[regime] = []
                regime_tgts[regime] = []

            pred = metrics.get("predicted_return", 0.0)
            actual = metrics.get("actual_return", 0.0)
            regime_preds[regime].append(pred)
            regime_tgts[regime].append(actual)

        result = {}
        for regime in regime_preds:
            preds = np.array(regime_preds[regime])
            tgts = np.array(regime_tgts[regime])
            n = len(preds)
            if n < 5:
                continue

            ic = float(np.corrcoef(preds, tgts)[0, 1]) if np.std(preds) > 1e-10 else 0.0
            da = float(np.mean((preds > 0) == (tgts > 0)))
            result[regime] = {"ic": ic, "directional_accuracy": da, "n_samples": n}

        return result if result else self._default_regime_metrics()
