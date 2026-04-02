"""Attention prior from agent feedback — adjusts MetaMLP attention biases.

After each walk-forward window, we have:
  - Critic assessments per ticker (regime-specific confidence)
  - Executor trade journal with realized P&L per sub-model direction

This module computes per-sub-model performance scores and translates
them into attention bias vectors that nudge the MetaMLP's attention
network toward sub-models that performed best in the previous window.

The bias is applied to the *output layer* of the attention network
(the ``nn.Linear(32, n_sub_models)`` layer) so that the softmax
starts with a prior favoring historically-stronger models.

Persistence layout:
    data/attention_prior/
        latest_bias.json       # Latest attention bias vector
        bias_history.jsonl     # Append-only log of per-window biases
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Canonical sub-model order matching HierarchicalForecaster
DEFAULT_SUB_MODEL_NAMES = ["lstm_d", "tft_d", "lstm_m", "tft_m"]


class AttentionPriorComputer:
    """Computes attention bias vectors from Critic + Executor feedback.

    Workflow:
      1. ``compute_from_walk_forward_window()`` — reads predictions/targets
         from a completed window and computes per-sub-model IC and
         directional accuracy.
      2. ``compute_from_trade_journal()`` — reads the Executor's trade
         journal and computes realized P&L attribution per sub-model.
      3. ``blend_and_save()`` — combines both signals into a single bias
         vector and persists it for the next window.

    The bias vector has shape ``(n_sub_models,)`` and is added to the
    attention network's output layer bias (pre-softmax logits).  Positive
    values increase attention, negative decrease it.  The scale is
    controlled by ``bias_strength``.
    """

    def __init__(
        self,
        prior_dir: str = "data/attention_prior",
        sub_model_names: Optional[List[str]] = None,
        bias_strength: float = 0.5,
        momentum: float = 0.7,
    ):
        """
        Args:
            prior_dir:        Where to persist bias vectors
            sub_model_names:  Ordered list of sub-model names
            bias_strength:    Scale factor for the bias (higher = stronger prior)
            momentum:         EMA factor for blending with previous bias
                              (0 = no memory, 1 = ignore new evidence)
        """
        self.prior_dir = Path(prior_dir)
        self.sub_model_names = sub_model_names or DEFAULT_SUB_MODEL_NAMES
        self.bias_strength = bias_strength
        self.momentum = momentum
        self.n_sub = len(self.sub_model_names)

        self.prior_dir.mkdir(parents=True, exist_ok=True)

        # Current bias vector (loaded from disk if available)
        self._current_bias: Optional[np.ndarray] = None
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self):
        """Load latest bias from disk."""
        path = self.prior_dir / "latest_bias.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                bias_dict = data.get("bias", {})
                bias = np.zeros(self.n_sub)
                for i, name in enumerate(self.sub_model_names):
                    bias[i] = bias_dict.get(name, 0.0)
                self._current_bias = bias
                logger.info(
                    "Loaded attention bias: %s",
                    {n: round(float(b), 3) for n, b in zip(self.sub_model_names, bias)},
                )
            except Exception as e:
                logger.warning("Failed to load attention bias: %s", e)

    def save(self, bias: np.ndarray, window_idx: int = -1,
             metrics: Optional[Dict] = None):
        """Persist bias vector to disk."""
        bias_dict = {
            name: round(float(bias[i]), 6)
            for i, name in enumerate(self.sub_model_names)
        }

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "window_idx": window_idx,
            "bias": bias_dict,
            "bias_strength": self.bias_strength,
            "momentum": self.momentum,
            "metrics": metrics or {},
        }

        # Save latest
        with open(self.prior_dir / "latest_bias.json", "w") as f:
            json.dump(data, f, indent=2)

        # Append to history
        with open(self.prior_dir / "bias_history.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")

        self._current_bias = bias

    # ── Compute from sub-model predictions ────────────────────────────

    def compute_from_predictions(
        self,
        sub_model_predictions: Dict[str, np.ndarray],
        targets: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-sub-model metrics from raw predictions.

        Args:
            sub_model_predictions:  {model_name: (N,) array of predictions}
            targets:                (N,) array of actual returns

        Returns:
            {model_name: {"ic": float, "directional_accuracy": float, "mse": float}}
        """
        from scipy import stats

        metrics = {}
        for name in self.sub_model_names:
            preds = sub_model_predictions.get(name)
            if preds is None or len(preds) < 5:
                metrics[name] = {"ic": 0.0, "directional_accuracy": 0.5, "mse": 0.0}
                continue

            # IC
            if np.std(preds) < 1e-10 or np.std(targets) < 1e-10:
                ic = 0.0
            else:
                ic = float(np.corrcoef(preds, targets)[0, 1])
                if np.isnan(ic):
                    ic = 0.0

            # Directional accuracy
            dir_acc = float(np.mean((preds > 0) == (targets > 0)))

            # MSE
            mse = float(np.mean((preds - targets) ** 2))

            metrics[name] = {"ic": ic, "directional_accuracy": dir_acc, "mse": mse}

        return metrics

    # ── Compute from trade journal ────────────────────────────────────

    def compute_from_trade_journal(
        self,
        journal_path: str = "data/trade_journal/trade_journal.jsonl",
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute per-sub-model P&L attribution from the trade journal.

        Reads JSONL entries and estimates which sub-models contributed most
        to profitable vs unprofitable trades by correlating sub-model
        agreement/spread with realized returns.

        Returns:
            {model_name: attribution_score}  (higher = better)
        """
        path = Path(journal_path)
        if not path.exists():
            logger.info("No trade journal found at %s — skipping P&L attribution", path)
            return {}

        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        if not entries:
            return {}

        # Filter to window if specified
        if window_start or window_end:
            filtered = []
            for e in entries:
                ts = e.get("timestamp", "")
                if window_start and ts < window_start:
                    continue
                if window_end and ts > window_end:
                    continue
                filtered.append(e)
            entries = filtered

        if not entries:
            return {}

        # For each trade: if actual_return is populated, attribute
        # performance based on whether sub_model_agreement was high
        # (all models agreed → shared credit) vs low (disagreement →
        # credit to models aligned with actual outcome)
        attribution = {name: 0.0 for name in self.sub_model_names}
        n_trades = 0

        for entry in entries:
            actual_return = entry.get("actual_return")
            if actual_return is None:
                continue

            agreement = entry.get("sub_model_agreement", 0.5)
            pred_return = entry.get("predicted_return", 0.0)
            n_trades += 1

            # Simple attribution: profitable trade → credit to all models
            # proportional to agreement; unprofitable → debit proportional
            # to agreement
            profit_signal = 1.0 if actual_return > 0 else -1.0
            per_model_credit = profit_signal * agreement / self.n_sub
            for name in self.sub_model_names:
                attribution[name] += per_model_credit

        # Normalize
        if n_trades > 0:
            for name in attribution:
                attribution[name] /= n_trades

        logger.info("Trade journal attribution (%d trades): %s",
                     n_trades,
                     {n: round(v, 4) for n, v in attribution.items()})
        return attribution

    # ── Blend signals into bias vector ────────────────────────────────

    def compute_bias(
        self,
        sub_model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        journal_attribution: Optional[Dict[str, float]] = None,
        window_idx: int = -1,
    ) -> np.ndarray:
        """Combine IC-based scores and journal attribution into an attention bias.

        The bias is a vector of shape ``(n_sub_models,)`` that gets added
        to the attention network's pre-softmax logits.

        Process:
          1. Normalize per-model IC to z-scores → ``ic_signal``
          2. Normalize journal attribution → ``pnl_signal``
          3. Weighted blend (70% IC, 30% P&L) → ``raw_bias``
          4. Scale by ``bias_strength``
          5. EMA with previous bias (``momentum``)

        Returns:
            (n_sub_models,) numpy array
        """
        raw_scores = np.zeros(self.n_sub)

        # --- IC signal (70% weight) ---
        if sub_model_metrics:
            ics = np.array([
                sub_model_metrics.get(name, {}).get("ic", 0.0)
                for name in self.sub_model_names
            ])
            ic_std = ics.std()
            if ic_std > 1e-8:
                ic_z = (ics - ics.mean()) / ic_std
            else:
                ic_z = np.zeros(self.n_sub)
            raw_scores += 0.7 * ic_z

        # --- P&L attribution signal (30% weight) ---
        if journal_attribution:
            pnl_scores = np.array([
                journal_attribution.get(name, 0.0)
                for name in self.sub_model_names
            ])
            pnl_std = pnl_scores.std()
            if pnl_std > 1e-8:
                pnl_z = (pnl_scores - pnl_scores.mean()) / pnl_std
            else:
                pnl_z = np.zeros(self.n_sub)
            raw_scores += 0.3 * pnl_z

        # Scale
        bias = raw_scores * self.bias_strength

        # EMA with previous bias
        if self._current_bias is not None and len(self._current_bias) == self.n_sub:
            bias = self.momentum * self._current_bias + (1 - self.momentum) * bias

        # Clamp to prevent extreme biases
        bias = np.clip(bias, -2.0, 2.0)

        # Log
        bias_dict = {
            name: round(float(bias[i]), 4)
            for i, name in enumerate(self.sub_model_names)
        }
        logger.info("Attention bias for window %d: %s", window_idx, bias_dict)

        # Persist
        merged_metrics = {}
        if sub_model_metrics:
            merged_metrics["sub_model_ic"] = {
                n: round(m.get("ic", 0.0), 4) for n, m in sub_model_metrics.items()
            }
        if journal_attribution:
            merged_metrics["journal_attribution"] = {
                n: round(v, 4) for n, v in journal_attribution.items()
            }
        self.save(bias, window_idx=window_idx, metrics=merged_metrics)

        return bias

    # ── Convenience: load as torch tensor ─────────────────────────────

    def get_bias_tensor(self) -> Optional["torch.Tensor"]:
        """Return the current bias as a PyTorch tensor, or None."""
        import torch
        if self._current_bias is None:
            return None
        return torch.tensor(self._current_bias, dtype=torch.float32)

    def get_latest_bias(self) -> Optional[np.ndarray]:
        """Return the current bias as a numpy array, or None."""
        return self._current_bias.copy() if self._current_bias is not None else None
