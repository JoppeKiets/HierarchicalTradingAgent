"""Feature trust tracker — aggregates Analyst feature assessments over time.

After each Analyst run, the Analyst flags features as suspicious or supportive.
This module aggregates those signals across multiple runs (typically weeks of
paper trading) to build a long-term trust score per feature.

Trust scores are persisted to disk so they survive restarts and can be used by:
  - The Analyst: to automatically down-weight or drop distrusted features
  - The training pipeline: to mask low-trust features in the feature tensor

Persistence layout:
    data/feature_feedback/
        trust_scores.json           # Latest aggregated trust scores
        trust_history.jsonl         # Append-only log of every run's assessments
        feature_weights.json        # Derived multiplicative weights for training
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeatureTrustTracker:
    """Maintains rolling trust scores for every feature the Analyst sees.

    Each feature has three counters:
      - n_suspicious:  times flagged as suspicious
      - n_supportive:  times flagged as supportive
      - n_seen:        total times the feature was observed

    Trust score = (n_supportive - n_suspicious) / n_seen ∈ [-1, 1]

    Feature weight (for training) = max(0, (trust_score + 1) / 2)
        → 0.0 when trust = -1  (fully distrusted — drop)
        → 0.5 when trust =  0  (neutral)
        → 1.0 when trust = +1  (fully trusted)

    A configurable ``drop_threshold`` (default -0.5) determines which
    features to recommend for removal entirely.
    """

    def __init__(
        self,
        feedback_dir: str = "data/feature_feedback",
        drop_threshold: float = -0.5,
        decay_factor: float = 0.95,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.drop_threshold = drop_threshold
        self.decay_factor = decay_factor  # Exponential decay on old counts

        # Per-feature counters
        self.counters: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"n_suspicious": 0.0, "n_supportive": 0.0, "n_seen": 0.0}
        )

        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self):
        """Load existing trust state from disk."""
        path = self.feedback_dir / "trust_scores.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                for feat, counts in data.get("counters", {}).items():
                    self.counters[feat] = {
                        "n_suspicious": counts.get("n_suspicious", 0.0),
                        "n_supportive": counts.get("n_supportive", 0.0),
                        "n_seen": counts.get("n_seen", 0.0),
                    }
                logger.info(
                    "Loaded feature trust scores for %d features from %s",
                    len(self.counters), path,
                )
            except Exception as e:
                logger.warning("Failed to load trust scores: %s", e)

    def save(self):
        """Persist current state to disk."""
        path = self.feedback_dir / "trust_scores.json"
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_features": len(self.counters),
            "counters": dict(self.counters),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Also save derived weights
        self.save_feature_weights()

    def save_feature_weights(self):
        """Save multiplicative feature weights for the training pipeline."""
        weights = self.get_feature_weights()
        path = self.feedback_dir / "feature_weights.json"
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_features": len(weights),
            "drop_threshold": self.drop_threshold,
            "weights": weights,
            "dropped_features": self.get_dropped_features(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Update ────────────────────────────────────────────────────────

    def record_run(
        self,
        suspicious_features: List[str],
        supportive_features: List[str],
        all_features: List[str],
        ticker: str = "",
        run_id: str = "",
    ):
        """Record one Analyst run's feature assessments.

        Args:
            suspicious_features:  Feature names the Analyst flagged as suspicious
            supportive_features:  Feature names the Analyst flagged as supportive
            all_features:         All feature names observed in this run
            ticker:               Ticker symbol (for logging)
            run_id:               Pipeline run identifier
        """
        # Apply exponential decay to existing counts (ages old evidence)
        for feat in self.counters:
            self.counters[feat]["n_suspicious"] *= self.decay_factor
            self.counters[feat]["n_supportive"] *= self.decay_factor
            self.counters[feat]["n_seen"] *= self.decay_factor

        # Update counters with new observations
        for feat in all_features:
            self.counters[feat]["n_seen"] += 1.0
        for feat in suspicious_features:
            self.counters[feat]["n_suspicious"] += 1.0
        for feat in supportive_features:
            self.counters[feat]["n_supportive"] += 1.0

        # Append to history log
        self._append_history(suspicious_features, supportive_features,
                             all_features, ticker, run_id)

    def _append_history(
        self,
        suspicious: List[str],
        supportive: List[str],
        all_feats: List[str],
        ticker: str,
        run_id: str,
    ):
        """Append one entry to the JSONL history log."""
        history_path = self.feedback_dir / "trust_history.jsonl"
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "ticker": ticker,
            "n_suspicious": len(suspicious),
            "n_supportive": len(supportive),
            "n_total": len(all_feats),
            "suspicious": suspicious,
            "supportive": supportive,
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Query ─────────────────────────────────────────────────────────

    def get_trust_score(self, feature: str) -> float:
        """Trust score for a single feature ∈ [-1, 1].

        Returns 0.0 (neutral) for unknown features.
        """
        c = self.counters.get(feature)
        if c is None or c["n_seen"] < 1e-6:
            return 0.0
        return (c["n_supportive"] - c["n_suspicious"]) / c["n_seen"]

    def get_all_trust_scores(self) -> Dict[str, float]:
        """Trust scores for all tracked features."""
        return {feat: self.get_trust_score(feat) for feat in self.counters}

    def get_feature_weights(self) -> Dict[str, float]:
        """Multiplicative weights ∈ [0, 1] for training.

        weight = max(0, (trust_score + 1) / 2)
        """
        weights = {}
        for feat in self.counters:
            ts = self.get_trust_score(feat)
            weights[feat] = max(0.0, (ts + 1.0) / 2.0)
        return weights

    def get_dropped_features(self) -> List[str]:
        """Features below ``drop_threshold`` → recommend dropping."""
        dropped = []
        for feat in self.counters:
            if self.get_trust_score(feat) < self.drop_threshold:
                dropped.append(feat)
        return sorted(dropped)

    def get_suspicious_ranking(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-K most distrusted features (lowest trust scores)."""
        scores = self.get_all_trust_scores()
        ranked = sorted(scores.items(), key=lambda x: x[1])
        return ranked[:top_k]

    def get_supportive_ranking(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-K most trusted features (highest trust scores)."""
        scores = self.get_all_trust_scores()
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def summary(self) -> Dict[str, Any]:
        """Quick summary for logging."""
        scores = self.get_all_trust_scores()
        if not scores:
            return {"n_features": 0}

        vals = list(scores.values())
        dropped = self.get_dropped_features()
        return {
            "n_features": len(scores),
            "n_dropped": len(dropped),
            "trust_mean": round(sum(vals) / len(vals), 3),
            "trust_min": round(min(vals), 3),
            "trust_max": round(max(vals), 3),
            "dropped_features": dropped[:10],  # First 10 for brevity
        }
