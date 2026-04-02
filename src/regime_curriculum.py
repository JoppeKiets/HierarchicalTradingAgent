#!/usr/bin/env python3
"""Regime-Aware Curriculum Learning for Hierarchical Forecaster.

Pipeline
--------
1.  **Cluster** historical market periods into canonical regimes
    (bull, bear, choppy, crisis) using KMeans on the 8-dim regime-feature
    vectors already computed by ``_build_regime_dataframe``.

2.  **Label** every training sample with its regime cluster ID so
    per-regime IC can be tracked.

3.  **Oversample** regimes where a sub-model's IC is lowest: compute a
    weakness score per cluster and return ``torch.Tensor`` sample weights
    suitable for ``WeightedRandomSampler``.

4.  **Persist** the fitted cluster model alongside the feature cache so
    every run uses the same regime partition.

Public API
----------
- ``RegimeClusterer``         — fits / loads / persists the cluster model
- ``build_regime_curriculum`` — one-shot helper used by ``run_pipeline``
- ``compute_curriculum_weights`` — converts per-regime IC → sample weights
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical regime names
# ---------------------------------------------------------------------------

# Human-readable labels assigned to each cluster centre by heuristic analysis
# of SPY return (regime_spy_ret5) and VIX level (regime_vix_level) axes.
# The mapping is determined *after* fitting and can be overridden externally.

REGIME_LABEL_ORDER = ["bull_low_vol", "bull_high_vol", "choppy", "bear_low_vol", "bear_high_vol", "crisis"]


# ---------------------------------------------------------------------------
# RegimeClusterer
# ---------------------------------------------------------------------------

class RegimeClusterer:
    """KMeans-based market-regime clusterer.

    Fits on the 8-dim regime vectors produced by
    ``_build_regime_dataframe`` and assigns each trading date to one of
    ``n_regimes`` clusters, labelled with human-readable names.

    Usage::

        clusterer = RegimeClusterer(n_regimes=6)
        clusterer.fit(regime_df)      # pd.DataFrame indexed by date
        label = clusterer.label(ordinal_date)   # e.g. "bear_high_vol"
        all_map = clusterer.date_to_label       # dict[int, str]

    Persistence::

        clusterer.save(path)
        clusterer = RegimeClusterer.load(path)
    """

    # Feature names that drive the regime axis (subset of 8-dim vector)
    _KEY_FEATURES = [
        "regime_spy_ret5",    # trend signal
        "regime_spy_vol20",   # volatility signal
        "regime_vix_level",   # panic / fear signal
        "regime_spy_breadth", # market breadth
    ]

    def __init__(self, n_regimes: int = 6, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state

        self._km = None            # sklearn KMeans object
        self._cluster_to_label: Dict[int, str] = {}
        self._date_to_label: Dict[int, str] = {}    # ordinal → label
        self._label_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, regime_df) -> "RegimeClusterer":
        """Fit KMeans on ``regime_df`` (DateFrame indexed by calendar date).

        The DataFrame must contain at least the 4 key feature columns.
        Missing columns default to zero.  The full 8-dim vector is used
        for clustering; the 4-dim projection is used only for labelling.

        Parameters
        ----------
        regime_df : pd.DataFrame
            Output of ``_build_regime_dataframe``.  Index is ``datetime.date``
            or ``datetime``.  Columns include ``regime_spy_ret5`` etc.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import RobustScaler
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for RegimeClusterer. "
                "Install with: pip install scikit-learn"
            ) from e

        import pandas as pd

        if regime_df is None or regime_df.empty:
            logger.warning("RegimeClusterer.fit: empty regime_df — using dummy clusters")
            self._date_to_label = {}
            self._cluster_to_label = {i: f"regime_{i}" for i in range(self.n_regimes)}
            return self

        # Use all available regime columns (fall back gracefully if missing)
        feature_cols = [c for c in regime_df.columns if c.startswith("regime_")]
        if not feature_cols:
            logger.warning("No regime_ columns found — skipping clustering")
            return self

        X = regime_df[feature_cols].fillna(0.0).values.astype(np.float32)

        # Robust scaling before clustering
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=20,
            max_iter=500,
        )
        labels_int = km.fit_predict(X_scaled)

        self._km = km
        self._scaler = scaler
        self._feature_cols = feature_cols

        # Assign human-readable regime labels to each cluster
        self._cluster_to_label = self._label_clusters(
            km.cluster_centers_,
            feature_cols,
        )

        # Build ordinal-date → label mapping
        dates = regime_df.index
        self._date_to_label = {}
        for i, d in enumerate(dates):
            try:
                ordinal = d.toordinal() if hasattr(d, "toordinal") else int(d)
            except Exception:
                continue
            self._date_to_label[ordinal] = self._cluster_to_label[int(labels_int[i])]

        # Count samples per regime
        self._label_counts = {}
        for label in self._date_to_label.values():
            self._label_counts[label] = self._label_counts.get(label, 0) + 1

        logger.info(
            "RegimeClusterer fitted on %d dates, %d regimes: %s",
            len(dates),
            self.n_regimes,
            {lbl: cnt for lbl, cnt in sorted(self._label_counts.items())},
        )
        return self

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------

    def label(self, ordinal_date: int) -> str:
        """Return the regime label for an ordinal date (0 if not found)."""
        return self._date_to_label.get(ordinal_date, "unknown")

    def predict(self, regime_vector: np.ndarray) -> str:
        """Predict regime label for a raw 8-dim regime vector (inference)."""
        if self._km is None:
            return "unknown"
        v = regime_vector.reshape(1, -1).astype(np.float32)
        v_scaled = self._scaler.transform(v)
        cluster_id = int(self._km.predict(v_scaled)[0])
        return self._cluster_to_label.get(cluster_id, "unknown")

    @property
    def date_to_label(self) -> Dict[int, str]:
        return self._date_to_label

    @property
    def all_labels(self) -> List[str]:
        return sorted(set(self._cluster_to_label.values()))

    @property
    def label_counts(self) -> Dict[str, int]:
        return dict(self._label_counts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_regimes": self.n_regimes,
            "random_state": self.random_state,
            "cluster_to_label": self._cluster_to_label,
            "date_to_label": {str(k): v for k, v in self._date_to_label.items()},
            "label_counts": self._label_counts,
            "km": self._km,
            "scaler": getattr(self, "_scaler", None),
            "feature_cols": getattr(self, "_feature_cols", []),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("RegimeClusterer saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeClusterer":
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(n_regimes=payload["n_regimes"], random_state=payload["random_state"])
        obj._cluster_to_label = {int(k): v for k, v in payload["cluster_to_label"].items()}
        obj._date_to_label = {int(k): v for k, v in payload["date_to_label"].items()}
        obj._label_counts = payload.get("label_counts", {})
        obj._km = payload.get("km")
        obj._scaler = payload.get("scaler")
        obj._feature_cols = payload.get("feature_cols", [])
        logger.info("RegimeClusterer loaded from %s  (%d dates)", path, len(obj._date_to_label))
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_clusters(
        self,
        centres: np.ndarray,
        feature_cols: List[str],
    ) -> Dict[int, str]:
        """Assign human-readable labels to cluster centres.

        Heuristic rules on SPY return and VIX axes:
          - spy_ret5  (z-score): >+0.5 = bull, < -0.5 = bear, else choppy
          - vix_level (z-score): >+0.75 = high_vol, else low_vol
          - extreme drawdown (vix > +1.5 AND spy_ret5 < -1) = crisis
        """
        ret5_idx = self._safe_col_idx(feature_cols, "regime_spy_ret5")
        vix_idx  = self._safe_col_idx(feature_cols, "regime_vix_level")
        vol20_idx = self._safe_col_idx(feature_cols, "regime_spy_vol20")

        labels: Dict[int, str] = {}
        label_usage: Dict[str, int] = {}

        for k, centre in enumerate(centres):
            ret5  = float(centre[ret5_idx])  if ret5_idx  >= 0 else 0.0
            vix   = float(centre[vix_idx])   if vix_idx   >= 0 else 0.0
            vol20 = float(centre[vol20_idx]) if vol20_idx >= 0 else 0.0

            # Crisis: severe bear + panic vol
            if ret5 < -1.0 and vix > 1.0:
                base = "crisis"
            elif ret5 < -0.5:
                base = "bear_high_vol" if (vix > 0.5 or vol20 > 0.5) else "bear_low_vol"
            elif ret5 > 0.5:
                base = "bull_high_vol" if (vix > 0.3 or vol20 > 0.3) else "bull_low_vol"
            else:
                base = "choppy"

            # Deduplicate: append suffix if label already used
            cnt = label_usage.get(base, 0)
            label = base if cnt == 0 else f"{base}_{cnt}"
            label_usage[base] = cnt + 1
            labels[k] = label

        logger.debug("Cluster labels: %s", labels)
        return labels

    @staticmethod
    def _safe_col_idx(cols: List[str], name: str) -> int:
        try:
            return cols.index(name)
        except ValueError:
            return -1


# ---------------------------------------------------------------------------
# One-shot builder
# ---------------------------------------------------------------------------

def build_regime_clusterer(
    data_cfg,
    n_regimes: int = 6,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> RegimeClusterer:
    """Build or load a ``RegimeClusterer`` for the given data configuration.

    The fitted model is cached at ``{cache_dir}/regime_clusterer.pkl`` so
    repeated calls are free.  Pass ``force=True`` to refit.

    Parameters
    ----------
    data_cfg : HierarchicalDataConfig
        Used to call ``_build_regime_dataframe``.
    n_regimes : int
        Number of clusters (default 6).
    cache_dir : str | None
        Directory to cache the fitted model.  Defaults to
        ``data_cfg.cache_dir``.
    force : bool
        Refit even if a cached model exists.
    """
    from src.hierarchical_data import _build_regime_dataframe

    cache = Path(cache_dir or data_cfg.cache_dir)
    pkl_path = cache / "regime_clusterer.pkl"

    if pkl_path.exists() and not force:
        try:
            return RegimeClusterer.load(pkl_path)
        except Exception as e:
            logger.warning("Could not load cached clusterer (%s) — refitting", e)

    regime_df = _build_regime_dataframe(data_cfg)
    clusterer = RegimeClusterer(n_regimes=n_regimes)
    clusterer.fit(regime_df)
    clusterer.save(pkl_path)
    return clusterer


# ---------------------------------------------------------------------------
# Curriculum weight computation
# ---------------------------------------------------------------------------

def compute_curriculum_weights(
    per_regime_ic: Dict[str, Dict[str, float]],
    base_weight: float = 1.0,
    max_weight: float = 5.0,
    ic_key: str = "ic",
    da_key: str = "directional_accuracy",
) -> Dict[str, float]:
    """Convert per-regime performance metrics → training sample weights.

    Regimes where the model performs *worst* (lowest IC, lowest DA) receive
    the *highest* weight so the next training epoch focuses on them.

    Parameters
    ----------
    per_regime_ic : dict
        ``{regime_label: {"ic": float, "directional_accuracy": float, ...}}``
    base_weight : float
        Weight assigned to the best-performing regime (default 1.0).
    max_weight : float
        Maximum weight for the worst-performing regime (default 5.0).

    Returns
    -------
    Dict[str, float]
        ``{regime_label: weight}`` — higher means the model is weaker there.
    """
    if not per_regime_ic:
        return {}

    # Weakness score per regime: higher = model struggles more
    weakness: Dict[str, float] = {}
    for regime, m in per_regime_ic.items():
        ic = m.get(ic_key, 0.0)
        da = m.get(da_key, 0.5)
        # Both axes contribute: bad IC and low DA both increase weakness
        w = (1.0 - da) + (1.0 - min(abs(ic), 1.0))
        weakness[regime] = w

    min_w = min(weakness.values())
    max_w = max(weakness.values())
    spread = max_w - min_w if max_w > min_w else 1.0

    return {
        regime: base_weight + ((w - min_w) / spread) * (max_weight - base_weight)
        for regime, w in weakness.items()
    }


# ---------------------------------------------------------------------------
# Per-sample weight tensor builder
# ---------------------------------------------------------------------------

def build_sample_weight_tensor(
    ordinal_dates: List[int],
    clusterer: RegimeClusterer,
    regime_weights: Dict[str, float],
    default_weight: float = 1.0,
) -> "torch.Tensor":  # type: ignore[name-defined]
    """Build a (N,) float32 weight tensor aligned to a list of ordinal dates.

    Parameters
    ----------
    ordinal_dates : list[int]
        Per-sample ordinal dates from the dataset index.
    clusterer : RegimeClusterer
        Fitted clusterer to map date → regime label.
    regime_weights : dict[str, float]
        Per-regime sample weights from ``compute_curriculum_weights``.
    default_weight : float
        Fallback for dates with unknown regime.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)`` float32.
    """
    import torch

    weights = []
    for od in ordinal_dates:
        label = clusterer.label(od)
        w = regime_weights.get(label, default_weight)
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Regime-aware DataLoader factory
# ---------------------------------------------------------------------------

def make_regime_weighted_loader(
    dataset,
    ordinal_dates: List[int],
    clusterer: RegimeClusterer,
    regime_weights: Dict[str, float],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> "DataLoader":  # type: ignore[name-defined]
    """Return a DataLoader that oversamples under-performing regimes.

    Uses ``WeightedRandomSampler`` with replacement so the expected number
    of samples per epoch equals ``len(dataset)``.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    ordinal_dates : list[int]
        Ordinal date for each item in ``dataset`` (same order).
    clusterer : RegimeClusterer
    regime_weights : dict[str, float]
        From ``compute_curriculum_weights``.
    batch_size, num_workers, pin_memory : forwarded to DataLoader.
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    sample_weights = build_sample_weight_tensor(
        ordinal_dates, clusterer, regime_weights
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),  # double required by PyTorch sampler
        num_samples=len(dataset),
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=3 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# Per-regime IC evaluation helper (standalone, no trainer dependency)
# ---------------------------------------------------------------------------

def evaluate_per_regime(
    preds: np.ndarray,
    targets: np.ndarray,
    ordinal_dates: np.ndarray,
    clusterer: RegimeClusterer,
    min_samples: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Compute IC, rank-IC, and DA broken down by regime cluster.

    Parameters
    ----------
    preds : np.ndarray  shape (N,)
    targets : np.ndarray  shape (N,)
    ordinal_dates : np.ndarray  shape (N,) int32/int64
    clusterer : RegimeClusterer
    min_samples : int
        Minimum samples in a regime to report meaningful metrics.

    Returns
    -------
    dict
        ``{regime_label: {"ic": float, "directional_accuracy": float,
                          "rank_ic": float, "n_samples": int}}``
    """
    from scipy import stats

    regime_preds: Dict[str, List[float]] = {}
    regime_tgts: Dict[str, List[float]] = {}

    for pred, tgt, od in zip(preds, targets, ordinal_dates):
        label = clusterer.label(int(od))
        regime_preds.setdefault(label, []).append(float(pred))
        regime_tgts.setdefault(label, []).append(float(tgt))

    result: Dict[str, Dict[str, float]] = {}
    for label in regime_preds:
        p = np.array(regime_preds[label])
        t = np.array(regime_tgts[label])
        n = len(p)

        if n < min_samples:
            continue

        ic = (
            float(np.corrcoef(p, t)[0, 1])
            if np.std(p) > 1e-10 and np.std(t) > 1e-10
            else 0.0
        )
        if n >= min_samples:
            _sr = stats.spearmanr(p, t)
            ric = float(_sr.statistic) if hasattr(_sr, "statistic") else float(_sr[0])  # type: ignore[index]
        else:
            ric = 0.0
        da = float(np.mean((p > 0) == (t > 0)))

        result[label] = {
            "ic": ic if np.isfinite(ic) else 0.0,
            "rank_ic": ric if np.isfinite(ric) else 0.0,
            "directional_accuracy": da,
            "n_samples": n,
        }

    return result


# ---------------------------------------------------------------------------
# Compact logging helper
# ---------------------------------------------------------------------------

def log_regime_stats(
    model_name: str,
    per_regime_ic: Dict[str, Dict[str, float]],
    regime_weights: Dict[str, float],
) -> None:
    """Log a compact table of regime IC + assigned curriculum weights."""
    logger.info("  ── Regime curriculum weights for [%s] ──", model_name)
    if not per_regime_ic:
        logger.info("    (no per-regime data available)")
        return
    for label in sorted(per_regime_ic.keys()):
        m = per_regime_ic[label]
        w = regime_weights.get(label, 1.0)
        logger.info(
            "    %-22s  IC=%+.4f  DA=%.2f%%  n=%4d  weight=%.2f",
            label,
            m.get("ic", 0.0),
            m.get("directional_accuracy", 0.5) * 100,
            m.get("n_samples", 0),
            w,
        )
