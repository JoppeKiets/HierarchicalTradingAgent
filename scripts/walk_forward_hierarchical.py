#!/usr/bin/env python3
"""Walk-forward validation for the Hierarchical Forecaster.

Trains and evaluates the model over multiple time windows to measure
robustness and consistency across different market regimes.

Strategy:
  1. Divide the timeline into K overlapping windows.
  2. For each window:
     a. Train on the first 70% of the window
     b. Validate on the next 15%
     c. Test on the final 15%
  3. Report per-window and aggregate metrics.

This gives a realistic estimate of how the model would perform if
retrained periodically (e.g., monthly).

Usage:
    # Default: 5 windows, phases 1-3
    python scripts/walk_forward_hierarchical.py

    # Quick test: 2 windows, fewer epochs
    python scripts/walk_forward_hierarchical.py --n-windows 2 --epochs 5

    # Full run with Phase 4
    python scripts/walk_forward_hierarchical.py --n-windows 5 --phases 0 1 2 3 4
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hierarchical_data import (
    HierarchicalDataConfig,
    LazyDailyDataset,
    LazyMinuteDataset,
    create_dataloaders,
    get_viable_tickers,
    preprocess_all,
    _build_regime_dataframe,
    REGIME_FEATURE_NAMES,
)
from src.news_data import LazyNewsDataset, NewsDataConfig
from src.hierarchical_models import (
    HierarchicalForecaster,
    HierarchicalModelConfig,
)
from agents.feedback.attention_prior import AttentionPriorComputer

logger = logging.getLogger(__name__)


# ============================================================================
# Walk-forward window dataset
# ============================================================================

class WindowDailyDataset(LazyDailyDataset):
    """LazyDailyDataset restricted to a specific ordinal date range."""

    def __init__(
        self,
        tickers: List[str],
        cfg: HierarchicalDataConfig,
        date_lo: int,
        date_hi: int,
    ):
        # Bypass parent __init__ to do our own filtering
        self.cfg = cfg
        self.cache = Path(cfg.cache_dir) / "daily"
        self.split_name = "custom"
        self.index = []
        self.n_features = 0

        warmup = max(cfg.daily_seq_len, cfg.daily_norm_window, 60)

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            tgt_path = self.cache / f"{ticker}_targets.npy"
            date_path = self.cache / f"{ticker}_dates.npy"
            if not feat_path.exists() or not tgt_path.exists() or not date_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            tgt = np.load(tgt_path, mmap_mode="r")
            dates = np.load(date_path, mmap_mode="r")
            n_rows, n_feat = feat.shape
            self.n_features = n_feat

            for i in range(warmup, n_rows - cfg.forecast_horizon, cfg.daily_stride):
                if i - cfg.daily_seq_len >= 0 and not np.isnan(tgt[i]):
                    d = int(dates[i])
                    if date_lo <= d < date_hi:
                        self.index.append((ticker, i))

    def _temporal_bounds(self, warmup, end, cfg):
        # Not used — we filter by date in __init__
        return warmup, end


class WindowMinuteDataset(LazyMinuteDataset):
    """LazyMinuteDataset restricted to a specific ordinal date range."""

    def __init__(
        self,
        tickers: List[str],
        cfg: HierarchicalDataConfig,
        date_lo: int,
        date_hi: int,
    ):
        self.cfg = cfg
        self.cache = Path(cfg.cache_dir) / "minute"
        self.split_name = "custom"
        self.index = []
        self.n_features = 0

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            tgt_path = self.cache / f"{ticker}_targets.npy"
            date_path = self.cache / f"{ticker}_dates.npy"
            if not feat_path.exists() or not tgt_path.exists() or not date_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            tgt = np.load(tgt_path, mmap_mode="r")
            dates = np.load(date_path, mmap_mode="r")
            n_rows, n_feat = feat.shape
            self.n_features = n_feat

            for i in range(cfg.minute_seq_len, n_rows, cfg.minute_stride):
                if not np.isnan(tgt[i - 1]):
                    d = int(dates[i - 1])
                    if date_lo <= d < date_hi:
                        self.index.append((ticker, i))

    def _temporal_bounds(self, warmup, n_rows, cfg):
        return warmup, n_rows


class WindowNewsDataset(LazyNewsDataset):
    """LazyNewsDataset restricted to a specific ordinal date range."""

    def __init__(
        self,
        tickers: List[str],
        news_cfg: NewsDataConfig,
        date_lo: int,
        date_hi: int,
        daily_cache_dir: str = "data/feature_cache/daily",
    ):
        self.news_cfg = news_cfg
        self.cache = Path(news_cfg.news_seq_cache_dir)
        self.daily_cache = Path(daily_cache_dir)
        self.split_name = "custom"
        self.index = []
        self.n_features = news_cfg.total_dim

        warmup = max(news_cfg.seq_len, news_cfg.norm_window, 60)

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            daily_tgt_path = self.daily_cache / f"{ticker}_targets.npy"
            date_path = self.cache / f"{ticker}_dates.npy"
            if not feat_path.exists() or not daily_tgt_path.exists() or not date_path.exists():
                continue
            feat = np.load(feat_path, mmap_mode="r")
            if np.abs(feat).sum() < 1e-6:
                continue
            n_rows = feat.shape[0]
            tgt = np.load(daily_tgt_path, mmap_mode="r")
            dates = np.load(date_path, mmap_mode="r")
            end = n_rows - news_cfg.forecast_horizon
            for i in range(warmup, end, news_cfg.stride):
                if i - news_cfg.seq_len >= 0 and not np.isnan(tgt[i]):
                    d = int(dates[i])
                    if date_lo <= d < date_hi:
                        self.index.append((ticker, i))

    def _temporal_bounds(self, warmup, end, cfg):
        return warmup, end


def get_minute_date_range(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
) -> Optional[Tuple[int, int]]:
    """Return (min_ordinal, max_ordinal) of all cached minute data, or None."""
    cache = Path(cfg.cache_dir) / "minute"
    all_dates = []
    for ticker in tickers:
        date_path = cache / f"{ticker}_dates.npy"
        if not date_path.exists():
            continue
        dates = np.load(date_path, mmap_mode="r")
        nonzero = dates[dates > 0]
        if len(nonzero):
            all_dates.extend(nonzero.tolist())
    if not all_dates:
        return None
    return int(min(all_dates)), int(max(all_dates))


def make_minute_loaders(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    batch_size_minute: int = 64,
    num_workers: int = 4,
    purge_horizon: int = 5,
) -> Optional[Dict]:
    """Create train/val/test DataLoaders covering the full minute data range.

    The split is done temporally on the minute date range, completely
    independently of the daily walk-forward windows.

    Returns None if there is insufficient minute data (<50 train samples).
    """
    from torch.utils.data import DataLoader

    date_range = get_minute_date_range(tickers, cfg)
    if date_range is None:
        logger.warning("  make_minute_loaders: no minute cache found")
        return None

    lo, hi = date_range
    span = hi - lo
    train_end  = lo + int(span * train_frac) - purge_horizon
    val_start  = lo + int(span * train_frac) + purge_horizon
    val_end    = lo + int(span * (train_frac + val_frac))

    import datetime
    logger.info(
        f"  Minute date range: {datetime.date.fromordinal(lo)} → {datetime.date.fromordinal(hi)} "
        f"({span} days)"
    )
    logger.info(
        f"  Minute split:  train → {datetime.date.fromordinal(train_end)}"
        f"  |  val {datetime.date.fromordinal(val_start)} → {datetime.date.fromordinal(val_end)}"
        f"  |  test → {datetime.date.fromordinal(hi)}"
    )

    result: Dict = {"minute": {}}
    splits = [("train", lo, train_end), ("val", val_start, val_end), ("test", val_end, hi)]
    for split_name, s_lo, s_hi in splits:
        shuffle = split_name == "train"
        ds = WindowMinuteDataset(tickers, cfg, s_lo, s_hi)
        result["minute"][split_name] = DataLoader(
            ds,
            batch_size=batch_size_minute,
            shuffle=(shuffle and len(ds) > 0),
            num_workers=num_workers if len(ds) > 0 else 0,
            pin_memory=len(ds) > 0,
            drop_last=(shuffle and len(ds) > 0),
        )

    n_train = len(result["minute"]["train"].dataset)
    n_val   = len(result["minute"]["val"].dataset)
    n_test  = len(result["minute"]["test"].dataset)
    result["minute_n_features"] = max(
        result["minute"]["train"].dataset.n_features,
        result["minute"]["val"].dataset.n_features,
        1,
    )
    logger.info(f"  Minute loaders: train={n_train}, val={n_val}, test={n_test}, "
                f"n_features={result['minute_n_features']}")

    if n_train < 50:
        logger.warning(f"  Insufficient minute train samples ({n_train}), skipping minute loaders")
        return None

    return result


def make_window_loaders(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
    train_dates: Tuple[int, int],
    val_dates: Tuple[int, int],
    test_dates: Tuple[int, int],
    batch_size_daily: int = 128,
    batch_size_minute: int = 64,
    num_workers: int = 4,
) -> Dict:
    """Create DataLoaders for a single walk-forward window."""
    from torch.utils.data import DataLoader

    result = {"daily": {}, "minute": {}, "news": {}}

    news_cfg = NewsDataConfig()
    daily_cache_dir_str = str(Path(cfg.cache_dir) / "daily")

    for split_name, (lo, hi) in [
        ("train", train_dates), ("val", val_dates), ("test", test_dates),
    ]:
        shuffle = split_name == "train"

        daily_ds = WindowDailyDataset(tickers, cfg, lo, hi)
        minute_ds = WindowMinuteDataset(tickers, cfg, lo, hi)
        news_ds = WindowNewsDataset(tickers, news_cfg, lo, hi, daily_cache_dir=daily_cache_dir_str)

        # Handle empty datasets gracefully (minute data is often sparse)
        result["daily"][split_name] = DataLoader(
            daily_ds, batch_size=batch_size_daily,
            shuffle=(shuffle and len(daily_ds) > 0),
            num_workers=num_workers if len(daily_ds) > 0 else 0,
            pin_memory=len(daily_ds) > 0, drop_last=(shuffle and len(daily_ds) > 0),
        )
        result["minute"][split_name] = DataLoader(
            minute_ds, batch_size=batch_size_minute,
            shuffle=(shuffle and len(minute_ds) > 0),
            num_workers=num_workers if len(minute_ds) > 0 else 0,
            pin_memory=len(minute_ds) > 0, drop_last=(shuffle and len(minute_ds) > 0),
        )
        result["news"][split_name] = DataLoader(
            news_ds, batch_size=batch_size_daily,
            shuffle=(shuffle and len(news_ds) > 0),
            num_workers=num_workers if len(news_ds) > 0 else 0,
            pin_memory=len(news_ds) > 0, drop_last=False,
        ) if len(news_ds) > 0 else None

    result["daily_n_features"] = max(
        result["daily"]["train"].dataset.n_features,
        result["daily"]["val"].dataset.n_features,
        1,
    )
    result["minute_n_features"] = max(
        result["minute"]["train"].dataset.n_features,
        result["minute"]["val"].dataset.n_features,
        1,
    )

    return result


# ============================================================================
# Compute date ranges for walk-forward windows
# ============================================================================

def compute_windows(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
    n_windows: int = 5,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    expanding_window: bool = True,
    purge_horizon: int = 0,
) -> List[Dict[str, Tuple[int, int]]]:
    """Compute walk-forward windows from the available date range.

    Implements expanding windows (or rolling if expanding_window=False) with
    critical purging to prevent leakage.

    Args:
        tickers: List of stock tickers
        cfg: Data configuration
        n_windows: Number of windows
        train_frac, val_frac: Split fractions
        expanding_window: If True, fold 0 uses earliest data, fold N uses all data (expanding).
                         If False, windows roll forward with fixed span (rolling).
        purge_horizon: Days to purge from end of train and start of val to prevent leakage.
                       If 0, defaults to cfg.forecast_horizon.

    Returns a list of dicts, each with 'train', 'val', 'test' date tuples
    (ordinal_lo, ordinal_hi) with purging applied.
    """
    if purge_horizon == 0:
        purge_horizon = cfg.forecast_horizon

    # Find the overall date range from cached data (daily + minute)
    all_dates = set()
    for subdir in ["daily", "minute"]:
        cache_dir = Path(cfg.cache_dir) / subdir
        if not cache_dir.exists():
            continue
        for ticker in tickers:
            date_path = cache_dir / f"{ticker}_dates.npy"
            if date_path.exists():
                dates = np.load(date_path, mmap_mode="r")
                nonzero = dates[dates > 0]
                if len(nonzero) > 0:
                    all_dates.update(nonzero.tolist())

    if not all_dates:
        logger.error("No date data found in cache")
        return []

    sorted_dates = sorted(all_dates)
    min_date, max_date = sorted_dates[0], sorted_dates[-1]
    total_span = max_date - min_date

    logger.info(f"Date range: ordinal {min_date}—{max_date} "
                f"({total_span} days, ~{total_span/365:.1f} years)")
    logger.info(f"Purge horizon: {purge_horizon} days (forecast_horizon={cfg.forecast_horizon})")

    windows = []
    
    if expanding_window:
        # EXPANDING WINDOWS: Fold 0 = earliest data, Fold N = all data
        # Each fold expands from the start
        for w in range(n_windows):
            # Fold w spans from min_date to a progressively later end date
            frac = (w + 1) / n_windows  # 0.2, 0.4, 0.6, 0.8, 1.0 for 5 windows
            w_start = min_date
            w_end = min_date + int(total_span * frac)
            w_end = min(w_end, max_date)

            w_span = w_end - w_start
            train_end_base = w_start + int(w_span * train_frac)
            val_end_base = train_end_base + int(w_span * val_frac)

            # PURGE: Remove last purge_horizon days from train, first purge_horizon from val
            train_end_purged = train_end_base - purge_horizon
            val_start_purged = train_end_base + purge_horizon

            windows.append({
                "train": (w_start, train_end_purged),
                "val": (val_start_purged, val_end_base),
                "test": (val_end_base, w_end),
            })

            import datetime
            logger.info(f"  Window {w} (EXPANDING): "
                        f"train={datetime.date.fromordinal(w_start)}→"
                        f"{datetime.date.fromordinal(train_end_purged)}, "
                        f"val={datetime.date.fromordinal(val_start_purged)}→"
                        f"{datetime.date.fromordinal(val_end_base)}, "
                        f"test→{datetime.date.fromordinal(min(w_end, 999999))}")
    else:
        # ROLLING WINDOWS: Fixed span, shifting forward
        window_span = int(total_span * 0.6)   # Each window covers 60% of data
        step = (total_span - window_span) // max(n_windows - 1, 1)

        for w in range(n_windows):
            w_start = min_date + w * step
            w_end = w_start + window_span
            w_end = min(w_end, max_date)

            w_span = w_end - w_start
            train_end_base = w_start + int(w_span * train_frac)
            val_end_base = train_end_base + int(w_span * val_frac)

            # PURGE
            train_end_purged = train_end_base - purge_horizon
            val_start_purged = train_end_base + purge_horizon

            windows.append({
                "train": (w_start, train_end_purged),
                "val": (val_start_purged, val_end_base),
                "test": (val_end_base, w_end),
            })

            import datetime
            logger.info(f"  Window {w} (ROLLING): "
                        f"train={datetime.date.fromordinal(w_start)}→"
                        f"{datetime.date.fromordinal(train_end_purged)}, "
                        f"val={datetime.date.fromordinal(val_start_purged)}→"
                        f"{datetime.date.fromordinal(val_end_base)}, "
                        f"test→{datetime.date.fromordinal(min(w_end, 999999))}")

    return windows


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    from scipy import stats

    if len(preds) < 5:
        return {"ic": 0.0, "rank_ic": 0.0, "directional_accuracy": 0.5,
                "mse": 0.0, "n_samples": len(preds)}

    mse = float(np.mean((preds - targets) ** 2))
    if np.std(preds) < 1e-10 or np.std(targets) < 1e-10:
        ic, ric = 0.0, 0.0
    else:
        ic = float(np.corrcoef(preds, targets)[0, 1])
        ric = float(stats.spearmanr(preds, targets).correlation)

    dir_acc = float(np.mean((preds > 0) == (targets > 0)))

    return {"ic": ic, "rank_ic": ric, "directional_accuracy": dir_acc,
            "mse": mse, "n_samples": len(preds)}


# ============================================================================
# Compute overlapping dates between daily and minute data
# ============================================================================

def compute_overlapping_dates(
    daily_loader,
    minute_loader,
) -> Tuple[set, set, set]:
    """Find dates that have both daily and minute data.
    
    Args:
        daily_loader: DataLoader for daily data
        minute_loader: DataLoader for minute data
    
    Returns:
        (daily_dates, minute_dates, overlap_dates) where overlap_dates = daily ∩ minute
    """
    daily_dates = set()
    minute_dates = set()
    
    # Collect daily dates
    for batch in daily_loader:
        _, _, _, ordinal_dates, _ = batch
        for od in ordinal_dates:
            daily_dates.add(int(od))
    
    # Collect minute dates
    for batch in minute_loader:
        _, _, ordinal_dates, _ = batch
        for od in ordinal_dates:
            minute_dates.add(int(od))
    
    # Find intersection
    overlap_dates = daily_dates & minute_dates
    
    logger.info(
        f"Date overlap: {len(daily_dates)} daily dates, "
        f"{len(minute_dates)} minute dates, {len(overlap_dates)} overlapping"
    )
    
    return daily_dates, minute_dates, overlap_dates


# ============================================================================
# Collect predictions (same logic as evaluate_hierarchical.py)
# ============================================================================

@torch.no_grad()
def collect_test_predictions(
    forecaster: HierarchicalForecaster,
    daily_loader, minute_loader,
    data_cfg: HierarchicalDataConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full model and return (preds, targets, dates)."""
    forecaster.eval()
    E = forecaster.cfg.embedding_dim
    R = forecaster.cfg.regime_dim
    n_sub = forecaster.meta.n_sub_models

    regime_df = _build_regime_dataframe(data_cfg)
    regime_lookup = {}
    if not regime_df.empty:
        for d, row in regime_df.iterrows():
            if hasattr(d, 'toordinal'):
                regime_lookup[d.toordinal()] = row.values.astype(np.float32)

    # Collect daily model outputs
    daily_data = {}
    for batch in daily_loader:
        x, y, raw_y, ordinal_dates, tickers_batch = batch
        x = x.to(device)
        d1 = forecaster.lstm_d(x)
        d2 = forecaster.tft_d(x)
        for i in range(len(y)):
            key = (tickers_batch[i], int(ordinal_dates[i]))
            daily_data[key] = {
                "predictions": {
                    "lstm_d": d1["prediction"][i].cpu(),
                    "tft_d": d2["prediction"][i].cpu(),
                },
                "embeddings": {
                    "lstm_d": d1["embedding"][i].cpu(),
                    "tft_d": d2["embedding"][i].cpu(),
                },
                "target": y[i],          # z-scored (for IC computation)
                "raw_target": raw_y[i],  # raw return (for P&L backtest)
            }
        # Add TCN_D if present
        if hasattr(forecaster, "tcn_d"):
            d3 = forecaster.tcn_d(x)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                daily_data[key]["predictions"]["tcn_d"] = d3["prediction"][i].cpu()
                daily_data[key]["embeddings"]["tcn_d"] = d3["embedding"][i].cpu()

    # Collect minute model outputs (only if minute models are in the forecaster)
    minute_data = {}
    if minute_loader is not None and "lstm_m" in forecaster.sub_model_names:
        for batch in minute_loader:
            x, y, _, ordinal_dates, tickers_batch = batch
            x = x.to(device)
            m1 = forecaster.lstm_m(x)
            m2 = forecaster.tft_m(x)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                minute_data[key] = {
                    "predictions": {
                        "lstm_m": m1["prediction"][i].cpu(),
                        "tft_m": m2["prediction"][i].cpu(),
                    },
                    "embeddings": {
                        "lstm_m": m1["embedding"][i].cpu(),
                        "tft_m": m2["embedding"][i].cpu(),
                    },
                }

    zero_emb = torch.zeros(E)
    aligned_preds, aligned_embs, aligned_regimes = [], [], []
    aligned_tgts, aligned_raw_tgts, aligned_dates = [], [], []

    # Build prediction and embedding vectors according to forecaster's sub_model_names
    for key, dd in daily_data.items():
        _, ord_date = key
        md = minute_data.get(key, {})

        # Build pred/emb vectors in the order specified by forecaster.sub_model_names
        pred_list = []
        emb_list = []
        for model_name in forecaster.sub_model_names:
            if model_name in dd["predictions"]:
                pred_list.append(dd["predictions"][model_name])
                emb_list.append(dd["embeddings"][model_name])
            elif model_name in md.get("predictions", {}):
                pred_list.append(md["predictions"][model_name])
                emb_list.append(md["embeddings"][model_name])
            else:
                # Model prediction not available (e.g., news, fund_mlp not in this pipeline)
                pred_list.append(torch.tensor(0.0))
                emb_list.append(zero_emb)

        pred_vec = torch.stack(pred_list)  # (n_sub_models,)
        emb_vec = torch.cat(emb_list)      # (n_sub_models * E,)

        regime_vec = (torch.from_numpy(regime_lookup[ord_date])
                      if ord_date in regime_lookup else torch.zeros(R))

        aligned_preds.append(pred_vec)
        aligned_embs.append(emb_vec)
        aligned_regimes.append(regime_vec)
        aligned_tgts.append(dd["target"].item())
        aligned_raw_tgts.append(dd.get("raw_target", dd["target"]).item())
        aligned_dates.append(ord_date)

    if not aligned_preds:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Batched meta inference
    all_pred_t = torch.stack(aligned_preds)
    all_emb_t = torch.stack(aligned_embs)
    all_regime_t = torch.stack(aligned_regimes)

    preds_out = []
    batch_size = 1024
    forecaster.meta.eval()
    for start in range(0, len(all_pred_t), batch_size):
        end = min(start + batch_size, len(all_pred_t))
        p_b = all_pred_t[start:end].to(device)
        e_b = all_emb_t[start:end].to(device)
        r_b = all_regime_t[start:end].to(device)
        emb_list = [e_b[:, i*E:(i+1)*E] for i in range(n_sub)]  # Dynamic number of sub-models
        meta_out = forecaster.meta(p_b, emb_list, r_b)
        preds_out.append(meta_out["prediction"].cpu().numpy())

    return (
        np.concatenate(preds_out),
        np.array(aligned_tgts),        # z-scored targets (for IC)
        np.array(aligned_raw_tgts),    # raw returns (for P&L backtest)
        np.array(aligned_dates),
    )


@torch.no_grad()
def collect_sub_model_predictions(
    forecaster: HierarchicalForecaster,
    daily_loader, minute_loader,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Collect per-sub-model prediction arrays for attention prior computation.

    Returns:
        sub_preds:  {model_name: (N,) array of scalar predictions}
        targets:    (N,) array of actual returns
    """
    forecaster.eval()
    E = forecaster.cfg.embedding_dim

    # Collect daily model outputs keyed by (ticker, ord_date)
    daily_results: Dict[Tuple, Dict] = {}
    for batch in daily_loader:
        x, y, _, ordinal_dates, tickers_batch = batch
        x = x.to(device)
        d1 = forecaster.lstm_d(x)
        d2 = forecaster.tft_d(x)
        for i in range(len(y)):
            key = (tickers_batch[i], int(ordinal_dates[i]))
            daily_results[key] = {
                "lstm_d": float(d1["prediction"][i].cpu()),
                "tft_d": float(d2["prediction"][i].cpu()),
                "target": float(y[i]),
            }

    # Collect minute model outputs
    minute_results: Dict[Tuple, Dict] = {}
    for batch in minute_loader:
        x, y, _, ordinal_dates, tickers_batch = batch
        x = x.to(device)
        m1 = forecaster.lstm_m(x)
        m2 = forecaster.tft_m(x)
        for i in range(len(y)):
            key = (tickers_batch[i], int(ordinal_dates[i]))
            minute_results[key] = {
                "lstm_m": float(m1["prediction"][i].cpu()),
                "tft_m": float(m2["prediction"][i].cpu()),
            }

    # Align: iterate daily keys, fill minute with 0 if missing
    sub_names = ["lstm_d", "tft_d", "lstm_m", "tft_m"]
    per_model: Dict[str, list] = {n: [] for n in sub_names}
    targets: list = []

    for key, dd in daily_results.items():
        md = minute_results.get(key, {})
        per_model["lstm_d"].append(dd["lstm_d"])
        per_model["tft_d"].append(dd["tft_d"])
        per_model["lstm_m"].append(md.get("lstm_m", 0.0))
        per_model["tft_m"].append(md.get("tft_m", 0.0))
        targets.append(dd["target"])

    sub_preds = {name: np.array(vals) for name, vals in per_model.items()}
    return sub_preds, np.array(targets)


# ============================================================================
# Long-short backtest (same as evaluate_hierarchical.py)
# ============================================================================

def backtest_long_short(preds, targets, dates, top_k=20,
                        commission_bps: float = 5.0, slippage_bps: float = 5.0):
    """Long-short backtest with optional transaction-cost deduction.

    Args:
        commission_bps: One-way commission in basis points (default 5 bps).
        slippage_bps:   One-way slippage in basis points (default 5 bps).
        Costs are applied as: daily_net_return = gross_return - turnover * (commission + slippage) / 10_000
        where turnover = fraction of the portfolio that changed vs. the previous day.
    """
    unique_dates = sorted(set(dates))
    daily_returns = []
    turnovers = []
    prev_longs: set = set()
    prev_shorts: set = set()

    for d in unique_dates:
        mask = np.where(dates == d)[0]
        dp, dt = preds[mask], targets[mask]
        k = min(top_k, max(1, len(dp) // 4))
        rank = np.argsort(dp)
        long_idx  = set(mask[rank[-k:]])
        short_idx = set(mask[rank[:k]])

        lr = np.mean(dt[rank[-k:]])
        sr = np.mean(dt[rank[:k]])
        gross = (lr - sr) / 2.0

        # Turnover: fraction of slots that changed relative to previous day
        if prev_longs or prev_shorts:
            changed = len((long_idx - prev_longs) | (short_idx - prev_shorts))
            total_slots = max(len(long_idx) + len(short_idx), 1)
            turnover = changed / total_slots
        else:
            turnover = 1.0  # first day: full entry cost
        turnovers.append(turnover)

        cost = turnover * (commission_bps + slippage_bps) / 10_000
        daily_returns.append(gross - cost)

        prev_longs, prev_shorts = long_idx, short_idx

    dr = np.array(daily_returns)
    if len(dr) < 5:
        return {"sharpe": 0.0, "n_days": len(dr)}
    cum = np.cumprod(1 + dr)
    total = cum[-1] - 1.0
    ann_ret = (1 + total) ** (252.0 / len(dr)) - 1
    ann_vol = np.std(dr) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))
    avg_turnover = float(np.mean(turnovers))
    return {"sharpe": round(sharpe, 4), "max_drawdown": round(max_dd, 4),
            "total_return": round(total, 4), "n_days": len(dr),
            "avg_daily_turnover": round(avg_turnover, 4),
            "commission_bps": commission_bps, "slippage_bps": slippage_bps}


# ============================================================================
# Stitched Out-of-Sample Curve (Critical Validation Metric)
# ============================================================================

def compute_stitched_oos_curve(
    window_results: List[Dict],
    output_dir: str,
    top_k: int = 20,
) -> Dict:
    """Compute the stitched out-of-sample equity curve by concatenating 
    test predictions from all folds in chronological order.
    
    This is the PRIMARY validation metric: if the model is real, this curve
    should be smooth and profitable without look-ahead bias.
    
    Returns dict with:
        - sharpe, total_return, max_drawdown (on stitched curve)
        - daily_returns, dates, cumulative_returns (for plotting)
        - n_days: total days in stitched curve
    """
    all_preds = []
    all_targets = []
    all_dates = []
    
    for result in window_results:
        # Would need to re-load test predictions per fold
        # For now, we return a placeholder; full implementation loads fold checkpoints
        pass
    
    if not all_preds:
        logger.warning("No stitched predictions available")
        return {}
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_dates = np.concatenate(all_dates)
    
    # Sort by date to get chronological OOS curve
    sort_idx = np.argsort(all_dates)
    all_preds = all_preds[sort_idx]
    all_targets = all_targets[sort_idx]
    all_dates = all_dates[sort_idx]
    
    # Compute long-short backtest
    unique_dates = sorted(set(all_dates))
    daily_returns = []
    
    for d in unique_dates:
        mask = all_dates == d
        dp, dt = all_preds[mask], all_targets[mask]
        if len(dp) < 2:
            continue
        k = min(top_k, max(1, len(dp) // 4))
        rank = np.argsort(dp)
        lr = np.mean(dt[rank[-k:]])
        sr = np.mean(dt[rank[:k]])
        daily_returns.append((lr - sr) / 2.0)
    
    daily_returns = np.array(daily_returns)
    
    if len(daily_returns) < 5:
        return {"sharpe": 0.0, "n_days": len(daily_returns), "error": "insufficient data"}
    
    cum_returns = np.cumprod(1 + daily_returns)
    total_ret = cum_returns[-1] - 1.0
    ann_ret = (1 + total_ret) ** (252.0 / len(daily_returns)) - 1
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    
    peak = np.maximum.accumulate(cum_returns)
    max_dd = float(np.min((cum_returns - peak) / peak)) if len(cum_returns) > 0 else 0.0
    
    return {
        "sharpe": round(sharpe, 4),
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "max_drawdown": round(max_dd, 4),
        "n_days": len(daily_returns),
        "daily_returns": daily_returns,
        "cumulative_returns": cum_returns,
    }



# ============================================================================
# Main walk-forward pipeline
# ============================================================================

def run_walk_forward(
    n_windows: int = 5,
    phases: List[int] = None,
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 128,
    top_k: int = 20,
    output_dir: str = "results/walk_forward_hierarchical",
    num_workers: int = 4,
    force_preprocess: bool = False,
    expanding_window: bool = True,
    purge_horizon: int = 0,
    skip_models: List[str] = None,
    resume_from_fold: int = -1,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
):
    if phases is None:
        phases = [1, 2, 3]
    if skip_models is None:
        skip_models = []
    skip_models = [m.lower() for m in skip_models]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Walk-forward: {n_windows} windows, phases={phases}, "
                f"epochs={epochs}")

    # Base data config (for preprocessing and ticker discovery)
    data_cfg = HierarchicalDataConfig(split_mode="temporal")

    # Discover tickers
    tickers = get_viable_tickers(data_cfg)
    logger.info(f"Viable tickers: {len(tickers)}")

    # Preprocess if needed
    if 0 in phases or force_preprocess:
        preprocess_all(tickers, data_cfg, force=force_preprocess)

    # Compute windows
    windows = compute_windows(
        tickers, data_cfg, n_windows=n_windows,
        expanding_window=expanding_window,
        purge_horizon=purge_horizon,
    )
    if not windows:
        logger.error("Could not compute any windows")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Import training components
    from src.hierarchical_config import TrainConfig
    from src.hierarchical_trainers import SubModelTrainer, MetaTrainer
    from src.hierarchical_finetuner import JointFineTuner

    # Determine sub_model_names from first forecaster (they're consistent across windows)
    # Create a temporary forecaster to get the actual sub_model_names
    temp_model_cfg = HierarchicalModelConfig()
    temp_forecaster = HierarchicalForecaster(temp_model_cfg)
    actual_sub_model_names = temp_forecaster.sub_model_names
    del temp_forecaster, temp_model_cfg

    # Attention prior: accumulates bias across windows
    prior_dir = os.path.join(output_dir, "attention_prior")
    attention_prior = AttentionPriorComputer(
        prior_dir=prior_dir,
        sub_model_names=actual_sub_model_names,
    )

    # ── Train minute models ONCE on their own date range ───────────────────
    # Minute data (2026-01-26 → 2026-04-06) is completely separate from the
    # daily walk-forward windows (1980 → 2026).  Training them inside each
    # window fold would always yield empty datasets for historical windows.
    # Instead we train them once here, save the weights, and load them into
    # every fresh forecaster before Phase 3 meta training.
    minute_loaders = None
    minute_weights_path = os.path.join(output_dir, "minute_models", "minute_weights.pt")
    os.makedirs(os.path.join(output_dir, "minute_models"), exist_ok=True)

    if 2 in phases and not any(m in skip_models for m in ["lstm_m", "tft_m"]):
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 2: Training minute models on minute-specific date range")
        logger.info(f"{'='*70}")

        minute_loaders = make_minute_loaders(
            tickers, data_cfg,
            batch_size_minute=batch_size // 2,
            num_workers=num_workers,
        )

        if minute_loaders is not None:
            minute_tcfg = TrainConfig(
                output_dir=os.path.join(output_dir, "minute_models"),
                log_dir=os.path.join(output_dir, "minute_models", "logs"),
                epochs_phase2=epochs,
                patience=patience,
                batch_size_minute=batch_size // 2,
                num_workers=num_workers,
            )
            os.makedirs(minute_tcfg.output_dir, exist_ok=True)

            # Build a throw-away forecaster just to get the minute sub-models
            _m_n_feat = minute_loaders["minute_n_features"]
            minute_model_cfg = HierarchicalModelConfig(minute_input_dim=_m_n_feat)
            minute_forecaster_tmp = HierarchicalForecaster(minute_model_cfg).to(device)

            for model, name in [
                (minute_forecaster_tmp.lstm_m, "LSTM_M"),
                (minute_forecaster_tmp.tft_m,  "TFT_M"),
            ]:
                trainer = SubModelTrainer(model, name, device, minute_tcfg)
                trainer.train(
                    minute_loaders["minute"]["train"],
                    minute_loaders["minute"]["val"],
                    minute_tcfg.epochs_phase2,
                    minute_tcfg.output_dir,
                )

            # Save only the minute sub-model state dicts
            torch.save({
                "lstm_m": minute_forecaster_tmp.lstm_m.state_dict(),
                "tft_m":  minute_forecaster_tmp.tft_m.state_dict(),
            }, minute_weights_path)
            logger.info(f"  Saved minute model weights → {minute_weights_path}")
            del minute_forecaster_tmp
        else:
            logger.warning("  Phase 2 skipped: no usable minute data found")
    else:
        # If Phase 2 was not requested this run but weights exist from a previous run, reuse them
        if os.path.exists(minute_weights_path):
            logger.info(f"  Reusing existing minute weights from {minute_weights_path}")
            minute_loaders = make_minute_loaders(
                tickers, data_cfg,
                batch_size_minute=batch_size // 2,
                num_workers=num_workers,
            )

    all_window_results = []

    for w_idx, window in enumerate(windows):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# WINDOW {w_idx}/{n_windows}")
        logger.info(f"{'#'*70}")

        # TASK-013: resume — reload saved predictions and skip retraining
        if resume_from_fold >= 0 and w_idx < resume_from_fold:
            preds_path = os.path.join(output_dir, f"window_{w_idx}", f"fold_{w_idx}_preds.npz")
            if os.path.exists(preds_path):
                saved = np.load(preds_path, allow_pickle=True)
                all_window_results.append(saved["window_result"].item())
                logger.info(f"  Skipped (resume): loaded saved results from {preds_path}")
                continue
            else:
                logger.warning(f"  resume_from_fold={resume_from_fold} but no saved preds at {preds_path}; training fold {w_idx}")

        t0 = time.time()

        # Create window-specific dataloaders
        loaders = make_window_loaders(
            tickers, data_cfg,
            train_dates=window["train"],
            val_dates=window["val"],
            test_dates=window["test"],
            batch_size_daily=batch_size,
            batch_size_minute=batch_size // 2,
            num_workers=num_workers,
        )

        n_train = len(loaders["daily"]["train"].dataset)
        n_val = len(loaders["daily"]["val"].dataset)
        n_test = len(loaders["daily"]["test"].dataset)
        logger.info(f"  Daily:  train={n_train}, val={n_val}, test={n_test}")

        n_m_train = len(loaders["minute"]["train"].dataset)
        n_m_val = len(loaders["minute"]["val"].dataset)
        n_m_test = len(loaders["minute"]["test"].dataset)
        logger.info(f"  Minute: train={n_m_train}, val={n_m_val}, test={n_m_test}")

        if n_train < 100 or n_test < 10:
            logger.warning(f"  Skipping window {w_idx}: insufficient data")
            continue

        # Create fresh model for this window
        model_cfg = HierarchicalModelConfig(
            daily_input_dim=loaders["daily_n_features"],
            minute_input_dim=loaders["minute_n_features"],
            use_minute_models="lstm_m" not in skip_models and "tft_m" not in skip_models,
            use_news_model="news" not in skip_models,
        )
        if skip_models:
            logger.info(f"  Skipping sub-models: {skip_models} (use_minute={model_cfg.use_minute_models}, use_news={model_cfg.use_news_model})")
        forecaster = HierarchicalForecaster(model_cfg).to(device)
        
        # Device assignment verified by PyTorch

        # Apply attention bias from previous window (warm-start the attention prior)
        bias_tensor = attention_prior.get_bias_tensor()
        if bias_tensor is not None and w_idx > 0:
            logger.info(f"  Applying attention bias from window {w_idx - 1}")
            forecaster.meta.apply_attention_bias(bias_tensor.to(device))

        tcfg = TrainConfig(
            output_dir=os.path.join(output_dir, f"window_{w_idx}"),
            log_dir=os.path.join(output_dir, f"window_{w_idx}", "logs"),
            epochs_phase1=epochs,
            epochs_phase2=epochs,
            epochs_phase3=max(epochs // 2, 5),
            epochs_phase4=max(epochs // 3, 3),
            patience=patience,
            batch_size_daily=batch_size,
            batch_size_minute=batch_size // 2,
            num_workers=num_workers,
        )
        os.makedirs(tcfg.output_dir, exist_ok=True)

        # Phase 1: daily models
        if 1 in phases:
            logger.info(f"\n  Phase 1: Training daily models...")
            for model, name in [(forecaster.lstm_d, "LSTM_D"), (forecaster.tft_d, "TFT_D")]:
                trainer = SubModelTrainer(model, name, device, tcfg)
                trainer.train(
                    loaders["daily"]["train"], loaders["daily"]["val"],
                    tcfg.epochs_phase1, tcfg.output_dir,
                )
        
        # Re-ensure forecaster is on device after training
        forecaster = forecaster.to(device)

        # Phase 2: minute models were already trained before the fold loop
        # (on the minute-specific date range).  Nothing to do here per-fold.

        # Phase 3: meta model
        if 3 in phases:
            logger.info(f"\n  Phase 3: Training meta model...")
            forecaster = forecaster.to(device)

            # Load pre-trained minute model weights (trained on minute date range)
            if os.path.exists(minute_weights_path):
                minute_sd = torch.load(minute_weights_path, map_location=device)
                missing_lstm = forecaster.lstm_m.load_state_dict(minute_sd["lstm_m"], strict=False)
                missing_tft  = forecaster.tft_m.load_state_dict(minute_sd["tft_m"],  strict=False)
                logger.info(f"  Loaded minute weights → lstm_m missing={missing_lstm.missing_keys}, "
                            f"tft_m missing={missing_tft.missing_keys}")
            else:
                logger.warning("  No minute weights found — minute sub-models will be random init")

            # Use the minute-specific loaders (cover the real minute date range),
            # not the fold window loaders (which are empty for historical windows).
            m_train_dl = minute_loaders["minute"]["train"] if minute_loaders else loaders["minute"]["train"]
            m_val_dl   = minute_loaders["minute"]["val"]   if minute_loaders else loaders["minute"]["val"]

            meta_trainer = MetaTrainer(forecaster.meta, device, tcfg)
            meta_trainer.train(
                forecaster,
                loaders["daily"]["train"], m_train_dl,
                loaders["daily"]["val"],   m_val_dl,
                tcfg.epochs_phase3, tcfg.output_dir,
                data_cfg=data_cfg,
                n_train_news_dl=loaders["news"]["train"] if loaders.get("news") and loaders["news"].get("train") else None,
                n_val_news_dl=loaders["news"]["val"]   if loaders.get("news") and loaders["news"].get("val")   else None,
            )

        # Phase 4: joint fine-tuning
        if 4 in phases:
            logger.info(f"\n  Phase 4: Joint fine-tuning...")
            # Ensure forecaster is on device
            forecaster = forecaster.to(device)
            joint_trainer = JointFineTuner(forecaster, device, tcfg, data_cfg)
            joint_trainer.train(
                loaders["daily"]["train"], loaders["minute"]["train"],
                loaders["daily"]["val"], loaders["minute"]["val"],
                tcfg.epochs_phase4, tcfg.output_dir,
            )

        # Phase 5: fusion training on overlapping dates
        if 5 in phases:
            logger.info(f"\n  Phase 5: Fusion training on overlapping dates...")
            # Find overlapping dates between daily and minute training data
            daily_dates, minute_dates, overlap_dates = compute_overlapping_dates(
                loaders["daily"]["train"], loaders["minute"]["train"]
            )
            logger.info(f"    Daily dates: {len(daily_dates)}, Minute dates: {len(minute_dates)}, "
                       f"Overlap: {len(overlap_dates)}")
            
            if len(overlap_dates) > 10:
                # Extract daily and minute forecasters from main forecaster
                from src.hierarchical_models import DailyForecaster, MinuteForecaster, FusionMLP
                from src.hierarchical_trainers import FusionTrainer
                
                daily_forecaster = DailyForecaster(
                    lstm=forecaster.models[0],  # First LSTM is daily
                    tft=forecaster.models[1],   # First TFT is daily
                    news=forecaster.news_model,
                    meta=forecaster.meta,
                    cfg=data_cfg,
                    device=device,
                )
                
                minute_forecaster = MinuteForecaster(
                    lstm=forecaster.models[2],  # Third LSTM is minute
                    tft=forecaster.models[3],   # Third TFT is minute
                    meta=forecaster.meta,
                    cfg=data_cfg,
                    device=device,
                )
                
                fusion_mlp = FusionMLP(1, data_cfg.regime_dim if hasattr(data_cfg, 'regime_dim') else 8)
                fusion_mlp = fusion_mlp.to(device)
                
                fusion_trainer = FusionTrainer(
                    daily_forecaster, minute_forecaster, fusion_mlp, device, tcfg
                )
                
                fusion_trainer.train(
                    loaders["daily"]["train"], loaders["minute"]["train"],
                    overlap_dates, data_cfg,
                    epochs=max(tcfg.epochs_phase4 // 2, 5),
                    output_dir=tcfg.output_dir,
                )
            else:
                logger.warning(f"  Skipping Phase 5: insufficient overlapping dates ({len(overlap_dates)})")

        # Evaluate on validation set to get IC for weighting
        logger.info(f"\n  Evaluating on validation set for IC weighting...")
        # Ensure forecaster is on device before inference
        forecaster = forecaster.to(device)
        val_preds, val_targets, _, val_dates = collect_test_predictions(
            forecaster, loaders["daily"]["val"], loaders["minute"]["val"],
            data_cfg, device,
        )
        val_metrics = compute_metrics(val_preds, val_targets) if len(val_preds) > 5 else {}
        val_ic = val_metrics.get("ic", 0.0)

        # Evaluate on test set
        logger.info(f"\n  Evaluating on test window...")
        # Ensure forecaster is on device before inference
        forecaster = forecaster.to(device)
        preds, targets, raw_targets, dates = collect_test_predictions(
            forecaster, loaders["daily"]["test"], loaders["minute"]["test"],
            data_cfg, device,
        )

        if len(preds) < 5:
            logger.warning(f"  Window {w_idx}: too few test predictions ({len(preds)})")
            continue

        metrics = compute_metrics(preds, targets)  # targets = z-scores → correct for IC

        # TASK-004: cross-sectional z-score of predictions per date before backtest
        from scipy.stats import zscore as _zscore
        preds_cs = preds.copy()
        for _d in np.unique(dates):
            _mask = dates == _d
            if _mask.sum() > 1:
                preds_cs[_mask] = _zscore(preds[_mask])

        ls = backtest_long_short(preds_cs, raw_targets, dates, top_k=top_k,
                                   commission_bps=commission_bps, slippage_bps=slippage_bps)  # raw_targets = clipped returns → correct for P&L

        elapsed = time.time() - t0

        window_result = {
            "window": w_idx,
            "dates": {k: (int(v[0]), int(v[1])) for k, v in window.items()},
            "val_ic": round(val_ic, 4),  # For ensemble weighting
            "regression_metrics": metrics,
            "long_short": ls,
            "elapsed_seconds": round(elapsed, 1),
            "n_train": n_train,
            "n_test": n_test,
        }
        all_window_results.append(window_result)

        logger.info(f"\n  Window {w_idx} results:")
        logger.info(f"    Val IC={val_ic:.4f} (for ensemble weighting)")
        logger.info(f"    Test IC={metrics['ic']:.4f}, RankIC={metrics['rank_ic']:.4f}, "
                    f"DirAcc={metrics['directional_accuracy']:.3f}")
        logger.info(f"    L/S Sharpe={ls['sharpe']:.4f}, MaxDD={ls.get('max_drawdown', 0):.4f}, "
                    f"AvgTurnover={ls.get('avg_daily_turnover', 0):.3f} "
                    f"(cost={commission_bps:.0f}+{slippage_bps:.0f} bps)")
        logger.info(f"    Time: {elapsed:.0f}s")

        # TASK-013: save fold predictions for --resume-from-fold
        preds_save_path = os.path.join(tcfg.output_dir, f"fold_{w_idx}_preds.npz")
        np.savez(preds_save_path, window_result=np.array(window_result, dtype=object))
        logger.info(f"  Saved fold preds: {preds_save_path}")

        # Save fold checkpoint with fold index
        fold_checkpoint_path = os.path.join(tcfg.output_dir, f"fold_{w_idx}_forecaster.pt")
        forecaster.save(fold_checkpoint_path)
        logger.info(f"  Saved fold model: {fold_checkpoint_path}")

        # ── Compute attention prior for next window ──────────────────
        #    Collect per-sub-model predictions on the test set and
        #    compute IC per model.  This feeds the attention bias for
        #    the next window's MetaMLP initialization.
        logger.info(f"  Computing attention prior from window {w_idx} test set...")
        try:
            sub_preds, sub_targets = collect_sub_model_predictions(
                forecaster,
                loaders["daily"]["test"], loaders["minute"]["test"],
                device,
            )
            if len(sub_targets) >= 5:
                sub_metrics = attention_prior.compute_from_predictions(
                    sub_preds, sub_targets,
                )
                attention_prior.compute_bias(
                    sub_model_metrics=sub_metrics,
                    window_idx=w_idx,
                )
                logger.info(f"  Attention prior updated for next window")
            else:
                logger.info(f"  Too few test samples for attention prior ({len(sub_targets)})")
        except Exception as e:
            logger.warning(f"  Attention prior computation failed: {e}")

    # ─── Aggregate results ────────────────────────────────────────────
    if not all_window_results:
        logger.error("No windows completed successfully")
        return

    # Phase 2 minute training now happens BEFORE the fold loop (see above).
    # The old post-fold block has been removed to avoid duplicate training.

    logger.info(f"\n{'='*70}")
    logger.info("WALK-FORWARD SUMMARY")
    logger.info(f"{'='*70}")

    ics = [r["regression_metrics"]["ic"] for r in all_window_results]
    rics = [r["regression_metrics"]["rank_ic"] for r in all_window_results]
    das = [r["regression_metrics"]["directional_accuracy"] for r in all_window_results]
    sharpes = [r["long_short"]["sharpe"] for r in all_window_results]
    val_ics = [r.get("val_ic", 0.0) for r in all_window_results]

    # Find best fold by validation IC
    best_fold_idx = np.argmax(val_ics) if val_ics else 0
    best_fold_result = all_window_results[best_fold_idx]

    logger.info(f"\n  {'Window':<10} {'ValIC':>8} {'TestIC':>8} {'RankIC':>8} {'DirAcc':>8} {'Sharpe':>8}")
    logger.info("  " + "-" * 54)
    for r in all_window_results:
        m = r["regression_metrics"]
        ls = r["long_short"]
        val_ic_str = f"{r.get('val_ic', 0.0):>8.4f}"
        marker = " *BEST*" if r["window"] == best_fold_idx else ""
        logger.info(f"  W{r['window']:<9} {val_ic_str} {m['ic']:>8.4f} {m['rank_ic']:>8.4f} "
                    f"{m['directional_accuracy']:>8.3f} {ls['sharpe']:>8.4f}{marker}")

    logger.info("  " + "-" * 54)
    logger.info(f"  {'Mean':<10} {np.mean(val_ics):>8.4f} {np.mean(ics):>8.4f} {np.mean(rics):>8.4f} "
                f"{np.mean(das):>8.3f} {float(np.nanmean([s for s in sharpes if not (isinstance(s,float) and s!=s)])):>8.4f}")
    logger.info(f"  {'Std':<10} {np.std(val_ics):>8.4f} {np.std(ics):>8.4f} {np.std(rics):>8.4f} "
                f"{np.std(das):>8.3f} {np.nanstd(sharpes):>8.4f}")
    logger.info(f"  {'Min':<10} {np.min(val_ics):>8.4f} {np.min(ics):>8.4f} {np.min(rics):>8.4f} "
                f"{np.min(das):>8.3f} {np.nanmin(sharpes):>8.4f}")
    logger.info(f"  {'Max':<10} {np.max(val_ics):>8.4f} {np.max(ics):>8.4f} {np.max(rics):>8.4f} "
                f"{np.max(das):>8.3f} {np.nanmax(sharpes):>8.4f}")

    consistency = sum(1 for s in sharpes if isinstance(s, (int, float)) and not (s != s) and s > 0)
    logger.info(f"\n  Consistency: {consistency}/{len(sharpes)} windows positive Sharpe")
    logger.info(f"  Mean Test IC:     {np.mean(ics):.4f} ± {np.std(ics):.4f}")
    logger.info(f"  Mean Sharpe:      {float(np.nanmean([s for s in sharpes if not (isinstance(s,float) and s!=s)])):.4f} ± {np.nanstd(sharpes):.4f}")
    logger.info(f"\n  BEST FOLD: Fold {best_fold_idx} (Val IC={best_fold_result.get('val_ic', 0.0):.4f})")
    
    # Compute IC-weighted average
    val_ics_arr = np.array(val_ics)
    if np.sum(val_ics_arr) > 0:
        weights = val_ics_arr / np.sum(val_ics_arr)  # Normalize to sum=1
        ic_weighted_mean = np.average(ics, weights=weights)
        sharpe_weighted_mean = np.average(sharpes, weights=weights)
        logger.info(f"  IC-weighted Test IC:  {ic_weighted_mean:.4f}")
        logger.info(f"  IC-weighted Sharpe:   {sharpe_weighted_mean:.4f}")

    # Save aggregate with best fold and IC weights
    summary = {
        "n_windows": len(all_window_results),
        "best_fold_idx": int(best_fold_idx),
        "best_fold_val_ic": round(float(best_fold_result.get("val_ic", 0.0)), 4),
        "phases": phases,
        "epochs": epochs,
        "expanding_window": expanding_window,
        "purge_horizon": purge_horizon,
        "per_window": all_window_results,
        "aggregate": {
            "ic_mean": round(float(np.mean(ics)), 4),
            "ic_std": round(float(np.std(ics)), 4),
            "rank_ic_mean": round(float(np.mean(rics)), 4),
            "dir_acc_mean": round(float(np.mean(das)), 4),
            "sharpe_mean": round(float(float(np.nanmean([s for s in sharpes if not (isinstance(s,float) and s!=s)]))), 4),
            "sharpe_std": round(float(np.nanstd(sharpes)), 4),
            "consistency": f"{consistency}/{len(sharpes)}",
            "val_ic_weights": [round(float(w), 4) for w in (val_ics_arr / np.sum(val_ics_arr) if np.sum(val_ics_arr) > 0 else [0]*len(val_ics_arr))],
        },
    }

    out_path = os.path.join(output_dir, "walk_forward_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n  Results saved → {out_path}")

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for Hierarchical Forecaster")
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results/walk_forward_hierarchical")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--rolling", action="store_true",
                        help="Use rolling (fixed-span) windows instead of expanding windows")
    parser.add_argument("--purge-horizon", type=int, default=0,
                        help="Days to purge between train/val to prevent leakage (0 = use forecast_horizon)")
    parser.add_argument("--skip-models", type=str, nargs="+", default=[],
                        metavar="MODEL",
                        help="Sub-models to disable, e.g. --skip-models lstm_m tft_m news")
    # TASK-013: fold resume
    parser.add_argument("--resume-from-fold", type=int, default=-1,
                        help="Skip folds 0..N-1 (load saved preds) and train from fold N onwards")
    # TASK-014: transaction costs
    parser.add_argument("--commission-bps", type=float, default=5.0,
                        help="One-way commission in basis points (default 5)")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="One-way slippage in basis points (default 5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "walk_forward.log")),
            logging.StreamHandler(),
        ],
    )

    run_walk_forward(
        n_windows=args.n_windows,
        phases=args.phases,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        force_preprocess=args.force_preprocess,
        expanding_window=not args.rolling,
        purge_horizon=args.purge_horizon,
        skip_models=args.skip_models,
        resume_from_fold=args.resume_from_fold,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )


if __name__ == "__main__":
    main()
