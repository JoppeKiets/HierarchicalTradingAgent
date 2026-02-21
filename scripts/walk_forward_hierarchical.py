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
from src.hierarchical_models import (
    HierarchicalForecaster,
    HierarchicalModelConfig,
)

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

    result = {"daily": {}, "minute": {}}

    for split_name, (lo, hi) in [
        ("train", train_dates), ("val", val_dates), ("test", test_dates),
    ]:
        shuffle = split_name == "train"

        daily_ds = WindowDailyDataset(tickers, cfg, lo, hi)
        minute_ds = WindowMinuteDataset(tickers, cfg, lo, hi)

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
) -> List[Dict[str, Tuple[int, int]]]:
    """Compute walk-forward windows from the available date range.

    Scans both daily and minute caches so windows can cover the full
    timeline even when daily and minute data have different date ranges.

    Returns a list of dicts, each with 'train', 'val', 'test' date tuples
    (ordinal_lo, ordinal_hi).
    """
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

    # Each window spans a fraction of the total timeline,
    # windows overlap by shifting the start forward
    window_span = int(total_span * 0.6)   # Each window covers 60% of data
    step = (total_span - window_span) // max(n_windows - 1, 1)

    windows = []
    for w in range(n_windows):
        w_start = min_date + w * step
        w_end = w_start + window_span

        # Clamp to available range
        w_end = min(w_end, max_date)

        w_span = w_end - w_start
        train_end = w_start + int(w_span * train_frac)
        val_end = train_end + int(w_span * val_frac)

        windows.append({
            "train": (w_start, train_end),
            "val": (train_end, val_end),
            "test": (val_end, w_end),
        })

        import datetime
        logger.info(f"  Window {w}: "
                    f"train={datetime.date.fromordinal(w_start)}→"
                    f"{datetime.date.fromordinal(train_end)}, "
                    f"val→{datetime.date.fromordinal(val_end)}, "
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

    regime_df = _build_regime_dataframe(data_cfg)
    regime_lookup = {}
    if not regime_df.empty:
        for d, row in regime_df.iterrows():
            if hasattr(d, 'toordinal'):
                regime_lookup[d.toordinal()] = row.values.astype(np.float32)

    daily_data = {}
    for batch in daily_loader:
        x, y, ordinal_dates, tickers_batch = batch
        x = x.to(device)
        d1 = forecaster.lstm_d(x)
        d2 = forecaster.tft_d(x)
        for i in range(len(y)):
            key = (tickers_batch[i], int(ordinal_dates[i]))
            daily_data[key] = {
                "pred_lstm": d1["prediction"][i].cpu(),
                "pred_tft": d2["prediction"][i].cpu(),
                "emb_lstm": d1["embedding"][i].cpu(),
                "emb_tft": d2["embedding"][i].cpu(),
                "target": y[i],
            }

    minute_data = {}
    for batch in minute_loader:
        x, y, ordinal_dates, tickers_batch = batch
        x = x.to(device)
        m1 = forecaster.lstm_m(x)
        m2 = forecaster.tft_m(x)
        for i in range(len(y)):
            key = (tickers_batch[i], int(ordinal_dates[i]))
            minute_data[key] = {
                "pred_lstm": m1["prediction"][i].cpu(),
                "pred_tft": m2["prediction"][i].cpu(),
                "emb_lstm": m1["embedding"][i].cpu(),
                "emb_tft": m2["embedding"][i].cpu(),
            }

    zero_emb = torch.zeros(E)
    aligned_preds, aligned_embs, aligned_regimes = [], [], []
    aligned_tgts, aligned_dates = [], []

    for key, dd in daily_data.items():
        _, ord_date = key
        md = minute_data.get(key)

        if md is not None:
            pred_vec = torch.stack([
                dd["pred_lstm"], dd["pred_tft"],
                md["pred_lstm"], md["pred_tft"],
            ])
            emb_vec = torch.cat([
                dd["emb_lstm"], dd["emb_tft"],
                md["emb_lstm"], md["emb_tft"],
            ])
        else:
            pred_vec = torch.stack([
                dd["pred_lstm"], dd["pred_tft"],
                torch.tensor(0.0), torch.tensor(0.0),
            ])
            emb_vec = torch.cat([
                dd["emb_lstm"], dd["emb_tft"], zero_emb, zero_emb,
            ])

        regime_vec = (torch.from_numpy(regime_lookup[ord_date])
                      if ord_date in regime_lookup else torch.zeros(R))

        aligned_preds.append(pred_vec)
        aligned_embs.append(emb_vec)
        aligned_regimes.append(regime_vec)
        aligned_tgts.append(dd["target"].item())
        aligned_dates.append(ord_date)

    if not aligned_preds:
        return np.array([]), np.array([]), np.array([])

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
        emb_list = [e_b[:, i*E:(i+1)*E] for i in range(4)]
        meta_out = forecaster.meta(p_b, emb_list, r_b)
        preds_out.append(meta_out["prediction"].cpu().numpy())

    return np.concatenate(preds_out), np.array(aligned_tgts), np.array(aligned_dates)


# ============================================================================
# Long-short backtest (same as evaluate_hierarchical.py)
# ============================================================================

def backtest_long_short(preds, targets, dates, top_k=20):
    unique_dates = sorted(set(dates))
    daily_returns = []
    for d in unique_dates:
        mask = dates == d
        dp, dt = preds[mask], targets[mask]
        k = min(top_k, max(1, len(dp) // 4))
        rank = np.argsort(dp)
        lr = np.mean(dt[rank[-k:]])
        sr = np.mean(dt[rank[:k]])
        daily_returns.append((lr - sr) / 2.0)

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
    return {"sharpe": round(sharpe, 4), "max_drawdown": round(max_dd, 4),
            "total_return": round(total, 4), "n_days": len(dr)}


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
):
    if phases is None:
        phases = [1, 2, 3]

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
    windows = compute_windows(tickers, data_cfg, n_windows=n_windows)
    if not windows:
        logger.error("Could not compute any windows")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Import training components
    from train_hierarchical import (
        TrainConfig, SubModelTrainer, MetaTrainer, JointFineTuner
    )

    all_window_results = []

    for w_idx, window in enumerate(windows):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# WINDOW {w_idx}/{n_windows}")
        logger.info(f"{'#'*70}")

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
        )
        forecaster = HierarchicalForecaster(model_cfg).to(device)

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

        # Phase 2: minute models
        if 2 in phases and n_m_train > 50:
            logger.info(f"\n  Phase 2: Training minute models...")
            for model, name in [(forecaster.lstm_m, "LSTM_M"), (forecaster.tft_m, "TFT_M")]:
                trainer = SubModelTrainer(model, name, device, tcfg)
                trainer.train(
                    loaders["minute"]["train"], loaders["minute"]["val"],
                    tcfg.epochs_phase2, tcfg.output_dir,
                )
        elif 2 in phases:
            logger.info(f"  Skipping Phase 2: insufficient minute data ({n_m_train})")

        # Phase 3: meta model
        if 3 in phases:
            logger.info(f"\n  Phase 3: Training meta model...")
            meta_trainer = MetaTrainer(forecaster.meta, device, tcfg)
            meta_trainer.train(
                forecaster,
                loaders["daily"]["train"], loaders["minute"]["train"],
                loaders["daily"]["val"], loaders["minute"]["val"],
                tcfg.epochs_phase3, tcfg.output_dir,
                data_cfg=data_cfg,
            )

        # Phase 4: joint fine-tuning
        if 4 in phases:
            logger.info(f"\n  Phase 4: Joint fine-tuning...")
            joint_trainer = JointFineTuner(forecaster, device, tcfg, data_cfg)
            joint_trainer.train(
                loaders["daily"]["train"], loaders["minute"]["train"],
                loaders["daily"]["val"], loaders["minute"]["val"],
                tcfg.epochs_phase4, tcfg.output_dir,
            )

        # Evaluate on test set
        logger.info(f"\n  Evaluating on test window...")
        preds, targets, dates = collect_test_predictions(
            forecaster, loaders["daily"]["test"], loaders["minute"]["test"],
            data_cfg, device,
        )

        if len(preds) < 5:
            logger.warning(f"  Window {w_idx}: too few test predictions ({len(preds)})")
            continue

        metrics = compute_metrics(preds, targets)
        ls = backtest_long_short(preds, targets, dates, top_k=top_k)

        elapsed = time.time() - t0

        window_result = {
            "window": w_idx,
            "dates": {k: (int(v[0]), int(v[1])) for k, v in window.items()},
            "regression_metrics": metrics,
            "long_short": ls,
            "elapsed_seconds": round(elapsed, 1),
            "n_train": n_train,
            "n_test": n_test,
        }
        all_window_results.append(window_result)

        logger.info(f"\n  Window {w_idx} results:")
        logger.info(f"    IC={metrics['ic']:.4f}, RankIC={metrics['rank_ic']:.4f}, "
                    f"DirAcc={metrics['directional_accuracy']:.3f}")
        logger.info(f"    L/S Sharpe={ls['sharpe']:.4f}, MaxDD={ls.get('max_drawdown', 0):.4f}")
        logger.info(f"    Time: {elapsed:.0f}s")

        # Save per-window checkpoint
        forecaster.save(os.path.join(tcfg.output_dir, "forecaster.pt"))

    # ─── Aggregate results ────────────────────────────────────────────
    if not all_window_results:
        logger.error("No windows completed successfully")
        return

    logger.info(f"\n{'='*70}")
    logger.info("WALK-FORWARD SUMMARY")
    logger.info(f"{'='*70}")

    ics = [r["regression_metrics"]["ic"] for r in all_window_results]
    rics = [r["regression_metrics"]["rank_ic"] for r in all_window_results]
    das = [r["regression_metrics"]["directional_accuracy"] for r in all_window_results]
    sharpes = [r["long_short"]["sharpe"] for r in all_window_results]

    logger.info(f"\n  {'Window':<10} {'IC':>8} {'RankIC':>8} {'DirAcc':>8} {'Sharpe':>8}")
    logger.info("  " + "-" * 46)
    for r in all_window_results:
        m = r["regression_metrics"]
        ls = r["long_short"]
        logger.info(f"  W{r['window']:<9} {m['ic']:>8.4f} {m['rank_ic']:>8.4f} "
                    f"{m['directional_accuracy']:>8.3f} {ls['sharpe']:>8.4f}")

    logger.info("  " + "-" * 46)
    logger.info(f"  {'Mean':<10} {np.mean(ics):>8.4f} {np.mean(rics):>8.4f} "
                f"{np.mean(das):>8.3f} {np.mean(sharpes):>8.4f}")
    logger.info(f"  {'Std':<10} {np.std(ics):>8.4f} {np.std(rics):>8.4f} "
                f"{np.std(das):>8.3f} {np.std(sharpes):>8.4f}")
    logger.info(f"  {'Min':<10} {np.min(ics):>8.4f} {np.min(rics):>8.4f} "
                f"{np.min(das):>8.3f} {np.min(sharpes):>8.4f}")
    logger.info(f"  {'Max':<10} {np.max(ics):>8.4f} {np.max(rics):>8.4f} "
                f"{np.max(das):>8.3f} {np.max(sharpes):>8.4f}")

    consistency = sum(1 for s in sharpes if s > 0)
    logger.info(f"\n  Consistency: {consistency}/{len(sharpes)} windows positive Sharpe")
    logger.info(f"  Mean IC:     {np.mean(ics):.4f} ± {np.std(ics):.4f}")
    logger.info(f"  Mean Sharpe: {np.mean(sharpes):.4f} ± {np.std(sharpes):.4f}")

    # Save aggregate
    summary = {
        "n_windows": len(all_window_results),
        "phases": phases,
        "epochs": epochs,
        "per_window": all_window_results,
        "aggregate": {
            "ic_mean": round(float(np.mean(ics)), 4),
            "ic_std": round(float(np.std(ics)), 4),
            "rank_ic_mean": round(float(np.mean(rics)), 4),
            "dir_acc_mean": round(float(np.mean(das)), 4),
            "sharpe_mean": round(float(np.mean(sharpes)), 4),
            "sharpe_std": round(float(np.std(sharpes)), 4),
            "consistency": f"{consistency}/{len(sharpes)}",
        },
    }

    out_path = os.path.join(output_dir, "walk_forward_results.json")
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
    )


if __name__ == "__main__":
    main()
