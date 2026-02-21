#!/usr/bin/env python3
"""Generate all diagnostic plots for a completed hierarchical model run.

Plots produced (saved to {model_dir}/plots/):
  1. training_curves.png   — loss + IC per epoch for all 5 models
  2. pred_vs_actual.png    — scatter of predicted vs actual returns per model
  3. calibration.png       — decile calibration: mean actual return per pred decile
  4. ic_over_time.png      — rolling 30-day IC on test set ordered by date
  5. pred_distribution.png — histogram of predictions vs actuals per model
  6. multi_horizon_ic.png  — IC at horizons 1,3,5,10,15 days (post-hoc, no retraining)
  7. equity_curve.png      — simulated long-only top-K portfolio vs buy-and-hold

Usage:
    python scripts/plot_results.py --model-dir models/hierarchical_v5
    python scripts/plot_results.py --model-dir models/hierarchical_v5 --top-k 20
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

COLORS = {
    "LSTM_D": "#2196F3",   # blue
    "TFT_D":  "#4CAF50",   # green
    "LSTM_M": "#FF9800",   # orange
    "TFT_M":  "#9C27B0",   # purple
    "Meta":   "#F44336",   # red
    "META_ENSEMBLE": "#F44336",
}
MODEL_ORDER = ["LSTM_D", "TFT_D", "LSTM_M", "TFT_M", "Meta"]


# ============================================================================
# Data loading helpers
# ============================================================================

def load_history(model_dir: str) -> Dict[str, Dict]:
    """Load all *_history.csv files from model_dir."""
    histories = {}
    for name in ["LSTM_D", "TFT_D", "LSTM_M", "TFT_M", "Meta"]:
        path = Path(model_dir) / f"{name}_history.csv"
        if not path.exists():
            continue
        data = {"epoch": [], "train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["epoch"].append(int(row["epoch"]))
                data["train_loss"].append(float(row["train_loss"]))
                data["val_loss"].append(float(row["val_loss"]))
                ic = row.get("val_ic", "") or "0"
                ric = row.get("val_rank_ic", "") or "0"
                data["val_ic"].append(float(ic) if ic not in ("", "nan") else 0.0)
                data["val_rank_ic"].append(float(ric) if ric not in ("", "nan") else 0.0)
        histories[name] = data
    return histories


def load_test_results(model_dir: str) -> Optional[Dict]:
    path = Path(model_dir) / "test_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_test_predictions(model_dir: str, data_cfg_kwargs: dict) -> Optional[Dict]:
    """Run the final model on the test split and return raw prediction arrays."""
    try:
        import torch
        from src.hierarchical_data import HierarchicalDataConfig, create_dataloaders, \
            get_viable_tickers, split_tickers
        from src.hierarchical_models import HierarchicalForecaster
    except ImportError as e:
        logger.warning(f"Cannot collect predictions (import error): {e}")
        return None

    model_path = Path(model_dir) / "forecaster_final.pt"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = HierarchicalForecaster.load(str(model_path), device=str(device))
    forecaster.to(device)
    forecaster.eval()

    data_cfg = HierarchicalDataConfig(**data_cfg_kwargs)
    tickers = get_viable_tickers(data_cfg)
    splits = split_tickers(tickers, data_cfg)
    loaders = create_dataloaders(splits, data_cfg, batch_size_daily=64,
                                  batch_size_minute=32, num_workers=0)

    results = {m: {"preds": [], "targets": [], "dates": []}
               for m in ["lstm_d", "tft_d", "lstm_m", "tft_m"]}

    with torch.no_grad():
        # Daily models
        for batch in loaders["daily"]["test"]:
            x, y, ord_dates, _ = batch
            x = x.to(device)
            d1 = forecaster.lstm_d(x)
            d2 = forecaster.tft_d(x)
            results["lstm_d"]["preds"].extend(d1["prediction"].cpu().numpy())
            results["lstm_d"]["targets"].extend(y.numpy())
            results["lstm_d"]["dates"].extend(ord_dates.numpy())
            results["tft_d"]["preds"].extend(d2["prediction"].cpu().numpy())
            results["tft_d"]["targets"].extend(y.numpy())
            results["tft_d"]["dates"].extend(ord_dates.numpy())

        # Minute models
        for batch in loaders["minute"]["test"]:
            x, y, ord_dates, _ = batch
            x = x.to(device)
            m1 = forecaster.lstm_m(x)
            m2 = forecaster.tft_m(x)
            results["lstm_m"]["preds"].extend(m1["prediction"].cpu().numpy())
            results["lstm_m"]["targets"].extend(y.numpy())
            results["lstm_m"]["dates"].extend(ord_dates.numpy())
            results["tft_m"]["preds"].extend(m2["prediction"].cpu().numpy())
            results["tft_m"]["targets"].extend(y.numpy())
            results["tft_m"]["dates"].extend(ord_dates.numpy())

    for m in results:
        results[m]["preds"] = np.array(results[m]["preds"])
        results[m]["targets"] = np.array(results[m]["targets"])
        results[m]["dates"] = np.array(results[m]["dates"])

    return results


def collect_multi_horizon_predictions(model_dir: str, data_cfg_kwargs: dict,
                                       horizons: List[int]) -> Optional[Dict]:
    """For each horizon h, reindex cached daily targets to close[t+h]/close[t]-1
    and evaluate LSTM_D predictions against those targets.
    No retraining — we just shift the target array."""
    try:
        import torch
        from src.hierarchical_data import HierarchicalDataConfig, get_viable_tickers, split_tickers
        from src.hierarchical_models import HierarchicalForecaster
    except ImportError as e:
        logger.warning(f"Cannot collect multi-horizon data: {e}")
        return None

    model_path = Path(model_dir) / "forecaster_final.pt"
    if not model_path.exists():
        return None

    data_cfg = HierarchicalDataConfig(**data_cfg_kwargs)
    cache = Path(data_cfg.cache_dir) / "daily"
    tickers = get_viable_tickers(data_cfg)
    splits = split_tickers(tickers, data_cfg)
    test_tickers = splits["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = HierarchicalForecaster.load(str(model_path), device=str(device))
    forecaster.to(device)
    forecaster.eval()

    seq_len = forecaster.cfg.daily_seq_len
    horizon_results = {h: {"preds": [], "targets": []} for h in horizons}

    with torch.no_grad():
        for ticker in test_tickers:
            feat_path = cache / f"{ticker}_features.npy"
            tgt_path  = cache / f"{ticker}_targets.npy"
            if not feat_path.exists() or not tgt_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            n = feat.shape[0]

            # Load close prices — prefer the cached targets file (already aligned
            # to the feature matrix) over re-reading the raw CSV.
            close = None
            tgt_arr = np.load(tgt_path, mmap_mode="r") if tgt_path.exists() else None

            price_file = Path(data_cfg.organized_dir) / ticker / "price_history.csv"
            if price_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(price_file)
                    df.columns = [c.strip().lower() for c in df.columns]
                    if "adj close" in df.columns:
                        df["close"] = df["adj close"].fillna(df["close"])
                    df = df.sort_values("date").reset_index(drop=True)
                    c = df["close"].values.astype(np.float64)
                    if len(c) == n:
                        close = c
                except Exception:
                    pass

            if close is None:
                continue  # can't compute price-based targets

            # Use the same boundary as training (forecast_horizon=1 means n-1 usable rows)
            fh = getattr(data_cfg, "forecast_horizon", 1)
            warmup = max(seq_len, 60)
            usable = (n - fh) - warmup
            if usable <= 0:
                continue
            val_end = warmup + int(usable * (data_cfg.temporal_train_frac
                                              + data_cfg.temporal_val_frac))
            test_end = n - fh   # last valid prediction index

            stride = data_cfg.daily_stride
            for i in range(val_end, test_end, stride):
                if i - seq_len < 0:
                    continue
                x = torch.from_numpy(feat[i - seq_len:i].copy()).unsqueeze(0).to(device)
                pred = forecaster.lstm_d(x)["prediction"].item()

                for h in horizons:
                    if i + h >= len(close) or close[i] <= 0:
                        continue
                    target = float(close[i + h] / close[i] - 1.0)
                    target = float(np.clip(target, -0.5, 0.5))
                    horizon_results[h]["preds"].append(pred)
                    horizon_results[h]["targets"].append(target)

    # Compute IC per horizon
    ic_by_horizon = {}
    for h, d in horizon_results.items():
        p = np.array(d["preds"])
        t = np.array(d["targets"])
        n_samples = len(p)
        pred_std = np.std(p) if len(p) > 0 else 0.0
        tgt_std  = np.std(t) if len(t) > 0 else 0.0
        if n_samples < 10 or pred_std < 1e-10 or tgt_std < 1e-10:
            logger.debug(f"h={h}: n={n_samples}, pred_std={pred_std:.2e}, tgt_std={tgt_std:.2e} → IC=0")
            ic_by_horizon[h] = 0.0
        else:
            raw_ic = float(np.corrcoef(p, t)[0, 1])
            ic_by_horizon[h] = 0.0 if np.isnan(raw_ic) else raw_ic
    return ic_by_horizon


# ============================================================================
# Plot 1: Training curves
# ============================================================================

def plot_training_curves(histories: Dict, out_dir: str):
    names = [n for n in MODEL_ORDER if n in histories]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 3, figsize=(18, 4 * len(names)))
    if len(names) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Training Curves", fontsize=15, fontweight="bold", y=1.01)

    for idx, name in enumerate(names):
        d = histories[name]
        epochs = d["epoch"]
        color = COLORS.get(name, "gray")

        # Loss
        ax = axes[idx, 0]
        ax.plot(epochs, d["train_loss"], label="Train", color=color, alpha=0.9, lw=2)
        ax.plot(epochs, d["val_loss"],   label="Val",   color=color, alpha=0.5, lw=2, linestyle="--")
        if d["val_loss"]:
            best = int(np.argmin(d["val_loss"]))
            ax.axvline(epochs[best], color="green", linestyle=":", alpha=0.7,
                       label=f"Best e{epochs[best]}")
        ax.set_title(f"{name} — Loss", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # IC
        ax = axes[idx, 1]
        ax.plot(epochs, d["val_ic"], color=color, lw=2, label="Val IC")
        if d["val_rank_ic"]:
            ax.plot(epochs, d["val_rank_ic"], color=color, lw=2, linestyle="--",
                    alpha=0.6, label="Val RankIC")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        if d["val_ic"]:
            best = int(np.argmax(d["val_ic"]))
            ax.axvline(epochs[best], color="green", linestyle=":", alpha=0.7,
                       label=f"Best IC={d['val_ic'][best]:.3f} e{epochs[best]}")
        ax.set_title(f"{name} — Val IC / RankIC", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("IC")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Overfitting ratio
        ax = axes[idx, 2]
        ratio = [v / max(t, 1e-12) for t, v in zip(d["train_loss"], d["val_loss"])]
        ax.plot(epochs, ratio, color=color, lw=2)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.axhline(1.2, color="orange", linewidth=0.8, linestyle="--", alpha=0.7, label="1.2×")
        ax.axhline(1.5, color="red",    linewidth=0.8, linestyle="--", alpha=0.7, label="1.5×")
        ax.set_title(f"{name} — Val/Train Loss Ratio", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Ratio")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 2: Predicted vs actual scatter
# ============================================================================

def plot_pred_vs_actual(pred_data: Dict, out_dir: str):
    model_map = {"lstm_d": "LSTM_D", "tft_d": "TFT_D",
                 "lstm_m": "LSTM_M", "tft_m": "TFT_M"}
    names = [k for k in model_map if k in pred_data and len(pred_data[k]["preds"]) > 0]
    if not names:
        return

    n_cols = min(len(names), 2)
    n_rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
    fig.suptitle("Predicted vs Actual Returns (Test Set)", fontsize=14, fontweight="bold")

    for idx, key in enumerate(names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        name = model_map[key]
        p = np.array(pred_data[key]["preds"])
        t = np.array(pred_data[key]["targets"])
        color = COLORS.get(name, "steelblue")

        # Clip extremes for readability
        xlim = np.percentile(np.abs(p), 99) * 1.1
        ylim = np.percentile(np.abs(t), 99) * 1.1

        ax.scatter(p, t, alpha=0.05, s=2, color=color, rasterized=True)
        # Regression line
        if np.std(p) > 1e-10:
            m_coef, b_coef = np.polyfit(p, t, 1)
            x_line = np.linspace(-xlim, xlim, 100)
            ax.plot(x_line, m_coef * x_line + b_coef, color="red", lw=2, label=f"slope={m_coef:.2f}")
        ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
        ic = np.corrcoef(p, t)[0, 1] if np.std(p) > 1e-10 and np.std(t) > 1e-10 else 0.0
        ax.set_title(f"{name}  (IC={ic:.3f}, n={len(p):,})", fontweight="bold")
        ax.set_xlabel("Predicted Return"); ax.set_ylabel("Actual Return")
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Hide unused panels
    for idx in range(len(names), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "pred_vs_actual.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 3: Calibration (decile plot)
# ============================================================================

def plot_calibration(pred_data: Dict, out_dir: str):
    model_map = {"lstm_d": "LSTM_D", "tft_d": "TFT_D",
                 "lstm_m": "LSTM_M", "tft_m": "TFT_M"}
    names = [k for k in model_map if k in pred_data and len(pred_data[k]["preds"]) > 50]
    if not names:
        return

    n_cols = min(len(names), 2)
    n_rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle("Calibration: Mean Actual Return per Prediction Decile (Test Set)",
                 fontsize=13, fontweight="bold")

    for idx, key in enumerate(names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        name = model_map[key]
        p = np.array(pred_data[key]["preds"])
        t = np.array(pred_data[key]["targets"])
        color = COLORS.get(name, "steelblue")

        n_deciles = 10
        decile_labels = list(range(1, n_deciles + 1))
        quantile_bins = np.percentile(p, np.linspace(0, 100, n_deciles + 1))
        # Ensure strictly increasing bins
        quantile_bins = np.unique(quantile_bins)
        if len(quantile_bins) < 3:
            ax.set_title(f"{name} — insufficient variance"); continue

        bin_indices = np.digitize(p, quantile_bins[1:-1])
        means, stds, counts = [], [], []
        actual_labels = []
        for b in range(len(quantile_bins) - 1):
            mask = bin_indices == b
            if mask.sum() < 5:
                continue
            means.append(t[mask].mean())
            stds.append(t[mask].std() / np.sqrt(mask.sum()))
            actual_labels.append(b + 1)

        if not means:
            continue

        bar_colors = ["#d32f2f" if m < 0 else "#388e3c" for m in means]
        ax.bar(actual_labels, means, yerr=stds, color=bar_colors, alpha=0.8,
               capsize=4, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", lw=1)
        # Trend line
        if len(actual_labels) > 2:
            z = np.polyfit(actual_labels, means, 1)
            xl = np.array([actual_labels[0], actual_labels[-1]])
            ax.plot(xl, np.polyval(z, xl), color="navy", lw=2, linestyle="--",
                    label=f"slope={z[0]*1e4:.1f}×10⁻⁴")
        ax.set_title(f"{name} — Decile Calibration", fontweight="bold")
        ax.set_xlabel("Prediction Decile (1=lowest, 10=highest)")
        ax.set_ylabel("Mean Actual Return")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    for idx in range(len(names), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 4: Rolling IC over time
# ============================================================================

def plot_ic_over_time(pred_data: Dict, out_dir: str, window: int = 30):

    model_map = {"lstm_d": "LSTM_D", "tft_d": "TFT_D",
                 "lstm_m": "LSTM_M", "tft_m": "TFT_M"}

    fig, ax = plt.subplots(figsize=(14, 5))
    any_plotted = False

    for key, name in model_map.items():
        if key not in pred_data or len(pred_data[key]["preds"]) < window * 2:
            continue
        p = np.array(pred_data[key]["preds"])
        t = np.array(pred_data[key]["targets"])
        dates = np.array(pred_data[key]["dates"])

        # Sort by date
        sort_idx = np.argsort(dates)
        p, t, dates = p[sort_idx], t[sort_idx], dates[sort_idx]

        # Rolling IC
        rolling_ic = []
        rolling_dates = []
        for i in range(window, len(p)):
            pw = p[i - window:i]
            tw = t[i - window:i]
            if np.std(pw) < 1e-10 or np.std(tw) < 1e-10:
                rolling_ic.append(0.0)
            else:
                rolling_ic.append(float(np.corrcoef(pw, tw)[0, 1]))
            rolling_dates.append(dates[i])

        if not rolling_ic:
            continue

        import datetime
        plot_dates = [datetime.date.fromordinal(int(d)) for d in rolling_dates]
        ax.plot(plot_dates, rolling_ic, label=name, color=COLORS.get(name, "gray"),
                lw=1.5, alpha=0.85)
        any_plotted = True

    if not any_plotted:
        plt.close(); return

    ax.axhline(0, color="black", lw=1)
    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], alpha=0.04, color="green")
    ax.fill_between(ax.get_xlim(), ax.get_ylim()[0], 0, alpha=0.04, color="red")
    ax.set_title(f"Rolling {window}-sample IC Over Time (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("IC")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ic_over_time.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 5: Prediction distribution
# ============================================================================

def plot_pred_distribution(pred_data: Dict, out_dir: str):
    model_map = {"lstm_d": "LSTM_D", "tft_d": "TFT_D",
                 "lstm_m": "LSTM_M", "tft_m": "TFT_M"}
    names = [k for k in model_map if k in pred_data and len(pred_data[k]["preds"]) > 0]
    if not names:
        return

    n_cols = min(len(names), 2)
    n_rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle("Prediction vs Actual Return Distribution (Test Set)",
                 fontsize=13, fontweight="bold")

    for idx, key in enumerate(names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        name = model_map[key]
        p = np.array(pred_data[key]["preds"])
        t = np.array(pred_data[key]["targets"])
        color = COLORS.get(name, "steelblue")

        clip = np.percentile(np.abs(t), 99)
        bins = np.linspace(-clip, clip, 60)
        ax.hist(t, bins=bins, alpha=0.5, color="gray",    label="Actual",    density=True)
        ax.hist(p, bins=bins, alpha=0.7, color=color,     label="Predicted", density=True)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(p.mean(), color=color, lw=1.5, linestyle="--",
                   label=f"Pred mean={p.mean()*100:.3f}%")
        ax.set_title(f"{name}  (pred std={p.std()*100:.3f}%)", fontweight="bold")
        ax.set_xlabel("Return"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for idx in range(len(names), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "pred_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 6: Multi-horizon IC
# ============================================================================

def plot_multi_horizon_ic(ic_by_horizon: Dict[int, float], out_dir: str):
    horizons = sorted(ic_by_horizon.keys())
    ics = [ic_by_horizon[h] for h in horizons]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#388e3c" if v >= 0 else "#d32f2f" for v in ics]
    bars = ax.bar(horizons, ics, color=bar_colors, alpha=0.85, width=0.8)
    ax.axhline(0, color="black", lw=1)

    for bar, ic in zip(bars, ics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                ic + (0.002 if ic >= 0 else -0.004),
                f"{ic:.3f}", ha="center", va="bottom" if ic >= 0 else "top",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Forecast Horizon (trading days)", fontsize=12)
    ax.set_ylabel("Information Coefficient (IC)", fontsize=12)
    ax.set_title("LSTM_D: IC vs Forecast Horizon (Post-hoc, No Retraining)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "multi_horizon_ic.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 7: Equity curve (simple top-K long portfolio)
# ============================================================================

def plot_equity_curve(pred_data: Dict, out_dir: str, top_k: int = 20):
    import datetime

    # Use daily LSTM_D since it has the most samples and date coverage
    key = "lstm_d"
    if key not in pred_data or len(pred_data[key]["preds"]) < 100:
        # Fall back to tft_d
        key = "tft_d"
    if key not in pred_data or len(pred_data[key]["preds"]) < 100:
        return

    p = np.array(pred_data[key]["preds"])
    t = np.array(pred_data[key]["targets"])
    dates = np.array(pred_data[key]["dates"])

    sort_idx = np.argsort(dates)
    p, t, dates = p[sort_idx], t[sort_idx], dates[sort_idx]

    unique_dates = np.unique(dates)
    if len(unique_dates) < 10:
        return

    portfolio_returns = []
    bh_returns = []
    plot_dates = []

    for d in unique_dates:
        mask = dates == d
        if mask.sum() < max(5, top_k):
            continue
        pd_slice = p[mask]
        td_slice = t[mask]

        # Top-K long
        top_idx = np.argsort(pd_slice)[-top_k:]
        portfolio_returns.append(td_slice[top_idx].mean())

        # Equal-weight all stocks (proxy for market)
        bh_returns.append(td_slice.mean())
        plot_dates.append(datetime.date.fromordinal(int(d)))

    if len(portfolio_returns) < 5:
        return

    portfolio_cumret = np.cumprod(1 + np.array(portfolio_returns))
    bh_cumret        = np.cumprod(1 + np.array(bh_returns))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Simulated Portfolio: Top-{top_k} Long vs Equal-Weight (Test Set)",
                 fontsize=13, fontweight="bold")

    ax1.plot(plot_dates, portfolio_cumret, label=f"Top-{top_k} Long (LSTM_D)",
             color=COLORS["LSTM_D"], lw=2)
    ax1.plot(plot_dates, bh_cumret, label="Equal-Weight All",
             color="gray", lw=2, linestyle="--")
    ax1.axhline(1.0, color="black", lw=0.8)
    ax1.set_ylabel("Cumulative Return (1 = start)", fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    # Daily returns bar chart
    ax2.bar(plot_dates, portfolio_returns,
            color=["#388e3c" if r >= 0 else "#d32f2f" for r in portfolio_returns],
            alpha=0.7, width=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Daily Return", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    # Stats annotation
    total_ret   = portfolio_cumret[-1] - 1
    bh_total    = bh_cumret[-1] - 1
    n_days      = len(portfolio_returns)
    ann_factor  = 252 / n_days if n_days > 0 else 1
    ann_ret     = (1 + total_ret) ** ann_factor - 1
    vol         = np.std(portfolio_returns) * np.sqrt(252)
    sharpe      = ann_ret / (vol + 1e-10)
    dd          = 1 - portfolio_cumret / np.maximum.accumulate(portfolio_cumret)
    max_dd      = dd.max()

    stats_text = (f"Total return: {total_ret*100:.1f}%  (EW: {bh_total*100:.1f}%)\n"
                  f"Ann. return:  {ann_ret*100:.1f}%\n"
                  f"Ann. vol:     {vol*100:.1f}%\n"
                  f"Sharpe:       {sharpe:.2f}\n"
                  f"Max drawdown: {max_dd*100:.1f}%")
    ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(out_dir, "equity_curve.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Summary bar chart (from test_results.json)
# ============================================================================

def plot_summary_bars(test_results: Dict, out_dir: str):
    if not test_results:
        return

    name_map = {
        "LSTM_D": "LSTM_D", "TFT_D": "TFT_D",
        "LSTM_M": "LSTM_M", "TFT_M": "TFT_M",
        "META_ENSEMBLE": "Meta",
    }
    names  = [name_map[k] for k in ["LSTM_D","TFT_D","LSTM_M","TFT_M","META_ENSEMBLE"]
              if k in test_results]
    ics    = [test_results[k]["ic"]        for k in ["LSTM_D","TFT_D","LSTM_M","TFT_M","META_ENSEMBLE"]
              if k in test_results]
    rics   = [test_results[k]["rank_ic"]   for k in ["LSTM_D","TFT_D","LSTM_M","TFT_M","META_ENSEMBLE"]
              if k in test_results]
    daccs  = [test_results[k]["directional_accuracy"]
              for k in ["LSTM_D","TFT_D","LSTM_M","TFT_M","META_ENSEMBLE"]
              if k in test_results]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Test Set Summary Metrics", fontsize=14, fontweight="bold")

    for ax, vals, title, yref in zip(
        axes,
        [ics, rics, daccs],
        ["Information Coefficient (IC)", "Rank IC", "Directional Accuracy"],
        [0.0, 0.0, 0.5],
    ):
        bar_colors = [COLORS.get(n, "steelblue") for n in names]
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.85)
        ax.axhline(yref, color="black", lw=1, linestyle="--")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.001 if v >= yref else -0.003),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= yref else "top",
                    fontsize=9, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "summary_metrics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Main
# ============================================================================

def run(model_dir: str, top_k: int = 20, horizons: List[int] = None,
        skip_predictions: bool = False, data_cfg_overrides: dict = None):
    if horizons is None:
        horizons = [1, 3, 5, 10, 15]

    # Resolve model_dir to absolute so sub-functions work regardless of cwd
    model_dir = str(Path(model_dir).resolve())

    out_dir = os.path.join(model_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots → {out_dir}")

    # 1. Training curves (needs only CSVs)
    print("  Plotting training curves...")
    histories = load_history(model_dir)
    plot_training_curves(histories, out_dir)

    # 2. Summary bar chart from JSON
    test_results = load_test_results(model_dir)
    if test_results:
        print("  Plotting summary metrics...")
        plot_summary_bars(test_results, out_dir)

    if skip_predictions:
        print("  Skipping prediction-based plots (--no-predictions flag set)")
        return

    # 3-7. Plots that need raw predictions from the model
    data_cfg_kwargs = dict(split_mode="temporal")
    if data_cfg_overrides:
        data_cfg_kwargs.update(data_cfg_overrides)

    print("  Collecting test predictions (this may take a few minutes)...")
    pred_data = collect_test_predictions(model_dir, data_cfg_kwargs)

    if pred_data is not None:
        print("  Plotting pred vs actual scatter...")
        plot_pred_vs_actual(pred_data, out_dir)
        print("  Plotting calibration...")
        plot_calibration(pred_data, out_dir)
        print("  Plotting IC over time...")
        plot_ic_over_time(pred_data, out_dir)
        print("  Plotting prediction distributions...")
        plot_pred_distribution(pred_data, out_dir)
        print("  Plotting equity curve...")
        plot_equity_curve(pred_data, out_dir, top_k=top_k)

    print(f"  Computing multi-horizon IC (horizons={horizons})...")
    ic_by_horizon = collect_multi_horizon_predictions(model_dir, data_cfg_kwargs, horizons)
    if ic_by_horizon:
        print(f"  IC by horizon: { {h: f'{v:.3f}' for h,v in ic_by_horizon.items()} }")
        plot_multi_horizon_ic(ic_by_horizon, out_dir)

    print(f"\nDone. {len(os.listdir(out_dir))} files in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot all diagnostics for a model run")
    parser.add_argument("--model-dir", required=True, help="e.g. models/hierarchical_v5")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K stocks for equity curve simulation")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 15],
                        help="Forecast horizons for multi-horizon IC plot")
    parser.add_argument("--no-predictions", action="store_true",
                        help="Skip plots that require running the model (faster)")
    parser.add_argument("--split-mode", type=str, default="temporal")
    parser.add_argument("--daily-seq-len", type=int, default=720)
    parser.add_argument("--minute-seq-len", type=int, default=780)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    data_cfg_overrides = {
        "split_mode": args.split_mode,
        "daily_seq_len": args.daily_seq_len,
        "minute_seq_len": args.minute_seq_len,
    }

    run(
        model_dir=args.model_dir,
        top_k=args.top_k,
        horizons=args.horizons,
        skip_predictions=args.no_predictions,
        data_cfg_overrides=data_cfg_overrides,
    )


if __name__ == "__main__":
    main()
