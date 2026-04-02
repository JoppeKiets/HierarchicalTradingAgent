#!/usr/bin/env python3
"""Generate all diagnostic plots for a completed hierarchical model run.

Fully dynamic — automatically discovers all sub-models present in the
checkpoint / history CSVs.  Works with any combination of:
    lstm_d, tft_d, tcn_d, lstm_m, tft_m, news, fund_mlp, gnn, Meta, Joint

Plots produced (saved to {model_dir}/plots/):
  1. training_curves.png   — loss + IC per epoch for every sub-model
  2. pred_vs_actual.png    — scatter of predicted vs actual returns
  3. calibration.png       — decile calibration: mean actual return per pred decile
  4. ic_over_time.png      — rolling 30-day IC on test set ordered by date
  5. pred_distribution.png — histogram of predictions vs actuals
  6. multi_horizon_ic.png  — IC at horizons 1,3,5,10,15 days (post-hoc)
  7. equity_curve.png      — simulated long-only top-K portfolio vs equal-weight
  8. summary_metrics.png   — bar chart of test-set IC / RankIC / DirAcc per model

Usage:
    python scripts/plot_results.py --model-dir models/hierarchical_v9
    python scripts/plot_results.py --model-dir models/hierarchical_v8 --top-k 20
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
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#888888",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# Dynamic colour palette — every model gets a consistent colour
_PALETTE = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#F44336",  # red
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
    "#E91E63",  # pink
    "#CDDC39",  # lime
    "#FF5722",  # deep orange
    "#3F51B5",  # indigo
]

# Preferred order when available
_PREFERRED_ORDER = [
    "LSTM_D", "TFT_D", "TCN_D",
    "LSTM_M", "TFT_M",
    "News", "FundMLP", "GNN",
    "Meta", "Meta Ensemble", "Joint",
]


def _display_name(raw: str) -> str:
    """Normalise a CSV stem / JSON key to a readable display name."""
    mapping = {
        "lstm_d": "LSTM_D", "tft_d": "TFT_D", "tcn_d": "TCN_D",
        "lstm_m": "LSTM_M", "tft_m": "TFT_M",
        "news": "News", "fund_mlp": "FundMLP", "gnn": "GNN",
        "meta": "Meta", "joint": "Joint",
        "META_ENSEMBLE": "Meta Ensemble",
        "LSTM_D": "LSTM_D", "TFT_D": "TFT_D", "TCN_D": "TCN_D",
        "LSTM_M": "LSTM_M", "TFT_M": "TFT_M",
        "News": "News", "FundMLP": "FundMLP", "GNN": "GNN",
        "Meta": "Meta", "Joint": "Joint",
    }
    return mapping.get(raw, raw)


def _color_for(name: str, idx: int = 0) -> str:
    """Return a colour for a given display name."""
    fixed = {
        "LSTM_D": "#2196F3", "TFT_D": "#4CAF50", "TCN_D": "#00BCD4",
        "LSTM_M": "#FF9800", "TFT_M": "#9C27B0",
        "News": "#795548", "FundMLP": "#E91E63", "GNN": "#607D8B",
        "Meta": "#F44336", "Meta Ensemble": "#F44336", "Joint": "#FF5722",
    }
    return fixed.get(name, _PALETTE[idx % len(_PALETTE)])


def _sort_names(names: List[str]) -> List[str]:
    """Sort model names by preferred order, unknowns at end."""
    order = {n: i for i, n in enumerate(_PREFERRED_ORDER)}
    return sorted(names, key=lambda n: order.get(n, 999))


# ============================================================================
# Data loading helpers
# ============================================================================

def load_history(model_dir: str) -> Dict[str, Dict]:
    """Load all *_history.csv files — discovers model names automatically."""
    histories = {}
    for csv_path in sorted(Path(model_dir).glob("*_history.csv")):
        raw_name = csv_path.stem.replace("_history", "")
        display = _display_name(raw_name)
        data = {"epoch": [], "train_loss": [], "val_loss": [],
                "val_ic": [], "val_rank_ic": []}
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data["epoch"].append(int(row["epoch"]))
                    data["train_loss"].append(float(row["train_loss"]))
                    data["val_loss"].append(float(row["val_loss"]))
                    ic = row.get("val_ic", "") or "0"
                    ric = row.get("val_rank_ic", "") or "0"
                    data["val_ic"].append(
                        float(ic) if ic not in ("", "nan") else 0.0)
                    data["val_rank_ic"].append(
                        float(ric) if ric not in ("", "nan") else 0.0)
        except Exception as e:
            logger.warning(f"Skipping {csv_path}: {e}")
            continue
        if data["epoch"]:
            histories[display] = data
    return histories


def load_test_results(model_dir: str) -> Optional[Dict]:
    path = Path(model_dir) / "test_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_test_predictions(model_dir: str,
                             data_cfg_kwargs: dict) -> Optional[Dict]:
    """Run every sub-model on the test split and return raw arrays.

    Fully dynamic — discovers sub-models from the loaded checkpoint.
    """
    try:
        import torch
        from src.hierarchical_data import (HierarchicalDataConfig,
                                           create_dataloaders,
                                           get_viable_tickers, split_tickers)
        from src.hierarchical_models import HierarchicalForecaster
    except ImportError as e:
        logger.warning(f"Cannot collect predictions (import error): {e}")
        return None

    model_dir_path = Path(model_dir)
    for candidate in ["finetune_best.pt", "forecaster_final.pt",
                       "checkpoint_phase4.pt", "checkpoint_phase3.pt",
                       "meta_best.pt", "checkpoint_phase2.pt",
                       "checkpoint_phase1.pt"]:
        model_path = model_dir_path / candidate
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            break
    else:
        logger.warning(f"No model checkpoint found in {model_dir}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = HierarchicalForecaster.load(str(model_path), device=str(device))
    forecaster.to(device).eval()

    data_cfg = HierarchicalDataConfig(**data_cfg_kwargs)

    split_path = model_dir_path / "ticker_split.json"
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
    else:
        tickers = get_viable_tickers(data_cfg)
        splits = split_tickers(tickers, data_cfg)

    loaders = create_dataloaders(
        splits, data_cfg, batch_size_daily=64,
        batch_size_minute=32, num_workers=0)

    # Optional news / fundamental loaders
    news_loader = None
    fund_loader = None
    if "news" in forecaster.sub_model_names:
        try:
            from src.news_data import NewsDataConfig, create_news_dataloaders
            news_cfg = NewsDataConfig(
                seq_len=data_cfg.daily_seq_len,
                stride=data_cfg.daily_stride,
                forecast_horizon=data_cfg.forecast_horizon,
                split_mode=data_cfg.split_mode,
                temporal_train_frac=data_cfg.temporal_train_frac,
                temporal_val_frac=data_cfg.temporal_val_frac,
                temporal_test_frac=data_cfg.temporal_test_frac,
            )
            news_loaders = create_news_dataloaders(
                splits, news_cfg,
                daily_cache_dir=str(Path(data_cfg.cache_dir) / "daily"),
                batch_size=32, num_workers=0)
            news_loader = news_loaders.get("test")
        except Exception:
            pass

    if "fund_mlp" in forecaster.sub_model_names:
        try:
            from src.hierarchical_data import create_fundamental_dataloaders
            fund_loaders = create_fundamental_dataloaders(
                splits, data_cfg, batch_size=256, num_workers=0)
            fund_loader = fund_loaders.get("test")
        except Exception:
            pass

    modality_loaders = {
        "daily": loaders["daily"]["test"],
        "minute": loaders["minute"]["test"],
        "news": news_loader,
        "fundamental": fund_loader,
    }

    results: Dict[str, Dict] = {}

    with torch.no_grad():
        for name in forecaster.sub_model_names:
            modality = HierarchicalForecaster.MODALITY.get(name, "daily")
            if modality == "graph":
                continue  # GNN cross-sectional — skip for per-sample plots
            loader = modality_loaders.get(modality)
            if loader is None:
                continue
            sub = forecaster.sub_models[name].to(device).eval()
            preds, targets, dates = [], [], []
            for batch in loader:
                x, y, ord_dates, _ = batch
                x = x.to(device)
                out = sub(x)
                preds.extend(out["prediction"].cpu().numpy())
                targets.extend(y.numpy())
                dates.extend(ord_dates.numpy())
            sub.cpu()
            results[name] = {
                "preds": np.array(preds),
                "targets": np.array(targets),
                "dates": np.array(dates),
            }

    return results


def collect_multi_horizon_predictions(
    model_dir: str, data_cfg_kwargs: dict, horizons: List[int],
) -> Optional[Dict[int, float]]:
    """Post-hoc multi-horizon IC using first available daily model."""
    try:
        import torch
        from src.hierarchical_data import (HierarchicalDataConfig,
                                           get_viable_tickers, split_tickers)
        from src.hierarchical_models import HierarchicalForecaster
    except ImportError:
        return None

    model_dir_path = Path(model_dir)
    for candidate in ["finetune_best.pt", "forecaster_final.pt",
                       "checkpoint_phase3.pt", "meta_best.pt"]:
        model_path = model_dir_path / candidate
        if model_path.exists():
            break
    else:
        return None

    data_cfg = HierarchicalDataConfig(**data_cfg_kwargs)
    cache = Path(data_cfg.cache_dir) / "daily"

    split_path = model_dir_path / "ticker_split.json"
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
    else:
        tickers = get_viable_tickers(data_cfg)
        splits = split_tickers(tickers, data_cfg)
    test_tickers = splits["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = HierarchicalForecaster.load(str(model_path), device=str(device))
    forecaster.to(device).eval()

    # Pick the first daily model available
    daily_model_name = None
    for n in ["lstm_d", "tft_d", "tcn_d"]:
        if n in forecaster.sub_model_names:
            daily_model_name = n
            break
    if daily_model_name is None:
        return None
    sub = forecaster.sub_models[daily_model_name].to(device).eval()

    seq_len = forecaster.cfg.daily_seq_len
    horizon_results = {h: {"preds": [], "targets": []} for h in horizons}

    with torch.no_grad():
        for ticker in test_tickers:
            feat_path = cache / f"{ticker}_features.npy"
            if not feat_path.exists():
                continue
            feat = np.load(feat_path, mmap_mode="r")
            n = feat.shape[0]

            price_file = (Path(data_cfg.organized_dir) / ticker /
                          "price_history.csv")
            if not price_file.exists():
                continue
            try:
                import pandas as pd
                df = pd.read_csv(price_file)
                df.columns = [c.strip().lower() for c in df.columns]
                if "adj close" in df.columns:
                    df["close"] = df["adj close"].fillna(df["close"])
                df = df.sort_values("date").reset_index(drop=True)
                close = df["close"].values.astype(np.float64)
                if len(close) != n:
                    continue
            except Exception:
                continue

            fh = getattr(data_cfg, "forecast_horizon", 1)
            warmup = max(seq_len, 60)
            usable = (n - fh) - warmup
            if usable <= 0:
                continue
            val_end = warmup + int(usable * (data_cfg.temporal_train_frac
                                              + data_cfg.temporal_val_frac))
            test_end = n - fh
            stride = data_cfg.daily_stride

            for i in range(val_end, test_end, stride):
                if i - seq_len < 0:
                    continue
                x = torch.from_numpy(
                    feat[i - seq_len:i].copy()
                ).unsqueeze(0).to(device)
                pred = sub(x)["prediction"].item()

                for h in horizons:
                    if i + h >= len(close) or close[i] <= 0:
                        continue
                    if np.isnan(close[i + h]) or np.isnan(close[i]):
                        continue
                    target = float(close[i + h] / close[i] - 1.0)
                    if not np.isfinite(target):
                        continue
                    target = float(np.clip(target, -0.5, 0.5))
                    horizon_results[h]["preds"].append(pred)
                    horizon_results[h]["targets"].append(target)

    sub.cpu()

    ic_by_horizon: Dict[int, float] = {}
    for h, d in horizon_results.items():
        p = np.array(d["preds"])
        t = np.array(d["targets"])
        mask = np.isfinite(p) & np.isfinite(t)
        p, t = p[mask], t[mask]
        if len(p) < 10 or np.std(p) < 1e-10 or np.std(t) < 1e-10:
            ic_by_horizon[h] = 0.0
        else:
            raw = float(np.corrcoef(p, t)[0, 1])
            ic_by_horizon[h] = 0.0 if np.isnan(raw) else raw
    return ic_by_horizon


# ============================================================================
# Plot 1: Training curves  (3-column: loss, IC, overfit ratio)
# ============================================================================

def plot_training_curves(histories: Dict, out_dir: str):
    names = _sort_names([n for n in histories])
    if not names:
        return

    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(20, 3.8 * n), squeeze=False)
    fig.suptitle("Training Curves — All Sub-Models",
                 fontsize=16, fontweight="bold", y=1.005)

    for idx, name in enumerate(names):
        d = histories[name]
        epochs = d["epoch"]
        color = _color_for(name, idx)

        # ---- Column 0: Loss ----
        ax = axes[idx, 0]
        ax.plot(epochs, d["train_loss"], label="Train loss",
                color=color, lw=2, alpha=0.9)
        ax.plot(epochs, d["val_loss"], label="Val loss",
                color=color, lw=2, alpha=0.5, linestyle="--")
        if d["val_loss"]:
            best = int(np.argmin(d["val_loss"]))
            ax.axvline(epochs[best], color="#388e3c", ls=":", lw=1.2,
                       label=f"Best val ← e{epochs[best]}")
            ax.scatter([epochs[best]], [d["val_loss"][best]], color="#388e3c",
                       zorder=5, s=50, marker="v")
        ax.set_title(f"{name} — Loss", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.25)

        # ---- Column 1: IC / Rank IC ----
        ax = axes[idx, 1]
        ax.plot(epochs, d["val_ic"], color=color, lw=2,
                label="Val IC", alpha=0.9)
        if any(v != 0 for v in d["val_rank_ic"]):
            ax.plot(epochs, d["val_rank_ic"], color=color, lw=1.5,
                    ls="--", alpha=0.55, label="Val Rank IC")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        if d["val_ic"]:
            best = int(np.argmax(d["val_ic"]))
            ax.axvline(epochs[best], color="#388e3c", ls=":", lw=1.2)
            ax.scatter([epochs[best]], [d["val_ic"][best]], color="#388e3c",
                       zorder=5, s=50, marker="^",
                       label=f"Best IC = {d['val_ic'][best]:.4f}")
        ax.set_title(f"{name} — Information Coefficient", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("IC")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.25)

        # ---- Column 2: Overfit ratio ----
        ax = axes[idx, 2]
        ratio = [v / max(t, 1e-12) for t, v in
                 zip(d["train_loss"], d["val_loss"])]
        ax.plot(epochs, ratio, color=color, lw=2)
        ax.axhline(1.0, color="black", lw=1, ls=":")
        ax.axhspan(1.0, 1.2, color="#c8e6c9", alpha=0.25,
                   label="Healthy (1–1.2×)")
        ax.axhspan(1.2, 1.5, color="#fff9c4", alpha=0.25,
                   label="Mild overfit (1.2–1.5×)")
        ax.axhspan(1.5, max(max(ratio, default=1.6), 1.6),
                   color="#ffcdd2", alpha=0.25, label="Strong overfit (>1.5×)")
        ax.set_title(f"{name} — Val / Train Ratio", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Ratio")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 2: Predicted vs actual scatter
# ============================================================================

def plot_pred_vs_actual(pred_data: Dict, out_dir: str):
    names = _sort_names([_display_name(k) for k in pred_data
                         if len(pred_data[k]["preds"]) > 0])
    raw_keys = {_display_name(k): k for k in pred_data}
    if not names:
        return

    n_cols = min(len(names), 3)
    n_rows = (len(names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
    fig.suptitle("Predicted vs Actual Returns  (Test Set)",
                 fontsize=15, fontweight="bold")

    for idx, name in enumerate(names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        key = raw_keys[name]
        p = pred_data[key]["preds"]
        t = pred_data[key]["targets"]
        color = _color_for(name, idx)

        xlim = np.percentile(np.abs(p), 99) * 1.2
        ylim = np.percentile(np.abs(t), 99) * 1.2

        ax.scatter(p, t, alpha=0.04, s=1.5, color=color, rasterized=True)

        # Regression line
        if np.std(p) > 1e-10:
            slope, intercept = np.polyfit(p, t, 1)
            xr = np.linspace(-xlim, xlim, 100)
            ax.plot(xr, slope * xr + intercept, color="#d32f2f", lw=2,
                    label=f"slope = {slope:.2f}")
        ax.plot([-xlim, xlim], [-xlim, xlim], color="black", lw=0.6, ls=":",
                alpha=0.4, label="y = x")
        ax.axhline(0, color="black", lw=0.4)
        ax.axvline(0, color="black", lw=0.4)

        ic = (np.corrcoef(p, t)[0, 1]
              if np.std(p) > 1e-10 and np.std(t) > 1e-10 else 0.0)
        ax.set_title(f"{name}   IC = {ic:.4f}   n = {len(p):,}",
                     fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    for idx in range(len(names), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "pred_vs_actual.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 3: Calibration (decile plot)
# ============================================================================

def plot_calibration(pred_data: Dict, out_dir: str):
    names = _sort_names([_display_name(k) for k in pred_data
                         if len(pred_data[k]["preds"]) > 50])
    raw_keys = {_display_name(k): k for k in pred_data}
    if not names:
        return

    n_cols = min(len(names), 3)
    n_rows = (len(names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(
        "Calibration — Mean Actual Return per Prediction Decile  (Test Set)",
        fontsize=14, fontweight="bold")

    for idx, name in enumerate(names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        key = raw_keys[name]
        p = pred_data[key]["preds"]
        t = pred_data[key]["targets"]

        qbins = np.percentile(p, np.linspace(0, 100, 11))
        qbins = np.unique(qbins)
        if len(qbins) < 3:
            ax.set_title(f"{name} — insufficient variance")
            continue

        bin_idx = np.digitize(p, qbins[1:-1])
        means, sems, labels = [], [], []
        for b in range(len(qbins) - 1):
            mask = bin_idx == b
            if mask.sum() < 5:
                continue
            means.append(t[mask].mean())
            sems.append(t[mask].std() / np.sqrt(mask.sum()))
            labels.append(b + 1)

        if not means:
            continue

        bar_c = ["#c62828" if m < 0 else "#2e7d32" for m in means]
        ax.bar(labels, means, yerr=sems, color=bar_c, alpha=0.82,
               capsize=4, edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1.0})
        ax.axhline(0, color="black", lw=1)
        if len(labels) > 2:
            z = np.polyfit(labels, means, 1)
            xl = np.array([labels[0], labels[-1]])
            ax.plot(xl, np.polyval(z, xl), color="navy", lw=2, ls="--",
                    label=f"slope = {z[0]*1e4:.2f} × 10⁻⁴")
        spread = means[-1] - means[0] if len(means) >= 2 else 0
        ax.set_title(f"{name}  (spread = {spread*100:.3f}%)",
                     fontweight="bold")
        ax.set_xlabel("Prediction Decile  (1 = most bearish)")
        ax.set_ylabel("Mean Actual Return")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y")

    for idx in range(len(names), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 4: Rolling IC over time
# ============================================================================

def plot_ic_over_time(pred_data: Dict, out_dir: str, window: int = 30):
    """Improved IC-over-time plot with two panels:

    Top panel: Monthly IC bars (mean actual IC per calendar month) with
               ±1σ error bars across the individual daily cross-sections.
               Much more stable than a rolling-window line; each bar
               represents dozens of dates so the estimate is reliable.

    Bottom panel: Regime-stratified IC — loads regime_ic_*.json files
                  written during training to show per-regime IC for each
                  sub-model.  If those files are absent, shows the
                  rolling 60-sample IC instead of the noisy 30-sample.
    """
    import datetime
    from matplotlib.lines import Line2D

    raw_map = {_display_name(k): k for k in pred_data}
    # Consider models with at least some data
    all_models = _sort_names([_display_name(k) for k in pred_data
                              if len(pred_data[k]["preds"]) >= 60])
    if not all_models:
        return

    # Build monthly buckets for all models
    monthly_data: Dict[str, Dict[str, Dict]] = {}
    for dname in all_models:
        raw = raw_map[dname]
        p = pred_data[raw]["preds"].copy()
        t = pred_data[raw]["targets"].copy()
        dates = pred_data[raw]["dates"].copy()

        monthly: Dict[str, Dict] = {}
        for pi, ti, di in zip(p, t, dates):
            d = datetime.date.fromordinal(int(di))
            ym = f"{d.year}-{d.month:02d}"
            if ym not in monthly:
                monthly[ym] = {"preds": [], "tgts": []}
            monthly[ym]["preds"].append(pi)
            monthly[ym]["tgts"].append(ti)
        monthly_data[dname] = monthly

    # Collect union of all months
    all_yms = sorted({ym for md in monthly_data.values() for ym in md})
    ym_dates = [datetime.date(int(ym.split("-")[0]), int(ym.split("-")[1]), 1)
                for ym in all_yms]

    # Check for regime_ic JSON files (unchanged behavior)
    regime_ic_available = False
    regime_ic_data: Dict[str, Dict[str, float]] = {}
    if out_dir:
        model_dir = os.path.dirname(out_dir)
        for dname in all_models:
            raw_key = raw_map[dname].lower().replace(" ", "_")
            json_path = os.path.join(model_dir, f"regime_ic_{raw_key}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        regime_ic_data[dname] = json.load(f)
                    regime_ic_available = True
                except Exception:
                    pass

    # Choose top-panel models: show only TCN_D (if present), else first available
    top_models = [m for m in all_models if m == "TCN_D"]
    if not top_models:
        top_models = [all_models[0]]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("IC Over Time — Monthly Aggregation & Regime Breakdown",
                 fontsize=14, fontweight="bold")

    # ---------- Top panel: monthly bars for selected models ----------
    ax1 = axes[0]
    bar_width = 20 / max(len(top_models), 1)
    offsets = np.linspace(-(len(top_models)-1)/2, (len(top_models)-1)/2,
                          len(top_models)) * bar_width

    for idx, dname in enumerate(top_models):
        monthly = monthly_data.get(dname, {})
        ics, errs, xs = [], [], []
        for ym, xd in zip(all_yms, ym_dates):
            if ym not in monthly:
                continue
            mp = np.array(monthly[ym]["preds"])
            mt = np.array(monthly[ym]["tgts"])
            if len(mp) < 5 or np.std(mp) < 1e-10 or np.std(mt) < 1e-10:
                continue
            # compute daily cross-sectional ICs within the month for error bars
            dates_in_month = sorted(set(
                datetime.date.fromordinal(int(d))
                for d in pred_data[raw_map[dname]]["dates"]
                if datetime.date.fromordinal(int(d)).strftime("%Y-%m") == ym
            ))
            daily_ics = []
            for day in dates_in_month:
                day_ord = day.toordinal()
                day_p = pred_data[raw_map[dname]]["preds"][
                    pred_data[raw_map[dname]]["dates"] == day_ord]
                day_t = pred_data[raw_map[dname]]["targets"][
                    pred_data[raw_map[dname]]["dates"] == day_ord]
                if len(day_p) > 2 and np.std(day_p) > 1e-10 and np.std(day_t) > 1e-10:
                    daily_ics.append(float(np.corrcoef(day_p, day_t)[0, 1]))
            if daily_ics:
                ic_val = float(np.mean(daily_ics))
                err_val = float(np.std(daily_ics))
            else:
                try:
                    ic_val = float(np.corrcoef(mp, mt)[0, 1])
                except Exception:
                    ic_val = 0.0
                err_val = 0.0
            ics.append(ic_val)
            errs.append(err_val)
            xs.append(xd)

        if not xs:
            continue
        color = _color_for(dname, idx)
        x_numeric = [matplotlib.dates.date2num(x) for x in xs]
        x_shifted = [xn + offsets[idx] for xn in x_numeric]
        bar_c = [color if ic >= 0 else "#c62828" for ic in ics]
        avg_ic = float(np.mean(ics))
        ax1.bar(x_shifted, ics, width=bar_width * 0.85,
                color=bar_c, alpha=0.75, label=f"{dname}  (avg={avg_ic:.3f})")
        ax1.errorbar(x_shifted, ics, yerr=errs, fmt="none",
                     ecolor="#666666", elinewidth=0.8, capsize=3, alpha=0.8)

        # 12-month rolling average (smoothed mean)
        if len(ics) >= 3:
            k = min(12, max(1, len(ics)))
            smoothed = np.convolve(ics, np.ones(k) / k, mode="same")
            ax1.plot(x_numeric, smoothed, color=color, lw=2.5, alpha=0.95,
                     linestyle="-", label=f"{dname} {k}-mo avg")

    # label zero line explicitly
    ax1.axhline(0, color="black", lw=1)
    try:
        xmax = matplotlib.dates.date2num(ym_dates[-1]) if ym_dates else None
        if xmax is not None:
            ax1.text(xmax, 0, "  IC = 0 (no skill)", ha="right", va="bottom",
                     fontsize=9, color="black")
    except Exception:
        pass

    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax1.set_title("Monthly IC  (bars = mean daily cross-sectional IC; error bars = ±1σ across days)",
                  fontweight="bold")
    ax1.set_ylabel("IC")

    # Add proxy legend entry for the error bars (they're plotted with fmt='none')
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#666666", lw=1.2))
    labels.append("±1σ across daily ICs")
    ax1.legend(handles, labels, fontsize=9, loc="best", ncol=2)
    ax1.grid(True, alpha=0.25, axis="y")

    # ---------- Bottom panel: Regime bars OR heatmap fallback ----------
    ax2 = axes[1]
    if regime_ic_available and regime_ic_data:
        # Keep previous regime bar behaviour
        all_regimes = sorted({r for md in regime_ic_data.values() for r in md})
        x = np.arange(len(all_regimes))
        bar_w = 0.8 / max(len(regime_ic_data), 1)
        for idx, dname in enumerate(all_models):
            if dname not in regime_ic_data:
                continue
            vals = [regime_ic_data[dname].get(r, 0.0) for r in all_regimes]
            bar_c = [_color_for(dname, idx) if v >= 0 else "#c62828" for v in vals]
            offset = (idx - (len(all_models) - 1) / 2) * bar_w
            ax2.bar(x + offset, vals, width=bar_w * 0.9, color=bar_c,
                    alpha=0.8, label=dname)
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_regimes, rotation=25, ha="right")
        ax2.set_title("Per-Regime IC  (from regime_ic_*.json training logs)",
                      fontweight="bold")
        ax2.axhline(0, color="black", lw=1)
        ax2.set_ylabel("IC")
        ax2.legend(fontsize=9, loc="best", ncol=2)
        ax2.grid(True, alpha=0.25, axis="y")
    else:
        # Heatmap: rows=models, cols=months, value=monthly IC
        n_models = len(all_models)
        n_months = len(all_yms)
        data_matrix = np.full((n_models, n_months), np.nan)
        for i, dname in enumerate(all_models):
            for j, ym in enumerate(all_yms):
                month = monthly_data.get(dname, {}).get(ym)
                if not month:
                    continue
                mp = np.array(month["preds"])
                mt = np.array(month["tgts"])
                if len(mp) < 3 or np.std(mp) < 1e-10 or np.std(mt) < 1e-10:
                    data_matrix[i, j] = np.nan
                    continue
                # compute mean of daily cross-sectional ICs if possible
                dates_in_month = sorted(set(
                    datetime.date.fromordinal(int(d))
                    for d in pred_data[raw_map[dname]]["dates"]
                    if datetime.date.fromordinal(int(d)).strftime("%Y-%m") == ym
                ))
                daily_ics = []
                for day in dates_in_month:
                    day_ord = day.toordinal()
                    day_p = pred_data[raw_map[dname]]["preds"][
                        pred_data[raw_map[dname]]["dates"] == day_ord]
                    day_t = pred_data[raw_map[dname]]["targets"][
                        pred_data[raw_map[dname]]["dates"] == day_ord]
                    if len(day_p) > 2 and np.std(day_p) > 1e-10 and np.std(day_t) > 1e-10:
                        daily_ics.append(float(np.corrcoef(day_p, day_t)[0, 1]))
                if daily_ics:
                    data_matrix[i, j] = float(np.mean(daily_ics))
                else:
                    try:
                        data_matrix[i, j] = float(np.corrcoef(mp, mt)[0, 1])
                    except Exception:
                        data_matrix[i, j] = np.nan

        cmap = plt.get_cmap("RdYlGn")
        im = ax2.imshow(data_matrix, aspect="auto", cmap=cmap,
                        origin="lower", interpolation="nearest")
        # xticks
        ax2.set_xticks(np.arange(n_months))
        ax2.set_xticklabels([datetime.date(int(ym.split("-")[0]), int(ym.split("-")[1]), 1).strftime("%b %Y")
                             for ym in all_yms], rotation=35, ha="right")
        # yticks
        ax2.set_yticks(np.arange(n_models))
        ax2.set_yticklabels(all_models)
        ax2.set_title("Monthly IC Heatmap — Rows=models, cols=months",
                      fontweight="bold")
        ax2.set_ylabel("Model")
        cbar = fig.colorbar(im, ax=ax2, orientation="vertical", fraction=0.05)
        cbar.set_label("IC")

        # % positive IC months annotation per model (left of heatmap)
        for i, dname in enumerate(all_models):
            row = data_matrix[i, :]
            valid = ~np.isnan(row)
            if valid.sum() == 0:
                pct = 0.0
            else:
                pct = float((row[valid] > 0).sum()) / float(valid.sum())
            ax2.text(-0.02, (i + 0.5) / max(n_models, 1), f"{pct:.0%}",
                     transform=ax2.transAxes, ha="right", va="center", fontsize=9,
                     fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "ic_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 5: Prediction distribution
# ============================================================================

def plot_pred_distribution(pred_data: Dict, out_dir: str):
    names = _sort_names([_display_name(k) for k in pred_data
                         if len(pred_data[k]["preds"]) > 0])
    raw_keys = {_display_name(k): k for k in pred_data}
    if not names:
        return

    n_cols = min(len(names), 3)
    n_rows = (len(names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Distribution — Predicted vs Actual Returns  (Test Set)",
                 fontsize=14, fontweight="bold")

    for idx, name in enumerate(names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        key = raw_keys[name]
        p = pred_data[key]["preds"]
        t = pred_data[key]["targets"]
        color = _color_for(name, idx)

        clip = np.percentile(np.abs(t), 99)
        bins = np.linspace(-clip, clip, 70)
        ax.hist(t, bins=bins, alpha=0.45, color="#bdbdbd", label="Actual",
                density=True, edgecolor="white", linewidth=0.3)
        ax.hist(p, bins=bins, alpha=0.70, color=color, label="Predicted",
                density=True, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(p.mean(), color=color, lw=1.5, ls="--",
                   label=f"Pred μ = {p.mean()*100:.4f}%")
        ax.set_title(f"{name}  (pred σ = {p.std()*100:.4f}%)",
                     fontweight="bold")
        ax.set_xlabel("Return"); ax.set_ylabel("Density")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    for idx in range(len(names), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "pred_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 6: Multi-horizon IC
# ============================================================================

def plot_multi_horizon_ic(ic_by_horizon: Dict[int, float], out_dir: str,
                          model_name: str = "Best Daily Model"):
    horizons = sorted(ic_by_horizon.keys())
    ics = [ic_by_horizon[h] for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_c = ["#2e7d32" if v >= 0 else "#c62828" for v in ics]
    bars = ax.bar(horizons, ics, color=bar_c, alpha=0.85, width=0.7,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=1)

    for bar, ic in zip(bars, ics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                ic + (0.003 if ic >= 0 else -0.005),
                f"{ic:.4f}", ha="center",
                va="bottom" if ic >= 0 else "top",
                fontsize=11, fontweight="bold")

    ax.set_xlabel("Forecast Horizon (trading days)", fontsize=12)
    ax.set_ylabel("Information Coefficient (IC)", fontsize=12)
    ax.set_title(
        f"{model_name} — IC vs Forecast Horizon  (Post-Hoc, No Retraining)",
        fontsize=13, fontweight="bold")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "multi_horizon_ic.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 7: Equity curve (top-K long portfolio)
# ============================================================================

def plot_equity_curve(pred_data: Dict, out_dir: str, top_k: int = 20):
    import datetime

    # Pick the best daily model with sufficient data
    best_key = None
    for candidate in ["lstm_d", "tft_d", "tcn_d"]:
        if candidate in pred_data and len(pred_data[candidate]["preds"]) >= 100:
            best_key = candidate
            break
    if best_key is None:
        return

    display = _display_name(best_key)
    p = pred_data[best_key]["preds"].copy()
    t = pred_data[best_key]["targets"].copy()
    dates = pred_data[best_key]["dates"].copy()

    order = np.argsort(dates)
    p, t, dates = p[order], t[order], dates[order]

    unique_dates = np.unique(dates)
    if len(unique_dates) < 10:
        return

    portfolio_rets, bh_rets, bottom_rets, plot_dates = [], [], [], []

    for d in unique_dates:
        mask = dates == d
        if mask.sum() < max(5, top_k):
            continue
        ps, ts = p[mask], t[mask]

        top_idx = np.argsort(ps)[-top_k:]
        bot_idx = np.argsort(ps)[:top_k]
        portfolio_rets.append(ts[top_idx].mean())
        bottom_rets.append(ts[bot_idx].mean())
        bh_rets.append(ts.mean())
        plot_dates.append(datetime.date.fromordinal(int(d)))

    if len(portfolio_rets) < 5:
        return

    top_cum = np.cumprod(1 + np.array(portfolio_rets))
    bh_cum = np.cumprod(1 + np.array(bh_rets))
    bot_cum = np.cumprod(1 + np.array(bottom_rets))
    ls_rets = np.array(portfolio_rets) - np.array(bottom_rets)
    ls_cum = np.cumprod(1 + ls_rets)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                              gridspec_kw={"height_ratios": [3, 1.2]})
    fig.suptitle(
        f"Simulated Portfolio  (Top-{top_k} Long)  —  {display}  (Test Set)",
        fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(plot_dates, top_cum, label=f"Top-{top_k} Long",
            color="#2e7d32", lw=2.2)
    ax.plot(plot_dates, bh_cum, label="Equal-Weight (market proxy)",
            color="#757575", lw=1.8, ls="--")
    ax.plot(plot_dates, bot_cum, label=f"Bottom-{top_k} Long",
            color="#c62828", lw=1.5, ls=":", alpha=0.7)
    ax.plot(plot_dates, ls_cum, label="Long-Short (top − bottom)",
            color="#1565c0", lw=1.8, alpha=0.8)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative Return (1.0 = start)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Stats box
    total_ret = top_cum[-1] - 1
    bh_total = bh_cum[-1] - 1
    n_days = len(portfolio_rets)
    ann_factor = 252 / n_days if n_days > 0 else 1
    ann_ret = (1 + total_ret) ** ann_factor - 1
    vol = np.std(portfolio_rets) * np.sqrt(252)
    sharpe = ann_ret / (vol + 1e-10)
    dd = 1 - top_cum / np.maximum.accumulate(top_cum)
    max_dd = dd.max()
    ls_total = ls_cum[-1] - 1
    ls_sharpe = ((np.mean(ls_rets) * 252) /
                 (np.std(ls_rets) * np.sqrt(252) + 1e-10))

    stats = (
        f"Top-{top_k} total:    {total_ret*100:+.1f}%\n"
        f"Market total:   {bh_total*100:+.1f}%\n"
        f"L/S total:      {ls_total*100:+.1f}%\n"
        f"Ann. return:    {ann_ret*100:+.1f}%\n"
        f"Ann. vol:       {vol*100:.1f}%\n"
        f"Sharpe:         {sharpe:.2f}\n"
        f"L/S Sharpe:     {ls_sharpe:.2f}\n"
        f"Max drawdown:   {max_dd*100:.1f}%\n"
        f"Days:           {n_days}"
    )
    ax.text(0.99, 0.03, stats, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc",
                      alpha=0.9))

    # Daily returns
    ax2 = axes[1]
    colors = ["#2e7d32" if r >= 0 else "#c62828" for r in portfolio_rets]
    ax2.bar(plot_dates, portfolio_rets, color=colors, alpha=0.7, width=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Daily Return"); ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Plot 8: Summary metrics bar chart
# ============================================================================

def plot_summary_bars(test_results: Dict, out_dir: str):
    if not test_results:
        return

    items = []
    for k, v in test_results.items():
        name = _display_name(k)
        if isinstance(v, dict):
            items.append((name, v))
    items.sort(key=lambda x: _PREFERRED_ORDER.index(x[0])
               if x[0] in _PREFERRED_ORDER else 999)

    names = [n for n, _ in items]
    ics = [v.get("ic", 0) for _, v in items]
    rics = [v.get("rank_ic", 0) for _, v in items]
    daccs = [v.get("directional_accuracy", 0.5) for _, v in items]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3,
                             figsize=(max(14, 3.5 * len(names)), 5.5))
    fig.suptitle("Test-Set Summary Metrics", fontsize=15, fontweight="bold")

    metric_list = [
        (ics, "Information Coefficient (IC)", 0.0),
        (rics, "Rank IC", 0.0),
        (daccs, "Directional Accuracy", 0.5),
    ]

    for ax, (vals, title, ref) in zip(axes, metric_list):
        bar_c = [_color_for(n, i) for i, n in enumerate(names)]
        bars = ax.bar(x, vals, color=bar_c, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(ref, color="black", lw=1, ls="--", alpha=0.6)
        for bar, v in zip(bars, vals):
            offset = 0.003 if v >= ref else -0.005
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= ref else "top",
                    fontsize=9, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "summary_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ============================================================================
# Diagnostic dashboard
# ============================================================================

def plot_diagnostic_dashboard(
    model_dir: str,
    test_results: Dict,
    pred_data: Dict,
    ic_by_horizon: Dict,
    out_dir: str,
):
    """Single-page PNG that combines the most critical diagnostics.

    Panels (2×3 grid):
      [0,0] GNN status  — edge count from graph dataset metadata
      [0,1] Meta-phase loss trajectory
      [1,0] IC over time  (monthly bars, first model only for clarity)
      [1,1] Calibration  (decile bars, first model only)
      [2,0] Multi-horizon IC (bar chart)
      [2,1] Summary metrics table
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
    fig.suptitle(
        f"Hierarchical Model — Diagnostic Dashboard\n{Path(model_dir).name}",
        fontsize=14, fontweight="bold"
    )

    # ── Panel 0,0 : GNN status ────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.axis("off")
    gnn_meta_path = Path(model_dir) / "gnn_graph_meta.json"
    if gnn_meta_path.exists():
        with open(gnn_meta_path) as f:
            gmeta = json.load(f)
        n_tickers    = gmeta.get("n_tickers", "?")
        n_edges      = gmeta.get("n_edges", "?")
        n_self_loops = gmeta.get("n_self_loops", 0)
        n_sector     = int(n_edges) - int(n_self_loops) if isinstance(n_edges, int) else "?"
        status       = "✅ OK" if (isinstance(n_sector, int) and n_sector > 0) else "⚠️  0 sector edges"
        lines = [
            f"GNN Graph Status: {status}",
            f"Tickers:       {n_tickers}",
            f"Total edges:   {n_edges}",
            f"Sector edges:  {n_sector}",
            f"Self-loops:    {n_self_loops}",
        ]
        color = "#e8f5e9" if "OK" in status else "#fff3e0"
    else:
        lines = [
            "GNN Graph Status: ⚠️ no metadata",
            "gnn_graph_meta.json not found.",
            "Run with --use-gnn to generate it.",
        ]
        color = "#fce4ec"
    ax00.text(0.05, 0.95, "\n".join(lines),
              transform=ax00.transAxes, fontsize=10,
              va="top", ha="left",
              bbox=dict(boxstyle="round", facecolor=color, alpha=0.9))
    ax00.set_title("GNN Edge Diagnostics", fontweight="bold")

    # ── Panel 0,1 : Meta / Joint loss trajectory ──────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    histories = load_history(model_dir)
    meta_keys  = [k for k in histories if "meta" in k.lower() or "joint" in k.lower()]
    if meta_keys:
        for key in meta_keys[:3]:
            df = histories[key]
            if "val_loss" in df.columns:
                ax01.plot(df.index, df["val_loss"],
                          label=_display_name(key), lw=1.5, alpha=0.85)
        ax01.set_title("Meta / Joint Val Loss", fontweight="bold")
        ax01.set_xlabel("Epoch")
        ax01.set_ylabel("Loss")
        ax01.legend(fontsize=9)
        ax01.grid(True, alpha=0.25)
    else:
        ax01.text(0.5, 0.5, "No meta/joint history found",
                  ha="center", va="center", transform=ax01.transAxes, fontsize=10)
        ax01.set_title("Meta / Joint Val Loss", fontweight="bold")

    # ── Panel 1,0 : Monthly IC bars (first model only) ───────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    first_model = next(iter(pred_data), None) if pred_data else None
    if first_model:
        rows      = pred_data[first_model]
        dates_raw = [r.get("date") for r in rows if r.get("date")]
        if dates_raw:
            import matplotlib.dates as mdates
            from datetime import datetime

            # Deduplicate by date → daily cross-sectional IC
            from collections import defaultdict
            day_data: dict = defaultdict(lambda: {"p": [], "t": []})
            for r in rows:
                if not r.get("date"):
                    continue
                d = str(r["date"])[:10]
                if np.isfinite(r.get("pred", np.nan)) and np.isfinite(r.get("target", np.nan)):
                    day_data[d]["p"].append(r["pred"])
                    day_data[d]["t"].append(r["target"])

            daily_ic = {}
            for d, vs in day_data.items():
                p, t = np.array(vs["p"]), np.array(vs["t"])
                if len(p) >= 5 and np.std(p) > 1e-10 and np.std(t) > 1e-10:
                    r_val = np.corrcoef(p, t)[0, 1]
                    if np.isfinite(r_val):
                        daily_ic[d] = r_val

            if daily_ic:
                sorted_days = sorted(daily_ic.keys())
                by_month: dict = defaultdict(list)
                for d in sorted_days:
                    month = d[:7]
                    by_month[month].append(daily_ic[d])

                months    = sorted(by_month.keys())
                m_means   = [np.mean(by_month[m]) for m in months]
                m_stds    = [np.std(by_month[m]) / np.sqrt(len(by_month[m]))
                             for m in months]
                m_dates   = [datetime.strptime(m + "-01", "%Y-%m-%d") for m in months]
                bar_colors = ["#2e7d32" if v >= 0 else "#c62828" for v in m_means]
                ax10.bar(mdates.date2num(m_dates), m_means,
                         color=bar_colors, alpha=0.80,
                         width=20, yerr=m_stds,
                         error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.5})
                ax10.axhline(0, color="black", lw=1)
                ax10.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax10.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax10.xaxis.get_majorticklabels(), rotation=35, ha="right")
        ax10.set_title(f"Monthly IC — {_display_name(first_model)}", fontweight="bold")
        ax10.set_ylabel("IC")
        ax10.grid(True, alpha=0.25, axis="y")
    else:
        ax10.text(0.5, 0.5, "No prediction data", ha="center", va="center",
                  transform=ax10.transAxes)
        ax10.set_title("Monthly IC", fontweight="bold")

    # ── Panel 1,1 : Calibration (first model only) ───────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    if first_model:
        rows = pred_data[first_model]
        preds   = np.array([r.get("pred",   np.nan) for r in rows])
        targets = np.array([r.get("target", np.nan) for r in rows])
        mask    = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[mask], targets[mask]
        if len(preds) >= 50:
            n_deciles = 10
            dec_labels, dec_means, dec_stds = [], [], []
            edges = np.percentile(preds, np.linspace(0, 100, n_deciles + 1))
            for i in range(n_deciles):
                m = (preds >= edges[i]) & (preds < edges[i + 1])
                if m.sum() > 0:
                    dec_means.append(float(np.mean(targets[m])))
                    dec_stds.append(float(np.std(targets[m]) / np.sqrt(m.sum())))
                    dec_labels.append(f"D{i+1}")
            bar_c11 = ["#2e7d32" if v >= 0 else "#c62828" for v in dec_means]
            ax11.bar(range(len(dec_means)), dec_means, color=bar_c11, alpha=0.8,
                     yerr=dec_stds, error_kw={"elinewidth": 1, "alpha": 0.5})
            ax11.axhline(0, color="black", lw=1)
            ax11.set_xticks(range(len(dec_labels)))
            ax11.set_xticklabels(dec_labels, fontsize=9)
    ax11.set_title(f"Decile Calibration — {_display_name(first_model) if first_model else 'N/A'}",
                   fontweight="bold")
    ax11.set_xlabel("Prediction decile")
    ax11.set_ylabel("Mean actual return")
    ax11.grid(True, alpha=0.25, axis="y")

    # ── Panel 2,0 : Multi-horizon IC ─────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    if ic_by_horizon:
        hlist = sorted(ic_by_horizon.keys())
        ic_v  = [ic_by_horizon[h] for h in hlist]
        bar_c20 = ["#2e7d32" if v >= 0 else "#c62828" for v in ic_v]
        ax20.bar([f"{h}d" for h in hlist], ic_v, color=bar_c20, alpha=0.85)
        ax20.axhline(0, color="black", lw=1)
        for i, (h, v) in enumerate(zip(hlist, ic_v)):
            ax20.text(i, v + (0.003 if v >= 0 else -0.005),
                      f"{v:+.3f}", ha="center",
                      va="bottom" if v >= 0 else "top", fontsize=9)
    else:
        ax20.text(0.5, 0.5, "No multi-horizon IC data", ha="center", va="center",
                  transform=ax20.transAxes)
    ax20.set_title("Multi-Horizon IC", fontweight="bold")
    ax20.set_xlabel("Horizon")
    ax20.set_ylabel("IC")
    ax20.grid(True, alpha=0.25, axis="y")

    # ── Panel 2,1 : Test metrics table ───────────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.axis("off")
    if test_results:
        rows_data = []
        for k, v in sorted(test_results.items()):
            if not isinstance(v, dict):
                continue
            ic_v   = v.get("ic", np.nan)
            ric    = v.get("rank_ic", np.nan)
            dacc   = v.get("directional_accuracy", np.nan)
            rows_data.append([_display_name(k),
                               f"{ic_v:.4f}" if np.isfinite(ic_v)   else "NaN",
                               f"{ric:.4f}"  if np.isfinite(ric)    else "NaN",
                               f"{dacc:.3f}" if np.isfinite(dacc)   else "NaN"])
        if rows_data:
            col_labels = ["Model", "IC", "Rank IC", "Dir Acc"]
            tbl = ax21.table(cellText=rows_data, colLabels=col_labels,
                             loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.4)
            # Colour IC column
            ic_col = 1
            for i, row in enumerate(rows_data):
                try:
                    v = float(row[ic_col])
                    cell_color = "#e8f5e9" if v > 0.05 else "#fce4ec" if v < 0 else "#fffde7"
                except ValueError:
                    cell_color = "#fce4ec"
                tbl[(i + 1, ic_col)].set_facecolor(cell_color)
    ax21.set_title("Test Metrics Summary", fontweight="bold")

    out_path = os.path.join(out_dir, "diagnostic_dashboard.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


# ============================================================================
# Main
# ============================================================================

def run(model_dir: str, top_k: int = 20, horizons: List[int] = None,
        skip_predictions: bool = False, data_cfg_overrides: dict = None):
    if horizons is None:
        horizons = [1, 3, 5, 10, 15]

    model_dir = str(Path(model_dir).resolve())
    out_dir = os.path.join(model_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Generating plots → {out_dir}")
    print(f"{'='*60}")

    # 1. Training curves
    print("\n📈 Training curves...")
    histories = load_history(model_dir)
    if histories:
        print(f"   Found {len(histories)} models: {list(histories.keys())}")
        plot_training_curves(histories, out_dir)
    else:
        print("   ⚠ No *_history.csv files found")

    # 2. Summary bars
    test_results = load_test_results(model_dir)
    if test_results:
        print("\n📊 Summary metrics...")
        plot_summary_bars(test_results, out_dir)

    if skip_predictions:
        print("\n  Skipping prediction-based plots (--no-predictions)")
        return

    # 3-7. Plots that need raw predictions
    data_cfg_kwargs = dict(split_mode="temporal")
    if data_cfg_overrides:
        data_cfg_kwargs.update(data_cfg_overrides)

    print("\n🔄 Collecting test-set predictions (may take a few minutes)...")
    pred_data = collect_test_predictions(model_dir, data_cfg_kwargs)

    if pred_data:
        found = [_display_name(k) for k in pred_data]
        print(f"   Got predictions for: {found}")
        print("\n📉 Pred vs actual scatter...")
        plot_pred_vs_actual(pred_data, out_dir)
        print("📏 Calibration...")
        plot_calibration(pred_data, out_dir)
        print("📅 Rolling IC over time...")
        plot_ic_over_time(pred_data, out_dir)
        print("📊 Prediction distributions...")
        plot_pred_distribution(pred_data, out_dir)
        print("💰 Equity curve...")
        plot_equity_curve(pred_data, out_dir, top_k=top_k)
    else:
        print("   ⚠ Could not collect predictions")

    print(f"\n🌐 Multi-horizon IC (horizons={horizons})...")
    ic_by_horizon = collect_multi_horizon_predictions(
        model_dir, data_cfg_kwargs, horizons)
    if ic_by_horizon:
        print(f"   IC: {', '.join(f'{h}d={v:.4f}' for h,v in sorted(ic_by_horizon.items()))}")
        plot_multi_horizon_ic(ic_by_horizon, out_dir)

    print("\n🔬 Diagnostic dashboard...")
    plot_diagnostic_dashboard(
        model_dir=model_dir,
        test_results=test_results,
        pred_data=pred_data if pred_data else {},
        ic_by_horizon=ic_by_horizon if ic_by_horizon else {},
        out_dir=out_dir,
    )

    n_files = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    print(f"\n{'='*60}")
    print(f"  ✅ Done — {n_files} plots in {out_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all diagnostic plots for a model run")
    parser.add_argument("--model-dir", required=True,
                        help="e.g. models/hierarchical_v9")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K stocks for equity curve")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 3, 5, 10, 15])
    parser.add_argument("--no-predictions", action="store_true",
                        help="Skip plots that require running the model")
    parser.add_argument("--split-mode", default="temporal")
    parser.add_argument("--daily-seq-len", type=int, default=720)
    parser.add_argument("--minute-seq-len", type=int, default=780)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    run(
        model_dir=args.model_dir,
        top_k=args.top_k,
        horizons=args.horizons,
        skip_predictions=args.no_predictions,
        data_cfg_overrides={
            "split_mode": args.split_mode,
            "daily_seq_len": args.daily_seq_len,
            "minute_seq_len": args.minute_seq_len,
        },
    )


if __name__ == "__main__":
    main()
