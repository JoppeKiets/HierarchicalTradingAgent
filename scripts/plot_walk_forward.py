#!/usr/bin/env python3
"""Plotting utilities for walk-forward validation results.

Generates diagnostic plots for WF-CV:
  - Fold comparison: IC/RankIC/DirAcc/Sharpe per fold
  - Equity curve and drawdown (stitched OOS or per-fold backtest)
  - Signal vs realized-return scatter / heatmap
  - Regime sensitivity (std-dev across folds)
  - Ensemble weights

Usage:
    python scripts/plot_walk_forward.py \
      --results-dir results/walk_forward_hierarchical \
      --output-dir  results/walk_forward_hierarchical/plots

Input: walk_forward_summary.json produced by walk_forward_hierarchical.py
       (optionally) per-window *_returns.npy arrays for equity curve

Expected summary schema (top-level keys):
    per_window : list of dicts, each with:
        window          : int
        val_ic          : float
        dates           : {"train": [lo, hi], "val": [...], "test": [...]}
        regression_metrics : {"ic", "rank_ic", "directional_accuracy", "n_samples"}
        long_short      : {"sharpe", "total_return", "max_drawdown", "n_days"}
    aggregate   : {"ic_mean", "ic_std", "rank_ic_mean", "dir_acc_mean",
                   "sharpe_mean", "sharpe_std", "consistency", "val_ic_weights"}
    best_fold_idx       : int
    best_fold_val_ic    : float
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_label(window_dict: Dict, split: str = "test") -> str:
    """Human-readable date range for a window split."""
    import datetime
    try:
        lo, hi = window_dict["dates"][split]
        d0 = datetime.date.fromordinal(int(lo))
        d1 = datetime.date.fromordinal(int(hi))
        return f"{d0:%b'%y}–{d1:%b'%y}"
    except Exception:
        return ""


def load_summary(results_dir: str) -> Dict:
    """Load walk-forward summary JSON."""
    summary_path = Path(results_dir) / "walk_forward_summary.json"
    with open(summary_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Per-fold IC / RankIC / DirAcc / Sharpe bar charts
# ---------------------------------------------------------------------------

def plot_fold_comparison(summary: Dict, output_dir: str) -> None:
    """Four-panel per-fold metrics bar chart with date-range x-labels.

    Expected per_window keys: window, val_ic, regression_metrics.*, long_short.sharpe, dates
    """
    per_window   = summary["per_window"]
    best_fold_idx = summary.get("best_fold_idx", 0)

    n        = len(per_window)
    x        = np.arange(n)
    ics      = [r["regression_metrics"]["ic"]                   for r in per_window]
    rics     = [r["regression_metrics"]["rank_ic"]              for r in per_window]
    sharpes  = [r["long_short"]["sharpe"]                       for r in per_window]
    val_ics  = [r.get("val_ic", 0.0)                           for r in per_window]
    xlabels  = [f"W{r['window']}\n{_date_label(r, 'test')}"   for r in per_window]

    def _band(ax, values, color="red"):
        m, s = np.mean(values), np.std(values)
        ax.axhline(m, color=color, linestyle="--", linewidth=1.8,
                   label=f"Mean {m:+.4f}")
        if s > 1e-8:
            ax.axhspan(m - s, m + s, alpha=0.15, color=color, label=f"±1σ {s:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Walk-Forward Validation — Fold Stability", fontsize=14, fontweight="bold")

    # IC
    ax = axes[0, 0]
    ax.bar(x, ics, color=["#1565C0" if v >= 0 else "#B71C1C" for v in ics], alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    _band(ax, ics)
    ax.set_title("Test IC"); ax.set_ylabel("IC")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Rank IC
    ax = axes[0, 1]
    ax.bar(x, rics, color=["#1B5E20" if v >= 0 else "#B71C1C" for v in rics], alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    _band(ax, rics, color="#388E3C")
    ax.set_title("Test Rank IC"); ax.set_ylabel("Rank IC")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Sharpe
    ax = axes[1, 0]
    ax.bar(x, sharpes, color=["#2E7D32" if s >= 0 else "#C62828" for s in sharpes], alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    _band(ax, sharpes, color="#6A1B9A")
    ax.set_title("L/S Sharpe"); ax.set_ylabel("Sharpe")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Val IC
    ax = axes[1, 1]
    bar_colors = ["#FF8F00" if i == best_fold_idx else "#7B1FA2" for i in range(n)]
    ax.bar(x, val_ics, color=bar_colors, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    y_max    = max(max(val_ics), 0)
    y_marker = y_max + abs(y_max) * 0.1 + 0.005
    ax.text(best_fold_idx, y_marker, "★ BEST", ha="center",
            fontsize=9, fontweight="bold", color="#FF8F00")
    ax.set_title("Val IC (Ensemble Weight)"); ax.set_ylabel("Val IC")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "fold_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Plot 2: Equity curve + drawdown
# ---------------------------------------------------------------------------

def plot_equity_curve(
    summary: Dict,
    output_dir: str,
    returns_arrays: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """Per-fold and stitched equity curve + drawdown.

    Args:
        summary: walk-forward summary dict
        output_dir: save directory
        returns_arrays: optional {window_idx: (n_days,) float daily-returns array}.
            If None, function approximates from total_return/n_days in the summary.
    """
    per_window = summary["per_window"]
    n          = len(per_window)
    pal        = plt.cm.tab10(np.linspace(0, 0.9, n))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("L/S Strategy — Equity Curves & Drawdown", fontsize=13, fontweight="bold")
    ax_eq, ax_dd = axes

    all_daily_rets = []

    for i, r in enumerate(per_window):
        w         = r["window"]
        n_days    = r["long_short"].get("n_days", 0)
        total_ret = r["long_short"].get("total_return", 0.0)

        if returns_arrays is not None and w in returns_arrays:
            daily_rets = np.asarray(returns_arrays[w], dtype=float)
        elif n_days > 0:
            # Flat approximation: arithmetic decomposition of total_return over n_days.
            # Geometric decomposition fails when total_return <= -1 (complex result),
            # so we fall back to a simple linear daily return.
            base = 1 + total_ret
            if base > 0:
                daily_r = float(np.real(base ** (1.0 / max(n_days, 1)))) - 1
            else:
                # total lost ≥ 100%: distribute linearly
                daily_r = total_ret / max(n_days, 1)
            daily_rets = np.full(n_days, daily_r)
        else:
            continue

        cum  = np.cumprod(1 + daily_rets)
        peak = np.maximum.accumulate(cum)
        dd   = (cum - peak) / np.where(peak > 1e-10, peak, 1e-10)
        xs   = np.arange(len(cum))

        label = (f"W{w} ({_date_label(r, 'test')}) "
                 f"Sharpe={r['long_short']['sharpe']:.2f}")
        ax_eq.plot(xs, cum, color=pal[i], linewidth=1.2, label=label, alpha=0.8)
        ax_dd.fill_between(xs, dd, 0, color=pal[i], alpha=0.25)
        ax_dd.plot(xs, dd, color=pal[i], linewidth=0.8)
        all_daily_rets.extend(daily_rets.tolist())

    # Stitched OOS
    if all_daily_rets:
        sc   = np.cumprod(1 + np.array(all_daily_rets))
        peak = np.maximum.accumulate(sc)
        sdd  = (sc - peak) / np.where(peak > 1e-10, peak, 1e-10)
        ax_eq.plot(sc,  color="black", linewidth=2.0, linestyle="--",
                   label=f"Stitched OOS total={sc[-1]-1:.2%}", zorder=5)
        ax_dd.plot(sdd, color="black", linewidth=1.8, linestyle="--",
                   label=f"Stitched MaxDD={sdd.min():.2%}", zorder=5)

    ax_eq.axhline(1.0, color="gray", linewidth=0.7)
    ax_eq.set_ylabel("Cum. Return (1 = start)")
    ax_eq.legend(fontsize=8, loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    ax_dd.axhline(0, color="gray", linewidth=0.7)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Days (per-fold, concatenated in order)")
    ax_dd.legend(fontsize=8, loc="lower left")
    ax_dd.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Plot 3: Signal vs return scatter / heatmap
# ---------------------------------------------------------------------------

def plot_signal_vs_return(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    window_idx: Optional[int] = None,
    n_bins: int = 20,
) -> None:
    """Scatter + quantile-bin mean return chart for predicted signal.

    Args:
        preds:      (N,) model output scores
        targets:    (N,) realized returns
        output_dir: save directory
        window_idx: fold index for filename/title
        n_bins:     number of quantile bins (x-axis of right panel)

    Example:
        plot_signal_vs_return(fold_preds, fold_targets, output_dir, window_idx=0)
    """
    if len(preds) < 20:
        logger.warning("Too few samples for signal_vs_return plot, skipping")
        return

    tag = f"W{window_idx}" if window_idx is not None else "all"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Signal vs Realized Return — {tag}", fontsize=13, fontweight="bold")

    # Scatter
    ax = axes[0]
    ax.scatter(preds, targets, alpha=0.2, s=8, color="#1976D2", rasterized=True)
    m, b = np.polyfit(preds, targets, 1)
    xs = np.linspace(preds.min(), preds.max(), 200)
    ax.plot(xs, m * xs + b, color="red", linewidth=1.5, label=f"OLS slope={m:.3f}")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Predicted signal"); ax.set_ylabel("Realized return")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Quantile-bin mean return
    ax = axes[1]
    df = pd.DataFrame({"pred": preds, "ret": targets})
    try:
        df["bin"] = pd.qcut(df["pred"], q=n_bins, labels=False, duplicates="drop")
    except Exception:
        df["bin"] = pd.cut(df["pred"], bins=n_bins, labels=False)
    g = df.groupby("bin")["ret"].mean().dropna()
    ax.bar(g.index, g.values,
           color=["#1B5E20" if v >= 0 else "#B71C1C" for v in g.values],
           alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Signal quantile bin (1–{n_bins})")
    ax.set_ylabel("Mean realized return")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, f"signal_vs_return_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Plot 4: Regime sensitivity
# ---------------------------------------------------------------------------

def plot_regime_sensitivity(summary: Dict, output_dir: str) -> None:
    """Metric std-dev bar chart. High std → regime-sensitive."""
    per_window = summary["per_window"]
    ics     = [r["regression_metrics"]["ic"]      for r in per_window]
    rics    = [r["regression_metrics"]["rank_ic"] for r in per_window]
    sharpes = [r["long_short"]["sharpe"]          for r in per_window]

    metrics = ["IC", "Rank IC", "Sharpe"]
    stds    = [np.std(ics), np.std(rics), np.std(sharpes)]
    colors  = ["#1976D2", "#388E3C", "#D32F2F"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Regime Sensitivity — Std Dev Across Folds\n"
                 "(High std → unstable across market regimes)",
                 fontsize=12, fontweight="bold")

    bars = ax.bar(metrics, stds, color=colors, alpha=0.75, edgecolor="black", linewidth=1.1)
    for bar, s in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(max(stds) * 0.01, 0.0005),
                f"{s:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    y_top = max(max(stds) * 1.35, 0.06)
    ax.axhspan(0,    0.01, alpha=0.10, color="green",  label="Robust    (<0.01)")
    ax.axhspan(0.01, 0.05, alpha=0.10, color="yellow", label="Moderate  (0.01–0.05)")
    ax.axhspan(0.05, y_top, alpha=0.10, color="red",   label="Unstable  (>0.05)")

    ax.set_ylim(0, y_top)
    ax.set_ylabel("Standard Deviation Across Folds")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "regime_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Plot 5: Ensemble weights
# ---------------------------------------------------------------------------

def plot_ensemble_weights(summary: Dict, output_dir: str) -> None:
    per_window      = summary["per_window"]
    val_ic_weights  = summary["aggregate"].get("val_ic_weights", [])

    if not val_ic_weights:
        logger.info("  Skipping ensemble_weights plot (no val_ic_weights)")
        return

    n       = len(per_window)
    x       = np.arange(n)
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n))
    xlabels = [f"W{r['window']}\n{_date_label(r, 'val')}" for r in per_window]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
    fig.suptitle("Ensemble Weights (normalized Val IC)", fontsize=12, fontweight="bold")

    bars = ax.bar(x, val_ic_weights, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)
    for bar, w in zip(bars, val_ic_weights):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(max(val_ic_weights) * 0.01, 0.002),
                f"{w:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Ensemble Weight"); ax.set_xlabel("Fold")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(0, max(max(val_ic_weights) * 1.2, 0.01))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "ensemble_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# Plot 6: Aggregate summary table
# ---------------------------------------------------------------------------

def plot_aggregate_summary(summary: Dict, output_dir: str) -> None:
    agg  = summary["aggregate"]
    bfi  = summary.get("best_fold_idx", "—")
    bfic = summary.get("best_fold_val_ic", 0.0)

    rows = [
        ["Metric",          "Mean",                         "Std",                          "Note"],
        ["IC",              f"{agg['ic_mean']:+.4f}",       f"{agg['ic_std']:.4f}",         "Pearson corr"],
        ["Rank IC",         f"{agg['rank_ic_mean']:+.4f}",  "—",                            "Spearman corr"],
        ["Dir. Accuracy",   f"{agg['dir_acc_mean']:.4f}",   "—",                            "frac correct sign"],
        ["L/S Sharpe",      f"{agg['sharpe_mean']:+.4f}",   f"{agg['sharpe_std']:.4f}",     "annualized"],
        ["Consistency",     agg["consistency"],              "—",                            "windows +Sharpe"],
        ["Best fold",       f"W{bfi}",                      f"Val IC={bfic:.4f}",           ""],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, cellLoc="center", loc="center",
                   colWidths=[0.28, 0.18, 0.18, 0.32])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#1565C0")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    for i in range(1, len(rows)):
        c = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(4):
            tbl[(i, j)].set_facecolor(c)

    fig.suptitle("Walk-Forward Aggregate Summary", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot walk-forward validation results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir",  type=str, default="")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir  = args.output_dir or str(results_dir / "plots")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading summary from {results_dir}")
    summary = load_summary(str(results_dir))

    logger.info(f"Generating plots → {output_dir}")
    plot_fold_comparison(summary, output_dir)
    plot_equity_curve(summary, output_dir)
    plot_regime_sensitivity(summary, output_dir)
    plot_ensemble_weights(summary, output_dir)
    plot_aggregate_summary(summary, output_dir)

    logger.info("\n✓ All walk-forward plots generated!")


if __name__ == "__main__":
    main()
