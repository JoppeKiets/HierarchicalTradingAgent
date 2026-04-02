#!/usr/bin/env python3
"""Evaluate a trained HierarchicalForecaster on test data.

Produces:
  1. Per-ticker and aggregate regression metrics (IC, Rank IC, Dir Acc)
  2. Portfolio-level backtest (Sharpe, Sortino, Max Drawdown)
  3. Sub-model contribution analysis (which model helps most?)
  4. Comparison with simple baselines (Buy & Hold, Momentum)

Usage:
    # Evaluate final model on test split
    python scripts/evaluate_hierarchical.py --model models/hierarchical/forecaster_final.pt

    # Evaluate with custom data config
    python scripts/evaluate_hierarchical.py --model models/hierarchical/forecaster_final.pt \\
        --split-mode temporal --top-k 50

    # Compare against baselines
    python scripts/evaluate_hierarchical.py --model models/hierarchical/forecaster_final.pt \\
        --baselines
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hierarchical_data import (
    HierarchicalDataConfig,
    LazyDailyDataset,
    LazyMinuteDataset,
    create_dataloaders,
    get_viable_tickers,
    split_tickers,
    _build_regime_dataframe,
    REGIME_FEATURE_NAMES,
)
from src.hierarchical_models import HierarchicalForecaster

logger = logging.getLogger(__name__)


# ============================================================================
# Regression metrics
# ============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Regression metrics: IC, RankIC, DirectionalAccuracy, MSE, MAE."""
    from scipy import stats

    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))

    if np.std(preds) < 1e-10 or np.std(targets) < 1e-10:
        ic, ric = 0.0, 0.0
    else:
        ic = float(np.corrcoef(preds, targets)[0, 1])
        ric = float(stats.spearmanr(preds, targets).correlation)

    pred_dir = (preds > 0).astype(float)
    target_dir = (targets > 0).astype(float)
    dir_acc = float(np.mean(pred_dir == target_dir))

    return {
        "ic": ic, "rank_ic": ric, "directional_accuracy": dir_acc,
        "mse": mse, "rmse": rmse, "mae": mae, "n_samples": len(preds),
    }


# ============================================================================
# Collect predictions from the full model
# ============================================================================

@torch.no_grad()
def collect_predictions(
    forecaster: HierarchicalForecaster,
    daily_loader, minute_loader,
    data_cfg: HierarchicalDataConfig,
    device: torch.device,
    news_loader=None,
    fund_loader=None,
    graph_loader=None,
) -> Dict[str, np.ndarray]:
    """Run the full hierarchical model and collect per-sample predictions.

    Registry-aware: iterates over ``forecaster.sub_model_names`` so it
    automatically supports any combination of sub-models (TCN_D, FundMLP, GNN, …).

    Returns dict with:
        meta_preds, targets, tickers, dates
        {name}_preds for each sub-model
        attention_weights  (N, n_sub_models)
    """
    forecaster.eval()
    E = forecaster.cfg.embedding_dim
    R = forecaster.cfg.regime_dim
    sub_names = forecaster.sub_model_names  # sorted list

    # Build regime lookup
    regime_df = _build_regime_dataframe(data_cfg)
    regime_lookup: Dict[int, np.ndarray] = {}
    if not regime_df.empty:
        for d, row in regime_df.iterrows():
            if hasattr(d, 'toordinal'):
                regime_lookup[d.toordinal()] = row.values.astype(np.float32)

    import gc

    # Data-source → loader mapping
    loader_map = {
        "daily":       daily_loader,
        "minute":      minute_loader,
        "news":        news_loader,
        "fundamental": fund_loader,
        "graph":       graph_loader,
    }

    # ── Collect outputs from each sub-model, keyed by (ticker, ordinal_date) ──
    # per_model[name][(ticker, date)] = {"pred": tensor, "emb": tensor, "target": tensor}
    per_model: Dict[str, Dict[Tuple, Dict]] = {}

    for name in sub_names:
        ds = HierarchicalForecaster.MODALITY[name]
        loader = loader_map.get(ds)
        if loader is None or len(loader.dataset) == 0:
            logger.info(f"  {name}: no data loader for source '{ds}' — will zero-fill")
            per_model[name] = {}
            continue

        model = forecaster.sub_models[name]
        model.to(device).eval()
        result = {}

        if ds == "graph":
            # GNN: cross-sectional batches
            for batch in loader:
                if isinstance(batch, dict):
                    nf = batch["node_features"].squeeze(0).to(device)
                    tgt = batch["targets"].squeeze(0)
                    mask = batch["mask"].squeeze(0)
                    ei = batch["edge_index"].squeeze(0).to(device)
                    od = int(batch["ordinal_date"])
                    tickers_list = batch["tickers"]
                else:
                    b = batch[0]
                    nf = b["node_features"].to(device)
                    tgt = b["targets"]
                    mask = b["mask"]
                    ei = b["edge_index"].to(device)
                    od = int(b["ordinal_date"])
                    tickers_list = b["tickers"]
                mask_dev = mask.to(device)
                out = model(nf, ei, mask_dev)
                valid_tickers = [tickers_list[j] for j in range(len(tickers_list)) if mask[j]]
                vi = 0
                for j in range(len(tickers_list)):
                    if mask[j]:
                        key = (valid_tickers[vi], od)
                        result[key] = {
                            "pred": out["prediction"][vi].cpu(),
                            "emb": out["embedding"][vi].cpu(),
                            "target": tgt[j],
                        }
                        vi += 1
        else:
            for batch in loader:
                x, y, ordinal_dates, tickers_batch = batch
                x = x.to(device)
                out = model(x)
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    result[key] = {
                        "pred": out["prediction"][i].cpu(),
                        "emb": out["embedding"][i].cpu(),
                        "target": y[i],
                    }
        per_model[name] = result
        model.cpu()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Determine the universe of keys from the "daily" source ──
    # Daily loaders always present; this determines the universe of samples.
    daily_names = [n for n in sub_names if HierarchicalForecaster.MODALITY[n] == "daily"]
    all_keys: set = set()
    for n in daily_names:
        all_keys |= set(per_model[n].keys())
    if not all_keys:
        empty = {f"{n}_preds": np.array([]) for n in sub_names}
        empty.update({"meta_preds": np.array([]), "targets": np.array([]),
                       "dates": np.array([]), "attention_weights": np.array([]),
                       "tickers": []})
        return empty

    # Use the first daily model to get targets
    target_source = daily_names[0]

    # ── Align all models ──
    zero_emb = torch.zeros(E)
    aligned_preds: Dict[str, List[float]] = {n: [] for n in sub_names}
    aligned_pred_vecs: List[torch.Tensor] = []
    aligned_emb_vecs: List[torch.Tensor] = []
    aligned_regimes: List[torch.Tensor] = []
    aligned_targets: List[float] = []
    aligned_tickers: List[str] = []
    aligned_dates: List[int] = []

    for key in sorted(all_keys):
        ticker, ord_date = key

        # Get target from first daily model that has this key
        target_val = None
        for n in daily_names:
            if key in per_model[n]:
                target_val = per_model[n][key]["target"]
                break
        if target_val is None:
            continue

        pred_list = []
        emb_list_raw = []
        for name in sub_names:
            entry = per_model[name].get(key)
            if entry is not None:
                pred_list.append(entry["pred"])
                emb_list_raw.append(entry["emb"])
                aligned_preds[name].append(entry["pred"].item())
            else:
                pred_list.append(torch.tensor(0.0))
                emb_list_raw.append(zero_emb)
                aligned_preds[name].append(0.0)

        pred_vec = torch.stack(pred_list)
        emb_vec = torch.cat(emb_list_raw)

        if ord_date in regime_lookup:
            regime_vec = torch.from_numpy(regime_lookup[ord_date])
        else:
            regime_vec = torch.zeros(R)

        aligned_pred_vecs.append(pred_vec)
        aligned_emb_vecs.append(emb_vec)
        aligned_regimes.append(regime_vec)
        aligned_targets.append(target_val.item() if hasattr(target_val, 'item') else float(target_val))
        aligned_tickers.append(ticker)
        aligned_dates.append(ord_date)

    if not aligned_pred_vecs:
        empty = {f"{n}_preds": np.array([]) for n in sub_names}
        empty.update({"meta_preds": np.array([]), "targets": np.array([]),
                       "dates": np.array([]), "attention_weights": np.array([]),
                       "tickers": []})
        return empty

    # ── Batched meta inference ──
    N_SUB = len(sub_names)
    all_pred_t = torch.stack(aligned_pred_vecs)       # (N, N_SUB)
    all_emb_t = torch.stack(aligned_emb_vecs)          # (N, N_SUB*E)
    all_regime_t = torch.stack(aligned_regimes)         # (N, R)

    meta_preds_out, attn_weights_out = [], []
    batch_size = 1024
    forecaster.meta.to(device).eval()

    for start in range(0, len(all_pred_t), batch_size):
        end = min(start + batch_size, len(all_pred_t))
        p_b = all_pred_t[start:end].to(device)
        e_b = all_emb_t[start:end].to(device)
        r_b = all_regime_t[start:end].to(device)
        emb_list = [e_b[:, i*E:(i+1)*E] for i in range(N_SUB)]

        meta_out = forecaster.meta(p_b, emb_list, r_b)
        meta_preds_out.append(meta_out["prediction"].cpu().numpy())
        attn_weights_out.append(meta_out["attention_weights"].cpu().numpy())

    forecaster.meta.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "meta_preds": np.concatenate(meta_preds_out),
        "attention_weights": np.concatenate(attn_weights_out),
        "targets": np.array(aligned_targets),
        "tickers": aligned_tickers,
        "dates": np.array(aligned_dates),
        "sub_model_names": sub_names,
    }
    # Add per-model predictions
    for name in sub_names:
        results[f"{name}_preds"] = np.array(aligned_preds[name])

    return results


# ============================================================================
# Portfolio backtest from predictions
# ============================================================================

def backtest_long_short(
    preds: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    tickers: List[str],
    top_k: int = 20,
) -> Dict[str, float]:
    """Simple long-short portfolio backtest.

    Strategy:
      - Each day, rank all stocks by predicted return.
      - Go long the top-k and short the bottom-k (equal-weight).
      - Daily P&L = mean(top-k actual returns) - mean(bottom-k actual returns)

    Returns summary statistics.
    """
    import datetime as dt

    unique_dates = sorted(set(dates))
    daily_returns = []

    for d in unique_dates:
        mask = dates == d
        day_preds = preds[mask]
        day_targets = targets[mask]

        if len(day_preds) < 2 * top_k:
            # Not enough stocks — use all of them
            k = max(1, len(day_preds) // 4)
        else:
            k = top_k

        rank = np.argsort(day_preds)
        long_idx = rank[-k:]
        short_idx = rank[:k]

        long_ret = np.mean(day_targets[long_idx])
        short_ret = np.mean(day_targets[short_idx])
        ls_ret = (long_ret - short_ret) / 2.0  # divide by 2 for proper scaling
        daily_returns.append(ls_ret)

    daily_returns = np.array(daily_returns)
    n_days = len(daily_returns)

    if n_days < 10:
        return {"sharpe": 0.0, "total_return": 0.0, "n_days": n_days,
                "error": "too few trading days"}

    # Cumulative return
    cum_ret = np.cumprod(1 + daily_returns)
    total_ret = cum_ret[-1] - 1.0

    # Annualized
    ann_ret = (1 + total_ret) ** (252.0 / n_days) - 1.0
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    down_vol = np.std(downside) * np.sqrt(252) if len(downside) > 1 else 1e-10
    sortino = ann_ret / down_vol

    # Max drawdown
    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / peak
    max_dd = float(np.min(dd))

    # Hit rate (positive daily return)
    hit_rate = float(np.mean(daily_returns > 0))

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate": round(hit_rate, 4),
        "n_days": n_days,
        "top_k": top_k,
        "mean_daily_ret": round(float(np.mean(daily_returns)), 6),
    }


def backtest_long_only(
    preds: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    tickers: List[str],
    top_k: int = 20,
) -> Dict[str, float]:
    """Long-only portfolio: buy the top-k stocks each day.

    Returns summary statistics.
    """
    unique_dates = sorted(set(dates))
    daily_returns = []

    for d in unique_dates:
        mask = dates == d
        day_preds = preds[mask]
        day_targets = targets[mask]

        if len(day_preds) < top_k:
            k = max(1, len(day_preds) // 2)
        else:
            k = top_k

        # Only go long if predicted positive
        positive_mask = day_preds > 0
        if positive_mask.sum() > 0:
            pos_preds = day_preds[positive_mask]
            pos_targets = day_targets[positive_mask]
            rank = np.argsort(pos_preds)
            top_idx = rank[-min(k, len(rank)):]
            daily_returns.append(np.mean(pos_targets[top_idx]))
        else:
            daily_returns.append(0.0)  # Stay in cash

    daily_returns = np.array(daily_returns)
    n_days = len(daily_returns)

    if n_days < 10:
        return {"sharpe": 0.0, "total_return": 0.0, "n_days": n_days}

    cum_ret = np.cumprod(1 + daily_returns)
    total_ret = cum_ret[-1] - 1.0
    ann_ret = (1 + total_ret) ** (252.0 / n_days) - 1.0
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / peak
    max_dd = float(np.min(dd))

    return {
        "sharpe": round(sharpe, 4),
        "total_return": round(total_ret, 4),
        "ann_return": round(ann_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "n_days": n_days,
        "strategy": "long_only",
    }


# ============================================================================
# Sub-model contribution analysis
# ============================================================================

def analyze_sub_model_contributions(results: Dict) -> Dict:
    """Analyze which sub-models contribute most to predictions.

    Registry-aware: reads sub_model_names from results dict.

    Computes:
      - Per-model IC and directional accuracy
      - Attention weight statistics
      - Meta model improvement over best sub-model
    """
    targets = results["targets"]
    sub_names = results.get("sub_model_names", ["lstm_d", "tft_d", "lstm_m", "tft_m"])
    analysis = {}

    # Per sub-model metrics
    for name in sub_names:
        key = f"{name}_preds"
        if key not in results or len(results[key]) == 0:
            continue
        preds = results[key]
        m = compute_metrics(preds, targets)
        analysis[name.upper()] = {
            "ic": m["ic"], "rank_ic": m["rank_ic"],
            "dir_acc": m["directional_accuracy"],
            "mse": m["mse"],
        }

    # Meta ensemble
    m = compute_metrics(results["meta_preds"], targets)
    analysis["Meta"] = {
        "ic": m["ic"], "rank_ic": m["rank_ic"],
        "dir_acc": m["directional_accuracy"],
        "mse": m["mse"],
    }

    # Attention weight analysis (dynamic number of sub-models)
    attn = results["attention_weights"]
    if attn.ndim == 2 and attn.shape[1] == len(sub_names):
        analysis["attention_stats"] = {
            "mean_weights": [round(float(x), 4) for x in attn.mean(axis=0)],
            "std_weights": [round(float(x), 4) for x in attn.std(axis=0)],
            "model_names": [n.upper() for n in sub_names],
        }

    # Equal-weight average of all sub-models
    sub_preds = [results[f"{n}_preds"] for n in sub_names if f"{n}_preds" in results]
    if sub_preds:
        equal_avg = np.mean(sub_preds, axis=0)
        eq_metrics = compute_metrics(equal_avg, targets)
        analysis["equal_weight_avg"] = {
            "ic": eq_metrics["ic"], "rank_ic": eq_metrics["rank_ic"],
            "dir_acc": eq_metrics["directional_accuracy"],
        }

    # Daily-only average
    daily_names = [n for n in sub_names if n in ("lstm_d", "tft_d", "tcn_d")]
    daily_sub_preds = [results[f"{n}_preds"] for n in daily_names if f"{n}_preds" in results]
    if daily_sub_preds:
        daily_avg = np.mean(daily_sub_preds, axis=0)
        da_metrics = compute_metrics(daily_avg, targets)
        analysis["daily_avg"] = {
            "ic": da_metrics["ic"], "rank_ic": da_metrics["rank_ic"],
            "dir_acc": da_metrics["directional_accuracy"],
        }

    return analysis


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate(
    model_path: str,
    data_cfg: HierarchicalDataConfig,
    output_dir: str = "results/hierarchical_eval",
    top_k: int = 20,
    run_baselines: bool = False,
):
    """Full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model onto CPU first; sub-models are moved to GPU one-at-a-time
    # during collect_predictions to avoid OOM on large datasets.
    logger.info(f"Loading model from {model_path}")
    forecaster = HierarchicalForecaster.load(model_path, device="cpu")
    forecaster.eval()

    total_params = sum(p.numel() for p in forecaster.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")

    # Set up data
    tickers = get_viable_tickers(data_cfg)
    splits = split_tickers(tickers, data_cfg)
    # Use small batches to avoid OOM: TFT attention on seq_len=720
    # needs O(batch * seq^2) VRAM — batch=256 blows up to ~5 GB.
    # num_workers=0 to avoid ConnectionResetError in background runs.
    loaders = create_dataloaders(
        splits, data_cfg,
        batch_size_daily=32, batch_size_minute=16, num_workers=0,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ─── 1. Collect predictions on test set ───────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Collecting predictions on test set")
    logger.info("=" * 60)

    results = collect_predictions(
        forecaster,
        loaders["daily"]["test"],
        loaders["minute"]["test"],
        data_cfg, device,
    )
    n_samples = len(results["targets"])
    logger.info(f"  Collected {n_samples:,} test predictions")

    if n_samples == 0:
        logger.error("No test predictions — check data config")
        return

    # ─── 2. Aggregate regression metrics ──────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Regression metrics")
    logger.info("=" * 60)

    agg_metrics = compute_metrics(results["meta_preds"], results["targets"])
    logger.info(f"  Meta model (aggregate):")
    logger.info(f"    IC             = {agg_metrics['ic']:.4f}")
    logger.info(f"    Rank IC        = {agg_metrics['rank_ic']:.4f}")
    logger.info(f"    Dir Accuracy   = {agg_metrics['directional_accuracy']:.3f}")
    logger.info(f"    RMSE           = {agg_metrics['rmse']:.6f}")
    logger.info(f"    N samples      = {agg_metrics['n_samples']:,}")

    # Per-ticker metrics
    unique_tickers = sorted(set(results["tickers"]))
    per_ticker = {}
    for t in unique_tickers:
        mask = np.array([tk == t for tk in results["tickers"]])
        if mask.sum() < 10:
            continue
        m = compute_metrics(results["meta_preds"][mask], results["targets"][mask])
        per_ticker[t] = m

    if per_ticker:
        ics = [m["ic"] for m in per_ticker.values()]
        logger.info(f"\n  Per-ticker IC: mean={np.mean(ics):.4f}, "
                    f"median={np.median(ics):.4f}, "
                    f"std={np.std(ics):.4f}, "
                    f"positive={sum(1 for x in ics if x > 0)}/{len(ics)}")

    # ─── 3. Sub-model contribution analysis ───────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Sub-model contribution analysis")
    logger.info("=" * 60)

    contributions = analyze_sub_model_contributions(results)

    sub_names = results.get("sub_model_names", ["lstm_d", "tft_d", "lstm_m", "tft_m"])

    logger.info(f"\n  {'Model':<18} {'IC':>8} {'RankIC':>8} {'DirAcc':>8} {'MSE':>10}")
    logger.info("  " + "-" * 56)
    for name in [n.upper() for n in sub_names] + ["Meta"]:
        if name not in contributions:
            continue
        c = contributions[name]
        marker = " ★" if name == "Meta" else ""
        logger.info(f"  {name:<18} {c['ic']:>8.4f} {c['rank_ic']:>8.4f} "
                    f"{c['dir_acc']:>8.3f} {c['mse']:>10.6f}{marker}")

    if "equal_weight_avg" in contributions:
        eq = contributions["equal_weight_avg"]
        logger.info(f"  {'EqualWeightAvg':<18} {eq['ic']:>8.4f} {eq['rank_ic']:>8.4f} "
                    f"{eq['dir_acc']:>8.3f}")
    if "daily_avg" in contributions:
        da = contributions["daily_avg"]
        logger.info(f"  {'DailyAvg':<18} {da['ic']:>8.4f} {da['rank_ic']:>8.4f} "
                    f"{da['dir_acc']:>8.3f}")

    if "attention_stats" in contributions:
        attn = contributions["attention_stats"]
        logger.info(f"\n  Attention weights (mean ± std):")
        for name, mu, std in zip(attn["model_names"], attn["mean_weights"], attn["std_weights"]):
            bar = "█" * int(mu * 40)
            logger.info(f"    {name:<8} {mu:.3f} ± {std:.3f}  {bar}")

    # ─── 4. Portfolio backtest ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Portfolio backtest")
    logger.info("=" * 60)

    ls_result = backtest_long_short(
        results["meta_preds"], results["targets"],
        results["dates"], results["tickers"], top_k=top_k,
    )
    logger.info(f"\n  Long-Short portfolio (top/bottom {top_k}):")
    for k, v in ls_result.items():
        logger.info(f"    {k:<20} = {v}")

    lo_result = backtest_long_only(
        results["meta_preds"], results["targets"],
        results["dates"], results["tickers"], top_k=top_k,
    )
    logger.info(f"\n  Long-Only portfolio (top {top_k}):")
    for k, v in lo_result.items():
        logger.info(f"    {k:<20} = {v}")

    # ─── 5. Baselines comparison ──────────────────────────────────────
    baseline_results = {}
    if run_baselines:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Baseline comparisons")
        logger.info("=" * 60)

        # Random baseline
        rng = np.random.RandomState(42)
        random_preds = rng.randn(n_samples) * np.std(results["meta_preds"])
        rand_ls = backtest_long_short(
            random_preds, results["targets"],
            results["dates"], results["tickers"], top_k=top_k,
        )
        rand_metrics = compute_metrics(random_preds, results["targets"])
        baseline_results["Random"] = {**rand_ls, **rand_metrics}
        logger.info(f"  Random baseline: Sharpe={rand_ls['sharpe']:.4f}, "
                    f"IC={rand_metrics['ic']:.4f}")

        # Momentum baseline (use daily model's raw momentum signal)
        # Approximate: use previous-day target as momentum signal
        momentum_preds = np.zeros_like(results["targets"])
        for t in unique_tickers:
            mask = np.array([tk == t for tk in results["tickers"]])
            idxs = np.where(mask)[0]
            if len(idxs) > 1:
                # Sort by date
                date_order = np.argsort(results["dates"][idxs])
                sorted_idxs = idxs[date_order]
                # Use lagged target as momentum signal
                momentum_preds[sorted_idxs[1:]] = results["targets"][sorted_idxs[:-1]]

        mom_ls = backtest_long_short(
            momentum_preds, results["targets"],
            results["dates"], results["tickers"], top_k=top_k,
        )
        mom_metrics = compute_metrics(momentum_preds, results["targets"])
        baseline_results["Momentum(1d)"] = {**mom_ls, **mom_metrics}
        logger.info(f"  Momentum(1d):    Sharpe={mom_ls['sharpe']:.4f}, "
                    f"IC={mom_metrics['ic']:.4f}")

        # Zero baseline (always predict 0 — stays in cash)
        zero_preds = np.zeros_like(results["meta_preds"])
        zero_metrics = compute_metrics(zero_preds, results["targets"])
        baseline_results["Zero"] = zero_metrics
        logger.info(f"  Zero baseline:   IC={zero_metrics['ic']:.4f}, "
                    f"DirAcc={zero_metrics['directional_accuracy']:.3f}")

    # ─── 6. Save results ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Saving results")
    logger.info("=" * 60)

    output = {
        "model_path": model_path,
        "aggregate_metrics": agg_metrics,
        "per_ticker_metrics": per_ticker,
        "sub_model_contributions": contributions,
        "long_short_backtest": ls_result,
        "long_only_backtest": lo_result,
        "baselines": baseline_results,
        "config": {
            "split_mode": data_cfg.split_mode,
            "top_k": top_k,
            "n_test_tickers": len(unique_tickers),
            "n_test_samples": n_samples,
        },
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"  Results saved → {results_path}")

    # Save predictions for further analysis
    pred_path = os.path.join(output_dir, "predictions.npz")
    save_dict = {
        "meta_preds": results["meta_preds"],
        "targets": results["targets"],
        "dates": results["dates"],
        "attention_weights": results["attention_weights"],
    }
    for name in sub_names:
        key = f"{name}_preds"
        if key in results:
            save_dict[key] = results[key]
    np.savez_compressed(pred_path, **save_dict)
    logger.info(f"  Predictions saved → {pred_path}")

    # ─── Summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Test samples:        {n_samples:,}")
    logger.info(f"  Test tickers:        {len(unique_tickers)}")
    logger.info(f"  Meta IC:             {agg_metrics['ic']:.4f}")
    logger.info(f"  Meta Rank IC:        {agg_metrics['rank_ic']:.4f}")
    logger.info(f"  Meta Dir Accuracy:   {agg_metrics['directional_accuracy']:.3f}")
    logger.info(f"  L/S Sharpe (top {top_k}): {ls_result['sharpe']:.4f}")
    logger.info(f"  L/S Max Drawdown:    {ls_result['max_drawdown']:.4f}")
    logger.info(f"  Long-Only Sharpe:    {lo_result['sharpe']:.4f}")
    logger.info("=" * 60)

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical Forecaster")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to forecaster checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="results/hierarchical_eval")
    parser.add_argument("--split-mode", type=str, default="temporal",
                        choices=["ticker", "temporal"])
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K stocks for portfolio backtest")
    parser.add_argument("--baselines", action="store_true",
                        help="Run baseline comparisons")
    parser.add_argument("--daily-stride", type=int, default=1,
                        help="Daily stride (use 1 for dense test evaluation)")
    parser.add_argument("--minute-stride", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data_cfg = HierarchicalDataConfig(
        split_mode=args.split_mode,
        daily_stride=args.daily_stride,
        minute_stride=args.minute_stride,
    )

    evaluate(
        model_path=args.model,
        data_cfg=data_cfg,
        output_dir=args.output_dir,
        top_k=args.top_k,
        run_baselines=args.baselines,
    )


if __name__ == "__main__":
    main()
