#!/usr/bin/env python3
"""Generate stock predictions using a trained HierarchicalForecaster.

Loads a trained model and preprocessed feature cache, then produces a ranked
list of tickers with predicted next-day returns, attention weights, and
confidence scores.

Usage:
    # Rank all tickers using the latest cached features
    python scripts/predict.py --model models/hierarchical/forecaster_final.pt

    # Top 20 picks only
    python scripts/predict.py --model models/hierarchical/forecaster_final.pt --top-k 20

    # Save to JSON
    python scripts/predict.py --model models/hierarchical/forecaster_final.pt --output predictions.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hierarchical_data import (
    HierarchicalDataConfig,
    _build_regime_dataframe,
)
from src.hierarchical_models import HierarchicalForecaster

logger = logging.getLogger(__name__)


def load_latest_features(
    ticker: str,
    cfg: HierarchicalDataConfig,
) -> Dict:
    """Load the most recent feature window for a ticker.

    Returns dict with 'daily' and optionally 'minute' tensors,
    plus the latest ordinal date.
    """
    cache = Path(cfg.cache_dir)
    result = {"ticker": ticker, "daily": None, "minute": None, "date": 0}

    # Daily features
    daily_feat = cache / "daily" / f"{ticker}_features.npy"
    daily_dates = cache / "daily" / f"{ticker}_dates.npy"
    if daily_feat.exists() and daily_dates.exists():
        feat = np.load(daily_feat, mmap_mode="r")
        dates = np.load(daily_dates, mmap_mode="r")
        n = feat.shape[0]

        # Take the last daily_seq_len rows
        seq_len = cfg.daily_seq_len
        if n >= seq_len:
            window = feat[n - seq_len : n].copy()
            result["daily"] = torch.from_numpy(window).float()
            result["date"] = int(dates[n - 1])

    # Minute features
    minute_feat = cache / "minute" / f"{ticker}_features.npy"
    minute_dates = cache / "minute" / f"{ticker}_dates.npy"
    if minute_feat.exists() and minute_dates.exists():
        feat = np.load(minute_feat, mmap_mode="r")
        dates = np.load(minute_dates, mmap_mode="r")
        n = feat.shape[0]

        seq_len = cfg.minute_seq_len
        if n >= seq_len:
            window = feat[n - seq_len : n].copy()
            result["minute"] = torch.from_numpy(window).float()

    return result


@torch.no_grad()
def generate_predictions(
    forecaster: HierarchicalForecaster,
    tickers: List[str],
    cfg: HierarchicalDataConfig,
    device: torch.device,
) -> List[Dict]:
    """Generate predictions for all tickers.

    Returns list of dicts sorted by predicted return (descending).
    """
    forecaster.eval()
    E = forecaster.cfg.embedding_dim
    R = forecaster.cfg.regime_dim

    # Build regime lookup
    regime_df = _build_regime_dataframe(cfg)
    regime_lookup = {}
    if not regime_df.empty:
        for d, row in regime_df.iterrows():
            if hasattr(d, "toordinal"):
                regime_lookup[d.toordinal()] = row.values.astype(np.float32)

    predictions = []
    n_skipped = 0

    for ticker in tickers:
        data = load_latest_features(ticker, cfg)
        if data["daily"] is None:
            n_skipped += 1
            continue

        daily_x = data["daily"].unsqueeze(0).to(device)
        has_minute = data["minute"] is not None
        minute_x = data["minute"].unsqueeze(0).to(device) if has_minute else None

        # Run each sub-model via the registry — works for any combination of
        # lstm_d / tft_d / tcn_d / lstm_m / tft_m / news / fund_mlp / gnn
        sub_preds: Dict[str, torch.Tensor] = {}
        sub_embs: Dict[str, torch.Tensor] = {}
        for name in forecaster.sub_model_names:
            modality = forecaster.MODALITY[name]
            if modality == "daily":
                x_in = daily_x
            elif modality == "minute":
                if not has_minute:
                    sub_preds[name] = torch.zeros(1, device=device)
                    sub_embs[name] = torch.zeros(1, E, device=device)
                    continue
                x_in = minute_x
            else:
                # news / fundamental / graph — skip for now (no single-ticker loader)
                sub_preds[name] = torch.zeros(1, device=device)
                sub_embs[name] = torch.zeros(1, E, device=device)
                continue
            out = forecaster.sub_models[name](x_in)
            sub_preds[name] = out["prediction"]   # (1,)
            sub_embs[name] = out["embedding"]      # (1, E)

        # Build meta inputs in sub_model_names order
        names = forecaster.sub_model_names
        pred_vec = torch.stack([sub_preds[n][0] for n in names]).unsqueeze(0)  # (1, N)
        emb_list = [sub_embs[n] for n in names]                                # [(1, E), ...]

        # Regime
        ord_date = data["date"]
        if ord_date in regime_lookup:
            regime_vec = torch.from_numpy(regime_lookup[ord_date]).unsqueeze(0).to(device)
        else:
            regime_vec = torch.zeros(1, R, device=device)

        meta_out = forecaster.meta(pred_vec, emb_list, regime_vec)
        attn = meta_out["attention_weights"][0].cpu().numpy()

        entry = {
            "ticker": ticker,
            "predicted_return": float(meta_out["prediction"][0].cpu()),
            "attention_weights": {n: float(attn[i]) for i, n in enumerate(names)},
            "has_minute_data": has_minute,
            "data_date_ordinal": ord_date,
        }
        # Back-compat keys for downstream consumers
        for name in names:
            entry[f"{name}_pred"] = float(sub_preds[name][0].cpu())
        # Legacy aliases used by screener_tools / analyst
        entry.setdefault("lstm_d_pred", entry.get("lstm_d_pred", 0.0))
        entry.setdefault("tft_d_pred", entry.get("tft_d_pred", 0.0))
        entry.setdefault("lstm_m_pred", entry.get("lstm_m_pred", 0.0))
        entry.setdefault("tft_m_pred", entry.get("tft_m_pred", 0.0))
        predictions.append(entry)

    logger.info(f"Generated {len(predictions)} predictions, skipped {n_skipped}")

    # Sort by predicted return (descending)
    predictions.sort(key=lambda x: x["predicted_return"], reverse=True)

    # Add rank
    for i, p in enumerate(predictions):
        p["rank"] = i + 1

    return predictions


def display_predictions(predictions: List[Dict], top_k: int = None):
    """Pretty-print the prediction table."""
    if top_k:
        # Show top and bottom K
        top = predictions[:top_k]
        bottom = predictions[-top_k:]
    else:
        top = predictions
        bottom = []

    import datetime

    print(f"\n{'='*85}")
    print(f"  STOCK PREDICTIONS — {len(predictions)} tickers ranked")
    print(f"{'='*85}")

    header = f"  {'Rank':>4} {'Ticker':<8} {'Pred Ret':>10} {'LSTM_D':>8} {'TFT_D':>8} {'LSTM_M':>8} {'TFT_M':>8} {'Min?':>4}"
    print(f"\n  ── TOP PICKS ──")
    print(header)
    print("  " + "-" * 78)
    for p in top:
        pr = p["predicted_return"]
        print(f"  {p['rank']:4d} {p['ticker']:<8} {pr:>+10.4f} "
              f"{p['lstm_d_pred']:>+8.4f} {p['tft_d_pred']:>+8.4f} "
              f"{p['lstm_m_pred']:>+8.4f} {p['tft_m_pred']:>+8.4f} "
              f"{'✓' if p['has_minute_data'] else '':>4}")

    if bottom:
        print(f"\n  ── BOTTOM PICKS (short candidates) ──")
        print(header)
        print("  " + "-" * 78)
        for p in bottom:
            pr = p["predicted_return"]
            print(f"  {p['rank']:4d} {p['ticker']:<8} {pr:>+10.4f} "
                  f"{p['lstm_d_pred']:>+8.4f} {p['tft_d_pred']:>+8.4f} "
                  f"{p['lstm_m_pred']:>+8.4f} {p['tft_m_pred']:>+8.4f} "
                  f"{'✓' if p['has_minute_data'] else '':>4}")

    # Summary
    all_preds = [p["predicted_return"] for p in predictions]
    n_positive = sum(1 for p in all_preds if p > 0)
    n_minute = sum(1 for p in predictions if p["has_minute_data"])
    print(f"\n  Summary:")
    print(f"    Total tickers:  {len(predictions)}")
    print(f"    Bullish (>0):   {n_positive}")
    print(f"    Bearish (<0):   {len(predictions) - n_positive}")
    print(f"    With minute:    {n_minute}")
    print(f"    Mean pred:      {np.mean(all_preds):+.6f}")
    print(f"    Std pred:       {np.std(all_preds):.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate stock predictions")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained forecaster checkpoint")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Show top/bottom K predictions")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full predictions to JSON")
    parser.add_argument("--cache-dir", type=str, default="data/feature_cache",
                        help="Feature cache directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model}...")
    forecaster = HierarchicalForecaster.load(args.model, device=str(device))
    forecaster = forecaster.to(device)

    # Discover tickers from cache
    cfg = HierarchicalDataConfig(cache_dir=args.cache_dir)
    cache_daily = Path(cfg.cache_dir) / "daily"
    if not cache_daily.exists():
        logger.error(f"No feature cache at {cache_daily}. Run preprocessing first.")
        sys.exit(1)

    tickers = sorted(set(
        f.stem.replace("_features", "")
        for f in cache_daily.glob("*_features.npy")
    ))
    logger.info(f"Found {len(tickers)} tickers in cache")

    # Generate predictions
    predictions = generate_predictions(forecaster, tickers, cfg, device)

    # Display
    display_predictions(predictions, top_k=args.top_k)

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2, default=str)
        logger.info(f"Predictions saved → {args.output}")


if __name__ == "__main__":
    main()
