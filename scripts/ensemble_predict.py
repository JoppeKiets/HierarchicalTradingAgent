#!/usr/bin/env python3
"""Ensemble prediction using multiple walk-forward folds with IC weighting.

Loads the best fold model (highest validation IC) and uses it for inference.
Can also average predictions from top-K folds weighted by their validation ICs.

Usage:
    # Use best fold model
    python scripts/ensemble_predict.py --model-dir models/wf_v1 --best-only

    # Ensemble top-3 folds weighted by IC
    python scripts/ensemble_predict.py --model-dir models/wf_v1 --top-k 3

    # Output to JSON
    python scripts/ensemble_predict.py --model-dir models/wf_v1 \
      --output predictions_ensemble.json --top-k 3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hierarchical_data import (
    HierarchicalDataConfig,
    _build_regime_dataframe,
    get_viable_tickers,
)
from src.hierarchical_models import HierarchicalForecaster

logger = logging.getLogger(__name__)


def load_wf_summary(model_dir: Path) -> Dict:
    """Load the walk-forward summary JSON with fold validation ICs and best fold."""
    summary_path = model_dir / "walk_forward_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Walk-forward summary not found: {summary_path}")
    with open(summary_path) as f:
        return json.load(f)


def get_fold_models(
    model_dir: Path, summary: Dict, top_k: int = 1
) -> List[Tuple[int, float, Path]]:
    """Get fold models sorted by validation IC (descending).
    
    Returns list of (fold_idx, val_ic, checkpoint_path).
    """
    per_window = summary.get("per_window", [])
    fold_info = [
        (r["window"], r.get("val_ic", 0.0)) for r in per_window
    ]
    fold_info.sort(key=lambda x: x[1], reverse=True)
    
    top_folds = fold_info[:top_k]
    
    result: List[Tuple[int, float, Path]] = []
    for fold_idx, val_ic in top_folds:
        # Checkpoint is in window_N/fold_N_forecaster.pt
        checkpoint_path = model_dir / f"window_{fold_idx}" / f"fold_{fold_idx}_forecaster.pt"
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping fold {fold_idx}")
            continue
        result.append((fold_idx, val_ic, checkpoint_path))
    
    return result


def load_latest_features(
    ticker: str, cfg: HierarchicalDataConfig
) -> Dict[str, Optional[torch.Tensor]]:
    """Load latest cached features for a ticker (from scripts/predict.py)."""
    from scripts.predict import load_latest_features as lf
    return lf(ticker, cfg)


@torch.no_grad()
def generate_ensemble_predictions(
    model_dir: str,
    tickers: Optional[List[str]] = None,
    cfg: Optional[HierarchicalDataConfig] = None,
    device: Optional[torch.device] = None,
    best_only: bool = False,
    top_k: int = 3,
) -> List[Dict]:
    """Generate predictions using ensemble of fold models.
    
    Args:
        model_dir: Path to walk-forward results directory
        tickers: List of tickers (default: auto-discover)
        cfg: Data config (default: auto-create)
        device: PyTorch device (default: auto-select CUDA/CPU)
        best_only: If True, only use the best fold (no ensemble)
        top_k: Number of folds to ensemble (ignored if best_only=True)
    
    Returns list of predictions sorted by predicted return (descending).
    """
    model_dir = Path(model_dir)
    
    if cfg is None:
        cfg = HierarchicalDataConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if tickers is None:
        from src.hierarchical_data import get_viable_tickers
        tickers = get_viable_tickers(cfg)
    
    logger.info(f"Loading walk-forward summary from {model_dir}")
    summary = load_wf_summary(model_dir)
    
    # Get fold models
    n_folds_to_use = 1 if best_only else top_k
    fold_models = get_fold_models(model_dir, summary, top_k=n_folds_to_use)
    
    if not fold_models:
        raise ValueError(f"No fold models found in {model_dir}")
    
    logger.info(f"Using {len(fold_models)} fold models:")
    for fold_idx, val_ic, checkpoint_path in fold_models:
        logger.info(f"  Fold {fold_idx}: Val IC={val_ic:.4f} → {checkpoint_path.name}")
    
    # Compute ensemble weights (normalized by val IC)
    val_ics = np.array([ic for _, ic, _ in fold_models])
    weights = val_ics / np.sum(val_ics)  # Normalize to [0, 1]
    
    logger.info(f"Fold weights (by Val IC): {weights}")
    
    # Load all fold models
    models = []
    for fold_idx, val_ic, checkpoint_path in fold_models:
        logger.info(f"  Loading fold {fold_idx}...")
        model = HierarchicalForecaster.load(str(checkpoint_path), device=str(device))
        model.to(device).eval()
        models.append((model, weights[len(models)]))
    
    # Build regime lookup
    regime_df = _build_regime_dataframe(cfg)
    regime_lookup = {}
    if not regime_df.empty:
        for d, row in regime_df.iterrows():
            if hasattr(d, "toordinal"):
                regime_lookup[d.toordinal()] = row.values.astype(np.float32)
    
    # Generate ensemble predictions
    predictions = []
    n_skipped = 0
    E = models[0][0].cfg.embedding_dim
    R = models[0][0].cfg.regime_dim
    
    for ticker in tickers:
        data = load_latest_features(ticker, cfg)
        if data["daily"] is None:
            n_skipped += 1
            continue
        
        daily_x = data["daily"].unsqueeze(0).to(device)
        has_minute = data["minute"] is not None
        minute_x = data["minute"].unsqueeze(0).to(device) if has_minute else None
        ord_date = data["date"]
        
        # Get regime vector
        if ord_date in regime_lookup:
            regime_vec = torch.from_numpy(regime_lookup[ord_date]).unsqueeze(0).to(device)
        else:
            regime_vec = torch.zeros(1, R, device=device)
        
        # Ensemble: average predictions across folds
        ensemble_pred = 0.0
        ensemble_attention = None
        
        for model, weight in zip(models, weights):
            with torch.no_grad():
                # Collect sub-model predictions
                sub_preds = {}
                sub_embs = {}
                for name in model.sub_model_names:
                    modality = model.MODALITY[name]
                    if modality == "daily":
                        x_in = daily_x
                    elif modality == "minute":
                        if not has_minute:
                            sub_preds[name] = torch.zeros(1, device=device)
                            sub_embs[name] = torch.zeros(1, E, device=device)
                            continue
                        x_in = minute_x
                    else:
                        # news / fundamental / graph — skip for now
                        sub_preds[name] = torch.zeros(1, device=device)
                        sub_embs[name] = torch.zeros(1, E, device=device)
                        continue
                    
                    out = model.sub_models[name](x_in)
                    sub_preds[name] = out["prediction"]
                    sub_embs[name] = out["embedding"]
                
                # Build meta inputs in sub_model_names order
                names = model.sub_model_names
                pred_vec = torch.stack([sub_preds[n][0] for n in names]).unsqueeze(0)
                emb_list = [sub_embs[n] for n in names]
                
                # Meta model
                meta_out = model.meta(pred_vec, emb_list, regime_vec)
                fold_pred = float(meta_out["prediction"][0].cpu())
                fold_attn = meta_out["attention_weights"][0].cpu().numpy()
                
                ensemble_pred += weight * fold_pred
                
                if ensemble_attention is None:
                    ensemble_attention = weight * fold_attn
                else:
                    ensemble_attention += weight * fold_attn
        
        entry = {
            "ticker": ticker,
            "predicted_return": round(ensemble_pred, 6),
            "attention_weights": {
                n: round(float(ensemble_attention[i]), 4)
                for i, n in enumerate(model.sub_model_names)
            },
            "has_minute_data": has_minute,
            "data_date_ordinal": ord_date,
            "n_folds_ensemble": len(models),
            "fold_val_ics": [round(ic, 4) for _, ic, _ in fold_models],
        }
        
        # Back-compat keys
        for i, name in enumerate(model.sub_model_names):
            entry[f"{name}_pred"] = 0.0  # Not tracked in ensemble; use meta pred
        
        predictions.append(entry)
    
    logger.info(f"Generated {len(predictions)} ensemble predictions, skipped {n_skipped}")
    
    # Sort by predicted return (descending)
    predictions.sort(key=lambda x: x["predicted_return"], reverse=True)
    
    # Add rank
    for i, p in enumerate(predictions):
        p["rank"] = i + 1
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate ensemble predictions from WF folds")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to walk-forward results directory")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file (default: print to stdout)")
    parser.add_argument("--best-only", action="store_true",
                        help="Use only the best fold (no ensemble)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top folds to ensemble (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    try:
        predictions = generate_ensemble_predictions(
            args.model_dir,
            best_only=args.best_only,
            top_k=args.top_k,
        )
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(predictions[:20], f, indent=2)  # Save top 20
            logger.info(f"Saved top-20 predictions to {args.output}")
        else:
            # Print top 10
            print(json.dumps(predictions[:10], indent=2))
            print(f"\n... {len(predictions) - 10} more predictions omitted")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
