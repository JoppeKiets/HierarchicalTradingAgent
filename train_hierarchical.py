#!/usr/bin/env python3
"""Hierarchical forecaster training pipeline entry point.

This module has been refactored for maintainability:
- src/hierarchical_config.py - Configuration classes (TrainConfig)  
- src/hierarchical_metrics.py - Metrics utilities
- src/hierarchical_trainers.py - Trainer classes (SubModelTrainer, GNNSubModelTrainer, MetaTrainer)
- src/hierarchical_finetuner.py - Fine-tuning class (JointFineTuner)
- src/hierarchical_pipeline.py - Pipeline orchestration (run_pipeline)
- train_hierarchical.py - Main entry point (this file)

To see the full training code structure, examine the src/hierarchical_*.py modules.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from src.hierarchical_config import TrainConfig
from src.hierarchical_data import HierarchicalDataConfig
from src.hierarchical_models import HierarchicalModelConfig
from src.hierarchical_pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Forecaster Training")
    parser.add_argument("--phase", type=int, nargs="+", default=None,
                        help="Phases: 0=preprocess, 1=daily, 2=minute, 3=meta, 4=finetune. Default: all.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="models/hierarchical")
    parser.add_argument("--daily-stride", type=int, default=20)
    parser.add_argument("--minute-stride", type=int, default=30)
    parser.add_argument("--daily-seq-len", type=int, default=720)
    parser.add_argument("--minute-seq-len", type=int, default=780)
    parser.add_argument("--split-mode", type=str, default="temporal",
                        choices=["ticker", "temporal"],
                        help="Split strategy: 'ticker' (by company) or 'temporal' (by time)")
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--early-stop-metric", type=str, default="ic",
                        choices=["ic", "loss"],
                        help="Metric to use for early stopping (default: ic)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-news", dest="use_news", action="store_false",
                        help="Disable the FinBERT news encoder sub-model (Phase 1.5). Enabled by default.")
    parser.set_defaults(use_news=True)
    parser.add_argument("--use-tcn-d", action="store_true",
                        help="Enable Dilated TCN on daily data (trained alongside LSTM_D/TFT_D in Phase 1).")
    parser.add_argument("--use-fund-mlp", action="store_true",
                        help="Enable FundamentalMLP on quarterly data (Phase 1.6).")
    parser.add_argument("--use-gnn-features", action="store_true",
                        help="Inject precomputed GNN embeddings as auxiliary features (requires export_gnn_embeddings.py).")
    parser.add_argument("--skip-trained", action="store_true",
                        help="When resuming, skip Phase 1/2 training for ALL sub-models "
                             "that were already in the checkpoint.")
    parser.add_argument("--skip-models", type=str, nargs="+", default=None,
                        metavar="NAME",
                        help="Explicit list of sub-model names to skip in Phase 1/2. "
                             "Valid names: lstm_d tft_d tcn_d lstm_m tft_m news fund_mlp gnn. "
                             "Can be combined with --skip-trained.")
    parser.add_argument("--feature-set", type=str, default="full",
                        choices=["full", "raw_plus"],
                        help="Daily feature set: 'full' (40+ features) or 'raw_plus' (~15 stationary features).")
    parser.add_argument("--target-type", type=str, default="close_to_close",
                        choices=["close_to_close", "open_to_close"],
                        help="Prediction target: 'close_to_close' (next-day) or 'open_to_close' (intraday).")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Recompute cached features even if present.")
    parser.add_argument("--low-memory", action="store_true",
                        help="Reduce batch sizes and enable AMP for low-memory systems.")
    parser.add_argument("--v10", action="store_true",
                        help="V10 preset: lower TFT LR, warmup, tighter clipping, "
                             "smaller minute models, stronger minute regularization.")
    parser.add_argument("--regime-curriculum", action="store_true",
                        help="Enable regime-aware curriculum learning: cluster historical "
                             "periods into bull/bear/choppy/crisis regimes and oversample "
                             "regimes where each sub-model's IC is lowest.")
    parser.add_argument("--n-regimes", type=int, default=6,
                        help="Number of regime clusters for curriculum (default: 6).")
    
    # Walk-Forward Cross-Validation parameters
    parser.add_argument("--walk-forward", action="store_true",
                        help="Enable walk-forward validation (expanding windows with purging)")
    parser.add_argument("--n-wf-windows", type=int, default=5,
                        help="Number of walk-forward windows (folds) — default: 5")
    parser.add_argument("--wf-expanding", action="store_true", default=True,
                        help="Use expanding windows (default) instead of rolling windows")
    parser.add_argument("--wf-purge-horizon", type=int, default=0,
                        help="Days to purge between train/val (default: forecast_horizon)")
    
    args = parser.parse_args()

    log_dir = args.output_dir.replace("models/", "logs/")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger.info("=" * 70)
    logger.info("Hierarchical Forecaster Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")

    tcfg = TrainConfig(
        output_dir=args.output_dir,
        log_dir=log_dir,
        loss_fn=args.loss,
        patience=args.patience,
        num_workers=args.num_workers,
        early_stop_metric=args.early_stop_metric,
    )
    if args.low_memory:
        tcfg.reduce_memory()
    if args.epochs:
        tcfg.epochs_phase1 = args.epochs
        tcfg.epochs_phase2 = args.epochs
        tcfg.epochs_phase3 = args.epochs
        tcfg.epochs_phase4 = max(args.epochs // 5, 3)  # Phase 4 uses fewer epochs
    if args.batch_size:
        tcfg.batch_size_daily = args.batch_size
        tcfg.batch_size_minute = args.batch_size
    if args.lr:
        tcfg.lr = args.lr

    # V10 preset
    if args.v10:
        logger.info("🔧 V10 preset enabled — applying per-model tuning")
        tcfg.lr_tft = 5e-5
        tcfg.warmup_steps = 2000
        tcfg.grad_clip_tft = 0.5
        tcfg.lr_minute = 1e-4
        tcfg.weight_decay_minute = 1e-3
        tcfg.patience = 20
        tcfg.batch_size_lstm_d = 512
        tcfg.batch_size_tft_d = 192
        tcfg.batch_size_lstm_m = 512
        tcfg.batch_size_tft_m = 256

    logger.info(f"📊 Training config: AMP={tcfg.use_amp}, num_workers={tcfg.num_workers}")
    logger.info(f"   Batch sizes — daily_base={tcfg.batch_size_daily}, "
                f"LSTM_D={tcfg.batch_size_lstm_d or tcfg.batch_size_daily}, "
                f"TFT_D={tcfg.batch_size_tft_d or tcfg.batch_size_daily}, "
                f"minute_base={tcfg.batch_size_minute}, "
                f"LSTM_M={tcfg.batch_size_lstm_m or tcfg.batch_size_minute}, "
                f"TFT_M={tcfg.batch_size_tft_m or tcfg.batch_size_minute}")

    data_cfg = HierarchicalDataConfig(
        daily_seq_len=args.daily_seq_len,
        minute_seq_len=args.minute_seq_len,
        daily_stride=args.daily_stride,
        minute_stride=args.minute_stride,
        split_mode=args.split_mode,
        daily_feature_set=args.feature_set,
        daily_target_type=args.target_type,
    )

    model_cfg = HierarchicalModelConfig(
        daily_seq_len=args.daily_seq_len,
        minute_seq_len=args.minute_seq_len,
        use_news_model=args.use_news,
        use_tcn_d=args.use_tcn_d,
        use_fund_mlp=args.use_fund_mlp,
        use_gnn_features=args.use_gnn_features,
    )

    # V10: smaller minute models
    if args.v10:
        model_cfg.minute_hidden_dim = 64
        model_cfg.minute_n_layers = 1
        model_cfg.minute_dropout = 0.3
        logger.info(f"  Minute model: hidden={model_cfg.minute_hidden_dim}, "
                     f"layers={model_cfg.minute_n_layers}, dropout={model_cfg.minute_dropout}")

    # Walk-forward cross-validation
    if args.walk_forward:
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION MODE")
        logger.info("=" * 70)
        logger.info(f"  Windows: {args.n_wf_windows}")
        logger.info(f"  Mode: {'EXPANDING' if args.wf_expanding else 'ROLLING'}")
        logger.info(f"  Purge horizon: {args.wf_purge_horizon or data_cfg.forecast_horizon} days")
        
        from scripts.walk_forward_hierarchical import run_walk_forward
        
        run_walk_forward(
            n_windows=args.n_wf_windows,
            phases=args.phase,
            epochs=args.epochs or 20,
            patience=args.patience,
            batch_size=args.batch_size or 128,
            top_k=20,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            force_preprocess=args.force_preprocess,
            expanding_window=args.wf_expanding,
            purge_horizon=args.wf_purge_horizon,
        )
        return

    # Standard single-split training
    logger.info("=" * 70)
    logger.info("STANDARD TRAINING MODE (Single Train/Val/Test Split)")
    logger.info("=" * 70)
    
    run_pipeline(
        phases=args.phase,
        resume_path=args.resume,
        tcfg=tcfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        force_preprocess=args.force_preprocess,
        skip_trained=args.skip_trained,
        skip_models=set(args.skip_models) if args.skip_models else None,
        regime_curriculum=args.regime_curriculum,
        n_regimes=args.n_regimes,
    )


if __name__ == "__main__":
    main()
