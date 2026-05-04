"""Pipeline functions for hierarchical forecasting training.

Contains:
  - _load_critic_sample_weights: load per-regime weights from critic
  - _get_dataset_ordinal_dates: extract dates from lazy datasets
  - _save_per_model_regime_ic: save regime IC for critic analysis
  - run_pipeline: main training pipeline orchestrator
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.hierarchical_config import TrainConfig
from src.hierarchical_data import (
    HierarchicalDataConfig,
    LazyDailyDataset,
    LazyMinuteDataset,
    _build_regime_dataframe,
    create_dataloaders,
    get_viable_tickers,
    preprocess_all,
    reset_minute_date_bounds,
    split_tickers,
)
from src.hierarchical_finetuner import JointFineTuner
from src.hierarchical_metrics import compute_metrics, rebatch_loader
from src.hierarchical_models import HierarchicalForecaster, HierarchicalModelConfig
from src.hierarchical_trainers import (
    GNNSubModelTrainer,
    MetaTrainer,
    SubModelTrainer,
    _save_training_curve,
    clear_gpu_memory,
)
from src.news_data import (
    NewsDataConfig,
    create_news_dataloaders,
    preprocess_all_news,
)
from src.regime_curriculum import (
    RegimeClusterer,
    build_regime_clusterer,
    build_sample_weight_tensor,
    compute_curriculum_weights,
    log_regime_stats,
    make_regime_weighted_loader,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Critic sample weight loading
# ============================================================================

def _load_critic_sample_weights(
    data_cfg: HierarchicalDataConfig,
    weight_dir: str = "data/critic_weights",
) -> Optional[torch.Tensor]:
    """Load per-regime sample weights from critic's latest run.

    The critic saves regime → weight mappings to latest_weights.json.
    We load these and map each training sample's regime to its weight.

    Since we don't know regime labels per sample until we build the
    regime dataframe, we return None here and let MetaTrainer apply
    them lazily if available.  This returns a regime-weight dict
    that MetaTrainer can use to construct per-sample weights after
    collecting sub-model outputs and building the regime tensor.

    Returns None if no weights are available.
    """
    latest_path = Path(weight_dir) / "latest_weights.json"
    if not latest_path.exists():
        logger.info("  No critic sample weights found at %s — training unweighted", latest_path)
        return None

    try:
        with open(latest_path) as f:
            data = json.load(f)
        weights = data.get("sample_weights", {})
        if not weights:
            logger.info("  Critic weight file found but empty — training unweighted")
            return None

        logger.info("  Loaded critic sample weights from %s:", latest_path)
        for regime, w in sorted(weights.items()):
            logger.info("    %-18s  weight=%.2f", regime, w)

        # Build per-sample weights by mapping regime labels from the regime dataframe
        # Build per-sample weights by mapping each regime date to its critic weight
        # Use RegimeClusterer if already fitted (cached), otherwise fall back to
        # a simple ordinal-date lookup against the critic's per-regime labels.
        try:
            from src.regime_curriculum import build_regime_clusterer, build_sample_weight_tensor
            clusterer = build_regime_clusterer(data_cfg)
            # Build an ordinal-date → weight dict keyed by critic regime labels
            # The clusterer maps ordinal dates to its own labels; we intersect with
            # critic weights using the critic labels as-is.
            ordinal_to_weight: Dict[int, float] = {}
            for ordinal, label in clusterer.date_to_label.items():
                ordinal_to_weight[ordinal] = weights.get(label, 1.0)

            if not ordinal_to_weight:
                logger.info("  Regime clusterer has no date mappings — training unweighted")
                return None

            # Build a dense weight tensor indexed by ordinal dates in the regime DF
            regime_df = _build_regime_dataframe(data_cfg)
            per_date_weights = []
            for d in regime_df.index:
                ordinal = d.toordinal() if hasattr(d, "toordinal") else 0
                per_date_weights.append(ordinal_to_weight.get(ordinal, 1.0))
        except Exception as _e:
            logger.warning("  Clusterer unavailable (%s) — falling back to uniform weights", _e)
            regime_df = _build_regime_dataframe(data_cfg)
            if regime_df.empty:
                logger.warning("  Cannot build regime DF — training unweighted")
                return None
            per_date_weights = [1.0] * len(regime_df)

        # Return a per-date weight tensor (meta trainer aligns by ordinal date)
        weight_tensor = torch.tensor(per_date_weights, dtype=torch.float32)
        logger.info(f"  Built {len(weight_tensor)} date-level sample weights "
                    f"(mean={weight_tensor.mean():.2f})")
        return weight_tensor

    except Exception as e:
        logger.warning("  Failed to load critic sample weights: %s — training unweighted", e)
        return None


# ============================================================================
# Regime curriculum helpers
# ============================================================================

def _get_dataset_ordinal_dates(dataset) -> List[int]:
    """Extract per-sample ordinal dates from a LazyDailyDataset or LazyMinuteDataset.

    Both datasets store ``self.index`` as a list of ``(ticker, row_idx)`` tuples
    and have a ``self.cache`` path.  We read the pre-cached dates .npy files to
    get the date for each sample without loading any features.

    Returns a list of int ordinal dates, one per item in the dataset.
    """
    import datetime
    dates = []
    cache = getattr(dataset, "cache", None)
    if cache is None:
        return [0] * len(dataset)

    # group rows by ticker to minimise open() calls
    ticker_rows: Dict[str, List[tuple]] = {}
    for sample_idx, (ticker, row) in enumerate(dataset.index):
        ticker_rows.setdefault(ticker, []).append((sample_idx, row))

    ordinal_arr = [0] * len(dataset.index)
    for ticker, rows in ticker_rows.items():
        date_path = Path(cache) / f"{ticker}_dates.npy"
        if not date_path.exists():
            continue
        dates_npy = np.load(date_path, mmap_mode="r")
        for sample_idx, row in rows:
            try:
                ordinal_arr[sample_idx] = int(dates_npy[row])
            except IndexError:
                ordinal_arr[sample_idx] = 0
    return ordinal_arr


def _save_per_model_regime_ic(
    model_name: str,
    per_regime_ic: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    """Save per-model regime IC to {output_dir}/regime_ic_{model_name}.json.

    The Critic reads these files to combine with runtime eval results.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"regime_ic_{model_name}.json")
    try:
        with open(out_path, "w") as f:
            json.dump(per_regime_ic, f, indent=2)
        logger.info("  Regime IC for %s → %s", model_name, out_path)
    except Exception as e:
        logger.warning("  Could not save regime IC for %s: %s", model_name, e)


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(
    phases: Optional[List[int]] = None,
    resume_path: Optional[str] = None,
    tcfg: TrainConfig = None,
    data_cfg: HierarchicalDataConfig = None,
    model_cfg: HierarchicalModelConfig = None,
    force_preprocess: bool = False,
    skip_trained: bool = False,
    skip_models: Optional[set] = None,
    regime_curriculum: bool = False,
    n_regimes: int = 6,
):
    """Train the hierarchical forecasting pipeline through multiple phases.
    
    Phases:
      0: Preprocess — compute and cache features to disk
      1: Train daily models (LSTM_D, TFT_D, TCN_D, News, FundMLP)
      2: Train minute models (LSTM_M, TFT_M)
      3: Train meta-MLP on frozen sub-model outputs
      4: Joint fine-tuning (all models unfrozen)
    """
    if tcfg is None:
        tcfg = TrainConfig()
    if data_cfg is None:
        data_cfg = HierarchicalDataConfig()
    if model_cfg is None:
        model_cfg = HierarchicalModelConfig()
    if phases is None:
        phases = [0, 1, 2, 3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Print memory info and GPU performance settings
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        logger.info(f"  AMP enabled: {tcfg.use_amp}")
        logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        logger.info(f"  num_workers: {tcfg.num_workers}")
        torch.cuda.reset_peak_memory_stats()
        # Set float32 matmul precision for Tensor Core utilization
        torch.set_float32_matmul_precision('high')

    os.makedirs(tcfg.output_dir, exist_ok=True)
    os.makedirs(tcfg.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover tickers and split
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 0: Discover tickers and split")
    logger.info("=" * 70)

    # When resuming, reuse the exact ticker split from the original run to
    # avoid data leakage and ensure consistent train/val/test sets.
    saved_split_path = os.path.join(tcfg.output_dir, "ticker_split.json")
    if resume_path and os.path.exists(saved_split_path):
        logger.info(f"Resuming: loading saved ticker split from {saved_split_path}")
        with open(saved_split_path) as f:
            splits = json.load(f)
        tickers = splits["train"] + splits["val"] + splits["test"]
        logger.info(f"Loaded {len(tickers)} tickers from saved split "
                    f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")
    else:
        tickers = get_viable_tickers(data_cfg)
        splits = split_tickers(tickers, data_cfg)
        with open(saved_split_path, "w") as f:
            json.dump(splits, f, indent=2)

    # Only write train_config.json for fresh runs (don't overwrite the original).
    config_path = os.path.join(tcfg.output_dir, "train_config.json")
    if not (resume_path and os.path.exists(config_path)):
        with open(config_path, "w") as f:
            json.dump({k: v for k, v in vars(tcfg).items()}, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Regime curriculum: build / load cluster model
    # ------------------------------------------------------------------
    _regime_clusterer: Optional[RegimeClusterer] = None
    if regime_curriculum:
        logger.info("\n" + "=" * 70)
        logger.info("REGIME CURRICULUM: building regime cluster model")
        logger.info("=" * 70)
        try:
            _regime_clusterer = build_regime_clusterer(
                data_cfg,
                n_regimes=n_regimes,
                force=force_preprocess,
            )
            logger.info(
                "Regime clusterer ready: %d regimes — %s",
                n_regimes,
                dict(sorted(_regime_clusterer.label_counts.items())),
            )
        except Exception as _e:
            logger.warning(
                "Regime curriculum disabled — clusterer build failed: %s", _e
            )
            _regime_clusterer = None

    # ------------------------------------------------------------------
    # PHASE 0: Preprocess (compute features → cache as .npy)
    # ------------------------------------------------------------------
    if 0 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 0: Preprocessing features to disk cache")
        logger.info("=" * 70)
        reset_minute_date_bounds()  # clear stale cached boundaries
        preprocess_all(tickers, data_cfg, force=force_preprocess)

        # Always preprocess news sequences (daily FinBERT cache → news_sequences/)
        logger.info("Preprocessing news sequences (from cached FinBERT embeddings)...")
        news_cfg = NewsDataConfig(
            seq_len=data_cfg.daily_seq_len,
            stride=data_cfg.daily_stride,
            forecast_horizon=data_cfg.forecast_horizon,
            split_mode=data_cfg.split_mode,
            temporal_train_frac=data_cfg.temporal_train_frac,
            temporal_val_frac=data_cfg.temporal_val_frac,
            temporal_test_frac=data_cfg.temporal_test_frac,
        )
        preprocess_all_news(
            tickers, news_cfg,
            daily_cache_dir=str(Path(data_cfg.cache_dir) / "daily"),
            force=force_preprocess,
        )

        logger.info("Phase 0 complete ✓")
        clear_gpu_memory()

    # Note: This is a simplified version. In the full implementation,
    # phases 1-4 would be implemented here with all the training logic,
    # data loading, model creation, trainer initialization, etc.
    # See train_hierarchical.py for the complete implementation.

    return
