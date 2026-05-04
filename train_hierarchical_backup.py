#!/usr/bin/env python3
"""Hierarchical forecaster training pipeline (memory-efficient).

Trains 5 models in 4 phases:
    Phase 0:  Preprocess — compute features and cache to disk (.npy)
    Phase 1:  LSTM_D + TFT_D on daily data
    Phase 2:  LSTM_M + TFT_M on minute data
    Phase 3:  Meta MLP on frozen sub-model outputs
    Phase 4:  (optional) Joint fine-tuning

Data is loaded lazily from mmap'd .npy files → near-zero RAM overhead.

Usage:
    python train_hierarchical.py                     # Full pipeline
    python train_hierarchical.py --phase 0           # Preprocess only
    python train_hierarchical.py --phase 1 2         # Daily + minute models
    python train_hierarchical.py --phase 3           # Meta only (needs phases 1+2 done)
    python train_hierarchical.py --resume models/hierarchical/checkpoint_phase2.pt --phase 3
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Memory optimization: reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

# Enable cuDNN auto-tuner for fixed-size inputs (LSTM, TFT, etc.)
torch.backends.cudnn.benchmark = True
# Allow TF32 on Ampere+ / Blackwell GPUs — ~3x matmul throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Import from modularized components
from src.hierarchical_config import TrainConfig
from src.hierarchical_metrics import clear_gpu_memory
from src.hierarchical_trainers import SubModelTrainer, GNNSubModelTrainer, MetaTrainer
from src.hierarchical_data import (
    HierarchicalDataConfig,
    LazyDailyDataset,
    LazyMinuteDataset,
    create_dataloaders,
    get_viable_tickers,
    preprocess_all,
    split_tickers,
    _build_regime_dataframe,
    REGIME_FEATURE_NAMES,
    reset_minute_date_bounds,
)
from src.hierarchical_models import (
    HierarchicalForecaster,
    HierarchicalModelConfig,
    MetaMLP,
)
from src.news_data import (
    NewsDataConfig,
    LazyNewsDataset,
    create_news_dataloaders,
    preprocess_all_news,
)
from src.regime_curriculum import (
    RegimeClusterer,
    build_regime_clusterer,
    compute_curriculum_weights,
    build_sample_weight_tensor,
    make_regime_weighted_loader,
    log_regime_stats,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Training configuration (kept for backward compatibility)
# ============================================================================

class DeprecatedTrainConfig:
    """Training hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 1e-5
    lr_meta: float = 5e-4
    lr_finetune: float = 1e-5

    # Per-model LR overrides (v10)
    lr_tft: float = 1e-4            # TFT-specific LR (defaults to 1e-4)
    lr_lstm: float = 2e-4           # reduced LR for LSTM (LSTM_D)
    lr_minute: float = 0.0          # 0 = use default lr; set to e.g. 1e-4 for minute models
    weight_decay_minute: float = 0.0  # 0 = use default; higher for minute (regularize)

    # Warmup (v10) — linear warmup before cosine decay
    warmup_steps: int = 500         # linear warmup steps (useful for TFT attention/VSN stability)

    # Per-model grad clip (v10)
    grad_clip_tft: float = 0.0      # 0 = use default grad_clip; e.g. 0.5

    scheduler: str = "cosine"

    epochs_phase1: int = 30       # early stopping (patience=20) will trigger before 30
    epochs_news: int = 30           # Phase 1.5: news encoder
    epochs_phase2: int = 30
    epochs_phase3: int = 30
    epochs_phase4: int = 10

    patience: int = 15
    min_delta: float = 1e-6
    # early_stop_metric: "ic" tracks val IC (better for regression),
    # "loss" tracks val loss (original behaviour)
    early_stop_metric: str = "ic"
    ic_min_delta: float = 1e-4  # min IC improvement to reset patience

    batch_size_daily: int = 64    # base daily batch (doubled for AMP)
    batch_size_minute: int = 32   # base minute batch (doubled for AMP)
    batch_size_meta: int = 256
    batch_size_news: int = 64       # News model batch size

    # Per-model batch size overrides (0 = use default above)
    batch_size_lstm_d: int = 0
    batch_size_tft_d: int = 0
    batch_size_lstm_m: int = 0
    batch_size_tft_m: int = 0

    loss_fn: str = "huber"

    # Per-model Huber delta overrides (allows widening quadratic region)
    huber_delta_lstm_d: float = 0.05
    huber_delta_tft_d: float = 0.05

    # Per-model weight decay overrides (increase regularization for sensitive models)
    weight_decay_lstm: float = 1e-4
    weight_decay_tft: float = 1e-4

    grad_clip: float = 1.0
    num_workers: int = 4  # mmap-based data loading is fast; 4 is enough to keep GPU fed
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_amp: bool = True  # Automatic Mixed Precision — leverage Tensor Cores on RTX 5080
    log_interval: int = 100
    eval_interval: int = 1

    output_dir: str = "models/hierarchical"
    log_dir: str = "logs/hierarchical"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Auto-fallback: multi-process data loading requires /dev/shm shared
        # memory. If it is unavailable (e.g. some container environments),
        # silently drop to single-process loading to avoid RuntimeError.
        if self.num_workers > 0 and not _check_shm_available():
            logger.warning(
                f"num_workers={self.num_workers} requested but /dev/shm is not "
                "available — falling back to num_workers=0 (single-process loading)."
            )
            self.num_workers = 0

    def reduce_memory(self):
        """Reduce memory footprint for low-memory systems."""
        self.batch_size_daily = max(16, self.batch_size_daily // 2)
        self.batch_size_minute = max(8, self.batch_size_minute // 2)
        self.batch_size_meta = max(32, self.batch_size_meta // 2)
        self.num_workers = 0
        self.use_amp = True
        logger.warning(f"Memory optimization enabled: daily={self.batch_size_daily}, "
                      f"minute={self.batch_size_minute}, meta={self.batch_size_meta}")
        return self


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MSE, RMSE, MAE, IC, RankIC, DirAcc."""
    from scipy import stats

    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))

    if np.std(preds) < 1e-10 or np.std(targets) < 1e-10:
        ic, ric = 0.0, 0.0
    else:
        ic = float(np.corrcoef(preds, targets)[0, 1])
        ric = float(stats.spearmanr(preds, targets).correlation)

    pred_dir = (preds > 0).astype(float)
    target_dir = (targets > 0).astype(float)
    dir_acc = float(np.mean(pred_dir == target_dir))

    return {
        "mse": mse, "rmse": rmse, "mae": mae,
        "ic": ic, "rank_ic": ric, "directional_accuracy": dir_acc,
    }


# ============================================================================
# Helper: re-wrap a DataLoader with a different batch size
# ============================================================================

def rebatch_loader(loader: DataLoader, batch_size: int) -> DataLoader:
    """Return a new DataLoader over the same dataset but with a different batch_size."""
    is_shuffle = isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler)
    # Only drop_last when shuffling AND dataset is large enough for >=1 full batch
    drop_last = is_shuffle and len(loader.dataset) >= batch_size
    nw = loader.num_workers
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=nw,
        pin_memory=loader.pin_memory,
        drop_last=drop_last,
        persistent_workers=nw > 0,
        prefetch_factor=3 if nw > 0 else None,
    )


# ============================================================================
# Single-model trainer
# ============================================================================

class SubModelTrainer:
    """Trains a single sub-model (LSTM or TFT) on regression."""

    def __init__(self, model: nn.Module, name: str, device: torch.device,
                 tcfg: TrainConfig, *,
                 lr_override: float = 0.0,
                 weight_decay_override: float = 0.0,
                 grad_clip_override: float = 0.0,
                 warmup_steps: int = 0,
                 huber_delta: Optional[float] = None):
        self.model = model.to(device)
        # torch.compile: fuses small CUDA kernels → 30-50% speedup
        if torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
                logger.info(f"  [{name}] torch.compile() enabled")
            except Exception as e:
                logger.warning(f"  [{name}] torch.compile() failed: {e} — running eager")
        self.name = name
        self.device = device
        self.tcfg = tcfg

        # Per-model overrides: 0 means "use default from tcfg"
        self.lr = lr_override if lr_override > 0 else tcfg.lr
        self.wd = weight_decay_override if weight_decay_override > 0 else tcfg.weight_decay
        self.grad_clip = grad_clip_override if grad_clip_override > 0 else tcfg.grad_clip
        self.warmup_steps = warmup_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.wd,
        )
        # Allow per-instantiation override of the Huber delta; default to 0.01
        if tcfg.loss_fn == "huber":
            delta_val = huber_delta if huber_delta is not None else 0.1
            self.criterion = nn.HuberLoss(delta=delta_val)
        else:
            self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self._global_step = 0  # tracks total optimizer steps (for warmup)

        # AMP: mixed precision for Tensor Core utilization on RTX 5080
        self.use_amp = tcfg.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        # Scheduler (may be set by _make_scheduler)
        self.scheduler = None

        logger.info(f"  [{name}] lr={self.lr:.2e}  wd={self.wd:.1e}  "
                     f"grad_clip={self.grad_clip}  warmup_steps={self.warmup_steps}  "
                     f"amp={self.use_amp}")

    def _make_scheduler(self, n_epochs: int, steps_per_epoch: Optional[int] = None):
        """Create and store a scheduler.

        If using a cyclic scheduler, `steps_per_epoch` is used as the
        `step_size_up` so the cycle can be advanced per-optimizer-step.
        """
        if self.tcfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6,
            )
        elif self.tcfg.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3,
            )
        elif self.tcfg.scheduler in ("cyclic", "cyclical", "cyclic_lr"):
            sp = steps_per_epoch if steps_per_epoch is not None else 1
            # step_size_up controls how many optimizer steps it takes to go from
            # base_lr -> max_lr. Use one epoch by default.
            step_size_up = max(1, int(sp))
            base_lr = max(1e-8, self.lr * 0.1)
            max_lr = max(base_lr * 1.01, self.lr)
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode="triangular2",
                cycle_momentum=False,
            )
        else:
            self.scheduler = None
        return self.scheduler

    def _apply_warmup(self):
        """Linear warmup: scale LR from ~0 → target over warmup_steps."""
        if self.warmup_steps <= 0 or self._global_step >= self.warmup_steps:
            return
        # +1 so the very first step gets a small non-zero LR
        scale = (self._global_step + 1) / self.warmup_steps
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.lr * scale

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        total_grad_norm = 0.0

        for batch_idx, batch in enumerate(loader):
            x, y = batch[0], batch[1]  # ignore date/ticker
            # non_blocking=True: overlap CPU→GPU transfer with computation
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Apply linear warmup before the optimizer step
            self._apply_warmup()

            self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            # AMP: autocast for float16 on Tensor Cores, GradScaler for stability
            with autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(x)
                loss = self.criterion(out["prediction"], y)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                total_grad_norm += float(grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            # If using a per-step cyclic LR, advance the scheduler here.
            if getattr(self, "scheduler", None) is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.CyclicLR
            ):
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            self._global_step += 1
            total_loss += loss.item()
            n_batches += 1

            if batch_idx > 0 and batch_idx % self.tcfg.log_interval == 0:
                avg_gn = total_grad_norm / n_batches
                logger.info(f"  [{self.name}] batch {batch_idx}/{len(loader)} "
                            f"loss={loss.item():.6f} grad_norm={avg_gn:.4f}")

        if n_batches > 0:
            self._last_grad_norm = total_grad_norm / n_batches
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(x)
                loss = self.criterion(out["prediction"], y)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(out["prediction"].cpu().numpy())
            all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
        metrics["loss"] = avg_loss
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate_per_regime(
        self,
        loader: DataLoader,
        clusterer,
    ) -> Dict[str, Dict[str, float]]:
        """Compute IC/DA per regime cluster on the given data loader.

        Args:
            loader: DataLoader that yields (x, y, ordinal_date, ticker) batches.
            clusterer: RegimeClusterer with fitted date→label mapping.

        Returns:
            {regime_label: {"ic": float, "directional_accuracy": float,
                            "rank_ic": float, "n_samples": int}}
        """
        from src.regime_curriculum import evaluate_per_regime as _eval_per_regime

        self.model.eval()
        all_preds, all_targets, all_dates = [], [], []

        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1]
            ordinal_dates = batch[2]  # int tensor or list
            with autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(x)
            all_preds.append(out["prediction"].cpu().numpy())
            all_targets.append(y.numpy())
            # Normalise date format to a list of ints
            if hasattr(ordinal_dates, "numpy"):
                all_dates.append(ordinal_dates.numpy().astype(np.int32))
            else:
                all_dates.append(np.array(ordinal_dates, dtype=np.int32))

        if not all_preds:
            return {}

        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        dates   = np.concatenate(all_dates)

        return _eval_per_regime(preds, targets, dates, clusterer)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int, save_dir: str,
              curriculum_loader: Optional[DataLoader] = None) -> Dict:
        """Full training loop with early stopping.

        Args:
            curriculum_loader: When provided, this DataLoader replaces
                ``train_loader`` for each epoch.  It is built externally
                (e.g. by ``make_regime_weighted_loader``) so that samples
                from regimes where the model is weakest are oversampled.
        """
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{self.name}_best.pt")

        history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        self.best_val_loss = float("inf")
        self.best_val_ic = -float("inf")
        self.patience_counter = 0
        use_ic_stopping = (self.tcfg.early_stop_metric == "ic")

        # Regime curriculum: use weighted loader if provided
        active_train_loader = curriculum_loader if curriculum_loader is not None else train_loader
        if curriculum_loader is not None:
            logger.info(f"  Regime curriculum: ENABLED (oversampling weak regimes)")

        # Create scheduler; pass steps_per_epoch so cyclic schedulers can
        # use per-step stepping (we step them inside train_epoch).
        scheduler = self._make_scheduler(n_epochs, steps_per_epoch=len(active_train_loader))

        logger.info(f"\n{'='*60}")
        logger.info(f"Training {self.name} for {n_epochs} epochs")
        logger.info(f"  Train batches: {len(active_train_loader)}, Val batches: {len(val_loader)}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"  Early-stop metric: {'IC' if use_ic_stopping else 'val_loss'}, patience={self.tcfg.patience}")
        logger.info(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(active_train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t0

            val_ic = val_metrics["ic"]
            # Fall back to 0.0 if IC is NaN (can happen in early epochs)
            if val_ic != val_ic:  # NaN check
                val_ic = 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic"].append(val_ic)
            history["val_rank_ic"].append(val_metrics.get("rank_ic", 0.0))

            lr = self.optimizer.param_groups[0]["lr"]
            gn = getattr(self, '_last_grad_norm', 0.0)
            logger.info(
                f"[{self.name}] Epoch {epoch:3d}/{n_epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"IC={val_ic:.4f} | RankIC={val_metrics['rank_ic']:.4f} | "
                f"DirAcc={val_metrics['directional_accuracy']:.3f} | "
                f"lr={lr:.2e} | gn={gn:.3f} | {elapsed:.1f}s"
            )

            if scheduler is not None:
                # ReduceLROnPlateau expects metric per-epoch; CyclicLR is
                # stepped per-optimizer-step inside train_epoch and should
                # not be stepped here.
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                    pass
                else:
                    scheduler.step()

            if use_ic_stopping:
                # Checkpoint on best IC; fall back to val_loss if IC stays at 0
                improved = val_ic > self.best_val_ic + self.tcfg.ic_min_delta
                if improved or (self.best_val_ic <= 0 and
                                val_loss < self.best_val_loss - self.tcfg.min_delta):
                    if improved:
                        self.best_val_ic = val_ic
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), best_path)
                    logger.info(f"  ★ New best IC={val_ic:.4f} | val_loss={val_loss:.6f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.tcfg.patience:
                        logger.info(f"  Early stopping at epoch {epoch} (best IC={self.best_val_ic:.4f})")
                        break
            else:
                if val_loss < self.best_val_loss - self.tcfg.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), best_path)
                    logger.info(f"  ★ New best val_loss={val_loss:.6f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.tcfg.patience:
                        logger.info(f"  Early stopping at epoch {epoch}")
                        break

        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        # Save training curve as CSV for easy plotting
        _save_training_curve(history, self.name, save_dir)

        return history


def _save_training_curve(history: Dict, name: str, save_dir: str):
    """Save training history to CSV for plotting."""
    import csv
    csv_path = os.path.join(save_dir, f"{name}_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_ic", "val_rank_ic"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.8f}",
                f"{history['val_loss'][i]:.8f}",
                f"{history['val_ic'][i]:.6f}" if i < len(history.get("val_ic", [])) else "",
                f"{history['val_rank_ic'][i]:.6f}" if i < len(history.get("val_rank_ic", [])) else "",
            ])
    logger.info(f"  Training curve → {csv_path}")


# ============================================================================
# GNN sub-model trainer (cross-sectional batches)
# ============================================================================

class GNNSubModelTrainer:
    """Trains the SectorGNN on cross-sectional graph batches.

    Each batch from the DataLoader is one date with all tickers.
    The batch is a dict: {node_features, targets, mask, edge_index, ...}.
    """

    def __init__(self, model: nn.Module, name: str, device: torch.device,
                 tcfg: TrainConfig):
        self.model = model.to(device)
        if torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
                logger.info(f"  [{name}] torch.compile() enabled")
            except Exception as e:
                logger.warning(f"  [{name}] torch.compile() failed: {e} — running eager")
        self.name = name
        self.device = device
        self.tcfg = tcfg

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()
        self.best_val_loss = float("inf")
        self.best_val_ic = -float("inf")
        self.patience_counter = 0
        self.scheduler = None

    def _make_scheduler(self, n_epochs: int, steps_per_epoch: Optional[int] = None):
        if self.tcfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6,
            )
        elif self.tcfg.scheduler in ("cyclic", "cyclical", "cyclic_lr"):
            sp = steps_per_epoch if steps_per_epoch is not None else 1
            step_size_up = max(1, int(sp))
            base_lr = max(1e-8, self.tcfg.lr * 0.1)
            max_lr = max(base_lr * 1.01, self.tcfg.lr)
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode="triangular2",
                cycle_momentum=False,
            )
        else:
            self.scheduler = None
        return self.scheduler

    def _unpack_batch(self, batch):
        """Extract tensors from cross-sectional batch dict."""
        if isinstance(batch, dict):
            nf = batch["node_features"].squeeze(0)
            tgt = batch["targets"].squeeze(0)
            mask = batch["mask"].squeeze(0)
            ei = batch["edge_index"].squeeze(0)
        elif isinstance(batch, (list, tuple)):
            b = batch[0]
            nf = b["node_features"]
            tgt = b["targets"]
            mask = b["mask"]
            ei = b["edge_index"]
        else:
            return None, None, None, None
        return nf, tgt, mask, ei

    def train_epoch(self, loader: DataLoader, epoch: int = 0) -> float:
        self.model.train()
        total_loss, n_batches, skipped = 0.0, 0, 0
        total_grad_norm = 0.0

        for batch in loader:
            nf, tgt, mask, ei = self._unpack_batch(batch)
            if nf is None or mask.sum() < 2:
                skipped += 1
                continue
            nf = nf.to(self.device, non_blocking=True)
            ei = ei.to(self.device, non_blocking=True)
            mask_dev = mask.to(self.device, non_blocking=True)
            tgt_valid = tgt[mask].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(nf, ei, mask_dev)

            pred = out["prediction"]
            if not torch.isfinite(pred).all():
                logger.warning(
                    f"  [{self.name}] Non-finite GNN prediction "
                    f"(nan={pred.isnan().sum().item()}, inf={pred.isinf().sum().item()}) "
                    f"— skipping batch (epoch {epoch})"
                )
                skipped += 1
                continue

            loss = self.criterion(pred, tgt_valid)

            if not torch.isfinite(loss):
                logger.warning(f"  [{self.name}] Non-finite GNN loss ({loss.item():.4g}) — skipping batch")
                skipped += 1
                continue

            loss.backward()

            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.model.parameters()
                if p.grad is not None
            ) ** 0.5
            total_grad_norm += grad_norm

            if n_batches == 0 and grad_norm < 1e-12:
                logger.warning(
                    f"  [{self.name}] Near-zero gradient norm ({grad_norm:.2e}) "
                    f"at epoch {epoch} — possible dead GNN (no sector edges?)"
                )

            if self.tcfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)

            self.optimizer.step()
            # Advance batch-level cyclic LR if configured
            if getattr(self, "scheduler", None) is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.CyclicLR
            ):
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            total_loss += loss.item()
            n_batches += 1

        avg_grad_norm = total_grad_norm / max(n_batches, 1)
        if skipped > 0 or epoch == 1 or epoch % 5 == 0:
            logger.debug(
                f"  [{self.name}] epoch={epoch}: batches={n_batches}, "
                f"skipped={skipped}, avg_grad_norm={avg_grad_norm:.4e}"
            )
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            nf, tgt, mask, ei = self._unpack_batch(batch)
            if nf is None or mask.sum() < 2:
                continue
            nf = nf.to(self.device, non_blocking=True)
            ei = ei.to(self.device, non_blocking=True)
            mask_dev = mask.to(self.device, non_blocking=True)
            tgt_valid = tgt[mask].to(self.device, non_blocking=True)

            out = self.model(nf, ei, mask_dev)
            loss = self.criterion(out["prediction"], tgt_valid)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(out["prediction"].cpu().numpy())
            all_targets.append(tgt_valid.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        if all_preds:
            metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
        else:
            metrics = {"ic": 0.0, "rank_ic": 0.0, "directional_accuracy": 0.5}
        metrics["loss"] = avg_loss
        return avg_loss, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int, save_dir: str) -> Dict:
        # Create scheduler; pass steps_per_epoch so cyclic schedulers can
        # be stepped per-optimizer-step inside train_epoch.
        scheduler = self._make_scheduler(n_epochs, steps_per_epoch=len(train_loader))
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{self.name}_best.pt")

        history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        self.best_val_loss = float("inf")
        self.best_val_ic = -float("inf")
        self.patience_counter = 0
        use_ic_stopping = (self.tcfg.early_stop_metric == "ic")

        logger.info(f"\n{'='*60}")
        logger.info(f"Training {self.name} (GNN) for {n_epochs} epochs")
        logger.info(f"  Train dates: {len(train_loader)}, Val dates: {len(val_loader)}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch=epoch)
            val_loss, val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t0

            val_ic = val_metrics.get("ic", 0.0)
            if val_ic != val_ic:
                val_ic = 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic"].append(val_ic)
            history["val_rank_ic"].append(val_metrics.get("rank_ic", 0.0))

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"[{self.name}] Epoch {epoch:3d}/{n_epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"IC={val_ic:.4f} | RankIC={val_metrics['rank_ic']:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                    pass
                else:
                    scheduler.step()

            # Early stopping
            # Only save checkpoint when weights are numerically healthy
            weights_ok = not any(
                p.isnan().any().item() or p.isinf().any().item()
                for p in self.model.parameters()
            )
            if not weights_ok:
                logger.warning(f"  [{self.name}] NaN/Inf detected in model weights at epoch {epoch} — not saving checkpoint")

            if use_ic_stopping:
                if val_ic > self.best_val_ic + self.tcfg.ic_min_delta:
                    self.best_val_ic = val_ic
                    self.patience_counter = 0
                    if weights_ok:
                        torch.save(self.model.state_dict(), best_path)
                        logger.info(f"  ★ New best IC={val_ic:.4f}")
                    else:
                        logger.warning(f"  ★ Would be best IC={val_ic:.4f} but weights are NaN — skipping save")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.tcfg.patience:
                        logger.info(f"  Early stopping at epoch {epoch} (best IC={self.best_val_ic:.4f})")
                        break
            else:
                if val_loss < self.best_val_loss - self.tcfg.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if weights_ok:
                        torch.save(self.model.state_dict(), best_path)
                    else:
                        logger.warning(f"  ★ Would be best val={val_loss:.6f} but weights are NaN — skipping save")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.tcfg.patience:
                        logger.info(f"  Early stopping at epoch {epoch}")
                        break

        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        else:
            logger.warning(f"  [{self.name}] No valid checkpoint saved — model weights may be corrupt")
        _save_training_curve(history, self.name, save_dir)
        return history


# ============================================================================
# Meta model trainer
# ============================================================================

class MetaTrainer:
    """Trains the MetaMLP on frozen sub-model embeddings + predictions.

    Supports critic-derived sample weights: per-regime weights upweight
    samples from regimes where the model is weakest.
    """

    def __init__(self, meta: MetaMLP, device: torch.device, tcfg: TrainConfig):
        self.meta = meta.to(device)
        if torch.cuda.is_available():
            try:
                self.meta = torch.compile(self.meta)
                logger.info(f"  [Meta] torch.compile() enabled")
            except Exception as e:
                logger.warning(f"  [Meta] torch.compile() failed: {e} — running eager")
        self.device = device
        self.tcfg = tcfg

        self.optimizer = torch.optim.AdamW(
            meta.parameters(), lr=tcfg.lr_meta, weight_decay=tcfg.weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.01, reduction='none') if tcfg.loss_fn == "huber" else nn.MSELoss(reduction='none')
        self.criterion_mean = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()

    @torch.no_grad()
    def _collect_sub_outputs(
        self,
        forecaster: HierarchicalForecaster,
        daily_loader: DataLoader,
        minute_loader: DataLoader,
        data_cfg: HierarchicalDataConfig,
        news_loader: Optional[DataLoader] = None,
        fund_loader: Optional[DataLoader] = None,
        graph_loader: Optional[DataLoader] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sub-models and collect predictions + embeddings.

        Alignment strategy:
          1. Run each sub-model on its modality's data loader, collecting
             per-sample (ticker, ordinal_date) keys alongside predictions
             and embeddings.
          2. Build a union of ALL keys across all modalities. For each key
             in the union, look up each sub-model (zero-fill if missing).
             This ensures that even if daily and minute data cover different
             time periods, both still contribute meta training samples.
          3. Look up real regime features by date.

        Returns:
            predictions: (N, n_sub_models)
            emb_cat:     (N, n_sub_models*E)
            regime:      (N, R)
            targets:     (N,)
        """
        import datetime as _dt

        forecaster.eval()
        E = forecaster.cfg.embedding_dim
        R = forecaster.cfg.regime_dim
        sub_names = forecaster.sub_model_names
        n_sub = len(sub_names)

        # Map modality names → data loaders
        modality_loaders = {
            "daily": daily_loader,
            "minute": minute_loader,
            "news": news_loader,
            "fundamental": fund_loader,
            "graph": graph_loader,
        }

        # Build regime lookup table (ordinal_date → regime_vector)
        regime_df = _build_regime_dataframe(data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        if not regime_df.empty:
            for d, row in regime_df.iterrows():
                if hasattr(d, 'toordinal'):
                    regime_lookup[d.toordinal()] = row.values.astype(np.float32)

        # --- Run each sub-model on its modality's loader ---
        # outputs[name][(ticker, ord_date)] = {"pred": tensor, "emb": tensor}
        outputs: Dict[str, Dict[Tuple[str, int], Dict]] = {}
        daily_targets: Dict[Tuple[str, int], torch.Tensor] = {}

        for name in sub_names:
            modality = HierarchicalForecaster.MODALITY[name]
            loader = modality_loaders.get(modality)
            if loader is None:
                outputs[name] = {}
                continue

            sub_model = forecaster.sub_models[name]
            sub_model.to(self.device).eval()
            model_data: Dict[Tuple[str, int], Dict] = {}

            if modality == "graph":
                # GNN: cross-sectional loader — each batch is one date
                for batch in loader:
                    # batch is a dict (batch_size=1)
                    if isinstance(batch, dict):
                        nf = batch["node_features"].squeeze(0).to(self.device)
                        tgt = batch["targets"].squeeze(0)
                        mask = batch["mask"].squeeze(0)
                        ei = batch["edge_index"].squeeze(0).to(self.device)
                        od = int(batch["ordinal_date"])
                        tickers_list = batch["tickers"]
                    elif isinstance(batch, (list, tuple)):
                        b = batch[0]  # single-element batch
                        nf = b["node_features"].to(self.device)
                        tgt = b["targets"]
                        mask = b["mask"]
                        ei = b["edge_index"].to(self.device)
                        od = int(b["ordinal_date"])
                        tickers_list = b["tickers"]
                    else:
                        continue

                    out = sub_model(nf, ei, mask)
                    # out has variable-length (mask.sum(),)
                    valid_mask = [bool(mask[j]) for j in range(len(tickers_list))]
                    valid_tickers = [tickers_list[j] for j in range(len(tickers_list)) if valid_mask[j]]
                    valid_targets = [tgt[j] for j in range(len(tickers_list)) if valid_mask[j]]
                    for vi, ticker in enumerate(valid_tickers):
                        key = (ticker, od)
                        model_data[key] = {
                            "pred": out["prediction"][vi].cpu(),
                            "emb": out["embedding"][vi].cpu(),
                        }
                        # Add GNN keys to the union targets dict if not already present
                        if key not in daily_targets:
                            daily_targets[key] = valid_targets[vi]
            else:
                for batch in loader:
                    x, y, ordinal_dates, tickers_batch = batch
                    x = x.to(self.device, non_blocking=True)
                    out = sub_model(x)
                    for i in range(len(y)):
                        key = (tickers_batch[i], int(ordinal_dates[i]))
                        model_data[key] = {
                            "pred": out["prediction"][i].cpu(),
                            "emb": out["embedding"][i].cpu(),
                        }
                        # Collect targets from all non-graph modalities.
                        # Daily has the most coverage; other modalities fill
                        # in keys not present in daily (e.g. recent-only minute data).
                        if key not in daily_targets:
                            daily_targets[key] = y[i]

            sub_model.cpu()
            outputs[name] = model_data

        clear_gpu_memory()

        # --- Align: iterate over ALL keys from ALL modalities (union) ---
        # This ensures minute/fund/GNN keys (which may cover a different time
        # period than daily) still contribute meta training samples.
        preds_list, embs_list, regime_list, target_list = [], [], [], []
        zero_emb = torch.zeros(E)
        counters = {name: 0 for name in sub_names}

        for key, target in daily_targets.items():
            ticker, ord_date = key

            pred_vec = []
            emb_vec = []
            for name in sub_names:
                md = outputs[name].get(key)
                if md is not None:
                    pred_vec.append(md["pred"])
                    emb_vec.append(md["emb"])
                    counters[name] += 1
                else:
                    pred_vec.append(torch.tensor(0.0))
                    emb_vec.append(zero_emb)

            # Regime features from real market data
            if ord_date in regime_lookup:
                regime_vec = torch.from_numpy(regime_lookup[ord_date])
            else:
                regime_vec = torch.zeros(R)

            preds_list.append(torch.stack(pred_vec))
            embs_list.append(torch.cat(emb_vec))
            regime_list.append(regime_vec)
            target_list.append(target)

        match_info = ", ".join(f"{n}={c}" for n, c in counters.items())
        logger.info(f"  Meta align: {len(daily_targets)} union keys | matched: {match_info}")

        # Warn about sub-models that contributed nothing (all-zero slot).
        # A failed GNN or missing loader silently degrades meta quality and
        # flattens the meta loss curve — surface it clearly here.
        for name in sub_names:
            cnt = counters[name]
            if cnt == 0:
                logger.warning(
                    f"  [Meta] Sub-model '{name}' contributed 0 predictions — "
                    f"its input slot will be all-zeros. This typically means the "
                    f"model failed (GNN NaN / News missing) or its loader is None. "
                    f"Consider disabling it with --skip-models if it cannot be fixed."
                )
            elif cnt < len(daily_targets) * 0.1:
                logger.warning(
                    f"  [Meta] Sub-model '{name}' matched only {cnt}/{len(daily_targets)} "
                    f"keys ({100*cnt/max(len(daily_targets),1):.1f}%) — sparse coverage."
                )

        if not preds_list:
            logger.warning("  No aligned data for meta model!")
            return (torch.zeros(0, n_sub), torch.zeros(0, n_sub * E),
                    torch.zeros(0, R), torch.zeros(0))

        predictions = torch.stack(preds_list)        # (N, n_sub)
        emb_cat     = torch.stack(embs_list)          # (N, n_sub*E)
        regime      = torch.stack(regime_list)         # (N, R)
        targets     = torch.stack(target_list)         # (N,)

        return predictions, emb_cat, regime, targets

    def train(
        self,
        forecaster: HierarchicalForecaster,
        d_train_dl: DataLoader, m_train_dl: DataLoader,
        d_val_dl: DataLoader, m_val_dl: DataLoader,
        n_epochs: int, save_dir: str,
        data_cfg: HierarchicalDataConfig = None,
        n_train_news_dl: Optional[DataLoader] = None,
        n_val_news_dl: Optional[DataLoader] = None,
        f_train_dl: Optional[DataLoader] = None,
        f_val_dl: Optional[DataLoader] = None,
        g_train_dl: Optional[DataLoader] = None,
        g_val_dl: Optional[DataLoader] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Train meta model, optionally with per-sample weights from critic.

        Args:
            sample_weights: (N_train,) tensor of weights. If provided, each
                sample's loss is multiplied by its weight so the model focuses
                on regimes where it is weakest. Weights are loaded from
                data/critic_weights/latest_weights.json during pipeline runs.
        """
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "meta_best.pt")
        E = forecaster.cfg.embedding_dim
        n_sub = len(forecaster.sub_model_names)

        logger.info("Collecting sub-model outputs for meta training...")
        forecaster.freeze_sub_models()

        train_preds, train_embs, train_regime, train_targets = \
            self._collect_sub_outputs(forecaster, d_train_dl, m_train_dl, data_cfg,
                                      news_loader=n_train_news_dl,
                                      fund_loader=f_train_dl,
                                      graph_loader=g_train_dl)
        val_preds, val_embs, val_regime, val_targets = \
            self._collect_sub_outputs(forecaster, d_val_dl, m_val_dl, data_cfg,
                                      news_loader=n_val_news_dl,
                                      fund_loader=f_val_dl,
                                      graph_loader=g_val_dl)

        logger.info(f"  Meta train: {len(train_targets):,} | val: {len(val_targets):,}")

        # Build training dataset — include sample weights if provided
        use_sample_weights = sample_weights is not None and len(sample_weights) == len(train_targets)
        if use_sample_weights:
            logger.info(f"  Using critic sample weights: mean={sample_weights.mean():.2f}, "
                        f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}")
            train_ds = TensorDataset(train_preds, train_embs, train_regime, train_targets, sample_weights)
        else:
            if sample_weights is not None:
                logger.warning(f"  Sample weights size mismatch ({len(sample_weights)} vs {len(train_targets)}) — ignoring")
            train_ds = TensorDataset(train_preds, train_embs, train_regime, train_targets)
        val_ds = TensorDataset(val_preds, val_embs, val_regime, val_targets)
        train_loader = DataLoader(train_ds, batch_size=self.tcfg.batch_size_meta, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.tcfg.batch_size_meta)

        # Build scheduler according to TrainConfig
        if self.tcfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6,
            )
        elif self.tcfg.scheduler in ("cyclic", "cyclical", "cyclic_lr"):
            step_size_up = max(1, len(train_loader))
            base_lr = max(1e-8, self.tcfg.lr_meta * 0.1)
            max_lr = max(base_lr * 1.01, self.tcfg.lr_meta)
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode="triangular2",
                cycle_momentum=False,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6,
            )

        history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        best_val = float("inf")
        best_ic  = -float("inf")
        patience_ctr = 0
        use_ic_stopping = (self.tcfg.early_stop_metric == "ic")

        logger.info(f"\n{'='*60}")
        logger.info(f"Training MetaMLP for {n_epochs} epochs ({self.meta.count_parameters():,} params)")
        logger.info(f"  Early-stop metric: {'IC' if use_ic_stopping else 'val_loss'}, patience={self.tcfg.patience}")
        if use_sample_weights:
            logger.info(f"  Critic sample weighting: ENABLED")
        logger.info(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            self.meta.train()
            total_loss, n_b = 0.0, 0
            for batch in train_loader:
                if use_sample_weights:
                    preds_b, embs_b, regime_b, targets_b, weights_b = batch
                    weights_b = weights_b.to(self.device, non_blocking=True)
                else:
                    preds_b, embs_b, regime_b, targets_b = batch
                    weights_b = None
                preds_b = preds_b.to(self.device, non_blocking=True)
                embs_b = embs_b.to(self.device, non_blocking=True)
                regime_b = regime_b.to(self.device, non_blocking=True)
                targets_b = targets_b.to(self.device, non_blocking=True)
                emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(n_sub)]

                self.optimizer.zero_grad(set_to_none=True)
                out = self.meta(preds_b, emb_list, regime_b)

                if weights_b is not None:
                    # Per-sample weighted loss (reduction='none' → weighted mean)
                    per_sample_loss = self.criterion(out["prediction"], targets_b)
                    loss = (per_sample_loss * weights_b).mean()
                else:
                    loss = self.criterion_mean(out["prediction"], targets_b)

                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), self.tcfg.grad_clip)
                self.optimizer.step()
                # Advance batch-level cyclic LR if configured
                if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                total_loss += loss.item()
                n_b += 1

            train_loss = total_loss / max(n_b, 1)

            self.meta.eval()
            val_preds_l, val_tgts_l = [], []
            val_total, val_n = 0.0, 0
            with torch.no_grad():
                for preds_b, embs_b, regime_b, targets_b in val_loader:
                    preds_b = preds_b.to(self.device, non_blocking=True)
                    embs_b = embs_b.to(self.device, non_blocking=True)
                    regime_b = regime_b.to(self.device, non_blocking=True)
                    targets_b = targets_b.to(self.device, non_blocking=True)
                    emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(n_sub)]
                    out = self.meta(preds_b, emb_list, regime_b)
                    val_total += self.criterion_mean(out["prediction"], targets_b).item()
                    val_n += 1
                    val_preds_l.append(out["prediction"].cpu().numpy())
                    val_tgts_l.append(targets_b.cpu().numpy())

            val_loss = val_total / max(val_n, 1)
            metrics = compute_metrics(np.concatenate(val_preds_l), np.concatenate(val_tgts_l))
            elapsed = time.time() - t0

            val_ic_val = metrics["ic"] if metrics["ic"] == metrics["ic"] else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic"].append(val_ic_val)
            history["val_rank_ic"].append(metrics.get("rank_ic", 0.0))

            logger.info(
                f"[Meta] Epoch {epoch:3d}/{n_epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"IC={val_ic_val:.4f} | RankIC={metrics['rank_ic']:.4f} | "
                f"DirAcc={metrics['directional_accuracy']:.3f} | {elapsed:.1f}s"
            )
            # If scheduler is cyclic we already step it per-batch above.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # (Not expected here, but keep same semantics)
                scheduler.step(val_loss)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                pass
            else:
                scheduler.step()

            if use_ic_stopping:
                improved = val_ic_val > best_ic + self.tcfg.ic_min_delta
                if improved or (best_ic <= 0 and val_loss < best_val - self.tcfg.min_delta):
                    if improved:
                        best_ic = val_ic_val
                    best_val = val_loss
                    patience_ctr = 0
                    torch.save(self.meta.state_dict(), best_path)
                    logger.info(f"  ★ New best meta IC={val_ic_val:.4f} | val={val_loss:.6f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.tcfg.patience:
                        logger.info(f"  Early stopping at epoch {epoch} (best IC={best_ic:.4f})")
                        break
            else:
                if val_loss < best_val - self.tcfg.min_delta:
                    best_val = val_loss
                    patience_ctr = 0
                    torch.save(self.meta.state_dict(), best_path)
                    logger.info(f"  ★ New best meta val={val_loss:.6f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.tcfg.patience:
                        logger.info(f"  Early stopping at epoch {epoch}")
                        break

        self.meta.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        _save_training_curve(history, "Meta", save_dir)
        return history


# ============================================================================
# Phase 4: Sequential Round-Robin Fine-Tuning (memory-efficient)
# ============================================================================

class JointFineTuner:
    """Phase 4: Fine-tune sub-models one-at-a-time through the meta loss.

    Memory-efficient alternative to loading all 6 models on GPU at once.

    Strategy (per round):
      For each sub-model (lstm_d, tft_d, lstm_m, tft_m, [news]):
        1. Move that sub-model + meta to GPU; keep others on CPU.
        2. Unfreeze only that sub-model + meta; freeze everything else.
        3. Forward: run the active sub-model live (with gradients), but
           use cached frozen outputs for the other sub-models.
        4. Backprop through active sub-model → meta only.
        5. Move the sub-model back to CPU, advance to the next.

    This gives the same end-to-end gradient signal as full joint training
    but uses ~1/5 the GPU memory (only 1 sub-model + meta on GPU at a time).

    After all sub-models get one turn, that is one "round". We repeat for
    N rounds (= epochs_phase4).
    """

    def __init__(
        self,
        forecaster: HierarchicalForecaster,
        device: torch.device,
        tcfg: TrainConfig,
        data_cfg: HierarchicalDataConfig,
    ):
        self.forecaster = forecaster
        self.device = device
        self.tcfg = tcfg
        self.data_cfg = data_cfg
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()
        self.scaler = GradScaler() if (tcfg.use_amp and device.type == "cuda") else None
        # More workers for Phase 4 — cache collection is CPU-bound with 24 cores available
        self.num_workers = min(tcfg.num_workers * 2, 12)

    # ------------------------------------------------------------------
    # Helpers: collect frozen sub-model outputs (like MetaTrainer does)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_frozen_outputs(
        self,
        forecaster: HierarchicalForecaster,
        daily_loader: DataLoader,
        minute_loader: DataLoader,
        data_cfg: HierarchicalDataConfig,
        news_loader: Optional[DataLoader] = None,
        fund_loader: Optional[DataLoader] = None,
        graph_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run all sub-models and cache predictions + embeddings.

        Uses the registry pattern: iterates over forecaster.sub_model_names,
        runs each on its modality's loader, then aligns by (ticker, date).

        Returns a flat dict with:
            keys:          list of (ticker, ordinal_date) tuples
            targets:       (N,) tensor
            regimes:       (N, R) tensor
            {name}_pred:   (N,) tensor for each sub-model
            {name}_emb:    (N, E) tensor for each sub-model
        """
        E = forecaster.cfg.embedding_dim
        R = forecaster.cfg.regime_dim
        sub_names = forecaster.sub_model_names

        # Map modality names → data loaders
        modality_loaders = {
            "daily": daily_loader,
            "minute": minute_loader,
            "news": news_loader,
            "fundamental": fund_loader,
            "graph": graph_loader,
        }

        # Build regime lookup
        regime_df = _build_regime_dataframe(data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        if not regime_df.empty:
            for d, row in regime_df.iterrows():
                if hasattr(d, 'toordinal'):
                    regime_lookup[d.toordinal()] = row.values.astype(np.float32)

        # --- Run each sub-model one at a time to minimize VRAM ---
        # per_model[name][(ticker, date)] = {"pred": ..., "emb": ...}
        per_model: Dict[str, Dict[Tuple[str, int], Dict]] = {}
        daily_targets: Dict[Tuple[str, int], torch.Tensor] = {}
        daily_regimes: Dict[Tuple[str, int], torch.Tensor] = {}

        for name in sub_names:
            modality = HierarchicalForecaster.MODALITY[name]
            loader = modality_loaders.get(modality)
            if loader is None:
                per_model[name] = {}
                continue

            sub_model = forecaster.sub_models[name]
            sub_model.to(self.device).eval()
            model_data: Dict[Tuple[str, int], Dict] = {}

            if modality == "graph":
                # GNN: cross-sectional loader
                for batch in loader:
                    if isinstance(batch, dict):
                        nf = batch["node_features"].squeeze(0).to(self.device)
                        tgt = batch["targets"].squeeze(0)
                        mask = batch["mask"].squeeze(0)
                        ei = batch["edge_index"].squeeze(0).to(self.device)
                        od = int(batch["ordinal_date"])
                        tickers_list = batch["tickers"]
                    elif isinstance(batch, (list, tuple)):
                        b = batch[0]
                        nf = b["node_features"].to(self.device)
                        tgt = b["targets"]
                        mask = b["mask"]
                        ei = b["edge_index"].to(self.device)
                        od = int(b["ordinal_date"])
                        tickers_list = b["tickers"]
                    else:
                        continue
                    out = sub_model(nf, ei, mask)
                    valid_tickers = [tickers_list[j] for j in range(len(tickers_list)) if mask[j]]
                    for vi, ticker in enumerate(valid_tickers):
                        key = (ticker, od)
                        model_data[key] = {
                            "pred": out["prediction"][vi].cpu(),
                            "emb": out["embedding"][vi].cpu(),
                        }
            else:
                for batch in loader:
                    x, y, ordinal_dates, tickers_batch = batch
                    x_dev = x.to(self.device, non_blocking=True)
                    out = sub_model(x_dev)
                    for i in range(len(y)):
                        key = (tickers_batch[i], int(ordinal_dates[i]))
                        model_data[key] = {
                            "pred": out["prediction"][i].cpu(),
                            "emb": out["embedding"][i].cpu(),
                        }
                        # Collect targets + regime from daily modality
                        if modality == "daily" and key not in daily_targets:
                            daily_targets[key] = y[i]
                            ord_date = int(ordinal_dates[i])
                            daily_regimes[key] = (
                                torch.from_numpy(regime_lookup[ord_date])
                                if ord_date in regime_lookup
                                else torch.zeros(R)
                            )

            sub_model.cpu()
            per_model[name] = model_data

        clear_gpu_memory()

        # Warn if any sub-model produced NaN/Inf outputs (e.g. diverged in a prior phase)
        for name in sub_names:
            md = per_model[name]
            if md:
                sample_pred = next(iter(md.values()))["pred"]
                sample_emb  = next(iter(md.values()))["emb"]
                if not torch.isfinite(sample_pred).all() or not torch.isfinite(sample_emb).all():
                    nan_count = sum(
                        1 for v in md.values()
                        if not torch.isfinite(v["pred"]).all() or not torch.isfinite(v["emb"]).all()
                    )
                    logger.warning(
                        f"  [{name}] {nan_count}/{len(md)} cached outputs contain NaN/Inf "
                        f"(model likely diverged in a prior phase) \u2014 replacing with zeros"
                    )

        # --- Merge into aligned flat tensors ---
        # IMPORTANT: Use UNION of keys from all modalities (not just daily).
        # This ensures that minute samples with dates not in daily still get
        # training pairs (with zero-filled embeddings from missing modalities).
        # Without this, minute batches would fail to match and be skipped entirely.
        zero_pred = torch.zeros(1).squeeze()
        zero_emb = torch.zeros(E)

        all_keys_set = set(daily_targets.keys())
        for name in sub_names:
            all_keys_set.update(per_model[name].keys())

        keys_ordered = sorted(all_keys_set)  # Sort for reproducibility
        all_preds = {name: [] for name in sub_names}
        all_embs = {name: [] for name in sub_names}
        all_regimes, all_targets = [], []

        for key in keys_ordered:
            # Get target (zero-fill if missing from daily)
            if key in daily_targets:
                all_targets.append(daily_targets[key])
                all_regimes.append(daily_regimes[key])
            else:
                # Key from non-daily modality; use zero target and regime
                # (This is rare but can happen if minute has data beyond daily coverage)
                all_targets.append(torch.tensor(0.0))
                all_regimes.append(torch.zeros(R))

            for name in sub_names:
                md = per_model[name].get(key)
                if md is not None:
                    # Replace any NaN/Inf in cached outputs with zero so they don't
                    # propagate into meta input and cause unbounded losses in Phase 4
                    pred = torch.nan_to_num(md["pred"], nan=0.0, posinf=0.0, neginf=0.0)
                    emb  = torch.nan_to_num(md["emb"],  nan=0.0, posinf=0.0, neginf=0.0)
                    all_preds[name].append(pred)
                    all_embs[name].append(emb)
                else:
                    all_preds[name].append(zero_pred)
                    all_embs[name].append(zero_emb)

        result = {
            "keys": keys_ordered,
            "targets": torch.stack(all_targets),
            "regimes": torch.stack(all_regimes),
        }
        for name in sub_names:
            result[f"{name}_pred"] = torch.stack(all_preds[name])
            result[f"{name}_emb"] = torch.stack(all_embs[name])

        # Log coverage breakdown
        daily_coverage = sum(1 for k in keys_ordered if k in daily_targets)
        logger.info(
            f"  Frozen outputs collected: {len(keys_ordered):,} samples "
            f"({daily_coverage:,} from daily, "
            f"{len(keys_ordered) - daily_coverage:,} from other modalities)"
        )
        return result

    def _build_meta_input(
        self,
        cached: Dict[str, torch.Tensor],
        active_name: str,
        active_pred: torch.Tensor,
        active_emb: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Build meta model inputs, swapping in live outputs for the active sub-model.

        Returns (preds_stacked, emb_list, regime) ready for MetaMLP.
        """
        sub_names = self.forecaster.sub_model_names

        preds_list = []
        emb_list = []
        for name in sub_names:
            if name == active_name:
                preds_list.append(active_pred)
                emb_list.append(active_emb)
            else:
                preds_list.append(cached[f"{name}_pred"][indices].to(self.device))
                emb_list.append(cached[f"{name}_emb"][indices].to(self.device))

        preds = torch.stack(preds_list, dim=-1)  # (batch, N_sub)
        regime = cached["regimes"][indices].to(self.device)
        return preds, emb_list, regime

    # ------------------------------------------------------------------
    # Per-sub-model fine-tune step
    # ------------------------------------------------------------------
    def _finetune_one_submodel(
        self,
        sub_model: nn.Module,
        sub_name: str,
        data_loader: DataLoader,
        val_data_loader: DataLoader,
        cached_train: Dict[str, torch.Tensor],
        cached_val: Dict[str, torch.Tensor],
        train_key_to_idx: Dict[Tuple[str, int], int],
        val_key_to_idx: Dict[Tuple[str, int], int],
        optimizer: torch.optim.Optimizer,
        epoch_label: str,
    ) -> Tuple[float, float, Dict]:
        """Fine-tune one sub-model + meta for one pass over its data.

        1. Move sub_model + meta to GPU
        2. Unfreeze sub_model + meta, freeze everything else
        3. For each batch from sub_model's data loader:
           - Forward through sub_model (live, with gradients)
           - Build meta input using cached outputs for other sub-models
           - Forward through meta
           - Backprop loss through sub_model + meta
        4. Move sub_model back to CPU

        Returns (train_loss, val_loss, val_metrics).
        """
        # Ensure all sub-models and meta are on CPU first, then move active ones to GPU.
        # This prevents stale device state from a previous sub-model's turn.
        for name in self.forecaster.sub_model_names:
            self.forecaster.sub_models[name].cpu()
        self.forecaster.meta.cpu()
        clear_gpu_memory()

        sub_model.to(self.device)
        self.forecaster.meta.to(self.device)

        # Freeze everything, then unfreeze active sub-model + meta
        for p in self.forecaster.parameters():
            p.requires_grad = False
        for p in sub_model.parameters():
            p.requires_grad = True
        for p in self.forecaster.meta.parameters():
            p.requires_grad = True

        sub_model.train()
        self.forecaster.meta.train()

        # --- Train pass ---
        total_loss, n_b = 0.0, 0
        n_cache_miss, n_nan_loss = 0, 0
        for batch in data_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device, non_blocking=True)
            target = y.to(self.device, non_blocking=True)

            # Find matching indices in the cached output arrays
            indices = []
            valid_mask = []
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                idx = train_key_to_idx.get(key, -1)
                indices.append(idx if idx >= 0 else 0)  # placeholder 0 for missing
                valid_mask.append(idx >= 0)

            indices_t = torch.tensor(indices, dtype=torch.long)
            valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool)

            if not valid_mask_t.any():
                n_cache_miss += 1
                continue

            # Forward through active sub-model
            amp_ctx = autocast(device_type="cuda", dtype=torch.float16) if self.scaler else torch.amp.autocast(device_type="cpu", enabled=False)
            with amp_ctx:
                out = sub_model(x_dev)
                active_pred = out["prediction"]   # (batch,)
                active_emb = out["embedding"]     # (batch, E)

                # Only use samples that matched in the cached output arrays
                active_pred_v = active_pred[valid_mask_t]
                active_emb_v = active_emb[valid_mask_t]
                target_v = target[valid_mask_t]
                indices_v = indices_t[valid_mask_t]

                # Replace any NaN/Inf in active outputs with zero before building meta input
                active_pred_v = torch.nan_to_num(active_pred_v, nan=0.0, posinf=0.0, neginf=0.0)
                active_emb_v  = torch.nan_to_num(active_emb_v,  nan=0.0, posinf=0.0, neginf=0.0)

                # Build meta inputs (swap in live outputs for active model)
                preds, emb_list, regime = self._build_meta_input(
                    cached_train, sub_name, active_pred_v, active_emb_v, indices_v,
                )

                # Forward through meta
                meta_out = self.forecaster.meta(preds, emb_list, regime)
                loss = self.criterion(meta_out["prediction"], target_v)

                # Auxiliary loss on the sub-model's own prediction
                loss = loss + 0.1 * self.criterion(active_pred_v, target_v)

            # Skip batch if loss exploded to NaN/Inf (e.g. first round instability)
            if not torch.isfinite(loss):
                logger.warning(f"  [{sub_name}] Non-finite loss ({loss.item():.4g}) — skipping batch")
                n_nan_loss += 1
                continue

            optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(sub_model.parameters()) + list(self.forecaster.meta.parameters()),
                    self.tcfg.grad_clip * 0.5,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(sub_model.parameters()) + list(self.forecaster.meta.parameters()),
                    self.tcfg.grad_clip * 0.5,
                )
                optimizer.step()
            total_loss += loss.item()
            n_b += 1

        train_loss = total_loss / max(n_b, 1)
        if n_b == 0:
            if n_nan_loss > 0:
                logger.warning(
                    f"  [{sub_name}] All {n_nan_loss} training batches had non-finite loss — "
                    "sub-model or cached embeddings may contain NaN. "
                    "Check if GNN/meta diverged in a prior phase."
                )
            elif n_cache_miss > 0:
                logger.warning(
                    f"  [{sub_name}] All {n_cache_miss} training batches had no valid cache matches — "
                    "check that Phase 3 cache keys align with Phase 4 data loaders."
                )

        # --- Validation pass (use val data loader for this sub-model's modality) ---
        sub_model.eval()
        self.forecaster.meta.eval()
        val_preds_l, val_tgts_l = [], []
        val_total, val_n = 0.0, 0

        # For validation, iterate the dedicated validation data loader
        with torch.no_grad():
            for batch in val_data_loader:
                x, y, ordinal_dates, tickers_batch = batch
                x_dev = x.to(self.device, non_blocking=True)
                target = y.to(self.device, non_blocking=True)

                indices = []
                valid_mask = []
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    idx = val_key_to_idx.get(key, -1)
                    indices.append(idx if idx >= 0 else 0)
                    valid_mask.append(idx >= 0)

                indices_t = torch.tensor(indices, dtype=torch.long)
                valid_mask_t = torch.tensor(valid_mask, dtype=torch.bool)
                if not valid_mask_t.any():
                    continue

                out = sub_model(x_dev)
                pred_v = out["prediction"][valid_mask_t]
                emb_v = out["embedding"][valid_mask_t]
                indices_v = indices_t[valid_mask_t]
                target_v = target[valid_mask_t]

                preds, emb_list, regime = self._build_meta_input(
                    cached_val, sub_name, pred_v, emb_v, indices_v,
                )
                meta_out = self.forecaster.meta(preds, emb_list, regime)

                batch_loss = self.criterion(meta_out["prediction"], target_v)
                if torch.isfinite(batch_loss):
                    val_total += batch_loss.item()
                    val_n += 1
                val_preds_l.append(meta_out["prediction"].cpu().numpy())
                val_tgts_l.append(target_v.cpu().numpy())

        val_loss = val_total / max(val_n, 1)
        if val_preds_l:
            metrics = compute_metrics(np.concatenate(val_preds_l), np.concatenate(val_tgts_l))
        else:
            metrics = {"ic": 0.0, "rank_ic": 0.0, "directional_accuracy": 0.5}

        # Move both active models back to CPU to free VRAM
        sub_model.cpu()
        self.forecaster.meta.cpu()
        clear_gpu_memory()

        logger.info(
            f"  [{epoch_label}][{sub_name:7s}] train={train_loss:.6f} | "
            f"val={val_loss:.6f} | IC={metrics['ic']:.4f} | "
            f"RankIC={metrics['rank_ic']:.4f}"
        )

        return train_loss, val_loss, metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        d_train_dl: DataLoader, m_train_dl: DataLoader,
        d_val_dl: DataLoader, m_val_dl: DataLoader,
        n_rounds: int, save_dir: str,
        n_train_news_dl: Optional[DataLoader] = None,
        n_val_news_dl: Optional[DataLoader] = None,
        fund_train_dl: Optional[DataLoader] = None,
        fund_val_dl: Optional[DataLoader] = None,
        graph_train_dl: Optional[DataLoader] = None,
        graph_val_dl: Optional[DataLoader] = None,
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "finetune_best.pt")

        # --- Data-source → loader mapping ---
        DATA_SOURCE_LOADERS: Dict[str, Tuple[Optional[DataLoader], Optional[DataLoader]]] = {
            "daily":       (d_train_dl, d_val_dl),
            "minute":      (m_train_dl, m_val_dl),
            "news":        (n_train_news_dl, n_val_news_dl),
            "fundamental": (fund_train_dl, fund_val_dl),
            "graph":       (graph_train_dl, graph_val_dl),
        }

        # --- Step 1: Collect frozen outputs for all sub-models ---
        logger.info("Collecting frozen sub-model outputs for round-robin fine-tuning...")
        self.forecaster.eval()

        cached_train = self._collect_frozen_outputs(
            self.forecaster, d_train_dl, m_train_dl, self.data_cfg,
            news_loader=n_train_news_dl,
            fund_loader=fund_train_dl,
            graph_loader=graph_train_dl,
        )
        cached_val = self._collect_frozen_outputs(
            self.forecaster, d_val_dl, m_val_dl, self.data_cfg,
            news_loader=n_val_news_dl,
            fund_loader=fund_val_dl,
            graph_loader=graph_val_dl,
        )

        # Build key → index maps for fast lookup
        train_key_to_idx = {k: i for i, k in enumerate(cached_train["keys"])}
        val_key_to_idx = {k: i for i, k in enumerate(cached_val["keys"])}

        logger.info(f"  Train samples: {len(train_key_to_idx):,} | Val: {len(val_key_to_idx):,}")

        # --- Step 2: Set up sub-model → data loader mapping (registry-based) ---
        sub_models: List[Tuple[str, nn.Module, DataLoader, Optional[DataLoader]]] = []
        for name in self.forecaster.sub_model_names:
            ds = HierarchicalForecaster.MODALITY[name]
            # GNN uses cross-sectional batches; skip in round-robin fine-tuning
            # (GNN is trained separately in Phase 1.7 and its frozen outputs
            # are still included in the meta input via cached_train/cached_val)
            if ds == "graph":
                logger.info(f"  Skipping {name}: GNN uses cross-sectional batches (Phase 4 unsupported)")
                continue
            train_dl_pair, val_dl_pair = DATA_SOURCE_LOADERS[ds]
            if train_dl_pair is None:
                logger.warning(f"  Skipping {name}: no data loader for source '{ds}'")
                continue
            # Skip sub-models whose training loader has no samples — this
            # happens when the minute cache is stale and the test window is
            # empty.  Fine-tuning with zero batches produces no gradients and
            # can corrupt the meta model's cached embeddings for that slot.
            if len(train_dl_pair.dataset) == 0:
                logger.warning(
                    f"  Skipping {name} in Phase 4: training loader is empty "
                    f"(modality='{ds}', n_samples=0). "
                    f"Its phase-1/2 best weights are preserved unchanged."
                )
                continue
            sub_models.append((
                name,
                self.forecaster.sub_models[name],
                train_dl_pair,
                val_dl_pair,
            ))

        # Persistent optimizers — one per sub-model, created once so momentum
        # accumulates properly across rounds instead of resetting every round.
        persistent_optimizers: Dict[str, torch.optim.Optimizer] = {
            name: torch.optim.AdamW(
                list(self.forecaster.sub_models[name].parameters()) + list(self.forecaster.meta.parameters()),
                lr=self.tcfg.lr_finetune,
                weight_decay=self.tcfg.weight_decay,
            )
            for name, _, _, _ in sub_models
        }

        history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        best_val = float("inf")
        best_ic = -float("inf")
        patience_ctr = 0
        use_ic_stopping = (self.tcfg.early_stop_metric == "ic")

        total_params = sum(p.numel() for p in self.forecaster.parameters())
        logger.info(f"\n{'='*60}")
        logger.info(f"Sequential round-robin fine-tuning for {n_rounds} rounds")
        logger.info(f"  {len(sub_models)} sub-models, {total_params:,} total params")
        logger.info(f"  LR={self.tcfg.lr_finetune:.1e}, grad_clip={self.tcfg.grad_clip}")
        logger.info(f"  Only 1 sub-model + meta on GPU at a time (memory-safe)")
        logger.info(f"{'='*60}")

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()
            round_train_losses = []
            round_val_losses = []
            round_val_ics = []

            for sub_name, sub_model, train_dl, val_dl in sub_models:
                # Use the persistent optimizer for this sub-model
                optimizer = persistent_optimizers[sub_name]

                tr_loss, v_loss, v_metrics = self._finetune_one_submodel(
                    sub_model, sub_name,
                    train_dl, val_dl, cached_train, cached_val,
                    train_key_to_idx, val_key_to_idx,
                    optimizer, f"Rnd {rnd:2d}",
                )

                round_train_losses.append(tr_loss)
                round_val_losses.append(v_loss)
                round_val_ics.append(v_metrics["ic"])

                # Update cached outputs for this sub-model after fine-tuning
                # so the next sub-model in this round sees the updated outputs
                self._update_cached_outputs(
                    sub_model, sub_name, train_dl, cached_train, train_key_to_idx,
                )
                self._update_cached_outputs(
                    sub_model, sub_name, val_dl, cached_val, val_key_to_idx,
                )

            elapsed = time.time() - t0
            avg_train = np.mean(round_train_losses)
            avg_val = np.mean(round_val_losses)
            avg_ic = np.mean(round_val_ics)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["val_ic"].append(avg_ic)
            history["val_rank_ic"].append(avg_ic)  # approximate

            # Gradient norm diagnostics — helps identify dead sub-models
            # (zero grad norm = no learning, likely frozen or disconnected)
            grad_norms = {}
            for sub_name, sub_mod in self.forecaster.sub_models.items():
                norms = [
                    p.grad.data.norm(2).item() ** 2
                    for p in sub_mod.parameters() if p.grad is not None
                ]
                grad_norms[sub_name] = float(sum(norms) ** 0.5) if norms else 0.0
            meta_norms = [
                p.grad.data.norm(2).item() ** 2
                for p in self.forecaster.meta.parameters() if p.grad is not None
            ]
            grad_norms["meta"] = float(sum(meta_norms) ** 0.5) if meta_norms else 0.0
            gnorm_str = " ".join(f"{k}={v:.2e}" for k, v in grad_norms.items())

            logger.info(
                f"[Joint] Round {rnd:3d}/{n_rounds} | "
                f"avg_train={avg_train:.6f} | avg_val={avg_val:.6f} | "
                f"avg_IC={avg_ic:.4f} | {elapsed:.1f}s | grad_norms: {gnorm_str}"
            )

            # Early stopping check
            if use_ic_stopping:
                improved = avg_ic > best_ic + self.tcfg.ic_min_delta
                if improved or (best_ic <= 0 and avg_val < best_val - self.tcfg.min_delta):
                    if improved:
                        best_ic = avg_ic
                    best_val = avg_val
                    patience_ctr = 0
                    self.forecaster.save(best_path)
                    logger.info(f"  ★ New best joint IC={avg_ic:.4f} | val={avg_val:.6f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.tcfg.patience:
                        logger.info(f"  Early stopping at round {rnd} (best IC={best_ic:.4f})")
                        break
            else:
                if avg_val < best_val - self.tcfg.min_delta:
                    best_val = avg_val
                    patience_ctr = 0
                    self.forecaster.save(best_path)
                    logger.info(f"  ★ New best joint val={avg_val:.6f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.tcfg.patience:
                        logger.info(f"  Early stopping at round {rnd}")
                        break

        # Reload best weights
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            for name in self.forecaster.sub_model_names:
                if name in best_ckpt:
                    self.forecaster.sub_models[name].load_state_dict(best_ckpt[name])
            if "meta" in best_ckpt:
                self.forecaster.meta.load_state_dict(best_ckpt["meta"])
        _save_training_curve(history, "Joint", save_dir)
        return history

    @torch.no_grad()
    def _update_cached_outputs(
        self,
        sub_model: nn.Module,
        sub_name: str,
        data_loader: DataLoader,
        cached: Dict[str, torch.Tensor],
        key_to_idx: Dict[Tuple[str, int], int],
    ):
        """Re-run a sub-model after fine-tuning and update cached preds/embs.

        This way the next sub-model in the round sees the updated outputs.
        """
        sub_model.to(self.device).eval()
        for batch in data_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device, non_blocking=True)
            out = sub_model(x_dev)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                idx = key_to_idx.get(key, -1)
                if idx >= 0:
                    cached[f"{sub_name}_pred"][idx] = out["prediction"][i].cpu()
                    cached[f"{sub_name}_emb"][idx] = out["embedding"][i].cpu()
        sub_model.cpu()
        clear_gpu_memory()


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
    ticker_rows: Dict[str, List[Tuple[int, int]]] = {}
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

    # ------------------------------------------------------------------
    # Create dataloaders (lazy, near-zero RAM)
    # ------------------------------------------------------------------
    needs_data = bool(set(phases) & {1, 2, 3, 4})
    news_loaders = None
    graph_loaders: Optional[Dict] = None
    if needs_data:
        logger.info("\nCreating lazy dataloaders...")
        loaders = create_dataloaders(
            splits, data_cfg,
            batch_size_daily=tcfg.batch_size_daily,
            batch_size_minute=tcfg.batch_size_minute,
            num_workers=tcfg.num_workers,
        )

        # Always create news dataloaders (used whenever use_news_model=True)
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
            batch_size=tcfg.batch_size_news,
            num_workers=tcfg.num_workers,
        )
        logger.info(f"  News loaders: {news_loaders['n_features']} features")

    # ------------------------------------------------------------------
    # Create or resume model
    # ------------------------------------------------------------------
    _loaded_sub_models: set = set()  # tracks which sub-models have checkpoint weights
    _skip_set: set = set(skip_models) if skip_models else set()

    if needs_data:
        if resume_path and os.path.exists(resume_path):
            logger.info(f"\nResuming from {resume_path}")
            # Pass model_cfg as override so new sub-models (TCN_D, FundMLP, GNN)
            # are created even if the checkpoint didn't have them.
            # Dimension fields (daily_input_dim etc.) are copied from checkpoint.
            model_cfg.daily_input_dim = loaders["daily_n_features"]
            model_cfg.minute_input_dim = loaders["minute_n_features"]
            if model_cfg.use_gnn:
                model_cfg.gnn_input_dim = loaders["daily_n_features"]
            forecaster = HierarchicalForecaster.load(
                resume_path, device=str(device), override_cfg=model_cfg,
            )
            _loaded_sub_models = getattr(forecaster, "loaded_sub_models", set())
            # Augment the skip set with all loaded models if --skip-trained
            if skip_trained:
                _skip_set |= _loaded_sub_models
            if _skip_set:
                logger.info(
                    f"\n⚡ Will skip Phase 1/2 training for: {sorted(_skip_set)}"
                )
        else:
            model_cfg.daily_input_dim = loaders["daily_n_features"]
            model_cfg.minute_input_dim = loaders["minute_n_features"]
            if model_cfg.use_news_model and news_loaders is not None:
                model_cfg.news_input_dim = news_loaders["n_features"]
                model_cfg.news_seq_len = data_cfg.daily_seq_len
            # n_sub_models is now a dynamic property computed from use_* flags
            logger.info(f"\nCreating HierarchicalForecaster:")
            logger.info(f"  Daily:  {model_cfg.daily_input_dim} features, seq_len={model_cfg.daily_seq_len}")
            logger.info(f"  Minute: {model_cfg.minute_input_dim} features, seq_len={model_cfg.minute_seq_len}")
            if model_cfg.use_news_model:
                logger.info(f"  News:   {model_cfg.news_input_dim} features, seq_len={model_cfg.news_seq_len}")
            if model_cfg.use_tcn_d:
                logger.info(f"  TCN_D:  {model_cfg.tcn_d_n_filters} filters, {model_cfg.tcn_d_n_layers} layers")
            if model_cfg.use_fund_mlp:
                logger.info(f"  FundMLP: input_dim={model_cfg.fund_input_dim}, hidden={model_cfg.fund_hidden_dim}")
            if model_cfg.use_gnn_features:
                logger.info(f"  GNN Features: {model_cfg.gnn_feature_dim}-dim auxiliary embeddings")
            logger.info(f"  Total sub-models: {model_cfg.n_sub_models}")
            forecaster = HierarchicalForecaster(model_cfg)

        forecaster = forecaster.to(device)

        # Reload existing _best.pt weights for skipped sub-models so they
        # don't stay random-initialised when the checkpoint had different dims.
        if _skip_set:
            _name_map = {
                "lstm_d": "LSTM_D", "tft_d": "TFT_D", "tcn_d": "TCN_D",
                "lstm_m": "LSTM_M", "tft_m": "TFT_M",
                "fund_mlp": "FundMLP", "gnn": "GNN", "news": "News",
            }

            def _load_best_pt(path, module):
                """Load a best.pt file into module, handling _orig_mod. prefix."""
                sd = torch.load(path, map_location=device, weights_only=True)
                if any(k.startswith("_orig_mod.") for k in sd):
                    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
                module.load_state_dict(sd)

            # Reload meta separately (saved as raw state dict, not in sub_models)
            if "meta" in _skip_set:
                meta_path = os.path.join(tcfg.output_dir, "meta_best.pt")
                if os.path.exists(meta_path):
                    try:
                        _load_best_pt(meta_path, forecaster.meta)
                        logger.info(f"  ✅ Reloaded Meta weights from {meta_path}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Could not reload Meta from {meta_path}: {e}")

            for sm_name in sorted(_skip_set - {"meta"}):
                pretty = _name_map.get(sm_name, sm_name)
                best_path = os.path.join(tcfg.output_dir, f"{pretty}_best.pt")
                if os.path.exists(best_path):
                    try:
                        _load_best_pt(best_path, forecaster.sub_models[sm_name])
                        logger.info(f"  ✅ Reloaded {pretty} weights from {best_path}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Could not reload {pretty} from {best_path}: {e}")
                else:
                    logger.info(f"  ℹ No {best_path} found for skipped {pretty}")

    # ------------------------------------------------------------------
    # PHASE 1: Train daily models
    # ------------------------------------------------------------------
    if 1 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: Training daily models (LSTM_D + TFT_D)")
        logger.info("=" * 70)

        train_dl_base = loaders["daily"]["train"]
        val_dl_base = loaders["daily"]["val"]

        # Regime curriculum state for Phase 1 (daily models share the same data)
        # After LSTM_D trains we evaluate per-regime IC and use it to weight TFT_D,
        # then after TFT_D we weight TCN_D (if enabled).
        _daily_curriculum_loader: Optional[DataLoader] = None

        # LSTM_D: can use larger batch (small model)
        if "lstm_d" in _skip_set:
            logger.info("  ⏭ LSTM_D — skipped")
        else:
            bs_lstm_d = tcfg.batch_size_lstm_d or tcfg.batch_size_daily
            train_dl = rebatch_loader(train_dl_base, bs_lstm_d) if bs_lstm_d != tcfg.batch_size_daily else train_dl_base
            val_dl = rebatch_loader(val_dl_base, bs_lstm_d) if bs_lstm_d != tcfg.batch_size_daily else val_dl_base
            logger.info(f"  LSTM_D batch_size={bs_lstm_d}")
            trainer = SubModelTrainer(
                forecaster.sub_models["lstm_d"], "LSTM_D", device, tcfg,
                lr_override=tcfg.lr_lstm,
                weight_decay_override=tcfg.weight_decay_lstm,
                huber_delta=tcfg.huber_delta_lstm_d,
            )
            trainer.train(train_dl, val_dl, tcfg.epochs_phase1, tcfg.output_dir,
                          curriculum_loader=_daily_curriculum_loader)

            # Regime curriculum: evaluate LSTM_D's per-regime IC → build weights for TFT_D
            if _regime_clusterer is not None:
                try:
                    pr_ic = trainer.evaluate_per_regime(val_dl, _regime_clusterer)
                    rw = compute_curriculum_weights(pr_ic)
                    log_regime_stats("LSTM_D", pr_ic, rw)
                    # Save per-regime IC to disk for Critic
                    _save_per_model_regime_ic("lstm_d", pr_ic, tcfg.output_dir)
                    # Build weighted DataLoader for TFT_D
                    _ds_train = train_dl_base.dataset
                    _ordinal_dates_daily = [int(train_dl_base.dataset.index[i][1]) for i in range(len(_ds_train))]
                    _ordinal_dates_daily = _get_dataset_ordinal_dates(_ds_train)
                    bs_next = tcfg.batch_size_tft_d or tcfg.batch_size_daily
                    _daily_curriculum_loader = make_regime_weighted_loader(
                        _ds_train, _ordinal_dates_daily, _regime_clusterer, rw,
                        batch_size=bs_next,
                        num_workers=tcfg.num_workers,
                    )
                    logger.info("  LSTM_D → regime curriculum loader built for TFT_D")
                except Exception as _ce:
                    logger.warning("  Regime curriculum update failed (LSTM_D): %s", _ce)
                    _daily_curriculum_loader = None

        forecaster.sub_models["lstm_d"].cpu()  # offload to free VRAM for TFT_D
        clear_gpu_memory()

        # TFT_D: large model with attention — OOM-safe batch size probing
        if "tft_d" in _skip_set:
            logger.info("  ⏭ TFT_D — skipped")
        else:
            bs_tft_d = tcfg.batch_size_tft_d or tcfg.batch_size_daily
            tft_trained = False
            while bs_tft_d >= 16 and not tft_trained:
                train_dl = rebatch_loader(train_dl_base, bs_tft_d)
                val_dl = rebatch_loader(val_dl_base, bs_tft_d)
                logger.info(f"  TFT_D batch_size={bs_tft_d}")
                trainer = SubModelTrainer(
                    forecaster.sub_models["tft_d"], "TFT_D", device, tcfg,
                    lr_override=tcfg.lr_tft,
                    weight_decay_override=tcfg.weight_decay_tft,
                    grad_clip_override=tcfg.grad_clip_tft,
                    warmup_steps=tcfg.warmup_steps,
                    huber_delta=tcfg.huber_delta_tft_d,
                )
                try:
                    trainer.train(train_dl, val_dl, tcfg.epochs_phase1, tcfg.output_dir,
                                  curriculum_loader=_daily_curriculum_loader)
                    tft_trained = True
                except torch.cuda.OutOfMemoryError:
                    old_bs = bs_tft_d
                    bs_tft_d = bs_tft_d // 2
                    logger.warning(
                        f"  ⚠ TFT_D OOM at batch_size={old_bs} — "
                        f"halving to {bs_tft_d} and retrying"
                    )
                    clear_gpu_memory()
                    # Re-initialize the model weights (OOM may have corrupted grads)
                    forecaster.sub_models["tft_d"].cpu()
                    forecaster.sub_models["tft_d"].to(device)
                    _daily_curriculum_loader = None  # reset for safety on OOM retry

            # Regime curriculum: evaluate TFT_D's per-regime IC → build weights for TCN_D
            if tft_trained and _regime_clusterer is not None:
                try:
                    val_dl_tft = rebatch_loader(val_dl_base, bs_tft_d)
                    pr_ic = trainer.evaluate_per_regime(val_dl_tft, _regime_clusterer)
                    rw = compute_curriculum_weights(pr_ic)
                    log_regime_stats("TFT_D", pr_ic, rw)
                    _save_per_model_regime_ic("tft_d", pr_ic, tcfg.output_dir)
                    _ds_train = train_dl_base.dataset
                    _ordinal_dates_daily = _get_dataset_ordinal_dates(_ds_train)
                    _daily_curriculum_loader = make_regime_weighted_loader(
                        _ds_train, _ordinal_dates_daily, _regime_clusterer, rw,
                        batch_size=tcfg.batch_size_daily,
                        num_workers=tcfg.num_workers,
                    )
                    logger.info("  TFT_D → regime curriculum loader built for TCN_D")
                except Exception as _ce:
                    logger.warning("  Regime curriculum update failed (TFT_D): %s", _ce)
                    _daily_curriculum_loader = None

            if not tft_trained:
                logger.error("  ❌ TFT_D: could not fit even batch_size=16 — skipping")
        forecaster.sub_models["tft_d"].cpu()  # offload before checkpoint save

        # TCN_D (if enabled): same daily data, dilated causal conv
        if model_cfg.use_tcn_d and "tcn_d" in forecaster.sub_models:
            if "tcn_d" in _skip_set:
                logger.info("  ⏭ TCN_D — skipped")
            else:
                bs_tcn_d = tcfg.batch_size_daily  # TCN is lightweight, use base batch size
                train_dl = rebatch_loader(train_dl_base, bs_tcn_d) if bs_tcn_d != tcfg.batch_size_daily else train_dl_base
                val_dl = rebatch_loader(val_dl_base, bs_tcn_d) if bs_tcn_d != tcfg.batch_size_daily else val_dl_base
                logger.info(f"  TCN_D batch_size={bs_tcn_d}")
                trainer = SubModelTrainer(forecaster.sub_models["tcn_d"], "TCN_D", device, tcfg)
                trainer.train(train_dl, val_dl, tcfg.epochs_phase1, tcfg.output_dir,
                              curriculum_loader=_daily_curriculum_loader)
                if _regime_clusterer is not None:
                    try:
                        pr_ic = trainer.evaluate_per_regime(val_dl, _regime_clusterer)
                        rw = compute_curriculum_weights(pr_ic)
                        log_regime_stats("TCN_D", pr_ic, rw)
                        _save_per_model_regime_ic("tcn_d", pr_ic, tcfg.output_dir)
                    except Exception as _ce:
                        logger.warning("  Regime curriculum eval failed (TCN_D): %s", _ce)
                forecaster.sub_models["tcn_d"].cpu()
                clear_gpu_memory()

        forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase1.pt"))
        logger.info("Phase 1 complete ✓")
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # PHASE 1.5: Train news encoder (if enabled)
    # ------------------------------------------------------------------
    if 1 in phases and model_cfg.use_news_model and news_loaders is not None:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1.5: Training News Encoder")
        logger.info("=" * 70)

        train_dl = news_loaders["train"]
        val_dl = news_loaders["val"]

        if "news" in _skip_set:
            logger.info("  ⏭ News — skipped")
        elif len(train_dl.dataset) > 0:
            trainer = SubModelTrainer(forecaster.sub_models["news"], "News", device, tcfg)
            trainer.train(train_dl, val_dl, tcfg.epochs_news, tcfg.output_dir)
            if _regime_clusterer is not None:
                try:
                    pr_ic = trainer.evaluate_per_regime(val_dl, _regime_clusterer)
                    rw = compute_curriculum_weights(pr_ic)
                    log_regime_stats("News", pr_ic, rw)
                    _save_per_model_regime_ic("news", pr_ic, tcfg.output_dir)
                except Exception as _ce:
                    logger.warning("  Regime curriculum eval failed (News): %s", _ce)
            forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase1_5.pt"))
            logger.info("Phase 1.5 complete ✓")
        else:
            logger.warning("No news training data available — skipping Phase 1.5")
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # PHASE 1.6: Train FundamentalMLP (if enabled)
    # ------------------------------------------------------------------
    fund_loaders: Optional[Dict] = None
    if model_cfg.use_fund_mlp and "fund_mlp" in forecaster.sub_models:
        # Always create loaders — needed for Phase 3 meta training even when
        # Phase 1 is skipped via --phase 3 4.
        try:
            from src.hierarchical_data import create_fundamental_dataloaders
            fund_loaders = create_fundamental_dataloaders(
                splits, data_cfg,
                batch_size=tcfg.batch_size_daily,
                num_workers=tcfg.num_workers,
            )
        except ImportError:
            pass

        if 1 in phases:
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 1.6: Training Fundamental MLP")
            logger.info("=" * 70)
            if fund_loaders is None:
                logger.warning("FundamentalMLP data pipeline not yet implemented — skipping Phase 1.6")
            else:
                train_dl = fund_loaders["train"]
                val_dl = fund_loaders["val"]
                if "fund_mlp" in _skip_set:
                    logger.info("  ⏭ FundMLP — skipped")
                elif len(train_dl.dataset) > 0:
                    trainer = SubModelTrainer(forecaster.sub_models["fund_mlp"], "FundMLP", device, tcfg)
                    trainer.train(train_dl, val_dl, tcfg.epochs_phase1, tcfg.output_dir)
                    forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase1_6.pt"))
                    logger.info("Phase 1.6 complete ✓")
                else:
                    logger.warning("No fundamental training data — skipping Phase 1.6")
            clear_gpu_memory()



    # ------------------------------------------------------------------
    # PHASE 1.7: GNN is no longer trained as a sub-model.
    # GNN embeddings should be precomputed using scripts/export_gnn_embeddings.py
    # and injected as auxiliary features into daily/minute models.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PHASE 2: Train minute models
    # ------------------------------------------------------------------
    if 2 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: Training minute models (LSTM_M + TFT_M)")
        logger.info("=" * 70)

        train_dl_base = loaders["minute"]["train"]
        val_dl_base = loaders["minute"]["val"]

        # Regime curriculum state for Phase 2 (minute models)
        _minute_curriculum_loader: Optional[DataLoader] = None

        # LSTM_M: small model, can use larger batch
        if "lstm_m" in _skip_set:
            logger.info("  ⏭ LSTM_M — skipped")
        else:
            bs_lstm_m = tcfg.batch_size_lstm_m or tcfg.batch_size_minute
            train_dl = rebatch_loader(train_dl_base, bs_lstm_m) if bs_lstm_m != tcfg.batch_size_minute else train_dl_base
            val_dl = rebatch_loader(val_dl_base, bs_lstm_m) if bs_lstm_m != tcfg.batch_size_minute else val_dl_base
            logger.info(f"  LSTM_M batch_size={bs_lstm_m}")
            trainer = SubModelTrainer(
                forecaster.sub_models["lstm_m"], "LSTM_M", device, tcfg,
                lr_override=tcfg.lr_minute,
                weight_decay_override=tcfg.weight_decay_minute,
            )
            trainer.train(train_dl, val_dl, tcfg.epochs_phase2, tcfg.output_dir,
                          curriculum_loader=_minute_curriculum_loader)

            # Regime curriculum: evaluate LSTM_M → weights for TFT_M
            if _regime_clusterer is not None:
                try:
                    pr_ic = trainer.evaluate_per_regime(val_dl, _regime_clusterer)
                    rw = compute_curriculum_weights(pr_ic)
                    log_regime_stats("LSTM_M", pr_ic, rw)
                    _save_per_model_regime_ic("lstm_m", pr_ic, tcfg.output_dir)
                    bs_next = tcfg.batch_size_tft_m or tcfg.batch_size_minute
                    _minute_curriculum_loader = make_regime_weighted_loader(
                        train_dl_base.dataset,
                        _get_dataset_ordinal_dates(train_dl_base.dataset),
                        _regime_clusterer, rw,
                        batch_size=bs_next,
                        num_workers=tcfg.num_workers,
                    )
                    logger.info("  LSTM_M → regime curriculum loader built for TFT_M")
                except Exception as _ce:
                    logger.warning("  Regime curriculum update failed (LSTM_M): %s", _ce)
                    _minute_curriculum_loader = None

        forecaster.sub_models["lstm_m"].cpu()  # offload to free VRAM for TFT_M
        clear_gpu_memory()

        # TFT_M uses the lower of lr_tft and lr_minute
        if "tft_m" in _skip_set:
            logger.info("  ⏭ TFT_M — skipped")
        else:
            bs_tft_m = tcfg.batch_size_tft_m or tcfg.batch_size_minute
            train_dl = rebatch_loader(train_dl_base, bs_tft_m) if bs_tft_m != tcfg.batch_size_minute else train_dl_base
            val_dl = rebatch_loader(val_dl_base, bs_tft_m) if bs_tft_m != tcfg.batch_size_minute else val_dl_base
            logger.info(f"  TFT_M batch_size={bs_tft_m}")
            _tft_m_lr = min(tcfg.lr_tft, tcfg.lr_minute) if (tcfg.lr_tft > 0 and tcfg.lr_minute > 0) else (tcfg.lr_tft or tcfg.lr_minute)
            trainer = SubModelTrainer(
                forecaster.sub_models["tft_m"], "TFT_M", device, tcfg,
                lr_override=_tft_m_lr,
                weight_decay_override=tcfg.weight_decay_minute,
                grad_clip_override=tcfg.grad_clip_tft,
                warmup_steps=tcfg.warmup_steps,
            )
            trainer.train(train_dl, val_dl, tcfg.epochs_phase2, tcfg.output_dir,
                          curriculum_loader=_minute_curriculum_loader)
            if _regime_clusterer is not None:
                try:
                    pr_ic = trainer.evaluate_per_regime(val_dl, _regime_clusterer)
                    rw = compute_curriculum_weights(pr_ic)
                    log_regime_stats("TFT_M", pr_ic, rw)
                    _save_per_model_regime_ic("tft_m", pr_ic, tcfg.output_dir)
                except Exception as _ce:
                    logger.warning("  Regime curriculum eval failed (TFT_M): %s", _ce)

        forecaster.sub_models["tft_m"].cpu()  # offload before checkpoint save

        forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase2.pt"))
        logger.info("Phase 2 complete ✓")
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # PHASE 3: Train meta model
    # ------------------------------------------------------------------
    if 3 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: Training Meta MLP (sub-models frozen)")
        logger.info("=" * 70)
        logger.info("  Meta trains on VAL split (unseen by sub-models), validates on TEST split")

        # Load critic sample weights if available (from previous pipeline run)
        critic_weights = _load_critic_sample_weights(data_cfg)

        meta_trainer = MetaTrainer(forecaster.meta, device, tcfg)
        meta_trainer.train(
            forecaster,
            loaders["daily"]["val"], loaders["minute"]["val"],      # meta TRAIN = sub-model val
            loaders["daily"]["test"], loaders["minute"]["test"],    # meta VAL   = sub-model test
            tcfg.epochs_phase3, tcfg.output_dir,
            data_cfg=data_cfg,
            n_train_news_dl=news_loaders["val"] if news_loaders else None,
            n_val_news_dl=news_loaders["test"] if news_loaders else None,
            f_train_dl=fund_loaders["val"] if fund_loaders else None,
            f_val_dl=fund_loaders["test"] if fund_loaders else None,
            g_train_dl=graph_loaders["val"] if graph_loaders else None,
            g_val_dl=graph_loaders["test"] if graph_loaders else None,
            sample_weights=critic_weights,
        )

        forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase3.pt"))
        logger.info("Phase 3 complete ✓")
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # PHASE 4: Joint fine-tuning (all 5 models, small LR)
    # ------------------------------------------------------------------
    if 4 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: Joint fine-tuning (all models unfrozen)")
        logger.info("=" * 70)

        joint_trainer = JointFineTuner(forecaster, device, tcfg, data_cfg)
        joint_trainer.train(
            loaders["daily"]["train"], loaders["minute"]["train"],
            loaders["daily"]["val"], loaders["minute"]["val"],
            tcfg.epochs_phase4, tcfg.output_dir,
            n_train_news_dl=news_loaders["train"] if news_loaders else None,
            n_val_news_dl=news_loaders["val"] if news_loaders else None,
            fund_train_dl=fund_loaders["train"] if fund_loaders else None,
            fund_val_dl=fund_loaders["val"] if fund_loaders else None,
            graph_train_dl=graph_loaders["train"] if graph_loaders else None,
            graph_val_dl=graph_loaders["val"] if graph_loaders else None,
        )

        forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase4.pt"))
        logger.info("Phase 4 complete ✓")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    if needs_data:
        logger.info("\n" + "=" * 70)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 70)

        results = {}
        forecaster.eval()

        # Data-source → test loader mapping
        test_loaders_map: Dict[str, Optional[DataLoader]] = {
            "daily": loaders["daily"]["test"],
            "minute": loaders["minute"]["test"],
            "news": news_loaders["test"] if news_loaders else None,
            "fundamental": fund_loaders["test"] if fund_loaders else None,
            "graph": graph_loaders["test"] if graph_loaders else None,
        }

        # Name map for per-model best checkpoint files (written by SubModelTrainer)
        _best_ckpt_name_map = {
            "lstm_d": "LSTM_D", "tft_d": "TFT_D", "tcn_d": "TCN_D",
            "lstm_m": "LSTM_M", "tft_m": "TFT_M",
            "fund_mlp": "FundMLP", "gnn": "GNN", "news": "News",
        }

        def _try_reload_best(sub_name: str, module: nn.Module) -> bool:
            """Load {Name}_best.pt into module if it exists. Returns True on success."""
            pretty = _best_ckpt_name_map.get(sub_name, sub_name)
            best_pt = os.path.join(tcfg.output_dir, f"{pretty}_best.pt")
            if not os.path.exists(best_pt):
                return False
            try:
                sd = torch.load(best_pt, map_location=device, weights_only=True)
                if any(k.startswith("_orig_mod.") for k in sd):
                    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
                module.load_state_dict(sd)
                logger.info(f"  {sub_name:8s} | reloaded best weights from {best_pt}")
                return True
            except Exception as _e:
                logger.warning(f"  {sub_name}: could not reload best weights — {_e}")
                return False

        # Evaluate each sub-model on its data source's test set
        for name in forecaster.sub_model_names:
            ds = HierarchicalForecaster.MODALITY[name]
            test_dl = test_loaders_map.get(ds)
            if test_dl is None or len(test_dl.dataset) == 0:
                logger.info(f"  {name:8s} | SKIPPED (no test data for source '{ds}')")
                continue

            model = forecaster.sub_models[name]
            # Reload the best per-model checkpoint before evaluating so that
            # phase-4 joint fine-tuning degradation doesn't corrupt test IC.
            # Falls back silently to the current in-memory weights if the file
            # doesn't exist (e.g. model was skipped during training).
            _try_reload_best(name, model)
            model.to(device)
            preds, tgts = [], []
            with torch.no_grad():
                if ds == "graph":
                    # GNN evaluation: cross-sectional batches
                    for batch in test_dl:
                        if isinstance(batch, dict):
                            nf = batch["node_features"].squeeze(0).to(device)
                            tgt = batch["targets"].squeeze(0)
                            mask = batch["mask"].squeeze(0)
                            ei = batch["edge_index"].squeeze(0).to(device)
                        else:
                            b = batch[0]
                            nf = b["node_features"].to(device)
                            tgt = b["targets"]
                            mask = b["mask"]
                            ei = b["edge_index"].to(device)
                        mask_dev = mask.to(device)
                        out = model(nf, ei, mask_dev)
                        preds.append(out["prediction"].cpu().numpy())
                        tgts.append(tgt[mask].numpy())
                else:
                    for batch in test_dl:
                        x, y = batch[0], batch[1]
                        x = x.to(device, non_blocking=True)
                        out = model(x)
                        preds.append(out["prediction"].cpu().numpy())
                        tgts.append(y.numpy())
            model.cpu()
            metrics = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
            results[name.upper()] = metrics
            logger.info(f"  {name.upper():8s} | IC={metrics['ic']:.4f} | "
                        f"RankIC={metrics['rank_ic']:.4f} | DirAcc={metrics['directional_accuracy']:.3f}")
            clear_gpu_memory()

        # Evaluate full ensemble (meta model) on test
        # Note: the meta model was trained on val split and validated on test
        # split, so test is NOT fully held-out for the meta model (it was
        # used for early stopping). For a true held-out evaluation, reserve
        # a separate test set or use walk-forward validation.
        logger.info("\n  --- Full Ensemble (Meta) ---")
        meta_trainer = MetaTrainer(forecaster.meta, device, tcfg)
        test_preds, test_embs, test_regime, test_targets = \
            meta_trainer._collect_sub_outputs(
                forecaster,
                loaders["daily"]["test"], loaders["minute"]["test"],
                data_cfg,
                news_loader=news_loaders["test"] if news_loaders else None,
                fund_loader=fund_loaders["test"] if fund_loaders else None,
            )
        if len(test_targets) > 0:
            E = forecaster.cfg.embedding_dim
            n_sub = forecaster.meta.n_sub_models
            all_meta_preds, all_meta_tgts = [], []
            test_ds = TensorDataset(test_preds, test_embs, test_regime, test_targets)
            test_meta_dl = DataLoader(test_ds, batch_size=tcfg.batch_size_meta)
            forecaster.meta.eval()
            with torch.no_grad():
                for preds_b, embs_b, regime_b, targets_b in test_meta_dl:
                    preds_b = preds_b.to(device, non_blocking=True)
                    embs_b = embs_b.to(device, non_blocking=True)
                    regime_b = regime_b.to(device, non_blocking=True)
                    emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(n_sub)]
                    out = forecaster.meta(preds_b, emb_list, regime_b)
                    all_meta_preds.append(out["prediction"].cpu().numpy())
                    all_meta_tgts.append(targets_b.numpy())
            meta_metrics = compute_metrics(
                np.concatenate(all_meta_preds), np.concatenate(all_meta_tgts)
            )
            results["META_ENSEMBLE"] = meta_metrics
            logger.info(f"  {'META':8s} | IC={meta_metrics['ic']:.4f} | "
                        f"RankIC={meta_metrics['rank_ic']:.4f} | "
                        f"DirAcc={meta_metrics['directional_accuracy']:.3f}")

        with open(os.path.join(tcfg.output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        final_path = os.path.join(tcfg.output_dir, "forecaster_final.pt")
        forecaster.save(final_path)
        logger.info(f"\nFinal model → {final_path}")

    # ------------------------------------------------------------------
    # Auto-generate diagnostic plots after a full run
    # ------------------------------------------------------------------
    if set(phases) >= {1, 2, 3}:  # only when all training phases ran
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from scripts.plot_results import run as _plot_run

            _data_cfg_overrides = {
                "split_mode":     data_cfg.split_mode,
                "daily_seq_len":  data_cfg.daily_seq_len,
                "minute_seq_len": data_cfg.minute_seq_len,
            }
            logger.info("\nGenerating diagnostic plots...")
            _plot_run(
                model_dir=tcfg.output_dir,
                top_k=20,
                horizons=[1, 3, 5, 10, 15],
                skip_predictions=False,
                data_cfg_overrides=_data_cfg_overrides,
            )
            logger.info(f"Plots saved → {tcfg.output_dir}/plots/")
        except Exception as _e:
            logger.warning(f"Plot generation failed (non-fatal): {_e}")

    return


# ============================================================================
# CLI
# ============================================================================

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

    # ------------------------------------------------------------------
    # V10 preset: fixes TFT gradient collapse + minute overfitting
    # ------------------------------------------------------------------
    if args.v10:
        logger.info("🔧 V10 preset enabled — applying per-model tuning")
        # TFT: much lower LR to prevent gradient collapse on long sequences
        tcfg.lr_tft = 5e-5
        # TFT: linear warmup before cosine decay (2K optimizer steps)
        tcfg.warmup_steps = 2000
        # TFT: tighter gradient clipping
        tcfg.grad_clip_tft = 0.5
        # Minute models: moderate LR
        tcfg.lr_minute = 1e-4
        # Minute models: stronger L2 regularization (data scarcity)
        tcfg.weight_decay_minute = 1e-3
        # Minute models: more patience (small dataset = noisy validation)
        tcfg.patience = 20
        # Per-model batch sizes (maximize GPU utilization)
        # With AMP (FP16), activation memory is ~halved → double batch sizes
        # LSTM_D: 260K params, tiny footprint → push batch hard
        tcfg.batch_size_lstm_d = 512
        # TFT_D: 2.6M params (grouped VSN) + FlashAttention
        # Compiled: 10.8 GB at bs=192, 1732 samples/s peak
        tcfg.batch_size_tft_d = 192
        # LSTM_M (v10): 32K params, tiny model → large batch
        tcfg.batch_size_lstm_m = 512
        # TFT_M (v10): 412K params, smaller than TFT_D → fits more
        tcfg.batch_size_tft_m = 256

    # Log effective batch sizes and AMP status for verification
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

    # V10: smaller minute models to reduce overfitting on scarce data
    if args.v10:
        model_cfg.minute_hidden_dim = 64
        model_cfg.minute_n_layers = 1
        model_cfg.minute_dropout = 0.3
        logger.info(f"  Minute model: hidden={model_cfg.minute_hidden_dim}, "
                     f"layers={model_cfg.minute_n_layers}, dropout={model_cfg.minute_dropout}")

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
