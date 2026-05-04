"""Trainer classes for hierarchical forecasting models.

Contains:
  - SubModelTrainer: trains single sub-models (LSTM, TFT, TCN, News, FundMLP)
  - GNNSubModelTrainer: trains cross-sectional graph neural networks
  - MetaTrainer: trains meta-MLP on frozen sub-model outputs
  - JointFineTuner: fine-tunes all models sequentially through meta loss
"""

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from src.hierarchical_config import TrainConfig
from src.hierarchical_data import (
    HierarchicalDataConfig,
    _build_regime_dataframe,
)
from src.hierarchical_metrics import compute_metrics
from src.hierarchical_models import HierarchicalForecaster, MetaMLP

logger = logging.getLogger(__name__)


def pairwise_rank_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable pairwise ranking loss (approx. Spearman via listwise margin).

    For each pair (i, j) where target_i > target_j, penalise if pred_i <= pred_j.
    Uses a soft margin so gradients flow smoothly.

    Args:
        preds:   (N,) predicted scores (must be 1-D)
        targets: (N,) ground-truth returns (must be 1-D)

    Returns:
        scalar loss (mean over all valid pairs)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    n = preds.shape[0]
    if n < 2:
        return preds.sum() * 0.0  # differentiable zero

    # Sample at most 512 pairs to keep memory O(512²) not O(N²)
    max_n = min(n, 512)
    if n > max_n:
        idx = torch.randperm(n, device=preds.device)[:max_n]
        preds = preds[idx]
        targets = targets[idx]
        n = max_n

    # (n, n) pairwise differences
    tgt_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # tgt_i - tgt_j
    pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)     # pred_i - pred_j

    # Only penalise pairs where |target diff| is meaningful (avoid noisy ties)
    sig = targets.std().clamp(min=1e-6)
    valid = tgt_diff > 0.1 * sig   # target_i > target_j by at least 10% of std

    if valid.sum() == 0:
        return preds.sum() * 0.0

    # Soft margin: log(1 + exp(-pred_diff)) where tgt_i > tgt_j
    margin_loss = torch.log1p(torch.exp(-pred_diff[valid]))
    return margin_loss.mean()


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


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
        # Prefer an explicit per-model weight decay when available (e.g. LSTM/TFT)
        if weight_decay_override > 0:
            self.wd = weight_decay_override
        else:
            # If this trainer is for an LSTM, prefer TrainConfig.weight_decay_lstm
            name_l = name.lower() if name is not None else ""
            if name_l.startswith("lstm") and hasattr(tcfg, "weight_decay_lstm"):
                self.wd = tcfg.weight_decay_lstm
            else:
                self.wd = tcfg.weight_decay
        self.grad_clip = grad_clip_override if grad_clip_override > 0 else tcfg.grad_clip
        self.warmup_steps = warmup_steps

        # AdamW with explicit weight decay (L2) for regularization.
        # Ensures LSTM trainers receive `weight_decay_lstm` by default.
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
        self.scaler = GradScaler(enabled=self.use_amp, init_scale=256, growth_factor=1.5, growth_interval=100)
        # Scheduler (may be set by _make_scheduler)
        self.scheduler = None

        logger.info(
            f"  [{name}] lr={self.lr:.2e}  wd={self.wd:.1e}  "
            f"grad_clip={self.grad_clip}  warmup_steps={self.warmup_steps}  "
            f"amp={self.use_amp}"
        )

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
            # Optionally support News dataset which returns a fifth element
            # ``has_news_frac`` — pass it to the model as `news_coverage`.
            news_cov = None
            if len(batch) >= 5 and isinstance(batch[4], (torch.Tensor, list, tuple)):
                # Only apply for the news trainer by name
                if self.name.lower().startswith("news"):
                    news_cov = batch[4]

            # non_blocking=True: overlap CPU→GPU transfer with computation
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if news_cov is not None and isinstance(news_cov, torch.Tensor):
                news_cov = news_cov.to(self.device, non_blocking=True)

            # Apply linear warmup before the optimizer step
            self._apply_warmup()

            self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            # AMP: autocast for float16 on Tensor Cores, GradScaler for stability
            with autocast(device_type="cuda", enabled=self.use_amp):
                if news_cov is not None:
                    out = self.model(x, news_coverage=news_cov)
                else:
                    out = self.model(x)
                base_loss = self.criterion(out["prediction"], y)
                rl_weight = getattr(self.tcfg, 'rank_loss_weight', 0.5)
                if rl_weight > 0.0 and out["prediction"].shape[0] >= 2:
                    rl = pairwise_rank_loss(out["prediction"].squeeze(), y)
                    loss = (1.0 - rl_weight) * base_loss + rl_weight * rl
                else:
                    loss = base_loss

            if not torch.isfinite(loss):
                logger.warning(f"  [{self.name}] NaN/Inf loss ({loss.item():.4g}) — skipping batch")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            self.scaler.scale(loss).backward()

            # Save loss before cleanup
            loss_val = loss.item()
            
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                total_grad_norm += float(grad_norm)

            # Skip optimizer step if gradients are non-finite (AMP overflow)
            if self.grad_clip > 0 and not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Clear intermediate activations to reduce memory usage
            del loss, x, y
            if batch_idx % 50 == 0:  # Periodically empty cache
                torch.cuda.empty_cache()
            
            total_loss += loss_val
            # If using a per-step cyclic LR, advance the scheduler here.
            if getattr(self, "scheduler", None) is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.CyclicLR
            ):
                try:
                    self.scheduler.step()
                except Exception:
                    pass
            self._global_step += 1
            n_batches += 1

            if batch_idx > 0 and batch_idx % self.tcfg.log_interval == 0:
                avg_gn = total_grad_norm / n_batches
                logger.info(f"  [{self.name}] batch {batch_idx}/{len(loader)} "
                            f"loss={loss_val:.6f} grad_norm={avg_gn:.4f}")

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
            news_cov = None
            if len(batch) >= 5 and isinstance(batch[4], (torch.Tensor, list, tuple)):
                if self.name.lower().startswith("news"):
                    news_cov = batch[4]

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if news_cov is not None and isinstance(news_cov, torch.Tensor):
                news_cov = news_cov.to(self.device, non_blocking=True)
            with autocast(device_type="cuda", enabled=self.use_amp):
                if news_cov is not None:
                    out = self.model(x, news_coverage=news_cov)
                else:
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
            # batch layout: (seq, z_target, raw_target, ordinal_date, ticker)
            # raw_target is batch[2]; ordinal_dates shifted to batch[3]
            ordinal_dates = batch[3] if len(batch) >= 5 else batch[2]
            news_cov = None
            if len(batch) >= 5 and isinstance(batch[4], (torch.Tensor, list, tuple)):
                if self.name.lower().startswith("news"):
                    news_cov = batch[4]
            if news_cov is not None and isinstance(news_cov, torch.Tensor):
                news_cov = news_cov.to(self.device, non_blocking=True)
            with autocast(device_type="cuda", enabled=self.use_amp):
                if news_cov is not None:
                    out = self.model(x, news_coverage=news_cov)
                else:
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
        # NOTE: torch.compile() causes OOM with large batch sizes; disabled for meta model
        # if torch.cuda.is_available():
        #     try:
        #         self.meta = torch.compile(self.meta)
        #         logger.info(f"  [Meta] torch.compile() enabled")
        #     except Exception as e:
        #         logger.warning(f"  [Meta] torch.compile() failed: {e} — running eager")
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
                    if modality == "news":
                        # News tuple: (seq, target, ordinal_date, ticker, news_density)
                        x, y, ordinal_dates, tickers_batch, news_cov = batch
                        x = x.to(self.device, non_blocking=True)
                        news_cov_dev = news_cov.to(self.device, non_blocking=True)
                        out = sub_model(x, news_coverage=news_cov_dev)
                    else:
                        # Daily/minute/fundamental: (seq, target, raw_target, ordinal_date, ticker)
                        x, y, _, ordinal_dates, tickers_batch = batch
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
            return (torch.zeros(0, len(sub_names)), torch.zeros(0, len(sub_names) * E),
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
# Fusion layer trainer (combines daily + minute meta outputs)
# ============================================================================

class FusionTrainer:
    """Trains FusionMLP to combine daily and minute meta predictions.
    
    Takes frozen outputs from daily_meta and minute_meta, learns to blend them.
    """
    
    def __init__(
        self,
        fusion_mlp,
        daily_forecaster,
        minute_forecaster,
        device: torch.device,
        tcfg: TrainConfig,
    ):
        self.fusion_mlp = fusion_mlp.to(device)
        self.daily_forecaster = daily_forecaster.to(device)
        self.minute_forecaster = minute_forecaster.to(device)
        self.device = device
        self.tcfg = tcfg
        
        # Freeze sub-models and meta models
        for param in self.daily_forecaster.parameters():
            param.requires_grad = False
        for param in self.minute_forecaster.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            fusion_mlp.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()
        self.scaler = GradScaler() if (tcfg.use_amp and device.type == "cuda") else None
    
    def train(
        self,
        daily_loader: DataLoader,
        minute_loader: DataLoader,
        overlap_dates: set,
        data_cfg: HierarchicalDataConfig,
        epochs: int,
        output_dir: str,
    ) -> Dict:
        """Train fusion layer on overlapping dates only.
        
        Args:
            daily_loader: DataLoader for daily data (full coverage)
            minute_loader: DataLoader for minute data (partial coverage)
            overlap_dates: Set of ordinal dates where both daily and minute exist
            data_cfg: Data configuration
            epochs: Number of training epochs
            output_dir: Directory to save results
        
        Returns:
            Training history (dict)
        """
        os.makedirs(output_dir, exist_ok=True)
        best_path = os.path.join(output_dir, "fusion_best.pt")
        
        # Build regime lookup
        regime_df = _build_regime_dataframe(data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        R = 8  # Default regime dimension
        if not regime_df.empty:
            for date_idx, row in regime_df.iterrows():
                # date_idx is typically a datetime.date object
                try:
                    ord_date: int = int(date_idx.toordinal())  # type: ignore
                    regime_lookup[ord_date] = np.array(row.values, dtype=np.float32)
                except (AttributeError, TypeError):
                    # Skip if can't convert to ordinal
                    continue


        
        # Collect predictions only for overlapping dates
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting predictions for fusion training")
        logger.info(f"{'='*60}")
        
        # Daily predictions
        daily_preds_by_key = {}
        self.daily_forecaster.eval()
        with torch.no_grad():
            for batch in daily_loader:
                x, y, _, ordinal_dates, tickers_batch = batch
                x = x.to(self.device)
                out = self.daily_forecaster(x, torch.zeros(x.shape[0], R, device=self.device))
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    daily_preds_by_key[key] = out["prediction"][i].cpu()
        
        # Minute predictions
        minute_preds_by_key = {}
        self.minute_forecaster.eval()
        with torch.no_grad():
            for batch in minute_loader:
                x, y, _, ordinal_dates, tickers_batch = batch
                x = x.to(self.device)
                out = self.minute_forecaster(x, torch.zeros(x.shape[0], R, device=self.device))
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    minute_preds_by_key[key] = out["prediction"][i].cpu()
        
        # Align on overlapping dates
        fusion_preds_list = []
        fusion_targets_list = []
        fusion_regimes_list = []
        
        for batch in daily_loader:
            x, y, _, ordinal_dates, tickers_batch = batch  # _ = raw_target (daily 5-tuple)
            for i in range(len(y)):
                od = int(ordinal_dates[i])
                ticker = tickers_batch[i]
                key = (ticker, od)
                
                # Skip if this date/ticker combo not in minute data
                if od not in overlap_dates or key not in minute_preds_by_key:
                    continue
                
                daily_pred = daily_preds_by_key.get(key)
                minute_pred = minute_preds_by_key.get(key)
                
                if daily_pred is None or minute_pred is None:
                    continue
                
                fusion_preds_list.append((daily_pred, minute_pred))
                fusion_targets_list.append(y[i].cpu())
                
                regime_vec = torch.from_numpy(regime_lookup.get(od, np.zeros(R, dtype=np.float32)))
                fusion_regimes_list.append(regime_vec)
        
        if len(fusion_preds_list) < 10:
            logger.warning(f"Only {len(fusion_preds_list)} overlapping samples for fusion training!")
            return {
                "train_loss": [],
                "val_loss": [],
                "val_ic": [],
                "val_rank_ic": [],
            }
        
        # Stack data
        daily_preds_t = torch.stack([p[0] for p in fusion_preds_list])
        minute_preds_t = torch.stack([p[1] for p in fusion_preds_list])
        targets_t = torch.stack(fusion_targets_list)
        regimes_t = torch.stack(fusion_regimes_list)
        
        # Create dataset
        fusion_dataset = TensorDataset(daily_preds_t, minute_preds_t, regimes_t, targets_t)
        fusion_loader = DataLoader(
            fusion_dataset,
            batch_size=min(self.tcfg.batch_size_daily, 64),
            shuffle=True,
        )
        
        logger.info(f"Fusion training: {len(fusion_dataset)} overlapping samples")
        logger.info(f"Fusion model: {self.fusion_mlp.count_parameters():,} parameters")
        
        # Training loop
        history = []
        best_loss = float('inf')
        patience_ctr = 0
        
        for epoch in range(epochs):
            # Train pass
            self.fusion_mlp.train()
            total_loss = 0.0
            n_batches = 0
            
            for daily_p, minute_p, regime, target in fusion_loader:
                daily_p = daily_p.to(self.device)
                minute_p = minute_p.to(self.device)
                regime = regime.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                    pred = self.fusion_mlp(daily_p, minute_p, regime)
                    loss = self.criterion(pred, target)
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.fusion_mlp.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.fusion_mlp.parameters(), 1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            train_loss = total_loss / max(n_batches, 1)
            
            # Validation pass
            self.fusion_mlp.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for daily_p, minute_p, regime, target in fusion_loader:
                    daily_p = daily_p.to(self.device)
                    minute_p = minute_p.to(self.device)
                    regime = regime.to(self.device)
                    
                    pred = self.fusion_mlp(daily_p, minute_p, regime)
                    loss = self.criterion(pred, target.to(self.device))
                    val_loss += loss.item()
                    
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(target.numpy())
            
            val_loss = val_loss / max(n_batches, 1)
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            
            # Compute IC
            if np.std(val_preds) > 1e-8 and np.std(val_targets) > 1e-8:
                val_ic = float(np.corrcoef(val_preds, val_targets)[0, 1])
                val_ic = 0.0 if np.isnan(val_ic) else val_ic
            else:
                val_ic = 0.0
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ic": val_ic,
            })
            
            logger.info(f"  [Fusion] Epoch {epoch+1:3d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f} | IC={val_ic:.4f}")
            
            # Early stopping
            if val_loss < best_loss - self.tcfg.min_delta:
                best_loss = val_loss
                patience_ctr = 0
                torch.save(self.fusion_mlp.state_dict(), best_path)
                logger.info(f"    ★ New best val={val_loss:.6f}")
            else:
                patience_ctr += 1
                if patience_ctr >= self.tcfg.patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break
        
        self.fusion_mlp.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        
        # Convert history list to dict for saving
        history_dict: Dict[str, List] = {
            "train_loss": [h["train_loss"] for h in history],
            "val_loss": [h["val_loss"] for h in history],
            "val_ic": [h["val_ic"] for h in history],
            "val_rank_ic": [0.0] * len(history),  # Not computed in fusion, placeholder
        }
        _save_training_curve(history_dict, "Fusion", output_dir)
        logger.info(f"Fusion training complete. Best loss: {best_loss:.6f}")
        
        return history_dict

