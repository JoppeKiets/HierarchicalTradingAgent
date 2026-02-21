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

# Memory optimization: reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
)
from src.hierarchical_models import (
    HierarchicalForecaster,
    HierarchicalModelConfig,
    MetaMLP,
)

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# Training configuration
# ============================================================================

class TrainConfig:
    """Training hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 1e-5
    lr_meta: float = 5e-4
    lr_finetune: float = 1e-5

    scheduler: str = "cosine"

    epochs_phase1: int = 100
    epochs_phase2: int = 100
    epochs_phase3: int = 50
    epochs_phase4: int = 10

    patience: int = 15
    min_delta: float = 1e-6
    # early_stop_metric: "ic" tracks val IC (better for regression),
    # "loss" tracks val loss (original behaviour)
    early_stop_metric: str = "ic"
    ic_min_delta: float = 1e-4  # min IC improvement to reset patience

    batch_size_daily: int = 32
    batch_size_minute: int = 16
    batch_size_meta: int = 128

    loss_fn: str = "huber"

    grad_clip: float = 1.0
    num_workers: int = 0
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_amp: bool = False  # Automatic Mixed Precision
    log_interval: int = 100
    eval_interval: int = 1

    output_dir: str = "models/hierarchical"
    log_dir: str = "logs/hierarchical"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

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
# Single-model trainer
# ============================================================================

class SubModelTrainer:
    """Trains a single sub-model (LSTM or TFT) on regression."""

    def __init__(self, model: nn.Module, name: str, device: torch.device, tcfg: TrainConfig):
        self.model = model.to(device)
        self.name = name
        self.device = device
        self.tcfg = tcfg

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _make_scheduler(self, n_epochs: int):
        if self.tcfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6,
            )
        elif self.tcfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3,
            )
        return None

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        total_grad_norm = 0.0

        for batch_idx, batch in enumerate(loader):
            x, y = batch[0], batch[1]  # ignore date/ticker
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out["prediction"], y)
            loss.backward()

            if self.tcfg.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                total_grad_norm += float(grad_norm)

            self.optimizer.step()
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
            x, y = x.to(self.device), y.to(self.device)
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int, save_dir: str) -> Dict:
        """Full training loop with early stopping."""
        scheduler = self._make_scheduler(n_epochs)
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{self.name}_best.pt")

        history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_rank_ic": []}
        self.best_val_loss = float("inf")
        self.best_val_ic   = -float("inf")
        self.patience_counter = 0
        use_ic_stopping = (self.tcfg.early_stop_metric == "ic")

        logger.info(f"\n{'='*60}")
        logger.info(f"Training {self.name} for {n_epochs} epochs")
        logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"  Early-stop metric: {'IC' if use_ic_stopping else 'val_loss'}, patience={self.tcfg.patience}")
        logger.info(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
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
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
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
# Meta model trainer
# ============================================================================

class MetaTrainer:
    """Trains the MetaMLP on frozen sub-model embeddings + predictions."""

    def __init__(self, meta: MetaMLP, device: torch.device, tcfg: TrainConfig):
        self.meta = meta.to(device)
        self.device = device
        self.tcfg = tcfg

        self.optimizer = torch.optim.AdamW(
            meta.parameters(), lr=tcfg.lr_meta, weight_decay=tcfg.weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()

    @torch.no_grad()
    def _collect_sub_outputs(
        self,
        forecaster: HierarchicalForecaster,
        daily_loader: DataLoader,
        minute_loader: DataLoader,
        data_cfg: HierarchicalDataConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sub-models and collect predictions + embeddings.

        Alignment strategy:
          1. Run daily & minute sub-models, collecting per-sample
             (ticker, ordinal_date) keys alongside predictions/embeddings.
          2. For each sample key present in *both* daily and minute results,
             build one aligned row.
          3. If a key only exists in daily (common — minute data is sparse),
             use the daily sub-model outputs and zero-fill minute fields so
             the meta model still gets daily signal.
          4. Look up real regime features by date.

        Returns:
            predictions: (N, 4)
            emb_cat:     (N, 4*E)
            regime:      (N, R)
            targets:     (N,)
        """
        import datetime as _dt

        forecaster.eval()
        E = forecaster.cfg.embedding_dim
        R = forecaster.cfg.regime_dim

        # Build regime lookup table (ordinal_date → regime_vector)
        regime_df = _build_regime_dataframe(data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        if not regime_df.empty:
            for d, row in regime_df.iterrows():
                if hasattr(d, 'toordinal'):
                    regime_lookup[d.toordinal()] = row.values.astype(np.float32)

        # --- Collect daily outputs keyed by (ticker, ordinal_date) ---
        daily_data: Dict[Tuple[str, int], Dict] = {}
        for batch in daily_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x = x.to(self.device)
            d1 = forecaster.lstm_d(x)
            d2 = forecaster.tft_d(x)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                daily_data[key] = {
                    "pred_lstm": d1["prediction"][i].cpu(),
                    "pred_tft":  d2["prediction"][i].cpu(),
                    "emb_lstm":  d1["embedding"][i].cpu(),
                    "emb_tft":   d2["embedding"][i].cpu(),
                    "target":    y[i],
                }

        # --- Collect minute outputs keyed by (ticker, ordinal_date) ---
        minute_data: Dict[Tuple[str, int], Dict] = {}
        for batch in minute_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x = x.to(self.device)
            m1 = forecaster.lstm_m(x)
            m2 = forecaster.tft_m(x)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                minute_data[key] = {
                    "pred_lstm": m1["prediction"][i].cpu(),
                    "pred_tft":  m2["prediction"][i].cpu(),
                    "emb_lstm":  m1["embedding"][i].cpu(),
                    "emb_tft":   m2["embedding"][i].cpu(),
                }

        # --- Align: iterate over all daily keys ---
        preds_list, embs_list, regime_list, target_list = [], [], [], []
        n_matched, n_daily_only = 0, 0
        zero_emb = torch.zeros(E)

        for key, dd in daily_data.items():
            ticker, ord_date = key
            md = minute_data.get(key)

            if md is not None:
                # Both daily and minute available for this (ticker, date)
                pred = torch.stack([
                    dd["pred_lstm"], dd["pred_tft"],
                    md["pred_lstm"], md["pred_tft"],
                ])
                emb = torch.cat([
                    dd["emb_lstm"], dd["emb_tft"],
                    md["emb_lstm"], md["emb_tft"],
                ])
                n_matched += 1
            else:
                # Daily only — zero-fill minute predictions/embeddings
                pred = torch.stack([
                    dd["pred_lstm"], dd["pred_tft"],
                    torch.tensor(0.0), torch.tensor(0.0),
                ])
                emb = torch.cat([
                    dd["emb_lstm"], dd["emb_tft"],
                    zero_emb, zero_emb,
                ])
                n_daily_only += 1

            # Regime features from real market data
            if ord_date in regime_lookup:
                regime_vec = torch.from_numpy(regime_lookup[ord_date])
            else:
                regime_vec = torch.zeros(R)

            preds_list.append(pred)
            embs_list.append(emb)
            regime_list.append(regime_vec)
            target_list.append(dd["target"])

        logger.info(f"  Meta align: {n_matched} matched (daily+minute), "
                    f"{n_daily_only} daily-only, {len(minute_data)} minute keys total")

        if not preds_list:
            logger.warning("  No aligned data for meta model!")
            return (torch.zeros(0, 4), torch.zeros(0, 4 * E),
                    torch.zeros(0, R), torch.zeros(0))

        predictions = torch.stack(preds_list)        # (N, 4)
        emb_cat     = torch.stack(embs_list)          # (N, 4*E)
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
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "meta_best.pt")
        E = forecaster.cfg.embedding_dim

        logger.info("Collecting sub-model outputs for meta training...")
        forecaster.freeze_sub_models()

        train_preds, train_embs, train_regime, train_targets = \
            self._collect_sub_outputs(forecaster, d_train_dl, m_train_dl, data_cfg)
        val_preds, val_embs, val_regime, val_targets = \
            self._collect_sub_outputs(forecaster, d_val_dl, m_val_dl, data_cfg)

        logger.info(f"  Meta train: {len(train_targets):,} | val: {len(val_targets):,}")

        train_ds = TensorDataset(train_preds, train_embs, train_regime, train_targets)
        val_ds = TensorDataset(val_preds, val_embs, val_regime, val_targets)
        train_loader = DataLoader(train_ds, batch_size=self.tcfg.batch_size_meta, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.tcfg.batch_size_meta)

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
        logger.info(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            self.meta.train()
            total_loss, n_b = 0.0, 0
            for preds_b, embs_b, regime_b, targets_b in train_loader:
                preds_b = preds_b.to(self.device)
                embs_b = embs_b.to(self.device)
                regime_b = regime_b.to(self.device)
                targets_b = targets_b.to(self.device)
                emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(4)]

                self.optimizer.zero_grad()
                out = self.meta(preds_b, emb_list, regime_b)
                loss = self.criterion(out["prediction"], targets_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), self.tcfg.grad_clip)
                self.optimizer.step()
                total_loss += loss.item()
                n_b += 1

            train_loss = total_loss / max(n_b, 1)

            self.meta.eval()
            val_preds_l, val_tgts_l = [], []
            val_total, val_n = 0.0, 0
            with torch.no_grad():
                for preds_b, embs_b, regime_b, targets_b in val_loader:
                    preds_b = preds_b.to(self.device)
                    embs_b = embs_b.to(self.device)
                    regime_b = regime_b.to(self.device)
                    targets_b = targets_b.to(self.device)
                    emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(4)]
                    out = self.meta(preds_b, emb_list, regime_b)
                    val_total += self.criterion(out["prediction"], targets_b).item()
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
# Phase 4: Joint fine-tuning
# ============================================================================

class JointFineTuner:
    """Phase 4: Fine-tune all 5 models end-to-end with small learning rate.

    Strategy:
      1. Unfreeze all parameters.
      2. Collect aligned (daily, minute, regime, target) tuples via the same
         key-based alignment as MetaTrainer.
      3. Run a combined forward pass through all 5 models and back-prop
         through the full graph — the meta loss gradient flows back into
         the sub-models, adapting them to the meta-model's needs.
      4. Use a much smaller LR than individual phases and aggressive
         gradient clipping to avoid catastrophic forgetting.
    """

    def __init__(
        self,
        forecaster: HierarchicalForecaster,
        device: torch.device,
        tcfg: TrainConfig,
        data_cfg: HierarchicalDataConfig,
    ):
        self.forecaster = forecaster.to(device)
        self.device = device
        self.tcfg = tcfg
        self.data_cfg = data_cfg

        # Very small LR to avoid destroying pre-trained weights
        self.optimizer = torch.optim.AdamW(
            forecaster.parameters(), lr=tcfg.lr_finetune, weight_decay=tcfg.weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=0.01) if tcfg.loss_fn == "huber" else nn.MSELoss()

    def _build_aligned_dataset(
        self,
        daily_loader: DataLoader,
        minute_loader: DataLoader,
    ) -> TensorDataset:
        """Build an aligned dataset for joint fine-tuning.

        Unlike MetaTrainer (which only needs predictions/embeddings), we need
        the raw input sequences so gradients can flow back through the
        sub-models. We match daily and minute samples by (ticker, date) keys.

        For samples with daily only (no minute match), we zero-fill the
        minute input so the meta model still trains on those rows.

        Returns:
            TensorDataset with (daily_x, minute_x, regime, target)
        """
        E = self.forecaster.cfg.embedding_dim
        R = self.forecaster.cfg.regime_dim

        # Build regime lookup
        regime_df = _build_regime_dataframe(self.data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        if not regime_df.empty:
            for d, row in regime_df.iterrows():
                if hasattr(d, 'toordinal'):
                    regime_lookup[d.toordinal()] = row.values.astype(np.float32)

        # Collect daily sequences keyed by (ticker, date)
        daily_data: Dict[Tuple[str, int], Dict] = {}
        for batch in daily_loader:
            x, y, ordinal_dates, tickers_batch = batch
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                daily_data[key] = {
                    "x": x[i],        # (seq_len, F_daily)
                    "target": y[i],
                }

        # Collect minute sequences keyed by (ticker, date)
        minute_data: Dict[Tuple[str, int], Dict] = {}
        for batch in minute_loader:
            x, y, ordinal_dates, tickers_batch = batch
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                # Keep only first match per key (avoid duplicates)
                if key not in minute_data:
                    minute_data[key] = {"x": x[i]}

        # Get dimensions from model config
        minute_seq_len = self.forecaster.cfg.minute_seq_len
        minute_feat_dim = self.forecaster.cfg.minute_input_dim

        # Align
        daily_xs, minute_xs, regimes, targets = [], [], [], []
        n_matched, n_daily_only = 0, 0
        zero_minute = torch.zeros(minute_seq_len, minute_feat_dim)

        for key, dd in daily_data.items():
            _, ord_date = key
            md = minute_data.get(key)

            daily_xs.append(dd["x"])
            targets.append(dd["target"])

            if md is not None:
                minute_xs.append(md["x"])
                n_matched += 1
            else:
                minute_xs.append(zero_minute)
                n_daily_only += 1

            if ord_date in regime_lookup:
                regimes.append(torch.from_numpy(regime_lookup[ord_date]))
            else:
                regimes.append(torch.zeros(R))

        logger.info(f"  Joint finetune align: {n_matched} matched, "
                    f"{n_daily_only} daily-only, {len(daily_data)} total")

        if not daily_xs:
            return TensorDataset(
                torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)
            )

        return TensorDataset(
            torch.stack(daily_xs),     # (N, daily_seq, F_daily)
            torch.stack(minute_xs),    # (N, minute_seq, F_minute)
            torch.stack(regimes),      # (N, R)
            torch.stack(targets),      # (N,)
        )

    def train(
        self,
        d_train_dl: DataLoader, m_train_dl: DataLoader,
        d_val_dl: DataLoader, m_val_dl: DataLoader,
        n_epochs: int, save_dir: str,
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "finetune_best.pt")

        logger.info("Collecting aligned sequences for joint fine-tuning...")
        self.forecaster.unfreeze_all()

        train_ds = self._build_aligned_dataset(d_train_dl, m_train_dl)
        val_ds = self._build_aligned_dataset(d_val_dl, m_val_dl)

        logger.info(f"  Joint train: {len(train_ds):,} | val: {len(val_ds):,}")

        train_loader = DataLoader(
            train_ds, batch_size=self.tcfg.batch_size_meta,
            shuffle=True, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.tcfg.batch_size_meta, pin_memory=True,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-7,
        )

        history = {"train_loss": [], "val_loss": [], "val_ic": []}
        best_val = float("inf")
        patience_ctr = 0

        total_params = sum(p.numel() for p in self.forecaster.parameters() if p.requires_grad)
        logger.info(f"\n{'='*60}")
        logger.info(f"Joint fine-tuning for {n_epochs} epochs ({total_params:,} params)")
        logger.info(f"  LR={self.tcfg.lr_finetune:.1e}, grad_clip={self.tcfg.grad_clip}")
        logger.info(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # --- Train ---
            self.forecaster.train()
            total_loss, n_b = 0.0, 0
            for daily_x, minute_x, regime, target in train_loader:
                daily_x = daily_x.to(self.device)
                minute_x = minute_x.to(self.device)
                regime = regime.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                out = self.forecaster(daily_x, minute_x, regime)
                loss = self.criterion(out["prediction"], target)

                # Optional: add sub-model auxiliary losses for stability
                aux_weight = 0.1
                for sub_key in ["lstm_d", "tft_d", "lstm_m", "tft_m"]:
                    sub_pred = out[sub_key]["prediction"]
                    loss = loss + aux_weight * self.criterion(sub_pred, target)

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.forecaster.parameters(), self.tcfg.grad_clip * 0.5
                )
                self.optimizer.step()
                total_loss += loss.item()
                n_b += 1

            train_loss = total_loss / max(n_b, 1)

            # --- Validate ---
            self.forecaster.eval()
            val_preds_l, val_tgts_l = [], []
            val_total, val_n = 0.0, 0
            with torch.no_grad():
                for daily_x, minute_x, regime, target in val_loader:
                    daily_x = daily_x.to(self.device)
                    minute_x = minute_x.to(self.device)
                    regime = regime.to(self.device)
                    target = target.to(self.device)

                    out = self.forecaster(daily_x, minute_x, regime)
                    val_total += self.criterion(out["prediction"], target).item()
                    val_n += 1
                    val_preds_l.append(out["prediction"].cpu().numpy())
                    val_tgts_l.append(target.cpu().numpy())

            val_loss = val_total / max(val_n, 1)
            metrics = compute_metrics(
                np.concatenate(val_preds_l), np.concatenate(val_tgts_l)
            )
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic"].append(metrics["ic"])

            logger.info(
                f"[Joint] Epoch {epoch:3d}/{n_epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"IC={metrics['ic']:.4f} | RankIC={metrics['rank_ic']:.4f} | "
                f"DirAcc={metrics['directional_accuracy']:.3f} | {elapsed:.1f}s"
            )
            scheduler.step()

            if val_loss < best_val - self.tcfg.min_delta:
                best_val = val_loss
                patience_ctr = 0
                self.forecaster.save(best_path)
                logger.info(f"  ★ New best joint val={val_loss:.6f}")
            else:
                patience_ctr += 1
                if patience_ctr >= self.tcfg.patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # Reload best weights
        best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.forecaster.lstm_d.load_state_dict(best_ckpt["lstm_d"])
        self.forecaster.tft_d.load_state_dict(best_ckpt["tft_d"])
        self.forecaster.lstm_m.load_state_dict(best_ckpt["lstm_m"])
        self.forecaster.tft_m.load_state_dict(best_ckpt["tft_m"])
        self.forecaster.meta.load_state_dict(best_ckpt["meta"])
        _save_training_curve(history, "Joint", save_dir)
        return history


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
    
    # Print memory info for debugging
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    os.makedirs(tcfg.output_dir, exist_ok=True)
    os.makedirs(tcfg.log_dir, exist_ok=True)

    with open(os.path.join(tcfg.output_dir, "train_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(tcfg).items()}, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Discover tickers and split
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 0: Discover tickers and split")
    logger.info("=" * 70)

    tickers = get_viable_tickers(data_cfg)
    splits = split_tickers(tickers, data_cfg)

    with open(os.path.join(tcfg.output_dir, "ticker_split.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # ------------------------------------------------------------------
    # PHASE 0: Preprocess (compute features → cache as .npy)
    # ------------------------------------------------------------------
    if 0 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 0: Preprocessing features to disk cache")
        logger.info("=" * 70)
        preprocess_all(tickers, data_cfg, force=force_preprocess)
        logger.info("Phase 0 complete ✓")
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # Create dataloaders (lazy, near-zero RAM)
    # ------------------------------------------------------------------
    needs_data = bool(set(phases) & {1, 2, 3, 4})
    if needs_data:
        logger.info("\nCreating lazy dataloaders...")
        loaders = create_dataloaders(
            splits, data_cfg,
            batch_size_daily=tcfg.batch_size_daily,
            batch_size_minute=tcfg.batch_size_minute,
            num_workers=tcfg.num_workers,
        )

    # ------------------------------------------------------------------
    # Create or resume model
    # ------------------------------------------------------------------
    if needs_data:
        if resume_path and os.path.exists(resume_path):
            logger.info(f"\nResuming from {resume_path}")
            forecaster = HierarchicalForecaster.load(resume_path, device=str(device))
        else:
            model_cfg.daily_input_dim = loaders["daily_n_features"]
            model_cfg.minute_input_dim = loaders["minute_n_features"]
            logger.info(f"\nCreating HierarchicalForecaster:")
            logger.info(f"  Daily:  {model_cfg.daily_input_dim} features, seq_len={model_cfg.daily_seq_len}")
            logger.info(f"  Minute: {model_cfg.minute_input_dim} features, seq_len={model_cfg.minute_seq_len}")
            forecaster = HierarchicalForecaster(model_cfg)

        forecaster = forecaster.to(device)

    # ------------------------------------------------------------------
    # PHASE 1: Train daily models
    # ------------------------------------------------------------------
    if 1 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: Training daily models (LSTM_D + TFT_D)")
        logger.info("=" * 70)

        train_dl = loaders["daily"]["train"]
        val_dl = loaders["daily"]["val"]

        trainer = SubModelTrainer(forecaster.lstm_d, "LSTM_D", device, tcfg)
        trainer.train(train_dl, val_dl, tcfg.epochs_phase1, tcfg.output_dir)

        trainer = SubModelTrainer(forecaster.tft_d, "TFT_D", device, tcfg)
        trainer.train(train_dl, val_dl, tcfg.epochs_phase1, tcfg.output_dir)

        forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase1.pt"))
        logger.info("Phase 1 complete ✓")
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # PHASE 2: Train minute models
    # ------------------------------------------------------------------
    if 2 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: Training minute models (LSTM_M + TFT_M)")
        logger.info("=" * 70)

        train_dl = loaders["minute"]["train"]
        val_dl = loaders["minute"]["val"]

        trainer = SubModelTrainer(forecaster.lstm_m, "LSTM_M", device, tcfg)
        trainer.train(train_dl, val_dl, tcfg.epochs_phase2, tcfg.output_dir)

        trainer = SubModelTrainer(forecaster.tft_m, "TFT_M", device, tcfg)
        trainer.train(train_dl, val_dl, tcfg.epochs_phase2, tcfg.output_dir)

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

        meta_trainer = MetaTrainer(forecaster.meta, device, tcfg)
        meta_trainer.train(
            forecaster,
            loaders["daily"]["val"], loaders["minute"]["val"],      # meta TRAIN = sub-model val
            loaders["daily"]["test"], loaders["minute"]["test"],    # meta VAL   = sub-model test
            tcfg.epochs_phase3, tcfg.output_dir,
            data_cfg=data_cfg,
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

        # Evaluate daily sub-models on test
        test_dl = loaders["daily"]["test"]
        if len(test_dl.dataset) > 0:
            for model, name in [(forecaster.lstm_d, "LSTM_D"), (forecaster.tft_d, "TFT_D")]:
                preds, tgts = [], []
                with torch.no_grad():
                    for batch in test_dl:
                        x, y = batch[0], batch[1]
                        x = x.to(device)
                        out = model(x)
                        preds.append(out["prediction"].cpu().numpy())
                        tgts.append(y.numpy())
                metrics = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
                results[name] = metrics
                logger.info(f"  {name:8s} | IC={metrics['ic']:.4f} | "
                            f"RankIC={metrics['rank_ic']:.4f} | DirAcc={metrics['directional_accuracy']:.3f}")

        # Evaluate minute sub-models on test
        test_dl = loaders["minute"]["test"]
        if len(test_dl.dataset) > 0:
            for model, name in [(forecaster.lstm_m, "LSTM_M"), (forecaster.tft_m, "TFT_M")]:
                preds, tgts = [], []
                with torch.no_grad():
                    for batch in test_dl:
                        x, y = batch[0], batch[1]
                        x = x.to(device)
                        out = model(x)
                        preds.append(out["prediction"].cpu().numpy())
                        tgts.append(y.numpy())
                metrics = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
                results[name] = metrics
                logger.info(f"  {name:8s} | IC={metrics['ic']:.4f} | "
                            f"RankIC={metrics['rank_ic']:.4f} | DirAcc={metrics['directional_accuracy']:.3f}")

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
            )
        if len(test_targets) > 0:
            E = forecaster.cfg.embedding_dim
            all_meta_preds, all_meta_tgts = [], []
            test_ds = TensorDataset(test_preds, test_embs, test_regime, test_targets)
            test_meta_dl = DataLoader(test_ds, batch_size=tcfg.batch_size_meta)
            forecaster.meta.eval()
            with torch.no_grad():
                for preds_b, embs_b, regime_b, targets_b in test_meta_dl:
                    preds_b = preds_b.to(device)
                    embs_b = embs_b.to(device)
                    regime_b = regime_b.to(device)
                    emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(4)]
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
    parser.add_argument("--daily-stride", type=int, default=5)
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
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Recompute cached features even if present.")
    parser.add_argument("--low-memory", action="store_true",
                        help="Reduce batch sizes and enable AMP for low-memory systems.")
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

    data_cfg = HierarchicalDataConfig(
        daily_seq_len=args.daily_seq_len,
        minute_seq_len=args.minute_seq_len,
        daily_stride=args.daily_stride,
        minute_stride=args.minute_stride,
        split_mode=args.split_mode,
    )

    model_cfg = HierarchicalModelConfig(
        daily_seq_len=args.daily_seq_len,
        minute_seq_len=args.minute_seq_len,
    )

    run_pipeline(
        phases=args.phase,
        resume_path=args.resume,
        tcfg=tcfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        force_preprocess=args.force_preprocess,
    )


if __name__ == "__main__":
    main()
