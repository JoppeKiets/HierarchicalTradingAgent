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

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _check_shm_available() -> bool:
    """Return True if /dev/shm is available and writable (needed for num_workers>0)."""
    import tempfile
    shm_path = "/dev/shm"
    if not os.path.isdir(shm_path):
        return False
    try:
        with tempfile.TemporaryFile(dir=shm_path):
            pass
        return True
    except OSError:
        return False


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
    epochs_news: int = 80           # Phase 1.5: news encoder
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
    batch_size_news: int = 32       # News model batch size

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
        news_loader: Optional[DataLoader] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sub-models and collect predictions + embeddings.

        Alignment strategy:
          1. Run daily & minute (& news) sub-models, collecting per-sample
             (ticker, ordinal_date) keys alongside predictions/embeddings.
          2. For each sample key present in *both* daily and minute results,
             build one aligned row.
          3. If a key only exists in daily (common — minute data is sparse),
             use the daily sub-model outputs and zero-fill minute fields so
             the meta model still gets daily signal.
          4. Look up real regime features by date.
          5. If news model is enabled, align news outputs by (ticker, date).

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
        use_news = forecaster.use_news and forecaster.news is not None
        n_sub = 5 if use_news else 4

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

        # --- Collect news outputs keyed by (ticker, ordinal_date) ---
        news_data: Dict[Tuple[str, int], Dict] = {}
        if use_news and news_loader is not None:
            for batch in news_loader:
                x, y, ordinal_dates, tickers_batch = batch
                x = x.to(self.device)
                n1 = forecaster.news(x)
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    news_data[key] = {
                        "pred": n1["prediction"][i].cpu(),
                        "emb":  n1["embedding"][i].cpu(),
                    }

        # --- Align: iterate over all daily keys ---
        preds_list, embs_list, regime_list, target_list = [], [], [], []
        n_matched, n_daily_only = 0, 0
        zero_emb = torch.zeros(E)

        for key, dd in daily_data.items():
            ticker, ord_date = key
            md = minute_data.get(key)
            nd = news_data.get(key) if use_news else None

            if md is not None:
                pred = [dd["pred_lstm"], dd["pred_tft"],
                        md["pred_lstm"], md["pred_tft"]]
                emb = [dd["emb_lstm"], dd["emb_tft"],
                       md["emb_lstm"], md["emb_tft"]]
                n_matched += 1
            else:
                pred = [dd["pred_lstm"], dd["pred_tft"],
                        torch.tensor(0.0), torch.tensor(0.0)]
                emb = [dd["emb_lstm"], dd["emb_tft"],
                       zero_emb, zero_emb]
                n_daily_only += 1

            # Add news model outputs if enabled
            if use_news:
                if nd is not None:
                    pred.append(nd["pred"])
                    emb.append(nd["emb"])
                else:
                    pred.append(torch.tensor(0.0))
                    emb.append(zero_emb)

            # Regime features from real market data
            if ord_date in regime_lookup:
                regime_vec = torch.from_numpy(regime_lookup[ord_date])
            else:
                regime_vec = torch.zeros(R)

            preds_list.append(torch.stack(pred))
            embs_list.append(torch.cat(emb))
            regime_list.append(regime_vec)
            target_list.append(dd["target"])

        n_news_matched = sum(1 for k in daily_data if k in news_data) if use_news else 0
        logger.info(f"  Meta align: {n_matched} matched (daily+minute), "
                    f"{n_daily_only} daily-only, {len(minute_data)} minute keys total"
                    + (f", {n_news_matched} news-matched" if use_news else ""))

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
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "meta_best.pt")
        E = forecaster.cfg.embedding_dim
        use_news = forecaster.use_news and forecaster.news is not None
        n_sub = 5 if use_news else 4

        logger.info("Collecting sub-model outputs for meta training...")
        forecaster.freeze_sub_models()

        train_preds, train_embs, train_regime, train_targets = \
            self._collect_sub_outputs(forecaster, d_train_dl, m_train_dl, data_cfg,
                                      news_loader=n_train_news_dl)
        val_preds, val_embs, val_regime, val_targets = \
            self._collect_sub_outputs(forecaster, d_val_dl, m_val_dl, data_cfg,
                                      news_loader=n_val_news_dl)

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
                emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(n_sub)]

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
                    emb_list = [embs_b[:, i*E:(i+1)*E] for i in range(n_sub)]
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
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run all sub-models on CPU/GPU and cache predictions + embeddings.

        Returns dict keyed by (ticker, ordinal_date) → {
            "lstm_d_pred", "lstm_d_emb", "tft_d_pred", "tft_d_emb",
            "lstm_m_pred", "lstm_m_emb", "tft_m_pred", "tft_m_emb",
            "news_pred", "news_emb" (if news enabled),
            "regime", "target",
        }
        But for efficiency we return separate tensors in a flat dict.
        """
        E = forecaster.cfg.embedding_dim
        R = forecaster.cfg.regime_dim
        use_news = forecaster.use_news and forecaster.news is not None

        # Build regime lookup
        regime_df = _build_regime_dataframe(data_cfg)
        regime_lookup: Dict[int, np.ndarray] = {}
        if not regime_df.empty:
            for d, row in regime_df.iterrows():
                if hasattr(d, 'toordinal'):
                    regime_lookup[d.toordinal()] = row.values.astype(np.float32)

        # ---------- Run daily models ----------
        daily_outputs: Dict[Tuple[str, int], Dict] = {}
        forecaster.lstm_d.to(self.device).eval()
        forecaster.tft_d.to(self.device).eval()
        for batch in daily_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device)
            d1 = forecaster.lstm_d(x_dev)
            d2 = forecaster.tft_d(x_dev)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                ord_date = int(ordinal_dates[i])
                regime = torch.from_numpy(regime_lookup[ord_date]) \
                    if ord_date in regime_lookup else torch.zeros(R)
                daily_outputs[key] = {
                    "lstm_d_pred": d1["prediction"][i].cpu(),
                    "lstm_d_emb": d1["embedding"][i].cpu(),
                    "tft_d_pred": d2["prediction"][i].cpu(),
                    "tft_d_emb": d2["embedding"][i].cpu(),
                    "target": y[i],
                    "regime": regime,
                }
        forecaster.lstm_d.cpu()
        forecaster.tft_d.cpu()
        clear_gpu_memory()

        # ---------- Run minute models ----------
        minute_outputs: Dict[Tuple[str, int], Dict] = {}
        forecaster.lstm_m.to(self.device).eval()
        forecaster.tft_m.to(self.device).eval()
        for batch in minute_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device)
            m1 = forecaster.lstm_m(x_dev)
            m2 = forecaster.tft_m(x_dev)
            for i in range(len(y)):
                key = (tickers_batch[i], int(ordinal_dates[i]))
                minute_outputs[key] = {
                    "lstm_m_pred": m1["prediction"][i].cpu(),
                    "lstm_m_emb": m1["embedding"][i].cpu(),
                    "tft_m_pred": m2["prediction"][i].cpu(),
                    "tft_m_emb": m2["embedding"][i].cpu(),
                }
        forecaster.lstm_m.cpu()
        forecaster.tft_m.cpu()
        clear_gpu_memory()

        # ---------- Run news model ----------
        news_outputs: Dict[Tuple[str, int], Dict] = {}
        if use_news and news_loader is not None:
            forecaster.news.to(self.device).eval()
            for batch in news_loader:
                x, y, ordinal_dates, tickers_batch = batch
                x_dev = x.to(self.device)
                n1 = forecaster.news(x_dev)
                for i in range(len(y)):
                    key = (tickers_batch[i], int(ordinal_dates[i]))
                    news_outputs[key] = {
                        "news_pred": n1["prediction"][i].cpu(),
                        "news_emb": n1["embedding"][i].cpu(),
                    }
            forecaster.news.cpu()
            clear_gpu_memory()

        # ---------- Merge into aligned flat tensors ----------
        zero_m_pred = torch.zeros(1).squeeze()
        zero_m_emb = torch.zeros(E)
        zero_n_pred = torch.zeros(1).squeeze()
        zero_n_emb = torch.zeros(E)

        keys_ordered = []
        all_preds = {name: [] for name in ["lstm_d", "tft_d", "lstm_m", "tft_m"]}
        all_embs = {name: [] for name in ["lstm_d", "tft_d", "lstm_m", "tft_m"]}
        if use_news:
            all_preds["news"] = []
            all_embs["news"] = []
        all_regimes, all_targets = [], []

        for key, dd in daily_outputs.items():
            keys_ordered.append(key)
            all_preds["lstm_d"].append(dd["lstm_d_pred"])
            all_preds["tft_d"].append(dd["tft_d_pred"])
            all_embs["lstm_d"].append(dd["lstm_d_emb"])
            all_embs["tft_d"].append(dd["tft_d_emb"])
            all_regimes.append(dd["regime"])
            all_targets.append(dd["target"])

            md = minute_outputs.get(key)
            if md is not None:
                all_preds["lstm_m"].append(md["lstm_m_pred"])
                all_preds["tft_m"].append(md["tft_m_pred"])
                all_embs["lstm_m"].append(md["lstm_m_emb"])
                all_embs["tft_m"].append(md["tft_m_emb"])
            else:
                all_preds["lstm_m"].append(zero_m_pred)
                all_preds["tft_m"].append(zero_m_pred)
                all_embs["lstm_m"].append(zero_m_emb)
                all_embs["tft_m"].append(zero_m_emb)

            if use_news:
                nd = news_outputs.get(key)
                if nd is not None:
                    all_preds["news"].append(nd["news_pred"])
                    all_embs["news"].append(nd["news_emb"])
                else:
                    all_preds["news"].append(zero_n_pred)
                    all_embs["news"].append(zero_n_emb)

        result = {
            "keys": keys_ordered,
            "targets": torch.stack(all_targets),
            "regimes": torch.stack(all_regimes),
        }
        for name in all_preds:
            result[f"{name}_pred"] = torch.stack(all_preds[name])
            result[f"{name}_emb"] = torch.stack(all_embs[name])

        logger.info(f"  Frozen outputs collected: {len(keys_ordered):,} samples")
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
        use_news = self.forecaster.use_news and self.forecaster.news is not None
        sub_names = ["lstm_d", "tft_d", "lstm_m", "tft_m"]
        if use_news:
            sub_names.append("news")

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
        # Move active models to GPU
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
        for batch in data_loader:
            x, y, ordinal_dates, tickers_batch = batch
            x_dev = x.to(self.device)
            target = y.to(self.device)

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
                continue

            # Forward through active sub-model
            out = sub_model(x_dev)
            active_pred = out["prediction"]   # (batch,)
            active_emb = out["embedding"]     # (batch, E)

            # Build meta inputs (swap in live outputs for active model)
            preds, emb_list, regime = self._build_meta_input(
                cached_train, sub_name, active_pred, active_emb, indices_t,
            )

            # Forward through meta
            meta_out = self.forecaster.meta(preds, emb_list, regime)
            loss = self.criterion(meta_out["prediction"], target)

            # Auxiliary loss on the sub-model's own prediction
            loss = loss + 0.1 * self.criterion(active_pred, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(sub_model.parameters()) + list(self.forecaster.meta.parameters()),
                self.tcfg.grad_clip * 0.5,
            )
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        train_loss = total_loss / max(n_b, 1)

        # --- Validation pass (use val data loader for this sub-model's modality) ---
        sub_model.eval()
        self.forecaster.meta.eval()
        val_preds_l, val_tgts_l = [], []
        val_total, val_n = 0.0, 0

        # For validation, we can reconstruct full meta predictions from cached
        # outputs + the live sub-model's updated predictions
        with torch.no_grad():
            for batch in data_loader:
                # Re-use same loader but only match val keys
                x, y, ordinal_dates, tickers_batch = batch
                x_dev = x.to(self.device)
                target = y.to(self.device)

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
                preds, emb_list, regime = self._build_meta_input(
                    cached_val, sub_name, out["prediction"], out["embedding"], indices_t,
                )
                meta_out = self.forecaster.meta(preds, emb_list, regime)

                # Only count valid (matched) samples
                val_total += self.criterion(
                    meta_out["prediction"][valid_mask_t],
                    target[valid_mask_t],
                ).item()
                val_n += 1
                val_preds_l.append(meta_out["prediction"][valid_mask_t].cpu().numpy())
                val_tgts_l.append(target[valid_mask_t].cpu().numpy())

        val_loss = val_total / max(val_n, 1)
        if val_preds_l:
            metrics = compute_metrics(np.concatenate(val_preds_l), np.concatenate(val_tgts_l))
        else:
            metrics = {"ic": 0.0, "rank_ic": 0.0, "directional_accuracy": 0.5}

        # Move sub-model back to CPU to free VRAM
        sub_model.cpu()
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
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "finetune_best.pt")

        use_news = self.forecaster.use_news and self.forecaster.news is not None

        # --- Step 1: Collect frozen outputs for all sub-models ---
        logger.info("Collecting frozen sub-model outputs for round-robin fine-tuning...")
        self.forecaster.eval()

        cached_train = self._collect_frozen_outputs(
            self.forecaster, d_train_dl, m_train_dl, self.data_cfg,
            news_loader=n_train_news_dl,
        )
        cached_val = self._collect_frozen_outputs(
            self.forecaster, d_val_dl, m_val_dl, self.data_cfg,
            news_loader=n_val_news_dl,
        )

        # Build key → index maps for fast lookup
        train_key_to_idx = {k: i for i, k in enumerate(cached_train["keys"])}
        val_key_to_idx = {k: i for i, k in enumerate(cached_val["keys"])}

        logger.info(f"  Train samples: {len(train_key_to_idx):,} | Val: {len(val_key_to_idx):,}")

        # --- Step 2: Set up sub-model → data loader mapping ---
        sub_models = [
            ("lstm_d", self.forecaster.lstm_d, d_train_dl, d_val_dl),
            ("tft_d",  self.forecaster.tft_d,  d_train_dl, d_val_dl),
            ("lstm_m", self.forecaster.lstm_m, m_train_dl, m_val_dl),
            ("tft_m",  self.forecaster.tft_m,  m_train_dl, m_val_dl),
        ]
        if use_news and n_train_news_dl is not None:
            sub_models.append(
                ("news", self.forecaster.news, n_train_news_dl, n_val_news_dl)
            )

        # Each sub-model gets its own optimizer (created fresh per round to
        # avoid stale momentum from previous rounds on frozen params)
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
                # Fresh optimizer per sub-model per round
                optimizer = torch.optim.AdamW(
                    list(sub_model.parameters()) + list(self.forecaster.meta.parameters()),
                    lr=self.tcfg.lr_finetune,
                    weight_decay=self.tcfg.weight_decay,
                )

                tr_loss, v_loss, v_metrics = self._finetune_one_submodel(
                    sub_model, sub_name,
                    train_dl, cached_train, cached_val,
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

            logger.info(
                f"[Joint] Round {rnd:3d}/{n_rounds} | "
                f"avg_train={avg_train:.6f} | avg_val={avg_val:.6f} | "
                f"avg_IC={avg_ic:.4f} | {elapsed:.1f}s"
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
            self.forecaster.lstm_d.load_state_dict(best_ckpt["lstm_d"])
            self.forecaster.tft_d.load_state_dict(best_ckpt["tft_d"])
            self.forecaster.lstm_m.load_state_dict(best_ckpt["lstm_m"])
            self.forecaster.tft_m.load_state_dict(best_ckpt["tft_m"])
            self.forecaster.meta.load_state_dict(best_ckpt["meta"])
            if "news" in best_ckpt and use_news:
                self.forecaster.news.load_state_dict(best_ckpt["news"])
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
            x_dev = x.to(self.device)
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
    # PHASE 0: Preprocess (compute features → cache as .npy)
    # ------------------------------------------------------------------
    if 0 in phases:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 0: Preprocessing features to disk cache")
        logger.info("=" * 70)
        reset_minute_date_bounds()  # clear stale cached boundaries
        preprocess_all(tickers, data_cfg, force=force_preprocess)

        # Also preprocess news sequences if news model is enabled
        if model_cfg.use_news_model:
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
    if needs_data:
        logger.info("\nCreating lazy dataloaders...")
        loaders = create_dataloaders(
            splits, data_cfg,
            batch_size_daily=tcfg.batch_size_daily,
            batch_size_minute=tcfg.batch_size_minute,
            num_workers=tcfg.num_workers,
        )

        # Create news dataloaders if news model is enabled
        if model_cfg.use_news_model:
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
    if needs_data:
        if resume_path and os.path.exists(resume_path):
            logger.info(f"\nResuming from {resume_path}")
            forecaster = HierarchicalForecaster.load(resume_path, device=str(device))
        else:
            model_cfg.daily_input_dim = loaders["daily_n_features"]
            model_cfg.minute_input_dim = loaders["minute_n_features"]
            if model_cfg.use_news_model and news_loaders is not None:
                model_cfg.news_input_dim = news_loaders["n_features"]
                model_cfg.news_seq_len = data_cfg.daily_seq_len
                model_cfg.n_sub_models = 5
            else:
                model_cfg.n_sub_models = 4
            logger.info(f"\nCreating HierarchicalForecaster:")
            logger.info(f"  Daily:  {model_cfg.daily_input_dim} features, seq_len={model_cfg.daily_seq_len}")
            logger.info(f"  Minute: {model_cfg.minute_input_dim} features, seq_len={model_cfg.minute_seq_len}")
            if model_cfg.use_news_model:
                logger.info(f"  News:   {model_cfg.news_input_dim} features, seq_len={model_cfg.news_seq_len}")
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
    # PHASE 1.5: Train news encoder (if enabled)
    # ------------------------------------------------------------------
    if 1 in phases and model_cfg.use_news_model and news_loaders is not None:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1.5: Training News Encoder")
        logger.info("=" * 70)

        train_dl = news_loaders["train"]
        val_dl = news_loaders["val"]

        if len(train_dl.dataset) > 0:
            trainer = SubModelTrainer(forecaster.news, "News", device, tcfg)
            trainer.train(train_dl, val_dl, tcfg.epochs_news, tcfg.output_dir)
            forecaster.save(os.path.join(tcfg.output_dir, "checkpoint_phase1_5.pt"))
            logger.info("Phase 1.5 complete ✓")
        else:
            logger.warning("No news training data available — skipping Phase 1.5")
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
            n_train_news_dl=news_loaders["val"] if news_loaders else None,
            n_val_news_dl=news_loaders["test"] if news_loaders else None,
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

        # Evaluate news sub-model on test (if enabled)
        if forecaster.use_news and forecaster.news is not None and news_loaders is not None:
            test_dl = news_loaders["test"]
            if len(test_dl.dataset) > 0:
                preds, tgts = [], []
                with torch.no_grad():
                    for batch in test_dl:
                        x, y = batch[0], batch[1]
                        x = x.to(device)
                        out = forecaster.news(x)
                        preds.append(out["prediction"].cpu().numpy())
                        tgts.append(y.numpy())
                metrics = compute_metrics(np.concatenate(preds), np.concatenate(tgts))
                results["NEWS"] = metrics
                logger.info(f"  {'NEWS':8s} | IC={metrics['ic']:.4f} | "
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
                news_loader=news_loaders["test"] if news_loaders else None,
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
                    preds_b = preds_b.to(device)
                    embs_b = embs_b.to(device)
                    regime_b = regime_b.to(device)
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
    parser.add_argument("--use-news", action="store_true",
                        help="Enable the FinBERT news encoder sub-model (Phase 1.5).")
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
        daily_feature_set=args.feature_set,
        daily_target_type=args.target_type,
    )

    model_cfg = HierarchicalModelConfig(
        daily_seq_len=args.daily_seq_len,
        minute_seq_len=args.minute_seq_len,
        use_news_model=args.use_news,
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
