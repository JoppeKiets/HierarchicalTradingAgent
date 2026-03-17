#!/usr/bin/env python3
"""Train v2 pipeline: Enhanced features + Temporal Fusion Transformer.

Fixes all Phase 1 diagnosis issues:
  - 51 features (was 7)
  - 3-class target (was 11)
  - Rolling z-score normalization
  - Class-weighted loss (focal loss)
  - Early stopping on validation loss
  - Proper time-series train/val split
  - Learning rate warmup + cosine decay

Usage:
    python train_v2.py
    python train_v2.py --tickers AAPL MSFT GOOGL NVDA TSLA --epochs 100
    python train_v2.py --n-classes 5 --hidden-dim 256
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from src.enhanced_features import prepare_enhanced_daily_data, FeatureConfig
from src.models.temporal_fusion_transformer import TemporalFusionTransformer, TFTConfig
from src.yahoo_data_loader import YahooDataLoader
from src.ticker_universe import TICKERS_50, get_diversified_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PurgedKFold:
    """Purged K-Fold Cross-Validation for Time Series.
    
    Unlike standard k-fold, this implementation:
    1. Maintains temporal ordering (no future data in training)
    2. Adds a "purge" gap between train and validation to prevent look-ahead bias
    3. Uses expanding or sliding windows for train set
    
    Based on: "Advances in Financial Machine Learning" (Marcos Lopez de Prado)
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 10, 
                 expanding: bool = True, min_train_size: int = 1000):
        """
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to skip between train and val (prevents look-ahead)
            expanding: If True, use expanding window (all past data). If False, sliding window.
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.expanding = expanding
        self.min_train_size = min_train_size
    
    def split(self, n_samples: int):
        """Generate train/val indices for each fold.
        
        Yields:
            (train_indices, val_indices) for each fold
        """
        # Calculate fold size
        fold_size = n_samples // (self.n_splits + 1)  # +1 to leave room for first train set
        
        for i in range(self.n_splits):
            # Validation fold starts at (i+1)/n_splits of the data
            val_start = (i + 1) * fold_size + self.min_train_size
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples - fold_size // 2:
                break
            
            # Training uses all data before validation minus purge gap
            if self.expanding:
                train_start = 0
            else:
                # Sliding window: same size as validation
                train_start = max(0, val_start - self.purge_gap - fold_size * 2)
            
            train_end = val_start - self.purge_gap
            
            if train_end <= train_start + self.min_train_size:
                continue
            
            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)
            
            yield train_indices, val_indices


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance with optional label smoothing.
    
    Down-weights easy examples, focuses on hard ones.
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, 
                 label_smoothing: float = 0.0, n_classes: int = 3):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                            self.label_smoothing / self.n_classes
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            if self.alpha is not None:
                log_probs = log_probs * self.alpha.unsqueeze(0)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def augment_sequence(seq: np.ndarray, noise_std: float = 0.01, 
                      time_warp_prob: float = 0.2) -> np.ndarray:
    """Apply data augmentation to a single sequence.
    
    Args:
        seq: (seq_len, n_features) array
        noise_std: Standard deviation of Gaussian noise to add
        time_warp_prob: Probability of applying time warping
    
    Returns:
        Augmented sequence
    """
    aug = seq.copy()
    
    # 1. Add small Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(*aug.shape).astype(np.float32) * noise_std
        aug = aug + noise
    
    # 2. Random time warping (slight temporal jitter)
    if np.random.random() < time_warp_prob:
        seq_len = len(aug)
        # Create slightly warped time indices
        orig_indices = np.arange(seq_len)
        warp = np.cumsum(np.random.uniform(0.9, 1.1, seq_len))
        warp = warp / warp[-1] * (seq_len - 1)
        # Interpolate features at warped positions
        for f in range(aug.shape[1]):
            aug[:, f] = np.interp(orig_indices, warp, aug[:, f])
    
    return aug


def augment_batch(sequences: np.ndarray, noise_std: float = 0.01,
                  time_warp_prob: float = 0.2, augment_prob: float = 0.5) -> np.ndarray:
    """Apply data augmentation to a batch of sequences.
    
    Args:
        sequences: (batch, seq_len, n_features) array
        noise_std: Standard deviation of Gaussian noise
        time_warp_prob: Probability of time warping per sequence
        augment_prob: Probability of augmenting each sequence at all
    
    Returns:
        Augmented sequences
    """
    augmented = sequences.copy()
    for i in range(len(augmented)):
        if np.random.random() < augment_prob:
            augmented[i] = augment_sequence(augmented[i], noise_std, time_warp_prob)
    return augmented


def load_multi_ticker_data(
    tickers: List[str],
    feature_config: FeatureConfig,
    period: str = "5y",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load and combine data from multiple tickers."""
    yahoo = YahooDataLoader()
    
    all_sequences = []
    all_targets = []
    all_returns = []
    feature_names = None
    
    for ticker in tickers:
        logger.info(f"Loading {ticker}...")
        try:
            df = yahoo.fetch_price_history(ticker, period=period)
            if df is None or df.empty or len(df) < 200:
                logger.warning(f"  Skipping {ticker}: insufficient data")
                continue
            
            seqs, targets, returns, names = prepare_enhanced_daily_data(df, feature_config)
            
            if len(seqs) == 0:
                logger.warning(f"  Skipping {ticker}: no sequences generated")
                continue
            
            all_sequences.append(seqs)
            all_targets.append(targets)
            all_returns.append(returns)
            
            if feature_names is None:
                feature_names = names
            
            logger.info(f"  {ticker}: {len(seqs)} sequences, {len(names)} features")
            
        except Exception as e:
            logger.error(f"  Error with {ticker}: {e}")
    
    if not all_sequences:
        raise ValueError("No data loaded from any ticker!")
    
    sequences = np.concatenate(all_sequences, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    returns = np.concatenate(all_returns, axis=0)
    
    logger.info(f"Total dataset: {len(sequences)} samples, {sequences.shape[2]} features")
    
    # Log class distribution
    unique, counts = np.unique(targets, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  Class {u}: {c} ({c/len(targets)*100:.1f}%)")
    
    return sequences, targets, returns, feature_names


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[CosineWarmupScheduler] = None,
    clip_grad: float = 1.0,
    augment_noise: float = 0.0,
    augment_prob: float = 0.0,
) -> Dict[str, float]:
    """Train one epoch with optional data augmentation."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_seqs, batch_targets, batch_returns in dataloader:
        # Apply data augmentation on CPU before moving to device
        if augment_noise > 0 and augment_prob > 0:
            batch_seqs_np = batch_seqs.numpy()
            batch_seqs_np = augment_batch(batch_seqs_np, noise_std=augment_noise,
                                          time_warp_prob=0.1, augment_prob=augment_prob)
            batch_seqs = torch.from_numpy(batch_seqs_np)
        
        batch_seqs = batch_seqs.to(device)
        batch_targets = batch_targets.to(device)
        batch_returns = batch_returns.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_seqs)
        
        # Classification loss
        policy_loss = criterion(outputs['action_logits'], batch_targets)
        
        # Value loss (predict actual returns)
        value_loss = nn.functional.mse_loss(outputs['value'], batch_returns)
        
        # Combined loss
        loss = policy_loss + 0.1 * value_loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * batch_seqs.size(0)
        predictions = outputs['action_logits'].argmax(dim=-1)
        total_correct += (predictions == batch_targets).sum().item()
        total_samples += batch_seqs.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for batch_seqs, batch_targets, batch_returns in dataloader:
        batch_seqs = batch_seqs.to(device)
        batch_targets = batch_targets.to(device)
        batch_returns = batch_returns.to(device)
        
        outputs = model(batch_seqs)
        
        policy_loss = criterion(outputs['action_logits'], batch_targets)
        value_loss = nn.functional.mse_loss(outputs['value'], batch_returns)
        loss = policy_loss + 0.1 * value_loss
        
        total_loss += loss.item() * batch_seqs.size(0)
        predictions = outputs['action_logits'].argmax(dim=-1)
        total_correct += (predictions == batch_targets).sum().item()
        total_samples += batch_seqs.size(0)
        
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(batch_targets.cpu().numpy())
    
    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    per_class_acc = {}
    unique_classes = np.unique(all_targets)
    for c in unique_classes:
        mask = all_targets == c
        if mask.sum() > 0:
            per_class_acc[int(c)] = float((all_preds[mask] == c).mean())
    
    # Prediction distribution
    pred_counts = np.bincount(all_preds, minlength=len(unique_classes))
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'per_class_accuracy': per_class_acc,
        'prediction_distribution': pred_counts.tolist(),
        'n_unique_preds': len(np.unique(all_preds)),
    }


def train(
    tickers: List[str],
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 5e-4,
    hidden_dim: int = 256,  # Bigger model for more data
    n_heads: int = 8,
    n_layers: int = 3,
    n_classes: int = 3,
    dropout: float = 0.5,  # Aggressive dropout
    patience: int = 20,
    val_split: float = 0.2,
    period: str = "10y",  # 10 years of history
    save_dir: str = "models/tft_v2",
    device_str: str = "auto",
    weight_decay: float = 1e-2,  # Aggressive weight decay
    label_smoothing: float = 0.1,
    augment_noise: float = 0.02,
    augment_prob: float = 0.3,
    stride: int = 5,  # Stride to reduce sequence overlap
):
    """Full training pipeline with regularization to combat overfitting."""
    
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Device: {device}")
    
    # Save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Feature config with stride
    feature_config = FeatureConfig(
        n_classes=n_classes,
        normalize=True,
        seq_len=60,
        forecast_horizon=5,
        stride=stride,  # Reduces sequence overlap
    )
    
    # Load data
    logger.info("Loading data...")
    sequences, targets, returns, feature_names = load_multi_ticker_data(
        tickers, feature_config, period
    )
    
    # Time-series split (no shuffling across time boundary)
    split_idx = int(len(sequences) * (1 - val_split))
    train_seqs, val_seqs = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    train_returns, val_returns = returns[:split_idx], returns[split_idx:]
    
    logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")
    
    # Compute class weights for focal loss
    unique, counts = np.unique(train_targets, return_counts=True)
    class_weights = torch.FloatTensor(len(train_targets) / (len(unique) * counts)).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_seqs),
        torch.from_numpy(train_targets),
        torch.from_numpy(train_returns),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_seqs),
        torch.from_numpy(val_targets),
        torch.from_numpy(val_returns),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    input_dim = sequences.shape[2]
    model_config = TFTConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
        seq_len=60,
        dropout=dropout,
    )
    model = TemporalFusionTransformer(model_config).to(device)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function with label smoothing
    criterion = FocalLoss(gamma=2.0, alpha=class_weights, 
                          label_smoothing=label_smoothing, n_classes=n_classes)
    
    # Optimizer with configurable weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    total_steps = epochs * len(train_loader)
    warmup_steps = min(len(train_loader) * 5, total_steps // 10)  # 5 epoch warmup
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
    logger.info(f"Model: TFT, hidden={hidden_dim}, heads={n_heads}, layers={n_layers}, dropout={dropout}")
    logger.info(f"Features: {input_dim}, Classes: {n_classes}")
    logger.info(f"Regularization: weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    logger.info(f"Data augmentation: noise={augment_noise}, prob={augment_prob}")
    logger.info(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train with data augmentation
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler,
                                    augment_noise=augment_noise, augment_prob=augment_prob)
        
        # Validate (no augmentation)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Record
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(scheduler.get_lr())
        
        # Log
        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} | "
            f"LR={scheduler.get_lr():.6f} | "
            f"{elapsed:.1f}s"
        )
        
        # Per-class breakdown every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Per-class val acc: {val_metrics['per_class_accuracy']}")
            logger.info(f"  Prediction dist:   {val_metrics['prediction_distribution']}")
            logger.info(f"  Unique preds:      {val_metrics['n_unique_preds']}")
        
        # Early stopping on validation accuracy (not loss, which is affected by label smoothing)
        # Also require model to use at least 2 classes (not collapsed)
        is_not_collapsed = val_metrics['n_unique_preds'] >= 2
        
        if val_metrics['accuracy'] > best_val_acc and is_not_collapsed:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), save_path / "tft_best.pt")
            logger.info(f"  ★ New best model saved (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, n_classes={val_metrics['n_unique_preds']})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                break
    
    total_time = time.time() - start_time
    
    # Save final model + config
    torch.save(model.state_dict(), save_path / "tft_final.pt")
    
    config_dict = {
        'model_type': 'TemporalFusionTransformer',
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'n_classes': n_classes,
        'dropout': dropout,
        'seq_len': 60,
        'forecast_horizon': 5,
        'n_parameters': model.count_parameters(),
        'tickers': tickers,
        'feature_names': feature_names,
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_acc,
        'training_time_seconds': total_time,
        'batch_size': batch_size,
        'lr': lr,
        # Regularization parameters
        'weight_decay': weight_decay,
        'label_smoothing': label_smoothing,
        'augment_noise': augment_noise,
        'augment_prob': augment_prob,
        'date': datetime.now().isoformat(),
    }
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {save_path}")
    
    # Final evaluation
    model.load_state_dict(torch.load(save_path / "tft_best.pt", weights_only=True))
    final_metrics = evaluate(model, val_loader, criterion, device)
    logger.info(f"\nFinal evaluation (best checkpoint):")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"  Per-class: {final_metrics['per_class_accuracy']}")
    logger.info(f"  Pred dist: {final_metrics['prediction_distribution']}")
    logger.info(f"  Unique predictions: {final_metrics['n_unique_preds']}")
    
    return history


def train_cv(
    tickers: List[str],
    n_folds: int = 5,
    purge_gap: int = 20,  # Skip 20 days between train and val to prevent look-ahead
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 5e-4,
    hidden_dim: int = 256,
    n_heads: int = 8,
    n_layers: int = 3,
    n_classes: int = 3,
    dropout: float = 0.5,
    patience: int = 15,
    period: str = "10y",
    save_dir: str = "models/tft_v2_cv",
    device_str: str = "auto",
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    augment_noise: float = 0.02,
    augment_prob: float = 0.3,
    stride: int = 5,
):
    """Train with Purged K-Fold Cross-Validation.
    
    This trains multiple models (one per fold) and reports average metrics.
    Final model is the best fold's model.
    """
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Device: {device}")
    logger.info(f"Starting Purged {n_folds}-Fold Cross-Validation (purge_gap={purge_gap})")
    
    # Save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Feature config with stride
    feature_config = FeatureConfig(
        n_classes=n_classes,
        normalize=True,
        seq_len=60,
        forecast_horizon=5,
        stride=stride,
    )
    
    # Load ALL data once
    logger.info("Loading data...")
    sequences, targets, returns, feature_names = load_multi_ticker_data(
        tickers, feature_config, period
    )
    
    logger.info(f"Total samples: {len(sequences)}")
    
    # Initialize CV
    cv = PurgedKFold(n_splits=n_folds, purge_gap=purge_gap, expanding=True)
    
    fold_results = []
    best_fold_acc = 0
    best_fold_idx = 0
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(len(sequences))):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*60}")
        logger.info(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        
        # Split data
        train_seqs = sequences[train_idx]
        train_targets = targets[train_idx]
        train_returns = returns[train_idx]
        val_seqs = sequences[val_idx]
        val_targets = targets[val_idx]
        val_returns = returns[val_idx]
        
        # Compute class weights
        unique, counts = np.unique(train_targets, return_counts=True)
        class_weights = torch.FloatTensor(len(train_targets) / (len(unique) * counts)).to(device)
        
        # Data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(train_seqs),
            torch.from_numpy(train_targets),
            torch.from_numpy(train_returns),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_seqs),
            torch.from_numpy(val_targets),
            torch.from_numpy(val_returns),
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Create fresh model for this fold
        input_dim = sequences.shape[2]
        model_config = TFTConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=n_classes,
            seq_len=60,
            dropout=dropout,
        )
        model = TemporalFusionTransformer(model_config).to(device)
        
        # Loss and optimizer
        criterion = FocalLoss(gamma=2.0, alpha=class_weights, 
                              label_smoothing=label_smoothing, n_classes=n_classes)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        total_steps = epochs * len(train_loader)
        warmup_steps = min(len(train_loader) * 3, total_steps // 10)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
        
        # Training loop for this fold
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train with augmentation
            train_seqs_aug = augment_batch(
                train_seqs, 
                noise_std=augment_noise, 
                augment_prob=augment_prob
            )
            aug_dataset = TensorDataset(
                torch.from_numpy(train_seqs_aug),
                torch.from_numpy(train_targets),
                torch.from_numpy(train_returns),
            )
            aug_loader = DataLoader(aug_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            
            train_metrics = train_epoch(model, aug_loader, optimizer, criterion, device, scheduler)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc and val_metrics['n_unique_preds'] >= 2:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                torch.save(model.state_dict(), save_path / f"tft_fold{fold_idx}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Fold {fold_idx+1} Epoch {epoch+1}: "
                           f"train_acc={train_metrics['accuracy']:.4f}, "
                           f"val_acc={val_metrics['accuracy']:.4f}")
        
        # Final evaluation for this fold
        model.load_state_dict(torch.load(save_path / f"tft_fold{fold_idx}_best.pt", weights_only=True))
        final_metrics = evaluate(model, val_loader, criterion, device)
        
        fold_results.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'val_accuracy': final_metrics['accuracy'],
            'per_class_accuracy': final_metrics['per_class_accuracy'],
            'n_unique_preds': final_metrics['n_unique_preds'],
        })
        
        logger.info(f"Fold {fold_idx+1} Final: val_acc={final_metrics['accuracy']:.4f}")
        
        if final_metrics['accuracy'] > best_fold_acc:
            best_fold_acc = final_metrics['accuracy']
            best_fold_idx = fold_idx
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    accs = [r['val_accuracy'] for r in fold_results]
    logger.info(f"Fold accuracies: {[f'{a:.4f}' for a in accs]}")
    logger.info(f"Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"Best fold: {best_fold_idx+1} with accuracy {best_fold_acc:.4f}")
    
    # Copy best fold model as final model
    import shutil
    shutil.copy(save_path / f"tft_fold{best_fold_idx}_best.pt", save_path / "tft_best.pt")
    
    # Save CV results
    cv_results = {
        'n_folds': n_folds,
        'purge_gap': purge_gap,
        'fold_results': fold_results,
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'best_fold': best_fold_idx,
        'config': {
            'hidden_dim': hidden_dim,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'stride': stride,
            'tickers': tickers,
        }
    }
    
    with open(save_path / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    return cv_results


def main():
    parser = argparse.ArgumentParser(description="Train TFT v2 model with regularization")
    parser.add_argument("--tickers", nargs="+", 
                       default=None,  # Will use TICKERS_50 if not specified
                       help="Tickers to train on (default: diversified 50 from ticker_universe)")
    parser.add_argument("--n-tickers", type=int, default=50, help="Number of diversified tickers if --tickers not specified")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension (increased for more data)")
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5, help="Aggressive dropout for regularization")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--period", type=str, default="10y", help="10 years of history")
    parser.add_argument("--save-dir", type=str, default="models/tft_v2")
    parser.add_argument("--device", type=str, default="auto")
    # Regularization parameters
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Aggressive weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--augment-noise", type=float, default=0.02, help="Gaussian noise std for augmentation")
    parser.add_argument("--augment-prob", type=float, default=0.3, help="Probability of augmenting each sample")
    # Stride for reducing sequence overlap
    parser.add_argument("--stride", type=int, default=5, help="Stride for sequence generation (reduces overlap)")
    # Cross-validation
    parser.add_argument("--cv", action="store_true", help="Use purged k-fold cross-validation")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--purge-gap", type=int, default=20, help="Gap between train and val in CV (days)")
    
    args = parser.parse_args()
    
    # Use diversified tickers if not specified
    if args.tickers is None:
        if args.n_tickers >= 50:
            tickers = TICKERS_50
        else:
            tickers = get_diversified_sample(args.n_tickers)
        logger.info(f"Using {len(tickers)} diversified tickers from ticker_universe")
    else:
        tickers = args.tickers
    
    if args.cv:
        # Use purged k-fold cross-validation
        train_cv(
            tickers=tickers,
            n_folds=args.n_folds,
            purge_gap=args.purge_gap,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_classes=args.n_classes,
            dropout=args.dropout,
            patience=args.patience,
            period=args.period,
            save_dir=args.save_dir + "_cv",
            device_str=args.device,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            augment_noise=args.augment_noise,
            augment_prob=args.augment_prob,
            stride=args.stride,
        )
    else:
        # Standard training with simple train/val split
        train(
            tickers=tickers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_classes=args.n_classes,
            dropout=args.dropout,
            patience=args.patience,
            period=args.period,
            save_dir=args.save_dir,
            device_str=args.device,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            augment_noise=args.augment_noise,
            augment_prob=args.augment_prob,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()