"""Metrics and utility functions for hierarchical forecaster training."""

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


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


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
