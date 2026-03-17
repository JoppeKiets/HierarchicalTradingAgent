#!/usr/bin/env python3
"""News data loading for the News Encoder sub-model.

Provides:
  - preprocess_news_ticker():  Build per-ticker news sequences from cached FinBERT embeddings
  - LazyNewsDataset:           PyTorch Dataset for news sequences (mmap'd, lazy)
  - create_news_dataloaders(): Convenience function

The news data pipeline mirrors the daily pipeline:
  - Same temporal split boundaries
  - Same stride / sequence length
  - Aligned by ordinal dates for meta-model fusion

Depends on: scripts/preprocess_news_embeddings.py having run first.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class NewsDataConfig:
    """Configuration for news data pipeline."""
    # Paths
    news_embedding_dir: str = "data/feature_cache/news"  # From preprocess_news_embeddings.py
    news_seq_cache_dir: str = "data/feature_cache/news_sequences"  # Processed sequences
    organized_dir: str = "data/organized"

    # Sequence parameters (should match daily config)
    seq_len: int = 720
    stride: int = 5
    forecast_horizon: int = 1

    # Embedding dimensions
    finbert_dim: int = 768
    sentiment_dim: int = 6
    total_dim: int = 774  # finbert_dim + sentiment_dim

    # Normalization
    normalize_embeddings: bool = True
    norm_window: int = 60

    # News lag (prevent look-ahead)
    news_lag_days: int = 1

    # Split (mirror daily config)
    split_mode: str = "temporal"
    temporal_train_frac: float = 0.70
    temporal_val_frac: float = 0.15
    temporal_test_frac: float = 0.15


def preprocess_news_ticker(
    ticker: str,
    news_cfg: NewsDataConfig,
    daily_cache_dir: str = "data/feature_cache/daily",
    force: bool = False,
) -> Optional[Dict]:
    """Build news feature sequences aligned to daily price dates.

    For each trading day in the daily cache:
      - Look up FinBERT embedding + sentiment for that day (lagged by news_lag_days)
      - If no news, fill with zeros
      - Save as (N_daily_rows, 774) array aligned 1:1 with daily features/targets

    This means the news sequence dataset has the exact same row indexing
    as the daily dataset, making alignment trivial.

    Saves:
        {news_seq_cache_dir}/{ticker}_features.npy   (T, 774)
        {news_seq_cache_dir}/{ticker}_dates.npy       (T,) ordinal dates (copied from daily)
    """
    cache = Path(news_cfg.news_seq_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    feat_path = cache / f"{ticker}_features.npy"
    date_path = cache / f"{ticker}_dates.npy"

    if not force and feat_path.exists() and date_path.exists():
        feat = np.load(feat_path, mmap_mode="r")
        return {"ticker": ticker, "n_rows": feat.shape[0], "n_features": feat.shape[1]}

    # Load daily dates (these define the timeline)
    daily_date_path = Path(daily_cache_dir) / f"{ticker}_dates.npy"
    if not daily_date_path.exists():
        return None
    daily_dates = np.load(daily_date_path, mmap_mode="r")  # ordinal dates
    n_daily = len(daily_dates)

    # Load cached FinBERT embeddings
    emb_dir = Path(news_cfg.news_embedding_dir)
    emb_path_src = emb_dir / f"{ticker}_embeddings.npy"
    sent_path_src = emb_dir / f"{ticker}_sentiment.npy"
    news_date_path = emb_dir / f"{ticker}_dates.npy"

    if not emb_path_src.exists() or not sent_path_src.exists() or not news_date_path.exists():
        # No news embeddings → fill with zeros
        features = np.zeros((n_daily, news_cfg.total_dim), dtype=np.float32)
        np.save(feat_path, features)
        np.save(date_path, daily_dates.copy())
        return {"ticker": ticker, "n_rows": n_daily, "n_features": news_cfg.total_dim, "has_news": False}

    news_embs = np.load(emb_path_src, mmap_mode="r")     # (n_news_days, 768)
    news_sents = np.load(sent_path_src, mmap_mode="r")    # (n_news_days, 6)
    news_dates = np.load(news_date_path, mmap_mode="r")   # (n_news_days,) ordinal

    # Build lookup: ordinal_date → (embedding, sentiment)
    news_lookup = {}
    for i in range(len(news_dates)):
        d = int(news_dates[i])
        news_lookup[d] = (news_embs[i].copy(), news_sents[i].copy())

    # Align to daily dates with lag
    features = np.zeros((n_daily, news_cfg.total_dim), dtype=np.float32)
    lag = news_cfg.news_lag_days
    n_matched = 0

    for row_idx in range(n_daily):
        # Look up news from `lag` days ago
        target_date = int(daily_dates[row_idx]) - lag
        if target_date in news_lookup:
            emb, sent = news_lookup[target_date]
            features[row_idx, :news_cfg.finbert_dim] = emb
            features[row_idx, news_cfg.finbert_dim:] = sent
            n_matched += 1

    # Optional: normalize embeddings (per-feature rolling z-score)
    if news_cfg.normalize_embeddings:
        features = _normalize_news_features(features, news_cfg)

    np.save(feat_path, features)
    np.save(date_path, daily_dates.copy())

    return {
        "ticker": ticker,
        "n_rows": n_daily,
        "n_features": news_cfg.total_dim,
        "has_news": True,
        "n_matched_days": n_matched,
        "match_rate": f"{n_matched / n_daily:.1%}" if n_daily > 0 else "0%",
    }


def _normalize_news_features(features: np.ndarray, cfg: NewsDataConfig) -> np.ndarray:
    """Normalize news features with rolling z-score.

    Only normalizes non-zero rows (days with news). Zero rows stay zero.
    This preserves the "no news" signal.
    """
    out = np.zeros_like(features)
    has_news = np.abs(features).sum(axis=1) > 1e-6

    for j in range(features.shape[1]):
        s = pd.Series(features[:, j])
        mu = s.rolling(cfg.norm_window, min_periods=1).mean()
        std = s.rolling(cfg.norm_window, min_periods=2).std().fillna(1.0).clip(lower=1e-8)
        normalized = ((s - mu) / std).clip(-5, 5).fillna(0).values
        # Only apply normalization to days with news
        out[:, j] = np.where(has_news, normalized, 0.0)

    return out.astype(np.float32)


def preprocess_all_news(
    tickers: List[str],
    news_cfg: NewsDataConfig,
    daily_cache_dir: str = "data/feature_cache/daily",
    force: bool = False,
) -> Dict:
    """Preprocess news sequences for all tickers."""
    meta = {}
    t0 = time.time()

    logger.info(f"Preprocessing news sequences for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        info = preprocess_news_ticker(ticker, news_cfg, daily_cache_dir, force=force)
        if info:
            meta[ticker] = info
        if (i + 1) % 100 == 0:
            logger.info(f"  News preprocessing: {i+1}/{len(tickers)} done")

    n_with_news = sum(1 for v in meta.values() if v.get("has_news", False))
    logger.info(f"  News sequences: {len(meta)} tickers total, {n_with_news} with news, "
                f"{time.time() - t0:.0f}s")
    return meta


# ============================================================================
# LazyNewsDataset
# ============================================================================

class LazyNewsDataset(Dataset):
    """Lazy-loading dataset for news sequences.

    Mirrors LazyDailyDataset: same tickers, same temporal bounds, same stride.
    Each item is a (seq_len, 774) news embedding sequence + target + date.

    The targets come from the daily cache (same as daily models), ensuring
    the news model predicts the same thing as daily models.
    """

    def __init__(self, tickers: List[str], news_cfg: NewsDataConfig,
                 daily_cache_dir: str = "data/feature_cache/daily",
                 split_name: str = "train"):
        self.news_cfg = news_cfg
        self.cache = Path(news_cfg.news_seq_cache_dir)
        self.daily_cache = Path(daily_cache_dir)
        self.split_name = split_name

        self.index: List[Tuple[str, int]] = []
        self.n_features = news_cfg.total_dim

        warmup = max(news_cfg.seq_len, news_cfg.norm_window, 60)

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            daily_tgt_path = self.daily_cache / f"{ticker}_targets.npy"
            if not feat_path.exists() or not daily_tgt_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            n_rows = feat.shape[0]
            tgt = np.load(daily_tgt_path, mmap_mode="r")
            end = n_rows - news_cfg.forecast_horizon

            # Temporal boundaries
            lo, hi = self._temporal_bounds(warmup, end, news_cfg)

            for i in range(lo, hi, news_cfg.stride):
                if i - news_cfg.seq_len >= 0 and not np.isnan(tgt[i]):
                    self.index.append((ticker, i))

        logger.info(f"LazyNewsDataset[{split_name}]: {len(self.index):,} sequences "
                    f"from {len(tickers)} tickers, {self.n_features} features")

    def _temporal_bounds(self, warmup: int, end: int, cfg: NewsDataConfig) -> Tuple[int, int]:
        if cfg.split_mode != "temporal":
            return warmup, end
        usable = end - warmup
        if usable <= 0:
            return warmup, warmup
        train_end = warmup + int(usable * cfg.temporal_train_frac)
        val_end = train_end + int(usable * cfg.temporal_val_frac)
        if self.split_name == "train":
            return warmup, train_end
        elif self.split_name == "val":
            return train_end, val_end
        else:
            return val_end, end

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ticker, row = self.index[idx]

        feat = np.load(self.cache / f"{ticker}_features.npy", mmap_mode="r")
        tgt = np.load(self.daily_cache / f"{ticker}_targets.npy", mmap_mode="r")

        seq = feat[row - self.news_cfg.seq_len: row].copy()
        target = float(tgt[row])

        # Load ordinal date for alignment
        date_path = self.cache / f"{ticker}_dates.npy"
        if date_path.exists():
            dates = np.load(date_path, mmap_mode="r")
            ordinal_date = int(dates[row])
        else:
            ordinal_date = 0

        return (
            torch.from_numpy(seq),
            torch.tensor(target, dtype=torch.float32),
            ordinal_date,
            ticker,
        )


def create_news_dataloaders(
    splits: Dict[str, List[str]],
    news_cfg: NewsDataConfig,
    daily_cache_dir: str = "data/feature_cache/daily",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict:
    """Create news DataLoaders for all splits.

    Returns:
        {
            'train': DataLoader,
            'val': DataLoader,
            'test': DataLoader,
            'n_features': int,
        }
    """
    result = {}

    for split_name in ["train", "val", "test"]:
        tickers = splits[split_name]
        shuffle = split_name == "train"

        ds = LazyNewsDataset(
            tickers, news_cfg,
            daily_cache_dir=daily_cache_dir,
            split_name=split_name,
        )

        result[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle if len(ds) > 0 else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle if len(ds) > 0 else False,
        )

    result["n_features"] = news_cfg.total_dim

    return result
