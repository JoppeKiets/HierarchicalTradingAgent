#!/usr/bin/env python3
"""Memory-efficient data loading for hierarchical model training.

Strategy:
  Phase A (preprocess, once):
    For each ticker, compute features and save as .npy to a cache directory.

  Phase B (training, lazy):
    A PyTorch Dataset reads from mmap'd .npy files, loading one sequence
    at a time.  This uses ~0 MB resident RAM regardless of dataset size.

Handles:
  - Reproducible 80/10/10 train/val/test split BY TICKER
  - Daily data from data/organized/{TICKER}/price_history.csv
  - Minute data from data/minute_history/{TICKER}.parquet
  - Feature computation for both timescales
"""

import json
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


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HierarchicalDataConfig:
    """Configuration for the hierarchical data pipeline."""

    # Paths
    organized_dir: str = "data/organized"
    minute_dir: str = "data/minute_history"
    cache_dir: str = "data/feature_cache"

    # Ticker selection
    min_daily_rows: int = 750
    min_minute_rows: int = 780     # ~2 trading days (down from 1170)
    max_tickers: int = 0

    # Sequence parameters
    daily_seq_len: int = 720
    minute_seq_len: int = 780      # ~2 trading days (down from 1170)

    # Stride controls dataset size
    daily_stride: int = 5
    minute_stride: int = 30        # smaller stride → more minute sequences

    # Target
    forecast_horizon: int = 1

    # Split ratios (by ticker count)
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10
    split_seed: int = 42

    # Split mode: "ticker" (original) or "temporal" (recommended)
    split_mode: str = "temporal"
    # For temporal split: fraction of each ticker's timeline
    temporal_train_frac: float = 0.70
    temporal_val_frac: float = 0.15
    temporal_test_frac: float = 0.15

    # Feature config for daily data
    daily_normalize: bool = True
    daily_norm_window: int = 60

    # Minute
    minute_normalize: bool = True
    minute_norm_window: int = 390

    # News features
    include_news_features: bool = True
    news_lag_days: int = 1  # shift by 1 day to avoid look-ahead leakage


# ============================================================================
# Ticker discovery and split
# ============================================================================

def get_viable_tickers(cfg: HierarchicalDataConfig) -> List[str]:
    """Find tickers with sufficient daily AND minute data."""
    organized = Path(cfg.organized_dir)
    minute = Path(cfg.minute_dir)

    viable = []
    for ticker_dir in sorted(organized.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        price_file = ticker_dir / "price_history.csv"
        minute_file = minute / f"{ticker}.parquet"

        if not price_file.exists() or not minute_file.exists():
            continue

        daily_rows = sum(1 for _ in open(price_file)) - 1
        if daily_rows < cfg.min_daily_rows:
            continue

        try:
            mdf = pd.read_parquet(minute_file)
            if len(mdf) < cfg.min_minute_rows:
                continue
        except Exception:
            continue

        viable.append(ticker)

    if cfg.max_tickers > 0:
        viable = viable[: cfg.max_tickers]

    logger.info(f"Found {len(viable)} viable tickers "
                f"(daily>={cfg.min_daily_rows}, minute>={cfg.min_minute_rows})")
    return viable


def split_tickers(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
) -> Dict[str, List[str]]:
    """Reproducible 80/10/10 split by company (ticker mode) or return all
    tickers for every split (temporal mode, splitting happens inside datasets)."""

    if cfg.split_mode == "temporal":
        # In temporal mode every ticker appears in every split; the
        # datasets themselves enforce the time boundaries.
        split = {
            "train": sorted(tickers),
            "val":   sorted(tickers),
            "test":  sorted(tickers),
        }
        logger.info(f"Temporal split mode: {len(tickers)} tickers in all splits "
                    f"(train={cfg.temporal_train_frac:.0%}, "
                    f"val={cfg.temporal_val_frac:.0%}, "
                    f"test={cfg.temporal_test_frac:.0%})")
        return split

    # --- Original ticker-based split ---
    rng = np.random.RandomState(cfg.split_seed)
    shuffled = list(tickers)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    split = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }
    logger.info(f"Ticker split: train={len(split['train'])}, "
                f"val={len(split['val'])}, test={len(split['test'])}")
    return split


# ============================================================================
# Feature computation helpers
# ============================================================================

def _compute_daily_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Compute ~51 daily features from OHLCV."""
    from src.enhanced_features import (
        FeatureConfig,
        compute_returns_features,
        compute_volatility_features,
        compute_trend_features,
        compute_momentum_features,
        compute_volume_features,
        compute_microstructure_features,
        compute_calendar_features,
    )
    cfg = FeatureConfig()
    parts = [
        compute_returns_features(df),
        compute_volatility_features(df, cfg),
        compute_trend_features(df, cfg),
        compute_momentum_features(df, cfg),
        compute_volume_features(df),
        compute_microstructure_features(df),
        compute_calendar_features(df),
    ]
    features = pd.concat(parts, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features.values.astype(np.float32), list(features.columns)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Parse timestamps robustly and return timezone-naive datetime."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert(None)
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def _compute_text_sentiment_proxy(text: str) -> float:
    """Lightweight lexicon-based sentiment proxy in [-1, 1]."""
    if not isinstance(text, str) or not text:
        return 0.0

    bullish_terms = [
        "beat", "strong", "growth", "upgrade", "bull", "outperform",
        "surge", "record", "buy", "positive", "gain", "profit",
    ]
    bearish_terms = [
        "miss", "weak", "downgrade", "bear", "underperform", "drop",
        "fall", "risk", "negative", "loss", "lawsuit", "concern",
    ]

    low = text.lower()
    pos = sum(low.count(term) for term in bullish_terms)
    neg = sum(low.count(term) for term in bearish_terms)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def _build_lagged_news_features(
    ticker: str,
    cfg: HierarchicalDataConfig,
    dates: pd.Series,
) -> Tuple[np.ndarray, List[str]]:
    """Build daily lagged news features aligned to provided dates.

    Features are aggregated per calendar day, then shifted by `news_lag_days`
    to prevent future leakage.
    """
    feature_names = [
        "news_count_lag1",
        "news_title_len_mean_lag1",
        "news_article_len_mean_lag1",
        "news_sentiment_mean_lag1",
        "news_sentiment_std_lag1",
    ]

    if not cfg.include_news_features:
        return np.zeros((len(dates), len(feature_names)), dtype=np.float32), feature_names

    news_file = Path(cfg.organized_dir) / ticker / "news_articles.csv"
    if not news_file.exists():
        return np.zeros((len(dates), len(feature_names)), dtype=np.float32), feature_names

    try:
        ndf = pd.read_csv(news_file)
    except Exception:
        return np.zeros((len(dates), len(feature_names)), dtype=np.float32), feature_names

    if ndf.empty:
        return np.zeros((len(dates), len(feature_names)), dtype=np.float32), feature_names

    date_col = next((c for c in ["Date", "date", "timestamp"] if c in ndf.columns), None)
    title_col = next((c for c in ["Article_title", "title", "Title"] if c in ndf.columns), None)
    body_col = next((c for c in ["Article", "text", "Text"] if c in ndf.columns), None)

    if date_col is None:
        return np.zeros((len(dates), len(feature_names)), dtype=np.float32), feature_names

    ndf["date"] = _safe_to_datetime(ndf[date_col]).dt.date
    ndf = ndf.dropna(subset=["date"])
    if ndf.empty:
        return np.zeros((len(dates), len(feature_names)), dtype=np.float32), feature_names

    ndf["title"] = ndf[title_col].astype(str) if title_col else ""
    ndf["body"] = ndf[body_col].astype(str) if body_col else ndf["title"]
    ndf["title_len"] = ndf["title"].str.len().astype(float)
    ndf["article_len"] = ndf["body"].str.len().astype(float)
    ndf["sentiment_proxy"] = (ndf["title"].fillna("") + " " + ndf["body"].fillna("")).map(
        _compute_text_sentiment_proxy
    )

    daily = ndf.groupby("date", as_index=False).agg(
        news_count=("title", "size"),
        news_title_len_mean=("title_len", "mean"),
        news_article_len_mean=("article_len", "mean"),
        news_sentiment_mean=("sentiment_proxy", "mean"),
        news_sentiment_std=("sentiment_proxy", "std"),
    )
    daily["news_sentiment_std"] = daily["news_sentiment_std"].fillna(0.0)

    target_dates = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce").dt.date})
    merged = target_dates.merge(daily, on="date", how="left").fillna(0.0)

    for col in [
        "news_count",
        "news_title_len_mean",
        "news_article_len_mean",
        "news_sentiment_mean",
        "news_sentiment_std",
    ]:
        merged[col] = merged[col].shift(cfg.news_lag_days).fillna(0.0)

    arr = merged[
        [
            "news_count",
            "news_title_len_mean",
            "news_article_len_mean",
            "news_sentiment_mean",
            "news_sentiment_std",
        ]
    ].values.astype(np.float32)
    return arr, feature_names


# ============================================================================
# Regime features  (market-wide signals for the MetaMLP)
# ============================================================================

_REGIME_CACHE: Optional[pd.DataFrame] = None  # module-level cache


def _build_regime_dataframe(cfg: HierarchicalDataConfig) -> pd.DataFrame:
    """Build a date-indexed DataFrame of market-wide regime features.

    Uses data from SPY, UVXY/VIXY, TLT, GLD already present in
    data/organized/.  Features are lagged by 1 day to prevent leakage
    and rolling-z-score normalised.

    Returns a DataFrame indexed by ``datetime.date`` with 8 columns.
    """
    global _REGIME_CACHE
    if _REGIME_CACHE is not None:
        return _REGIME_CACHE

    organized = Path(cfg.organized_dir)
    feature_names = [
        "regime_spy_ret5",       # SPY 5-day return (market trend)
        "regime_spy_vol20",      # SPY 20-day realised volatility
        "regime_vix_level",      # VIX proxy level (UVXY / VIXY)
        "regime_vix_ret5",       # VIX proxy 5-day change
        "regime_tlt_ret5",       # TLT 5-day return (rates trend)
        "regime_gld_ret5",       # GLD 5-day return (risk-off proxy)
        "regime_spy_breadth",    # SPY distance from 50-day SMA
        "regime_corr_spy_tlt",   # Rolling correlation SPY vs TLT
    ]

    def _load_close(ticker: str) -> Optional[pd.Series]:
        f = organized / ticker / "price_history.csv"
        if not f.exists():
            return None
        try:
            d = pd.read_csv(f, usecols=["date", "close"])
            d["date"] = pd.to_datetime(d["date"]).dt.date
            return d.set_index("date")["close"].sort_index()
        except Exception:
            return None

    spy = _load_close("SPY")
    # Try UVXY first, fall back to VIXY
    vix = _load_close("UVXY")
    if vix is None:
        vix = _load_close("VIXY")
    tlt = _load_close("TLT")
    gld = _load_close("GLD")

    if spy is None:
        logger.warning("SPY data not found — regime features will be zeros")
        _REGIME_CACHE = pd.DataFrame()
        return _REGIME_CACHE

    df = pd.DataFrame(index=spy.index)
    df["spy"] = spy
    if vix is not None:
        df["vix"] = vix
    if tlt is not None:
        df["tlt"] = tlt
    if gld is not None:
        df["gld"] = gld
    df = df.sort_index().ffill()

    # Compute features (all backward-looking)
    df["regime_spy_ret5"] = df["spy"].pct_change(5, fill_method=None)
    df["regime_spy_vol20"] = df["spy"].pct_change(fill_method=None).rolling(20, min_periods=5).std()
    df["regime_vix_level"] = df.get("vix", pd.Series(0, index=df.index))
    df["regime_vix_ret5"] = df.get("vix", pd.Series(0, index=df.index)).pct_change(5, fill_method=None)
    df["regime_tlt_ret5"] = df.get("tlt", pd.Series(0, index=df.index)).pct_change(5, fill_method=None)
    df["regime_gld_ret5"] = df.get("gld", pd.Series(0, index=df.index)).pct_change(5, fill_method=None)
    sma50 = df["spy"].rolling(50, min_periods=10).mean()
    df["regime_spy_breadth"] = (df["spy"] - sma50) / (sma50 + 1e-8)
    if tlt is not None:
        spy_ret = df["spy"].pct_change(fill_method=None)
        tlt_ret = df["tlt"].pct_change(fill_method=None)
        df["regime_corr_spy_tlt"] = spy_ret.rolling(20, min_periods=5).corr(tlt_ret)
    else:
        df["regime_corr_spy_tlt"] = 0.0

    regime = df[feature_names].copy()
    regime = regime.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Rolling z-score normalisation (60-day window)
    for col in feature_names:
        s = regime[col]
        mu = s.rolling(60, min_periods=5).mean()
        std = s.rolling(60, min_periods=5).std().clip(lower=1e-8)
        regime[col] = ((s - mu) / std).clip(-5, 5).fillna(0.0)

    # Lag by 1 day to prevent leakage
    regime = regime.shift(1).fillna(0.0)

    _REGIME_CACHE = regime
    logger.info(f"Regime features built: {len(regime)} dates, {len(feature_names)} features")
    return _REGIME_CACHE


REGIME_FEATURE_NAMES: List[str] = [
    "regime_spy_ret5", "regime_spy_vol20", "regime_vix_level",
    "regime_vix_ret5", "regime_tlt_ret5", "regime_gld_ret5",
    "regime_spy_breadth", "regime_corr_spy_tlt",
]


def get_regime_vector(date, cfg: HierarchicalDataConfig) -> np.ndarray:
    """Get the 8-dim regime vector for a single date."""
    regime_df = _build_regime_dataframe(cfg)
    if regime_df.empty or date not in regime_df.index:
        return np.zeros(len(REGIME_FEATURE_NAMES), dtype=np.float32)
    return regime_df.loc[date].values.astype(np.float32)


def _normalize_array(arr: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling z-score normalization per feature, clipped at +/-5."""
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        s = pd.Series(arr[:, j])
        mu = s.rolling(window, min_periods=1).mean()
        std = s.rolling(window, min_periods=2).std().fillna(1.0).clip(lower=1e-8)
        out[:, j] = ((s - mu) / std).clip(-5, 5).fillna(0).values
    return out.astype(np.float32)


def _compute_minute_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build minute features from parquet (OHLCV + pre-computed technicals)."""
    features = pd.DataFrame(index=df.index)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)

    features["return_1m"] = pd.Series(close).pct_change(fill_method=None).fillna(0).values
    features["log_volume"] = np.log1p(vol)
    features["spread"] = (high - low) / np.clip(close, 1e-8, None)
    features["vwap_proxy"] = ((high + low + close) / 3 - close) / np.clip(close, 1e-8, None)

    tech_cols = [
        "rsi", "macd", "macd_signal", "macd_diff",
        "ema_5", "ema_12", "ema_26",
        "bb_upper", "bb_lower", "bb_mid", "atr", "adx",
    ]
    for col in tech_cols:
        if col in df.columns:
            vals = df[col].values.astype(np.float64)
            if col.startswith("ema_") or col.startswith("bb_"):
                vals = (vals - close) / np.clip(close, 1e-8, None)
            features[col] = vals

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute - 9 * 60 - 30
        features["tod_sin"] = np.sin(2 * np.pi * minute_of_day.values / 390.0)
        features["tod_cos"] = np.cos(2 * np.pi * minute_of_day.values / 390.0)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features.values.astype(np.float32), list(features.columns)


# ============================================================================
# Preprocessing: save per-ticker features + targets to disk
# ============================================================================

def preprocess_daily_ticker(
    ticker: str,
    cfg: HierarchicalDataConfig,
    force: bool = False,
) -> Optional[Dict]:
    """Compute daily features + targets for one ticker, save to cache.

    Saves:
        {cache_dir}/daily/{ticker}_features.npy   (T, F)
        {cache_dir}/daily/{ticker}_targets.npy     (T,)
        {cache_dir}/daily/{ticker}_dates.npy       (T,)  ordinal dates for alignment
    """
    cache = Path(cfg.cache_dir) / "daily"
    cache.mkdir(parents=True, exist_ok=True)

    feat_path = cache / f"{ticker}_features.npy"
    tgt_path = cache / f"{ticker}_targets.npy"
    date_path = cache / f"{ticker}_dates.npy"

    if not force and feat_path.exists() and tgt_path.exists() and date_path.exists():
        feat = np.load(feat_path, mmap_mode="r")
        return {"ticker": ticker, "n_rows": feat.shape[0], "n_features": feat.shape[1]}

    price_file = Path(cfg.organized_dir) / ticker / "price_history.csv"
    try:
        df = pd.read_csv(price_file)
    except Exception as e:
        logger.warning(f"  {ticker}: read failed — {e}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    if "adj close" in df.columns:
        # Use adj close where available, fall back to regular close where NaN
        df["close"] = df["adj close"].fillna(df["close"])

    # Sort by date — CSVs may be in reverse chronological order from yfinance
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    feat_arr, names = _compute_daily_features(df)

    date_col = "date" if "date" in df.columns else None
    if date_col is not None:
        news_arr, news_names = _build_lagged_news_features(ticker, cfg, pd.to_datetime(df[date_col], errors="coerce"))
        if news_arr.shape[0] == feat_arr.shape[0]:
            feat_arr = np.concatenate([feat_arr, news_arr], axis=1)
            names = names + news_names
    if cfg.daily_normalize:
        feat_arr = _normalize_array(feat_arr, cfg.daily_norm_window)

    close = df["close"].values.astype(np.float64)
    targets = np.full(len(close), np.nan, dtype=np.float32)
    valid = close[:-cfg.forecast_horizon] > 0
    ret = np.where(valid,
        close[cfg.forecast_horizon:] / np.clip(close[:-cfg.forecast_horizon], 1e-8, None) - 1.0,
        0.0)
    targets[:-cfg.forecast_horizon] = np.clip(ret, -0.20, 0.20).astype(np.float32)

    # Save ordinal dates for meta-model alignment and regime lookup
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        ordinal_dates = np.array(
            [d.toordinal() if pd.notna(d) else 0 for d in dates], dtype=np.int32
        )
    else:
        ordinal_dates = np.arange(len(df), dtype=np.int32)

    np.save(feat_path, feat_arr)
    np.save(tgt_path, targets)
    np.save(date_path, ordinal_dates)

    return {
        "ticker": ticker,
        "n_rows": feat_arr.shape[0],
        "n_features": feat_arr.shape[1],
        "feature_names": names,
    }


def preprocess_minute_ticker(
    ticker: str,
    cfg: HierarchicalDataConfig,
    force: bool = False,
) -> Optional[Dict]:
    """Compute minute features + targets for one ticker, save to cache.

    Saves:
        {cache_dir}/minute/{ticker}_features.npy   (T, F)
        {cache_dir}/minute/{ticker}_targets.npy     (T,)
        {cache_dir}/minute/{ticker}_dates.npy       (T,)  ordinal dates
    """
    cache = Path(cfg.cache_dir) / "minute"
    cache.mkdir(parents=True, exist_ok=True)

    feat_path = cache / f"{ticker}_features.npy"
    tgt_path = cache / f"{ticker}_targets.npy"
    date_path = cache / f"{ticker}_dates.npy"

    if not force and feat_path.exists() and tgt_path.exists() and date_path.exists():
        feat = np.load(feat_path, mmap_mode="r")
        return {"ticker": ticker, "n_rows": feat.shape[0], "n_features": feat.shape[1]}

    minute_file = Path(cfg.minute_dir) / f"{ticker}.parquet"
    daily_file = Path(cfg.organized_dir) / ticker / "price_history.csv"

    if not minute_file.exists() or not daily_file.exists():
        return None

    try:
        mdf = pd.read_parquet(minute_file)
        daily = pd.read_csv(daily_file)
    except Exception as e:
        logger.warning(f"  {ticker}: load failed — {e}")
        return None

    if len(mdf) < cfg.minute_seq_len or "timestamp" not in mdf.columns:
        return None

    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], utc=True)
    mdf["date"] = mdf["timestamp"].dt.date

    daily.columns = [c.strip().lower() for c in daily.columns]
    if "adj close" in daily.columns:
        # Use adj close where available, fall back to regular close where NaN
        daily["close"] = daily["adj close"].fillna(daily["close"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date"] = daily["date"].dt.date
    daily_close = daily.set_index("date")["close"].to_dict()

    feat_arr, names = _compute_minute_features(mdf)

    # Add lagged daily news features and broadcast by minute date
    minute_dates = pd.to_datetime(mdf["timestamp"], utc=True).dt.tz_convert(None)
    news_arr, news_names = _build_lagged_news_features(ticker, cfg, minute_dates)
    if news_arr.shape[0] == feat_arr.shape[0]:
        feat_arr = np.concatenate([feat_arr, news_arr], axis=1)
        names = names + news_names
    if cfg.minute_normalize:
        feat_arr = _normalize_array(feat_arr, cfg.minute_norm_window)

    dates = mdf["date"].values
    unique_dates = sorted(set(dates))
    date_to_next = {unique_dates[j]: unique_dates[j + 1]
                    for j in range(len(unique_dates) - 1)}

    day_close = {}
    for d in unique_dates:
        if d in daily_close:
            day_close[d] = daily_close[d]
        else:
            mask = dates == d
            if mask.any():
                day_close[d] = float(mdf.loc[mask, "close"].iloc[-1])

    targets = np.full(len(mdf), np.nan, dtype=np.float32)
    for i, d in enumerate(dates):
        nd = date_to_next.get(d)
        if nd is None:
            continue
        ct = day_close.get(d)
        cn = day_close.get(nd)
        if ct is not None and cn is not None and ct != 0:
            targets[i] = float(cn / ct - 1.0)

    # Clip extreme targets to improve stability
    targets = np.clip(targets, -0.20, 0.20, out=targets)

    # Save ordinal dates for meta-model alignment and regime lookup
    ordinal_dates = np.array(
        [d.toordinal() if hasattr(d, 'toordinal') else 0 for d in dates],
        dtype=np.int32,
    )

    np.save(feat_path, feat_arr)
    np.save(tgt_path, targets)
    np.save(date_path, ordinal_dates)

    return {
        "ticker": ticker,
        "n_rows": feat_arr.shape[0],
        "n_features": feat_arr.shape[1],
        "feature_names": names,
    }


def preprocess_all(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
    kind: str = "both",
    force: bool = False,
) -> Dict:
    """Preprocess all tickers, save to cache. Run once."""
    meta = {"daily": {}, "minute": {}}
    t0 = time.time()

    if kind in ("daily", "both"):
        logger.info(f"Preprocessing daily features for {len(tickers)} tickers..." + (" (force)" if force else ""))
        for i, ticker in enumerate(tickers):
            info = preprocess_daily_ticker(ticker, cfg, force=force)
            if info:
                meta["daily"][ticker] = info
            if (i + 1) % 50 == 0:
                logger.info(f"  Daily: {i+1}/{len(tickers)} done")
        logger.info(f"  Daily: {len(meta['daily'])} tickers cached")

    if kind in ("minute", "both"):
        logger.info(f"Preprocessing minute features for {len(tickers)} tickers..." + (" (force)" if force else ""))
        for i, ticker in enumerate(tickers):
            info = preprocess_minute_ticker(ticker, cfg, force=force)
            if info:
                meta["minute"][ticker] = info
            if (i + 1) % 50 == 0:
                logger.info(f"  Minute: {i+1}/{len(tickers)} done")
        logger.info(f"  Minute: {len(meta['minute'])} tickers cached")

    logger.info(f"Preprocessing done in {time.time()-t0:.0f}s")

    meta_path = Path(cfg.cache_dir) / "metadata.json"
    serializable = {}
    for kind_key in meta:
        serializable[kind_key] = {}
        for t, info in meta[kind_key].items():
            serializable[kind_key][t] = {
                k: v for k, v in info.items()
                if not isinstance(v, (np.ndarray,))
            }
    with open(meta_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return meta


# ============================================================================
# Lazy-loading PyTorch Datasets
# ============================================================================

class LazyDailyDataset(Dataset):
    """Lazy-loading dataset for daily sequences.

    Each item loads a single (seq_len, n_features) slice from mmap'd .npy.
    Memory usage is O(batch_size * seq_len * features), not O(N * ...).

    When *split_name* is provided together with ``split_mode="temporal"``
    in *cfg*, only the temporal portion of each ticker is indexed.
    """

    def __init__(self, tickers: List[str], cfg: HierarchicalDataConfig,
                 split_name: str = "train"):
        self.cfg = cfg
        self.cache = Path(cfg.cache_dir) / "daily"
        self.split_name = split_name

        # Build index: (ticker_str, row_idx)
        self.index: List[Tuple[str, int]] = []
        self.n_features = 0

        warmup = max(cfg.daily_seq_len, cfg.daily_norm_window, 60)

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            tgt_path = self.cache / f"{ticker}_targets.npy"
            if not feat_path.exists() or not tgt_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            n_rows, n_feat = feat.shape
            self.n_features = n_feat

            tgt = np.load(tgt_path, mmap_mode="r")
            end = n_rows - cfg.forecast_horizon

            # Temporal boundaries for this ticker
            lo, hi = self._temporal_bounds(warmup, end, cfg)

            for i in range(lo, hi, cfg.daily_stride):
                if i - cfg.daily_seq_len >= 0 and not np.isnan(tgt[i]):
                    self.index.append((ticker, i))

        logger.info(f"LazyDailyDataset[{split_name}]: {len(self.index):,} sequences "
                    f"from {len(tickers)} tickers, {self.n_features} features")

    # ------------------------------------------------------------------
    def _temporal_bounds(self, warmup: int, end: int, cfg) -> Tuple[int, int]:
        """Return (lo, hi) row indices for the requested split."""
        if cfg.split_mode != "temporal":
            return warmup, end

        usable = end - warmup
        if usable <= 0:
            return warmup, warmup  # no data

        train_end = warmup + int(usable * cfg.temporal_train_frac)
        val_end   = train_end + int(usable * cfg.temporal_val_frac)
        test_end  = end  # rest

        if self.split_name == "train":
            return warmup, train_end
        elif self.split_name == "val":
            return train_end, val_end
        else:  # test
            return val_end, test_end

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ticker, row = self.index[idx]

        feat = np.load(self.cache / f"{ticker}_features.npy", mmap_mode="r")
        tgt = np.load(self.cache / f"{ticker}_targets.npy", mmap_mode="r")

        seq = feat[row - self.cfg.daily_seq_len : row].copy()  # copy from mmap
        target = float(tgt[row])

        # Load ordinal date for regime lookup / meta alignment
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


class LazyMinuteDataset(Dataset):
    """Lazy-loading dataset for minute sequences.

    Supports temporal splitting: when ``cfg.split_mode == "temporal"``,
    only the rows belonging to the requested *split_name* are indexed.
    """

    def __init__(self, tickers: List[str], cfg: HierarchicalDataConfig,
                 split_name: str = "train"):
        self.cfg = cfg
        self.cache = Path(cfg.cache_dir) / "minute"
        self.split_name = split_name

        self.index: List[Tuple[str, int]] = []
        self.n_features = 0

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            tgt_path = self.cache / f"{ticker}_targets.npy"
            if not feat_path.exists() or not tgt_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            tgt = np.load(tgt_path, mmap_mode="r")
            n_rows, n_feat = feat.shape
            self.n_features = n_feat

            lo, hi = self._temporal_bounds(cfg.minute_seq_len, n_rows, cfg)

            for i in range(lo, hi, cfg.minute_stride):
                if not np.isnan(tgt[i - 1]):
                    self.index.append((ticker, i))

        logger.info(f"LazyMinuteDataset[{split_name}]: {len(self.index):,} sequences "
                    f"from {len(tickers)} tickers, {self.n_features} features")

    # ------------------------------------------------------------------
    def _temporal_bounds(self, warmup: int, n_rows: int, cfg) -> Tuple[int, int]:
        """Return (lo, hi) row indices for the requested split."""
        if cfg.split_mode != "temporal":
            return warmup, n_rows

        usable = n_rows - warmup
        if usable <= 0:
            return warmup, warmup

        train_end = warmup + int(usable * cfg.temporal_train_frac)
        val_end   = train_end + int(usable * cfg.temporal_val_frac)
        test_end  = n_rows

        if self.split_name == "train":
            return warmup, train_end
        elif self.split_name == "val":
            return train_end, val_end
        else:
            return val_end, test_end

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ticker, row = self.index[idx]

        feat = np.load(self.cache / f"{ticker}_features.npy", mmap_mode="r")
        tgt = np.load(self.cache / f"{ticker}_targets.npy", mmap_mode="r")

        seq = feat[row - self.cfg.minute_seq_len : row].copy()
        target = float(tgt[row - 1])

        # Load ordinal date for regime lookup / meta alignment
        date_path = self.cache / f"{ticker}_dates.npy"
        if date_path.exists():
            dates = np.load(date_path, mmap_mode="r")
            ordinal_date = int(dates[row - 1])
        else:
            ordinal_date = 0

        return (
            torch.from_numpy(seq),
            torch.tensor(target, dtype=torch.float32),
            ordinal_date,
            ticker,
        )


# ============================================================================
# Convenience: create dataloaders
# ============================================================================

def create_dataloaders(
    splits: Dict[str, List[str]],
    cfg: HierarchicalDataConfig,
    batch_size_daily: int = 256,
    batch_size_minute: int = 128,
    num_workers: int = 4,
) -> Dict:
    """Create lazy DataLoaders for all splits.

    Returns:
        {
            'daily':  {'train': DataLoader, 'val': ..., 'test': ...},
            'minute': {'train': DataLoader, 'val': ..., 'test': ...},
            'daily_n_features': int,
            'minute_n_features': int,
        }
    """
    result = {"daily": {}, "minute": {}}

    for split_name in ["train", "val", "test"]:
        tickers = splits[split_name]
        shuffle = split_name == "train"

        daily_ds = LazyDailyDataset(tickers, cfg, split_name=split_name)
        minute_ds = LazyMinuteDataset(tickers, cfg, split_name=split_name)

        result["daily"][split_name] = DataLoader(
            daily_ds,
            batch_size=batch_size_daily,
            shuffle=shuffle if len(daily_ds) > 0 else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle if len(daily_ds) > 0 else False,
        )
        result["minute"][split_name] = DataLoader(
            minute_ds,
            batch_size=batch_size_minute,
            shuffle=shuffle if len(minute_ds) > 0 else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle if len(minute_ds) > 0 else False,
        )

    result["daily_n_features"] = result["daily"]["train"].dataset.n_features
    result["minute_n_features"] = result["minute"]["train"].dataset.n_features

    return result
