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

    # Stride controls dataset size (stride=20: ~99% overlap at seq=720 still)
    daily_stride: int = 20
    minute_stride: int = 30        # smaller stride → more minute sequences

    # Target
    forecast_horizon: int = 1
    minute_forecast_horizon: int = 390  # bars ahead for minute target (~1 trading day)

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
    

    # Feature set: "full" (40+ features, default) or "raw_plus" (~15 stationary features)
    daily_feature_set: str = "full"

    # Target type: "close_to_close" (default) or "open_to_close" (intraday)
    daily_target_type: str = "close_to_close"

    # Minute
    minute_normalize: bool = True
    

    # Use larger rolling windows by default for normalization
    daily_norm_window: int = 252
    minute_norm_window: int = 7800

    # News features
    include_news_features: bool = True
    news_lag_days: int = 1  # shift by 1 day to avoid look-ahead leakage
    # Use log-returns for targets instead of simple returns (False = simple)
    use_log_returns: bool = False
    # Target transform: 'raw' | 'ewma_zscore' | ...
    target_transform: str = "ewma_zscore"

    # Walk-Forward Validation parameters
    n_wf_windows: int = 5              # Number of expanding windows for WF CV
    wf_expanding_window: bool = True   # True = expanding (fold 0 earliest), False = rolling
    wf_purge_horizon: int = 0          # Will default to forecast_horizon if 0


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

def _compute_daily_features(
    df: pd.DataFrame,
    feature_set: str = "full",
) -> Tuple[np.ndarray, List[str]]:
    """Compute daily features from OHLCV.

    feature_set="full"     → 40+ feature pipeline (original behaviour)
    feature_set="raw_plus" → ~15 stationary features (recommended for v9+)
    """
    from src.enhanced_features import (
        FeatureConfig,
        compute_raw_plus_features,
        compute_returns_features,
        compute_volatility_features,
        compute_trend_features,
        compute_momentum_features,
        compute_volume_features,
        compute_microstructure_features,
        compute_calendar_features,
    )
    cfg = FeatureConfig(feature_set=feature_set)
    if feature_set == "raw_plus":
        features = compute_raw_plus_features(df, cfg)
    else:
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

    # Optional auto-generated features promoted by the Analyst loop.
    # This module is maintained by agents.feedback.auto_feature_engineer.
    try:
        from src.features.generated_features import compute_generated_features

        generated = compute_generated_features(features)
        if generated is not None and len(generated.columns) > 0:
            features = pd.concat([features, generated], axis=1)
            logger.info(
                "Added %d generated features: %s",
                len(generated.columns),
                list(generated.columns),
            )
    except Exception as e:
        logger.warning("Skipping generated features due to error: %s", e)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features.values.astype(np.float32), list(features.columns)


def _build_daily_sentiment_bridge(
    ticker: str,
    cfg: HierarchicalDataConfig,
    dates: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Build a 1-dim daily sentiment bridge for minute models.

    Loads daily news sentiment (from news cache) and broadcasts to minute
    resolution. This gives minute models *some* news context without
    requiring expensive minute-level news data.

    Args:
        ticker: Ticker symbol
        cfg: Config with paths
        dates: Ordinal dates array from minute data (N_minutes,)

    Returns:
        (sentiment_arr, feature_name)
        sentiment_arr: (N_minutes, 1) with values in [-1, 1], constant per day
        feature_name: "news_sentiment_daily_bridge"
    """
    # Try to load daily news embeddings
    news_emb_path = Path(cfg.cache_dir) / "news" / f"{ticker}_embeddings.npy"
    news_sent_path = Path(cfg.cache_dir) / "news" / f"{ticker}_sentiment.npy"
    news_date_path = Path(cfg.cache_dir) / "news" / f"{ticker}_dates.npy"

    n_bars = len(dates)
    feature_name = "news_sentiment_daily_bridge"

    if not (news_emb_path.exists() and news_sent_path.exists() and news_date_path.exists()):
        logger.debug(f"  {ticker}: daily news sentiment not cached, using zeros for bridge")
        return np.zeros((n_bars, 1), dtype=np.float32), feature_name

    try:
        news_dates = np.load(news_date_path, mmap_mode="r")  # (N_days,) ordinal
        news_sents = np.load(news_sent_path, mmap_mode="r")  # (N_days, 6)

        # Extract compound sentiment: pos_prob - neg_prob (index 0 and 1)
        # Format: [pos_mean, neg_mean, neu_mean, compound_mean, compound_std, count]
        if news_sents.shape[1] >= 4:
            # Use compound_mean (index 3)
            compound_sents = news_sents[:, 3].astype(np.float64)
        else:
            # Fallback: compute from pos/neg
            compound_sents = (news_sents[:, 0] - news_sents[:, 1]).astype(np.float64)

        # Build lookup: ordinal_date → sentiment value
        sentiment_lookup = {}
        for i in range(len(news_dates)):
            d = int(news_dates[i])
            sentiment_lookup[d] = float(compound_sents[i])

        # Align to minute dates with lag (1 day)
        sentiment_arr = np.zeros((n_bars, 1), dtype=np.float32)
        lag = cfg.news_lag_days
        n_matched = 0

        for bar_idx in range(n_bars):
            # Look up sentiment from `lag` days ago
            target_date = int(dates[bar_idx]) - lag
            if target_date in sentiment_lookup:
                sentiment_arr[bar_idx, 0] = float(sentiment_lookup[target_date])
                n_matched += 1

        logger.debug(
            f"  {ticker}: daily sentiment bridge — "
            f"{n_matched}/{n_bars} bars matched sentiment "
            f"(match_rate={n_matched/n_bars:.1%})"
        )
        return sentiment_arr, feature_name

    except Exception as e:
        logger.warning(f"  {ticker}: error building sentiment bridge — {e}, using zeros")
        return np.zeros((n_bars, 1), dtype=np.float32), feature_name


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


# ============================================================================
# Global minute date boundaries (for correct temporal splitting)
# ============================================================================

_MINUTE_DATE_BOUNDS: Optional[Dict[str, int]] = None  # ordinal day boundaries


def _compute_global_minute_date_bounds(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
) -> Dict[str, int]:
    """Compute global calendar-date split boundaries across ALL minute tickers.

    Instead of splitting each ticker's row indices independently (which causes
    the same market day to appear in different splits for different tickers),
    we collect every unique trading date across all tickers, sort them, and
    apply the 70/15/15 split to those dates.  All tickers share the same
    date boundaries, so no date ever appears in more than one split.

    Returns dict with keys 'train_end', 'val_end' (ordinal day integers).
    Train = dates < train_end, Val = train_end <= date < val_end, Test = date >= val_end.
    """
    global _MINUTE_DATE_BOUNDS
    if _MINUTE_DATE_BOUNDS is not None:
        return _MINUTE_DATE_BOUNDS

    cache = Path(cfg.cache_dir) / "minute"
    all_dates = set()
    for ticker in tickers:
        date_path = cache / f"{ticker}_dates.npy"
        if not date_path.exists():
            continue
        dates = np.load(date_path, mmap_mode="r")
        unique = set(int(d) for d in dates if d > 0)
        all_dates.update(unique)

    if not all_dates:
        logger.warning("No minute dates found — minute date bounds will be empty")
        _MINUTE_DATE_BOUNDS = {"train_end": 0, "val_end": 0}
        return _MINUTE_DATE_BOUNDS

    sorted_dates = sorted(all_dates)
    n = len(sorted_dates)
    train_idx = int(n * cfg.temporal_train_frac)
    val_idx = train_idx + int(n * cfg.temporal_val_frac)

    # Boundary dates: train gets dates[:train_idx], val gets dates[train_idx:val_idx],
    # test gets dates[val_idx:]
    train_end_date = sorted_dates[train_idx] if train_idx < n else sorted_dates[-1] + 1
    val_end_date = sorted_dates[val_idx] if val_idx < n else sorted_dates[-1] + 1

    import datetime
    def _ord_to_str(o):
        try:
            return datetime.date.fromordinal(o).isoformat()
        except Exception:
            return str(o)

    logger.info(
        f"Global minute date split: {n} unique dates, "
        f"train<{_ord_to_str(train_end_date)} ({train_idx}d), "
        f"val<{_ord_to_str(val_end_date)} ({val_idx - train_idx}d), "
        f"test≥{_ord_to_str(val_end_date)} ({n - val_idx}d)"
    )

    _MINUTE_DATE_BOUNDS = {
        "train_end": train_end_date,
        "val_end": val_end_date,
    }
    return _MINUTE_DATE_BOUNDS


def reset_minute_date_bounds():
    """Reset the cached minute date bounds (call before re-preprocessing)."""
    global _MINUTE_DATE_BOUNDS
    _MINUTE_DATE_BOUNDS = None


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
        min_periods = max(2, window // 2)
        mu = s.rolling(window, min_periods=min_periods).mean()
        std = s.rolling(window, min_periods=min_periods).std()
        # Fill early NaNs with global stats (more stable than tiny-window estimates)
        global_mu = s.mean()
        global_std = s.std() if s.std() > 1e-8 else 1.0
        mu = mu.fillna(global_mu)
        std = std.fillna(global_std).clip(lower=1e-8)
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

    feat_arr, names = _compute_daily_features(df, feature_set=cfg.daily_feature_set)

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

    if getattr(cfg, "daily_target_type", "close_to_close") == "open_to_close" and "open" in df.columns:
        # Intraday target: (close - open) / open for the row's own day.
        # Model enters at open, exits at close — no look-ahead leakage.
        open_ = df["open"].values.astype(np.float64)
        intra = np.where(open_ > 0, (close - open_) / open_, 0.0).astype(np.float32)
        targets[:] = np.clip(intra, -0.20, 0.20)
        raw_for_backtest = targets.copy()  # already raw returns for intraday target
    else:
        valid = close[:-cfg.forecast_horizon] > 0
        # Compute raw forward returns array (aligned to index i -> return for i -> i+H)
        if getattr(cfg, "use_log_returns", False):
            with np.errstate(divide='ignore', invalid='ignore'):
                ret = np.full(len(close), np.nan, dtype=np.float64)
                ret[:-cfg.forecast_horizon] = np.where(
                    valid,
                    np.log(
                        np.clip(close[cfg.forecast_horizon:], 1e-8, None)
                        / np.clip(close[:-cfg.forecast_horizon], 1e-8, None)
                    ),
                    np.nan,
                )
        else:
            ret = np.full(len(close), np.nan, dtype=np.float64)
            ret[:-cfg.forecast_horizon] = np.where(
                valid,
                close[cfg.forecast_horizon:] / np.clip(close[:-cfg.forecast_horizon], 1e-8, None) - 1.0,
                np.nan,
            )

        # Always compute raw clipped returns for backtesting P&L (never compound z-scores)
        raw_for_backtest = np.full(len(close), np.nan, dtype=np.float32)
        _raw_clip = np.where(np.isfinite(ret), np.clip(ret, -0.20, 0.20), np.nan)
        raw_for_backtest[:-cfg.forecast_horizon] = np.where(
            np.isfinite(_raw_clip[:-cfg.forecast_horizon]),
            _raw_clip[:-cfg.forecast_horizon],
            0.0,
        ).astype(np.float32)

        # Apply target transform
        transform = getattr(cfg, "target_transform", "ewma_zscore")
        if transform == "ewma_zscore":
            # Winsorize raw returns at +/-20% before sigma estimation
            winsor = np.clip(ret, -0.20, 0.20)

            # EWMA sigma with span = daily_norm_window, shifted by 1 to avoid look-ahead
            span = int(getattr(cfg, "daily_norm_window", 60))
            sigma_series = pd.Series(winsor).shift(1).ewm(span=span, adjust=False).std()
            sigma = sigma_series.fillna(1.0).clip(lower=1e-8).values

            # Standardized target: winsorized_return / ewma_sigma, then clip to +-5
            std_target = np.where(np.isfinite(sigma) & (sigma > 0), winsor / sigma, 0.0)
            targets[:-cfg.forecast_horizon] = np.clip(std_target[:-cfg.forecast_horizon], -5.0, 5.0).astype(np.float32)
        else:
            # Fallback: original behaviour (clip raw returns at 20%)
            raw = np.where(np.isnan(ret), 0.0, ret)
            targets[:-cfg.forecast_horizon] = np.clip(raw[:-cfg.forecast_horizon], -0.20, 0.20).astype(np.float32)

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
    # Save raw (unscaled) returns for backtesting — never use z-scores as P&L
    raw_tgt_path = cache / f"{ticker}_raw_targets.npy"
    np.save(raw_tgt_path, raw_for_backtest)

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

    minute_file = Path(cfg.minute_dir) / f"{ticker}.parquet"
    daily_file = Path(cfg.organized_dir) / ticker / "price_history.csv"

    if not force and feat_path.exists() and tgt_path.exists() and date_path.exists():
        # Auto-invalidate stale cache: if the raw parquet is newer than the
        # cached features, the cache was built from older data and must be
        # regenerated so the test split reflects the latest collected bars.
        cache_mtime = feat_path.stat().st_mtime
        if minute_file.exists() and minute_file.stat().st_mtime > cache_mtime:
            logger.debug(
                f"  {ticker}: minute cache is stale (parquet newer by "
                f"{minute_file.stat().st_mtime - cache_mtime:.0f}s) — regenerating"
            )
        else:
            feat = np.load(feat_path, mmap_mode="r")
            return {"ticker": ticker, "n_rows": feat.shape[0], "n_features": feat.shape[1]}

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

    # Save ordinal dates early (needed for sentiment bridge alignment)
    dates = mdf["date"].values
    ordinal_dates = np.array(
        [d.toordinal() if hasattr(d, 'toordinal') else 0 for d in dates],
        dtype=np.int32,
    )

    # ------------------------------------------------------------------
    # Optional: add daily sentiment bridge (1-dim news context)
    # ------------------------------------------------------------------
    # This is a lightweight alternative to full minute-level news.
    # Broadcasts daily news sentiment to all 390 minute bars.
    # Gives minute models *some* news context without expensive data collection.
    # Controlled by cfg.include_news_features.
    if cfg.include_news_features:
        sentiment_arr, sent_name = _build_daily_sentiment_bridge(
            ticker, cfg, ordinal_dates
        )
        feat_arr = np.concatenate([feat_arr, sentiment_arr], axis=1)
        names = names + [sent_name]

    # ------------------------------------------------------------------
    # Append 774-dim per-article news features (intraday-aligned)
    # ------------------------------------------------------------------
    # Produced by scripts/align_news_to_minute.py from the per-article
    # FinBERT cache at data/feature_cache/news_articles/.
    # Layout: [0:768] mean-pooled embedding, [768:774] sentiment summary
    # (pos_mean, neg_mean, neu_mean, compound_mean, compound_std, count).
    # Falls back to zeros if the cache hasn't been built yet.
    NEWS_DIM = 774
    minute_news_dir = Path(cfg.cache_dir) / "minute_news"
    news_feat_path  = minute_news_dir / f"{ticker}_news_features.npy"
    news_ts_path    = minute_news_dir / f"{ticker}_news_timestamps.npy"

    n_bars = len(mdf)
    if news_feat_path.exists() and news_ts_path.exists():
        cached_news = np.load(news_feat_path)          # (M, 774)
        cached_ts   = np.load(news_ts_path)            # (M,) unix sec

        if cached_news.shape[0] == n_bars:
            # Perfect length match — use directly
            news_arr = cached_news.astype(np.float32)
        else:
            # Align by timestamp via binary-search (handles row-count mismatch)
            bar_ts = (
                pd.to_datetime(mdf["timestamp"], utc=True).astype("int64") // 10**9
            ).values
            news_arr = np.zeros((n_bars, NEWS_DIM), dtype=np.float32)
            for i, bt in enumerate(bar_ts):
                idx = int(np.searchsorted(cached_ts, bt, side="right")) - 1
                if 0 <= idx < len(cached_news):
                    news_arr[i] = cached_news[idx]
    else:
        logger.debug(
            f"  {ticker}: minute_news cache missing — skipping news features. "
            f"Run scripts/align_news_to_minute.py to build it."
        )
        news_arr = np.zeros((n_bars, NEWS_DIM), dtype=np.float32)

    # NOTE: News features are now handled by the dedicated NewsEncoder sub-model,
    # NOT concatenated into minute features. This keeps minute_input_dim small (~18)
    # and prevents TFT_M from ballooning to 47M parameters.
    # Concatenating 774-dim news into minute features caused parameter explosion:
    # 18 + 774 = 792 features → 47M params instead of 1M.
    # news_names = [f"news_{i}" for i in range(NEWS_DIM)]
    # feat_arr   = np.concatenate([feat_arr, news_arr], axis=1)
    # names      = names + news_names

    if cfg.minute_normalize:
        feat_arr = _normalize_array(feat_arr, cfg.minute_norm_window)

    # Per-bar forward return targets: each bar predicts close[i+H]/close[i]-1
    # This gives every bar a unique target (no duplication within a day).
    close_vals = mdf["close"].values.astype(np.float64)
    H = cfg.minute_forecast_horizon
    targets = np.full(len(mdf), np.nan, dtype=np.float32)
    valid_mask = (close_vals[:-H] > 0) if H < len(close_vals) else np.array([], dtype=bool)
    if len(valid_mask) > 0:
        if getattr(cfg, "use_log_returns", False):
            with np.errstate(divide='ignore', invalid='ignore'):
                fwd_ret = np.where(
                    valid_mask,
                    np.log(np.clip(close_vals[H:], 1e-8, None) / np.clip(close_vals[:-H], 1e-8, None)),
                    np.nan,
                )
        else:
            fwd_ret = np.where(
                valid_mask,
                close_vals[H:] / np.clip(close_vals[:-H], 1e-8, None) - 1.0,
                np.nan,
            )
        # Place forward returns into full-length array for transform
        full_ret = np.full(len(mdf), np.nan, dtype=np.float64)
        full_ret[: len(fwd_ret)] = fwd_ret

        transform = getattr(cfg, "target_transform", "ewma_zscore")
        if transform == "ewma_zscore":
            # Winsorize at +/-20% before sigma estimation
            winsor = np.clip(full_ret, -0.20, 0.20)
            span = int(getattr(cfg, "minute_norm_window", 390))
            sigma_series = pd.Series(winsor).shift(1).ewm(span=span, adjust=False).std()
            sigma = sigma_series.fillna(1.0).clip(lower=1e-8).values
            std_target = np.where(np.isfinite(sigma) & (sigma > 0), winsor / sigma, 0.0)
            targets[: len(fwd_ret)] = np.clip(std_target[: len(fwd_ret)], -5.0, 5.0).astype(np.float32)
        else:
            targets[: len(fwd_ret)] = np.clip(fwd_ret, -0.20, 0.20).astype(np.float32)

    # ordinal_dates already computed at top of feature preprocessing
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

        # Load raw (unscaled) target for backtesting; fall back to z-scored if missing
        raw_tgt_path = self.cache / f"{ticker}_raw_targets.npy"
        if raw_tgt_path.exists():
            raw_tgt = np.load(raw_tgt_path, mmap_mode="r")
            raw_target = float(raw_tgt[row]) if np.isfinite(raw_tgt[row]) else 0.0
        else:
            raw_target = target  # fallback: will still be wrong if z-scored

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
            torch.tensor(raw_target, dtype=torch.float32),
            ordinal_date,
            ticker,
        )


class LazyMinuteDataset(Dataset):
    """Lazy-loading dataset for minute sequences.

    Supports temporal splitting: when ``cfg.split_mode == "temporal"``,
    **global calendar-date boundaries** are used so that no trading date
    ever appears in more than one split across all tickers.  This prevents
    same-date cross-ticker information leakage.
    """

    def __init__(self, tickers: List[str], cfg: HierarchicalDataConfig,
                 split_name: str = "train"):
        self.cfg = cfg
        self.cache = Path(cfg.cache_dir) / "minute"
        self.split_name = split_name

        self.index: List[Tuple[str, int]] = []
        self.n_features = 0

        # Pre-compute global date boundaries (shared across all tickers)
        if cfg.split_mode == "temporal":
            bounds = _compute_global_minute_date_bounds(tickers, cfg)
            self._date_bounds = bounds
        else:
            self._date_bounds = None

        for ticker in tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            tgt_path = self.cache / f"{ticker}_targets.npy"
            date_path = self.cache / f"{ticker}_dates.npy"
            if not feat_path.exists() or not tgt_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            tgt = np.load(tgt_path, mmap_mode="r")
            n_rows, n_feat = feat.shape
            self.n_features = n_feat

            # Load ordinal dates for global-date splitting
            if date_path.exists() and cfg.split_mode == "temporal":
                dates_arr = np.load(date_path, mmap_mode="r")
            else:
                dates_arr = None

            warmup = cfg.minute_seq_len
            for i in range(warmup, n_rows, cfg.minute_stride):
                if np.isnan(tgt[i - 1]):
                    continue
                # Global-date split: check if this row's date falls in our split
                if dates_arr is not None and self._date_bounds is not None:
                    row_date = int(dates_arr[i - 1])
                    if not self._date_in_split(row_date):
                        continue
                self.index.append((ticker, i))

        logger.info(f"LazyMinuteDataset[{split_name}]: {len(self.index):,} sequences "
                    f"from {len(tickers)} tickers, {self.n_features} features")

    # ------------------------------------------------------------------
    def _date_in_split(self, ordinal_date: int) -> bool:
        """Check if an ordinal date belongs to this split.

        Uses global calendar-date boundaries:
          train: date < train_end
          val:   train_end <= date < val_end
          test:  date >= val_end
        """
        if self._date_bounds is None:
            return True
        train_end = self._date_bounds["train_end"]
        val_end = self._date_bounds["val_end"]
        if self.split_name == "train":
            return ordinal_date < train_end
        elif self.split_name == "val":
            return train_end <= ordinal_date < val_end
        else:  # test
            return ordinal_date >= val_end

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ticker, row = self.index[idx]

        feat = np.load(self.cache / f"{ticker}_features.npy", mmap_mode="r")
        tgt = np.load(self.cache / f"{ticker}_targets.npy", mmap_mode="r")

        seq = feat[row - self.cfg.minute_seq_len : row].copy()
        target = float(tgt[row - 1])

        # Load ordinal date for regime lookup / meta alignment.
        # Use dates[row] — the TARGET bar's date — so this key matches the
        # daily dataset which also keys by the target bar's date (dates[row]).
        # The minute forecast_horizon is ~390 bars (1 trading day), so
        # dates[row-1] is 1 day behind the daily key; dates[row] aligns them.
        date_path = self.cache / f"{ticker}_dates.npy"
        if date_path.exists():
            dates = np.load(date_path, mmap_mode="r")
            # Clamp to valid range: row may equal len(dates)-1 when near end
            date_idx = min(row, len(dates) - 1)
            ordinal_date = int(dates[date_idx])
        else:
            ordinal_date = 0

        return (
            torch.from_numpy(seq),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),  # raw_target placeholder (minute has no separate raw)
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
    num_workers: int = 8,
) -> Dict:
    """Create lazy DataLoaders for all splits.

    Performance notes (RTX 5080 + Ryzen 9 7900X):
      - num_workers=8: saturates I/O pipeline without over-subscribing CPU
      - pin_memory=True: enables async CPU→GPU DMA transfers
      - persistent_workers=True: avoids respawning workers every epoch
      - prefetch_factor=3: keeps 3 batches ready per worker in the queue

    Returns:
        {
            'daily':  {'train': DataLoader, 'val': ..., 'test': ...},
            'minute': {'train': DataLoader, 'val': ..., 'test': ...},
            'daily_n_features': int,
            'minute_n_features': int,
        }
    """
    result = {"daily": {}, "minute": {}}
    use_persistent = num_workers > 0

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
            persistent_workers=use_persistent,
            prefetch_factor=3 if num_workers > 0 else None,
        )
        result["minute"][split_name] = DataLoader(
            minute_ds,
            batch_size=batch_size_minute,
            shuffle=shuffle if len(minute_ds) > 0 else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle if len(minute_ds) > 0 else False,
            persistent_workers=use_persistent,
            prefetch_factor=3 if num_workers > 0 else None,
        )

    result["daily_n_features"] = result["daily"]["train"].dataset.n_features
    result["minute_n_features"] = result["minute"]["train"].dataset.n_features

    return result


# ============================================================================
# Fundamental data: preprocessing + lazy dataset + dataloaders
# ============================================================================

def preprocess_fundamental_ticker(
    ticker: str,
    cfg: HierarchicalDataConfig,
    force: bool = False,
) -> Optional[Dict]:
    """Compute fundamental features for one ticker and save to cache.

    Uses the quarterly-financial pipeline from ``fundamental_features.py``
    to produce a (T, 14) array of daily-aligned fundamental ratios (forward-
    filled from quarterly reports).  Targets and dates are reused from the
    daily cache (must run ``preprocess_daily_ticker`` first).

    Saves:
        {cache_dir}/fundamental/{ticker}_features.npy   (T, F_fund)
    """
    cache = Path(cfg.cache_dir) / "fundamental"
    cache.mkdir(parents=True, exist_ok=True)

    feat_path = cache / f"{ticker}_features.npy"
    if not force and feat_path.exists():
        feat = np.load(feat_path, mmap_mode="r")
        return {"ticker": ticker, "n_rows": feat.shape[0], "n_features": feat.shape[1]}

    # Need daily price data and pre-existing daily cache (for targets/dates)
    daily_cache = Path(cfg.cache_dir) / "daily"
    daily_tgt_path = daily_cache / f"{ticker}_targets.npy"
    daily_date_path = daily_cache / f"{ticker}_dates.npy"
    if not daily_tgt_path.exists() or not daily_date_path.exists():
        return None

    price_file = Path(cfg.organized_dir) / ticker / "price_history.csv"
    if not price_file.exists():
        return None

    try:
        prices_df = pd.read_csv(price_file)
        prices_df.columns = [c.strip().lower() for c in prices_df.columns]
        if "date" in prices_df.columns:
            prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")
            prices_df = prices_df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"  {ticker}: read failed — {e}")
        return None

    try:
        from src.features.fundamental_features import compute_fundamental_features
        fund_df = compute_fundamental_features(ticker, prices_df)
    except Exception as e:
        logger.warning(f"  {ticker}: fundamental feature computation failed — {e}")
        return None

    if fund_df.empty:
        return None

    # Convert to numpy — columns are "fund_trailing_pe", etc.
    feat_arr = fund_df.values.astype(np.float32)

    # Validate length matches daily cache
    daily_tgt = np.load(daily_tgt_path, mmap_mode="r")
    if feat_arr.shape[0] != daily_tgt.shape[0]:
        logger.warning(
            f"  {ticker}: fundamental rows ({feat_arr.shape[0]}) != "
            f"daily rows ({daily_tgt.shape[0]}) — skipping"
        )
        return None

    np.save(feat_path, feat_arr)

    return {
        "ticker": ticker,
        "n_rows": feat_arr.shape[0],
        "n_features": feat_arr.shape[1],
        "feature_names": list(fund_df.columns),
    }


def preprocess_all_fundamentals(
    tickers: List[str],
    cfg: HierarchicalDataConfig,
    force: bool = False,
) -> Dict[str, Dict]:
    """Preprocess fundamental features for all tickers."""
    meta: Dict[str, Dict] = {}
    t0 = time.time()
    logger.info(
        f"Preprocessing fundamental features for {len(tickers)} tickers..."
        + (" (force)" if force else "")
    )
    for i, ticker in enumerate(tickers):
        info = preprocess_fundamental_ticker(ticker, cfg, force=force)
        if info:
            meta[ticker] = info
        if (i + 1) % 50 == 0:
            logger.info(f"  Fundamental: {i+1}/{len(tickers)} done ({len(meta)} ok)")

    logger.info(
        f"  Fundamental: {len(meta)} tickers cached in {time.time()-t0:.0f}s"
    )
    return meta


class LazyFundamentalDataset(Dataset):
    """Lazy-loading dataset for fundamental (tabular) features.

    Each item returns a **flat feature vector** (not a sequence) representing
    the latest-known fundamental ratios as of that date.  Targets and dates
    are reused from the daily cache.

    Returns: (fund_features_vec, target, ordinal_date, ticker)
        fund_features_vec: (F_fund,) — e.g. 14 fundamental ratios
    """

    def __init__(
        self,
        tickers: List[str],
        cfg: HierarchicalDataConfig,
        split_name: str = "train",
    ):
        self.cfg = cfg
        self.fund_cache = Path(cfg.cache_dir) / "fundamental"
        self.daily_cache = Path(cfg.cache_dir) / "daily"
        self.split_name = split_name

        self.index: List[Tuple[str, int]] = []
        self.n_features = 0

        warmup = max(cfg.daily_seq_len, cfg.daily_norm_window, 60)

        for ticker in tickers:
            fund_path = self.fund_cache / f"{ticker}_features.npy"
            tgt_path = self.daily_cache / f"{ticker}_targets.npy"
            date_path = self.daily_cache / f"{ticker}_dates.npy"

            if not fund_path.exists() or not tgt_path.exists():
                continue

            fund = np.load(fund_path, mmap_mode="r")
            tgt = np.load(tgt_path, mmap_mode="r")
            n_rows, n_feat = fund.shape
            self.n_features = n_feat

            end = n_rows - cfg.forecast_horizon

            # Temporal boundaries (same as daily — same targets/dates)
            lo, hi = self._temporal_bounds(warmup, end, cfg)

            for i in range(lo, hi, cfg.daily_stride):
                if not np.isnan(tgt[i]):
                    self.index.append((ticker, i))

        logger.info(
            f"LazyFundamentalDataset[{split_name}]: {len(self.index):,} samples "
            f"from {len(tickers)} tickers, {self.n_features} features"
        )

    def _temporal_bounds(self, warmup: int, end: int, cfg) -> Tuple[int, int]:
        """Return (lo, hi) row indices for the requested temporal split."""
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

        fund = np.load(
            self.fund_cache / f"{ticker}_features.npy", mmap_mode="r"
        )
        tgt = np.load(
            self.daily_cache / f"{ticker}_targets.npy", mmap_mode="r"
        )

        features_vec = fund[row].copy()  # (F_fund,)
        target = float(tgt[row])

        date_path = self.daily_cache / f"{ticker}_dates.npy"
        if date_path.exists():
            dates = np.load(date_path, mmap_mode="r")
            ordinal_date = int(dates[row])
        else:
            ordinal_date = 0

        return (
            torch.from_numpy(features_vec),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),  # raw_target placeholder
            ordinal_date,
            ticker,
        )


def create_fundamental_dataloaders(
    splits: Dict[str, List[str]],
    cfg: HierarchicalDataConfig,
    batch_size: int = 256,
    num_workers: int = 8,
) -> Dict:
    """Create DataLoaders for fundamental features (all splits).

    Returns:
        {
            'train': DataLoader, 'val': DataLoader, 'test': DataLoader,
            'n_features': int,
        }
    """
    result: Dict = {}
    use_persistent = num_workers > 0
    for split_name in ["train", "val", "test"]:
        tickers = splits[split_name]
        shuffle = split_name == "train"
        ds = LazyFundamentalDataset(tickers, cfg, split_name=split_name)
        result[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle if len(ds) > 0 else False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle if len(ds) > 0 else False,
            persistent_workers=use_persistent,
            prefetch_factor=3 if num_workers > 0 else None,
        )
    result["n_features"] = result["train"].dataset.n_features
    return result


# ============================================================================
# Graph data: cross-sectional dataset for GNN
# ============================================================================

def _load_sector_map(
    tickers: List[str],
    cache_dir: str = "data/metadata",
) -> Dict[str, str]:
    """Load ticker → sector mapping from metadata cache or yfinance.

    Tries the JSON cache first; falls back to yfinance for unknowns.
    """
    cache_path = Path(cache_dir) / "ticker_metadata.json"
    cached: Dict = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
        except Exception:
            pass

    sector_map: Dict[str, str] = {}
    unknown_tickers: List[str] = []

    for t in tickers:
        if t in cached and "sector" in cached[t]:
            sector_map[t] = cached[t]["sector"]
        else:
            unknown_tickers.append(t)

    # Batch-fetch unknowns from yfinance (with caching)
    if unknown_tickers:
        logger.info(f"Fetching sector info for {len(unknown_tickers)} tickers from yfinance...")
        for t in unknown_tickers:
            try:
                import yfinance as yf
                info = yf.Ticker(t).info or {}
                raw = info.get("sector", "Unknown")
                sector_map[t] = raw if raw else "Unknown"
                # Update cache
                if t not in cached:
                    cached[t] = {}
                cached[t]["sector"] = sector_map[t]
                cached[t]["industry"] = info.get("industry", "Unknown")
            except Exception:
                sector_map[t] = "Unknown"

        # Persist cache
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w") as f:
                json.dump(cached, f, indent=2)
        except Exception:
            pass

    return sector_map


def build_adjacency(
    tickers: List[str],
    sector_map: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse graph where stocks in the same sector are connected.

    Returns:
        edge_index:  (2, E) long tensor — COO format
        edge_weight: (E,) float tensor  — 1.0 for same-sector edges
    """
    sector_groups: Dict[str, List[int]] = {}
    for i, t in enumerate(tickers):
        sec = sector_map.get(t, "Unknown")
        sector_groups.setdefault(sec, []).append(i)

    src_list, dst_list = [], []
    for sec, members in sector_groups.items():
        if sec == "Unknown" or len(members) < 2:
            continue
        # Fully-connect members within sector (undirected)
        for a in members:
            for b in members:
                if a != b:
                    src_list.append(a)
                    dst_list.append(b)

    # Also add self-loops for every node
    for i in range(len(tickers)):
        src_list.append(i)
        dst_list.append(i)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


class CrossSectionalGraphDataset(Dataset):
    """Dataset that produces one *cross-sectional snapshot* per trading date.

    Each sample is a date ``d`` for which we have daily features for ≥2
    tickers.  The __getitem__ returns all tickers' features for that date
    as a batch, plus the sector-based adjacency graph.

    Returns a dict:
        node_features: (N_tickers, F) — daily features at date d
        targets:       (N_tickers,)   — next-day returns
        ordinal_date:  int
        tickers:       list[str]
        edge_index:    (2, E) long tensor
        edge_weight:   (E,) float tensor
        mask:          (N_tickers,) bool — True if ticker has data on this date
    """

    def __init__(
        self,
        tickers: List[str],
        cfg: HierarchicalDataConfig,
        split_name: str = "train",
        min_tickers_per_date: int = 10,
    ):
        self.cfg = cfg
        self.cache = Path(cfg.cache_dir) / "daily"
        self.split_name = split_name

        # Load sector map and build graph
        self.tickers = sorted(tickers)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        sector_map = _load_sector_map(self.tickers)
        self.edge_index, self.edge_weight = build_adjacency(self.tickers, sector_map)
        n_edges = self.edge_index.shape[1] if self.edge_index.numel() > 0 else 0
        n_sector_edges = n_edges - len(self.tickers)  # subtract self-loops
        if n_sector_edges <= 0:
            logger.warning(
                f"CrossSectionalGraphDataset[{split_name}]: 0 sector edges built "
                f"(all tickers mapped to 'Unknown' sector or sector_map empty). "
                f"GNN will only use self-loops — training will be ineffective. "
                f"Check data/metadata/ticker_metadata.json for sector coverage."
            )
        else:
            logger.info(
                f"CrossSectionalGraphDataset[{split_name}]: "
                f"{n_sector_edges} sector edges + {len(self.tickers)} self-loops"
            )
        self.n_features = 0

        # Build date → {ticker: row_idx} mapping
        warmup = max(cfg.daily_seq_len, cfg.daily_norm_window, 60)

        # date_data[ordinal_date] = {ticker: row_idx}
        date_data: Dict[int, Dict[str, int]] = {}

        for ticker in self.tickers:
            feat_path = self.cache / f"{ticker}_features.npy"
            tgt_path = self.cache / f"{ticker}_targets.npy"
            date_path = self.cache / f"{ticker}_dates.npy"
            if not feat_path.exists() or not tgt_path.exists() or not date_path.exists():
                continue

            feat = np.load(feat_path, mmap_mode="r")
            tgt = np.load(tgt_path, mmap_mode="r")
            dates_arr = np.load(date_path, mmap_mode="r")
            n_rows = feat.shape[0]
            self.n_features = feat.shape[1]

            end = n_rows - cfg.forecast_horizon
            lo, hi = self._temporal_bounds(warmup, end, cfg)

            for i in range(lo, hi, cfg.daily_stride):
                if np.isnan(tgt[i]):
                    continue
                od = int(dates_arr[i])
                if od <= 0:
                    continue
                if od not in date_data:
                    date_data[od] = {}
                date_data[od][ticker] = i

        # Keep only dates with enough tickers
        self.dates = sorted(
            od for od, tmap in date_data.items()
            if len(tmap) >= min_tickers_per_date
        )
        self.date_data = date_data

        logger.info(
            f"CrossSectionalGraphDataset[{split_name}]: {len(self.dates)} dates, "
            f"{len(self.tickers)} tickers, {self.n_features} features"
        )

    def _temporal_bounds(self, warmup: int, end: int, cfg) -> Tuple[int, int]:
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
        return len(self.dates)

    def __getitem__(self, idx: int):
        od = self.dates[idx]
        ticker_rows = self.date_data[od]

        N = len(self.tickers)
        F = self.n_features

        node_features = np.zeros((N, F), dtype=np.float32)
        targets = np.zeros(N, dtype=np.float32)
        mask = np.zeros(N, dtype=bool)

        for ticker, row in ticker_rows.items():
            ti = self.ticker_to_idx[ticker]
            feat = np.load(self.cache / f"{ticker}_features.npy", mmap_mode="r")
            tgt = np.load(self.cache / f"{ticker}_targets.npy", mmap_mode="r")
            # Use the feature vector at the specific row (most recent daily state)
            node_features[ti] = feat[row]
            targets[ti] = tgt[row]
            mask[ti] = True

        return {
            "node_features": torch.from_numpy(node_features),   # (N, F)
            "targets": torch.from_numpy(targets),                # (N,)
            "ordinal_date": od,
            "tickers": self.tickers,
            "edge_index": self.edge_index,       # shared, constant
            "edge_weight": self.edge_weight,
            "mask": torch.from_numpy(mask),      # (N,)
        }


def create_graph_dataloaders(
    splits: Dict[str, List[str]],
    cfg: HierarchicalDataConfig,
    batch_size: int = 1,
    num_workers: int = 0,
    min_tickers_per_date: int = 10,
) -> Dict:
    """Create DataLoaders for cross-sectional graph snapshots.

    Note: batch_size should typically be 1 because each sample is already
    a full cross-section (all tickers for one date).  The collate function
    is the default — stacking dicts is handled in the training loop.

    Returns:
        {
            'train': DataLoader, 'val': DataLoader, 'test': DataLoader,
            'n_features': int, 'n_tickers': int, 'edge_index': Tensor,
            'edge_weight': Tensor,
        }
    """
    result: Dict = {}
    ds_ref = None
    for split_name in ["train", "val", "test"]:
        tickers = splits[split_name]
        shuffle = split_name == "train"
        ds = CrossSectionalGraphDataset(
            tickers, cfg, split_name=split_name,
            min_tickers_per_date=min_tickers_per_date,
        )
        if ds_ref is None:
            ds_ref = ds
        result[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle if len(ds) > 0 else False,
            num_workers=num_workers,
            pin_memory=False,   # dict items handled manually
        )
    result["n_features"] = ds_ref.n_features if ds_ref else 0
    result["n_tickers"] = len(ds_ref.tickers) if ds_ref else 0
    result["edge_index"] = ds_ref.edge_index if ds_ref else torch.zeros(2, 0, dtype=torch.long)
    result["edge_weight"] = ds_ref.edge_weight if ds_ref else torch.zeros(0)
    return result
