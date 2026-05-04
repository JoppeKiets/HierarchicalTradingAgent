#!/usr/bin/env python3
"""Enhanced feature pipeline v2.

Replaces the 7-feature prepare_daily_data with a comprehensive 40+ feature pipeline.
Fixes key issues from Phase 1 diagnosis:
  1. Too few features (7 → 40+)
  2. No normalization (raw values → z-score per window)
  3. 11-class target too granular (→ 3-class or regression)
  4. return/log_return redundancy eliminated

Feature categories:
  - Price returns (multi-horizon)
  - Volatility (realized, Parkinson, Garman-Klass)
  - Trend (SMA crossovers, ADX)
  - Momentum (RSI, MACD, Stochastic, Williams %R)
  - Volume (OBV, VWAP ratio, volume momentum)
  - Market microstructure (range, gap, body ratio)
  - Calendar (day of week, month)

Usage:
    from src.enhanced_features import prepare_enhanced_daily_data, FeatureConfig
    config = FeatureConfig(n_classes=3, normalize=True)
    sequences, targets, returns = prepare_enhanced_daily_data(prices_df, config)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature pipeline."""
    seq_len: int = 60
    forecast_horizon: int = 5
    stride: int = 1               # Stride for sequence generation (1 = all, 5 = every 5th)

    # Feature set: "full" (40+ features, original) or "raw_plus" (~15 stationary features)
    feature_set: str = "full"

    # Target type: "close_to_close" (default) or "open_to_close" (same-day intraday return)
    target_type: str = "close_to_close"
    # Use log-returns for targets instead of simple returns (False = simple)
    use_log_returns: bool = False
    
    # Target scheme
    n_classes: int = 3            # 3 = down/flat/up, 5 = strong_down/.../strong_up, 0 = regression
    class_thresholds_3: tuple = (-0.01, 0.01)   # ±1% for 3-class (used only if use_percentile_thresholds=False)
    class_thresholds_5: tuple = (-0.03, -0.01, 0.01, 0.03)   # for 5-class (used only if use_percentile_thresholds=False)
    use_percentile_thresholds: bool = True  # If True, compute thresholds from data percentiles for balanced classes
    
    # Normalization
    normalize: bool = True
    norm_window: int = 252        # Rolling z-score window (default: 1 trading year)
    clip_value: float = 5.0       # Clip normalized features at ±5σ
    
    # Feature toggles (all on by default)
    use_returns: bool = True
    use_volatility: bool = True
    use_trend: bool = True
    use_momentum: bool = True
    use_volume: bool = True
    use_microstructure: bool = True
    use_calendar: bool = True
    
    # Extended feature toggles (off by default — require extra API calls)
    use_fundamentals: bool = False
    use_macro: bool = False
    use_sentiment: bool = False
    
    # Feature parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14


def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    return pd.Series(data).ewm(span=span, adjust=False).mean().values


def _sma(data: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with NaN handling."""
    s = pd.Series(data)
    min_periods = max(1, window // 2)
    rolled = s.rolling(window, min_periods=min_periods).mean()
    # Fill initial NaNs (insufficient window) with expanding mean to avoid
    # using very small-sample rolling means which are noisy.
    filled = rolled.fillna(s.expanding().mean())
    return filled.values


def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation."""
    return pd.Series(data).rolling(window, min_periods=2).std().fillna(0).values


def _prev(arr: np.ndarray, n: int = 1) -> np.ndarray:
    """Shift *arr* forward by *n* positions, filling the first *n* entries
    with ``arr[0]`` instead of wrapping (which ``np.roll`` does)."""
    out = np.empty_like(arr)
    out[n:] = arr[:-n]
    out[:n] = arr[0]
    return out


def compute_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon return features."""
    close = df["close"].values
    features = pd.DataFrame(index=df.index)
    
    # Simple returns at multiple horizons
    for h in [1, 2, 5, 10, 20]:
        features[f"return_{h}d"] = pd.Series(close).pct_change(h, fill_method=None).values
    
    # Log return (only 1-day, avoid redundancy with simple return)
    features["log_return_1d"] = np.log(pd.Series(close) / pd.Series(close).shift(1)).values
    
    return features


def compute_volatility_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Volatility features: realized, Parkinson, range-based."""
    close = np.maximum(df["close"].values, 1e-10)
    high = np.maximum(df["high"].values, 1e-10)
    low = np.maximum(df["low"].values, 1e-10)
    open_p = np.maximum(df["open"].values, 1e-10)
    ret = pd.Series(close).pct_change(fill_method=None).values
    
    features = pd.DataFrame(index=df.index)
    
    # Realized volatility at multiple windows
    for w in [5, 10, 20, 60]:
        features[f"realized_vol_{w}d"] = _rolling_std(ret, w) * np.sqrt(252)
    
    # Parkinson volatility (uses high/low, more efficient estimator)
    hl_ratio = np.log(high / low)
    parkinson = hl_ratio ** 2 / (4 * np.log(2))
    for w in [10, 20]:
        features[f"parkinson_vol_{w}d"] = np.sqrt(
            pd.Series(parkinson).rolling(w, min_periods=2).mean().fillna(0).values * 252
        )
    
    # Garman-Klass volatility (uses OHLC)
    gk = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_p) ** 2
    for w in [10, 20]:
        features[f"gk_vol_{w}d"] = np.sqrt(
            np.abs(pd.Series(gk).rolling(w, min_periods=2).mean().fillna(0).values) * 252
        )
    
    # Volatility ratio (short-term / long-term) — regime signal
    short_vol = _rolling_std(ret, 5) + 1e-10
    long_vol = _rolling_std(ret, 60) + 1e-10
    features["vol_ratio_5_60"] = short_vol / long_vol
    
    return features


def compute_trend_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Trend features: SMA crossovers, slope, ADX."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    
    features = pd.DataFrame(index=df.index)
    
    # SMA crossover signals (distance from SMA, normalized)
    for w in [10, 20, 50, 200]:
        sma = _sma(close, w)
        features[f"sma_{w}_dist"] = (close - sma) / (sma + 1e-10)
    
    # SMA slope (trend direction)
    for w in [20, 50]:
        sma = _sma(close, w)
        features[f"sma_{w}_slope"] = pd.Series(sma).pct_change(5, fill_method=None).fillna(0).values
    
    # Price position in N-day range
    for w in [20, 60]:
        roll_high = pd.Series(high).rolling(w, min_periods=1).max().values
        roll_low = pd.Series(low).rolling(w, min_periods=1).min().values
        rng = roll_high - roll_low + 1e-10
        features[f"price_position_{w}d"] = (close - roll_low) / rng
    
    # ADX (Average Directional Index) — trend strength
    period = cfg.adx_period
    dm_plus = np.maximum(np.diff(high, prepend=high[0]), 0)
    dm_minus = np.maximum(-np.diff(low, prepend=low[0]), 0)
    
    # Zero out where the other is larger
    mask = dm_plus > dm_minus
    dm_plus = np.where(mask, dm_plus, 0)
    dm_minus = np.where(~mask, dm_minus, 0)
    
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - _prev(close)), np.abs(low - _prev(close)))
    )
    
    atr = _ema(tr, period)
    di_plus = 100 * _ema(dm_plus, period) / (atr + 1e-10)
    di_minus = 100 * _ema(dm_minus, period) / (atr + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = _ema(dx, period)
    
    features["adx"] = adx / 100  # Normalize to [0, 1]
    features["di_diff"] = (di_plus - di_minus) / 100  # Signed trend strength
    
    return features


def compute_momentum_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Momentum features: RSI, MACD, Stochastic, Williams %R."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    
    features = pd.DataFrame(index=df.index)
    
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    period = cfg.rsi_period
    
    if period < n:
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, where=avg_loss > 0, out=np.ones(n))
    rsi = 100 - 100 / (1 + rs)
    features["rsi"] = rsi / 100  # Normalize to [0, 1]
    
    # MACD
    ema_fast = _ema(close, cfg.macd_fast)
    ema_slow = _ema(close, cfg.macd_slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, cfg.macd_signal)
    macd_hist = macd_line - signal_line
    
    # Normalize MACD by price level
    features["macd_norm"] = macd_line / (close + 1e-10)
    features["macd_signal_norm"] = signal_line / (close + 1e-10)
    features["macd_hist_norm"] = macd_hist / (close + 1e-10)
    
    # Stochastic Oscillator %K and %D
    for w in [14]:
        roll_low = pd.Series(low).rolling(w, min_periods=1).min().values
        roll_high = pd.Series(high).rolling(w, min_periods=1).max().values
        stoch_k = (close - roll_low) / (roll_high - roll_low + 1e-10)
        stoch_d = _sma(stoch_k, 3)
        features[f"stoch_k_{w}"] = stoch_k
        features[f"stoch_d_{w}"] = stoch_d
    
    # Williams %R
    features["williams_r"] = -(pd.Series(high).rolling(14, min_periods=1).max().values - close) / \
        (pd.Series(high).rolling(14, min_periods=1).max().values - 
         pd.Series(low).rolling(14, min_periods=1).min().values + 1e-10)
    
    # Rate of change at multiple horizons
    for h in [5, 10, 20]:
        prev_close = _prev(close, h)
        features[f"roc_{h}d"] = (close - prev_close) / (prev_close + 1e-10)
        features.loc[features.index[:h], f"roc_{h}d"] = 0  # Fix warmup
    
    return features


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume features: ratios, OBV slope, VWAP distance."""
    close = df["close"].values
    volume = df["volume"].values.astype(float)
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    
    features = pd.DataFrame(index=df.index)
    
    # Volume ratios
    for w in [5, 20]:
        vol_sma = _sma(volume, w)
        features[f"vol_ratio_{w}d"] = volume / (vol_sma + 1)
    
    # Volume trend (is volume increasing?)
    features["vol_change_5d"] = pd.Series(volume).pct_change(5, fill_method=None).fillna(0).values
    
    # On-Balance Volume (OBV) normalized slope
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    obv_sma = _sma(obv, 20)
    features["obv_slope"] = pd.Series(obv_sma).pct_change(5, fill_method=None).fillna(0).values
    
    # VWAP distance (approximate daily)
    typical_price = (high + low + close) / 3
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume) + 1
    vwap = cum_tp_vol / cum_vol
    features["vwap_dist"] = (close - vwap) / (vwap + 1e-10)
    
    return features


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market microstructure: candle patterns, gaps, ranges."""
    close = df["close"].values
    open_p = df["open"].values
    high = df["high"].values
    low = df["low"].values
    
    features = pd.DataFrame(index=df.index)
    
    # Body ratio (bullish/bearish candle strength)
    body = close - open_p
    candle_range = high - low + 1e-10
    features["body_ratio"] = body / candle_range
    
    # Upper/lower shadow ratios
    features["upper_shadow"] = (high - np.maximum(close, open_p)) / candle_range
    features["lower_shadow"] = (np.minimum(close, open_p) - low) / candle_range
    
    # Gap (overnight return)
    prev_close = _prev(close)
    features["gap"] = (open_p - prev_close) / (prev_close + 1e-10)
    features.iloc[0, features.columns.get_loc("gap")] = 0
    
    # ATR normalized
    prev_close = _prev(close)
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = _ema(tr, 14)
    features["atr_norm"] = atr / (close + 1e-10)
    
    # Intraday range relative to recent average
    daily_range = (high - low) / (close + 1e-10)
    avg_range = _sma(daily_range, 20)
    features["range_ratio"] = daily_range / (avg_range + 1e-10)
    
    return features


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar effects: day of week, month, turn of month."""
    features = pd.DataFrame(index=df.index)
    
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        # Assume index is datetime
        dates = pd.to_datetime(df.index)
    
    # Day of week (sin/cos encoding for cyclical nature)
    dow = dates.dt.dayofweek.values
    features["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    
    # Month (sin/cos encoding)
    month = dates.dt.month.values
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # Turn of month effect (last 3 and first 3 trading days)
    day = dates.dt.day.values
    features["turn_of_month"] = np.where((day <= 3) | (day >= 28), 1.0, 0.0)
    
    return features



def compute_raw_plus_features(df, cfg):
    """Simplified ~15-feature 'Raw+' set.

    Designed to be stationary and low-noise:
      - Log returns at 1, 5, 20-day horizons
      - Realized volatility (5d, 20d) and vol regime ratio
      - RSI (14-period, normalised to [-1, 1])
      - VWAP distance (intraday mean-reversion signal)
      - Relative volume (vs 20d SMA)
      - ATR normalised (market breadth / range)
      - Overnight gap
      - Intraday body ratio (bull/bear candle strength)
      - Calendar: day-of-week, turn-of-month (sin/cos)
    """
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    vol   = df["volume"].values.astype(np.float64)

    features = pd.DataFrame(index=df.index)

    # --- Log returns (stationary, minimal redundancy) ---
    for h in [1, 5, 20]:
        shifted = pd.Series(close).shift(h).values
        safe_shifted = np.where(shifted > 0, shifted, np.nan)
        features[f"log_ret_{h}d"] = np.where(
            safe_shifted > 0, np.log(close / safe_shifted), 0.0
        ).astype(np.float32)

    # --- Realized volatility ---
    ret1 = pd.Series(close).pct_change().fillna(0).values
    for w in [5, 20]:
        features[f"real_vol_{w}d"] = (_rolling_std(ret1, w) * np.sqrt(252)).astype(np.float32)

    # Volatility regime ratio (short vs long)
    short_vol = _rolling_std(ret1, 5) + 1e-10
    long_vol  = _rolling_std(ret1, 60) + 1e-10
    features["vol_regime"] = (short_vol / long_vol).astype(np.float32)

    # --- RSI (normalised to [-1, 1] to make it zero-centred) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    period = cfg.rsi_period
    n = len(close)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    if period < n:
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, where=avg_loss > 0, out=np.ones(n))
    rsi_raw = 100 - 100 / (1 + rs)
    features["rsi"] = ((rsi_raw / 50.0) - 1.0).astype(np.float32)

    # --- VWAP distance ---
    typical = (high + low + close) / 3.0
    cum_tv  = np.cumsum(typical * vol)
    cum_v   = np.cumsum(vol) + 1e-10
    vwap    = cum_tv / cum_v
    features["vwap_dist"] = ((close - vwap) / (vwap + 1e-10)).astype(np.float32)

    # --- Relative volume ---
    vol_sma20 = _sma(vol, 20)
    features["rel_vol"] = (vol / (vol_sma20 + 1.0)).astype(np.float32)

    # --- ATR normalised ---
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low  - np.roll(close, 1)),
        ),
    )
    atr = _ema(tr, 14)
    features["atr_norm"] = (atr / (close + 1e-10)).astype(np.float32)

    # --- Overnight gap ---
    gap = (open_ - np.roll(close, 1)) / (np.roll(close, 1) + 1e-10)
    gap[0] = 0.0
    features["gap"] = gap.astype(np.float32)

    # --- Body ratio (candle direction strength) ---
    candle_range = high - low + 1e-10
    features["body_ratio"] = ((close - open_) / candle_range).astype(np.float32)

    # --- Calendar (sin/cos for cyclicality) ---
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        dates = pd.to_datetime(df.index)
    dow   = dates.dt.dayofweek.values
    month = dates.dt.month.values
    day   = dates.dt.day.values
    features["dow_sin"]       = np.sin(2 * np.pi * dow   / 5).astype(np.float32)
    features["dow_cos"]       = np.cos(2 * np.pi * dow   / 5).astype(np.float32)
    features["month_sin"]     = np.sin(2 * np.pi * month / 12).astype(np.float32)
    features["month_cos"]     = np.cos(2 * np.pi * month / 12).astype(np.float32)
    features["turn_of_month"] = np.where((day <= 3) | (day >= 28), 1.0, 0.0).astype(np.float32)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def normalize_features(features_df: pd.DataFrame, window: int = 60, clip: float = 5.0) -> pd.DataFrame:
    """Apply rolling z-score normalization to all features.
    
    Each feature is normalized as: (x - rolling_mean) / rolling_std
    This prevents look-ahead bias while standardizing scales.
    """
    normalized = pd.DataFrame(index=features_df.index)
    
    min_periods = max(2, window // 2)
    for col in features_df.columns:
        series = features_df[col]
        roll_mean = series.rolling(window, min_periods=min_periods).mean()
        roll_std = series.rolling(window, min_periods=min_periods).std()

        # For the earliest rows where rolling produced NaN (insufficient data),
        # fall back to the global mean/std to avoid overly noisy estimates.
        global_mean = series.mean()
        global_std = series.std() if series.std() > 1e-8 else 1.0
        roll_mean = roll_mean.fillna(global_mean)
        roll_std = roll_std.fillna(global_std)

        # Z-score normalization
        z_scored = (series - roll_mean) / (roll_std + 1e-10)

        # Clip extreme values
        z_scored = z_scored.clip(-clip, clip)

        # Fill any remaining NaN with 0
        z_scored = z_scored.fillna(0)

        normalized[col] = z_scored
    
    return normalized


def compute_targets(
    prices_df: pd.DataFrame,
    cfg: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute target labels and raw returns.
    
    Returns:
        labels: int64 array of class labels (or float32 for regression)
        returns: float32 array of raw future returns
    """
    close = prices_df["close"].values
    n = len(close)

    if cfg.target_type == "open_to_close":
        open_prices = prices_df["open"].values
        future_returns = np.divide(
            close - open_prices,
            open_prices + 1e-10,
            out=np.zeros(n, dtype=np.float32),
            where=open_prices != 0,
        ).astype(np.float32)
    else:
        future_returns = np.zeros(n, dtype=np.float32)
        for i in range(n - cfg.forecast_horizon):
            if getattr(cfg, "use_log_returns", False):
                # log return: ln(P_t+h / P_t)
                future_returns[i] = np.log(close[i + cfg.forecast_horizon] / (close[i] + 1e-10))
            else:
                future_returns[i] = (close[i + cfg.forecast_horizon] - close[i]) / (close[i] + 1e-10)

    if cfg.n_classes == 0:
        return future_returns, future_returns

    labels = np.zeros(n, dtype=np.int64)
    valid_returns = future_returns if cfg.target_type == "open_to_close" else future_returns[:n - cfg.forecast_horizon]

    if cfg.n_classes == 3:
        if cfg.use_percentile_thresholds and len(valid_returns) > 0:
            lo = float(np.percentile(valid_returns, 33.3))
            hi = float(np.percentile(valid_returns, 66.7))
            logger.info(f"  Percentile thresholds (3-class): lo={lo:.4f}, hi={hi:.4f}")
        else:
            lo, hi = cfg.class_thresholds_3
        labels = np.where(future_returns < lo, 0,
                          np.where(future_returns > hi, 2, 1))

    elif cfg.n_classes == 5:
        if cfg.use_percentile_thresholds and len(valid_returns) > 0:
            t1 = float(np.percentile(valid_returns, 20))
            t2 = float(np.percentile(valid_returns, 40))
            t3 = float(np.percentile(valid_returns, 60))
            t4 = float(np.percentile(valid_returns, 80))
            logger.info(f"  Percentile thresholds (5-class): {t1:.4f}, {t2:.4f}, {t3:.4f}, {t4:.4f}")
        else:
            t1, t2, t3, t4 = cfg.class_thresholds_5
        labels = np.where(future_returns < t1, 0,
                          np.where(future_returns < t2, 1,
                          np.where(future_returns < t3, 2,
                          np.where(future_returns < t4, 3, 4))))

    elif cfg.n_classes == 11:
        thresholds = [-0.05, -0.03, -0.01, -0.005, 0, 0.005, 0.01, 0.03, 0.05, 0.10]
        labels = np.digitize(future_returns, thresholds)

    return labels.astype(np.int64), future_returns.astype(np.float32)



def prepare_enhanced_daily_data(
    prices_df: pd.DataFrame,
    cfg: FeatureConfig = None,
    extra_features_df: Optional[pd.DataFrame] = None,
    ticker: Optional[str] = None,
    macro_cache: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare enhanced daily data with 40+ features.
    
    Args:
        prices_df: DataFrame with columns [date, open, high, low, close, volume]
        cfg: Feature configuration
        extra_features_df: Optional additional features to merge by date
        ticker: Stock ticker symbol (needed for fundamentals/sentiment)
        macro_cache: Pre-fetched macro series dict (shared across tickers)
        
    Returns:
        sequences: (N, seq_len, n_features) float32
        targets: (N,) int64 labels or float32 for regression
        returns: (N,) float32 raw future returns
        feature_names: list of feature column names
    """
    if cfg is None:
        cfg = FeatureConfig()
    
    df = prices_df.copy()
    
    # Ensure required columns
    required = ["close", "high", "low", "open", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if cfg.feature_set == "raw_plus":
        all_features = compute_raw_plus_features(df, cfg)
    else:
        # Compute all feature groups
        feature_dfs = []

        if cfg.use_returns:
            feature_dfs.append(compute_returns_features(df))

        if cfg.use_volatility:
            feature_dfs.append(compute_volatility_features(df, cfg))

        if cfg.use_trend:
            feature_dfs.append(compute_trend_features(df, cfg))

        if cfg.use_momentum:
            feature_dfs.append(compute_momentum_features(df, cfg))

        if cfg.use_volume:
            feature_dfs.append(compute_volume_features(df))

        if cfg.use_microstructure:
            feature_dfs.append(compute_microstructure_features(df))

        if cfg.use_calendar:
            feature_dfs.append(compute_calendar_features(df))

        # Extended features (require extra API calls)
        if cfg.use_fundamentals and ticker:
            try:
                from src.features.fundamental_features import compute_fundamental_features
                fund_df = compute_fundamental_features(ticker, df)
                feature_dfs.append(fund_df)
            except Exception as exc:
                logger.warning(f"  Fundamental features failed for {ticker}: {exc}")

        if cfg.use_macro:
            try:
                from src.features.macro_features import compute_macro_features
                macro_df = compute_macro_features(df, _cache=macro_cache)
                feature_dfs.append(macro_df)
            except Exception as exc:
                logger.warning(f"  Macro features failed: {exc}")

        if cfg.use_sentiment and ticker:
            try:
                from src.features.sentiment_features import compute_sentiment_features
                sent_df = compute_sentiment_features(ticker, df)
                feature_dfs.append(sent_df)
            except Exception as exc:
                logger.warning(f"  Sentiment features failed for {ticker}: {exc}")
                from src.features.sentiment_features import SENTIMENT_FEATURE_NAMES
                dummy_sent = pd.DataFrame(
                    0.0, index=df.index, columns=SENTIMENT_FEATURE_NAMES
                )
                feature_dfs.append(dummy_sent)

        # Combine all features
        all_features = pd.concat(feature_dfs, axis=1)

    
    # Add extra features if provided
    if extra_features_df is not None and "date" in df.columns:
        extra = extra_features_df.copy()
        if "date" in extra.columns:
            extra["date"] = pd.to_datetime(extra["date"])
            df["date"] = pd.to_datetime(df["date"])
            extra = extra.set_index("date")
            all_features = all_features.join(
                extra.reindex(pd.to_datetime(df["date"]).values, method="ffill").reset_index(drop=True)
            )
    
    # Replace inf with NaN then fill
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(0)
    
    feature_names = list(all_features.columns)
    logger.info(f"Computed {len(feature_names)} features: {feature_names}")
    
    # Normalize
    if cfg.normalize:
        all_features = normalize_features(all_features, window=cfg.norm_window, clip=cfg.clip_value)
    
    # Compute targets
    labels, future_returns = compute_targets(df, cfg)
    
    # Build sequences with stride
    # Start after enough warmup for all indicators
    warmup = max(cfg.seq_len, 60, cfg.norm_window)
    end_idx = len(df) - cfg.forecast_horizon
    
    sequences = []
    targets = []
    returns = []
    
    feature_values = all_features.values.astype(np.float32)
    
    # Use stride to reduce sequence overlap (stride=1 keeps all, stride=5 keeps every 5th)
    for i in range(warmup, end_idx, cfg.stride):
        seq = feature_values[i - cfg.seq_len:i]
        if seq.shape[0] == cfg.seq_len:
            sequences.append(seq)
            targets.append(labels[i])
            returns.append(future_returns[i])
    
    if not sequences:
        logger.warning("No sequences generated — data too short?")
        return np.array([]), np.array([]), np.array([]), feature_names
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets)
    returns = np.array(returns, dtype=np.float32)
    
    logger.info(f"Generated {len(sequences)} sequences with {len(feature_names)} features (stride={cfg.stride})")
    
    # Log class distribution
    if cfg.n_classes > 0:
        unique, counts = np.unique(targets, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  Class {u}: {c} ({c/len(targets)*100:.1f}%)")
    
    return sequences, targets, returns, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing enhanced feature pipeline...\n")
    
    # Create synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.randn(n) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })
    
    # Test 3-class config
    cfg = FeatureConfig(n_classes=3, normalize=True)
    seqs, targets, rets, names = prepare_enhanced_daily_data(df, cfg)
    
    print(f"\nDataset shape: {seqs.shape}")
    print(f"Features ({len(names)}): {names}")
    print(f"Target classes: {np.unique(targets, return_counts=True)}")
    print(f"Return range: [{rets.min():.4f}, {rets.max():.4f}]")
    
    # Check normalization
    print(f"\nFeature stats (should be ~N(0,1) after normalization):")
    for i, name in enumerate(names[:5]):
        vals = seqs[:, -1, i]  # Last timestep
        print(f"  {name:<30} mean={vals.mean():.4f}, std={vals.std():.4f}")
    
    # Test 5-class config
    cfg5 = FeatureConfig(n_classes=5)
    seqs5, targets5, rets5, names5 = prepare_enhanced_daily_data(df, cfg5)
    print(f"\n5-class targets: {np.unique(targets5, return_counts=True)}")
    
    # Test regression config
    cfg_reg = FeatureConfig(n_classes=0)
    seqs_r, targets_r, rets_r, names_r = prepare_enhanced_daily_data(df, cfg_reg)
    print(f"\nRegression targets range: [{targets_r.min():.4f}, {targets_r.max():.4f}]")
    
    print("\n✅ Enhanced feature pipeline test passed!")
