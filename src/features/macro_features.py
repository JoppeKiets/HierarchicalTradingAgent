#!/usr/bin/env python3
"""Macro / cross-asset features for market regime conditioning.

Fetches data from Yahoo Finance for:
  - VIX term structure (VIX vs VIX3M → contango/backwardation)
  - Treasury yields (10Y, 2Y, yield curve slope)
  - Credit spreads (HYG vs LQD as proxy)
  - Gold, USD index, Oil as risk/safe-haven proxies
  - Market breadth (S&P 500 RSP/SPY ratio)

All features are resampled to business-day frequency and forward-filled to
align with stock price data.

Features (15 total):
  macro_vix, macro_vix_sma20_dist, macro_vix_term_structure,
  macro_yield_10y, macro_yield_2y, macro_yield_curve,
  macro_credit_spread, macro_gold_momentum, macro_dxy_momentum,
  macro_oil_momentum, macro_breadth, macro_sp500_return_20d,
  macro_sp500_vol_20d, macro_fear_greed_proxy, macro_risk_on_off

Usage:
    from src.features.macro_features import compute_macro_features
    macro_df = compute_macro_features(prices_df)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Yahoo Finance tickers for macro data
_MACRO_TICKERS = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "tnx": "^TNX",       # 10-Year Treasury yield
    "irx": "^IRX",       # 13-Week T-Bill (proxy for 2Y when ^TYX unavailable)
    "sp500": "^GSPC",    # S&P 500
    "rsp": "RSP",        # Equal-weight S&P 500 ETF (breadth proxy)
    "spy": "SPY",        # S&P 500 ETF
    "hyg": "HYG",        # High-yield corporate bond ETF
    "lqd": "LQD",        # Investment-grade corporate bond ETF
    "gld": "GLD",        # Gold ETF
    "uup": "UUP",        # US Dollar Index ETF
    "uso": "USO",        # Oil ETF
}

MACRO_FEATURE_NAMES: List[str] = [
    "macro_vix",
    "macro_vix_sma20_dist",
    "macro_vix_term_structure",
    "macro_yield_10y",
    "macro_yield_2y",
    "macro_yield_curve",
    "macro_credit_spread",
    "macro_gold_momentum",
    "macro_dxy_momentum",
    "macro_oil_momentum",
    "macro_breadth",
    "macro_sp500_return_20d",
    "macro_sp500_vol_20d",
    "macro_fear_greed_proxy",
    "macro_risk_on_off",
]


def _download_series(
    ticker: str,
    period: str = "12y",
    column: str = "Close",
) -> pd.Series:
    """Download a single price/level series from Yahoo Finance.

    Returns a Series indexed by datetime (timezone-naive, business-day).
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return pd.Series(dtype=float)
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        series = data[column].dropna()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series
    except Exception as exc:
        logger.debug(f"  Could not download {ticker}: {exc}")
        return pd.Series(dtype=float)


def _download_all_macro(period: str = "12y") -> Dict[str, pd.Series]:
    """Fetch all macro series in one pass."""
    out: Dict[str, pd.Series] = {}
    for name, ticker in _MACRO_TICKERS.items():
        series = _download_series(ticker, period=period)
        if not series.empty:
            out[name] = series
            logger.debug(f"  {name} ({ticker}): {len(series)} rows")
        else:
            logger.debug(f"  {name} ({ticker}): no data")
    return out


def compute_macro_features(
    prices_df: pd.DataFrame,
    period: str = "12y",
    _cache: Dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute macro/cross-asset features aligned to a stock's daily dates.

    Args:
        prices_df: Price DataFrame with a ``date`` column.
        period: How far back to fetch macro data.
        _cache: Pre-fetched macro series (for multi-ticker efficiency).

    Returns:
        DataFrame of macro features with the same index as *prices_df*.
    """
    n = len(prices_df)

    # Parse dates
    if "date" in prices_df.columns:
        dates = pd.to_datetime(prices_df["date"])
    else:
        dates = pd.to_datetime(prices_df.index)
    daily_idx = pd.DatetimeIndex(dates)

    # Fetch macro data (or use cache)
    macro = _cache if _cache is not None else _download_all_macro(period)

    # Helper: reindex a series to our daily dates via forward-fill
    def _align(series: pd.Series) -> np.ndarray:
        if series.empty:
            return np.full(n, np.nan)
        return series.reindex(daily_idx, method="ffill").values

    result = pd.DataFrame(index=prices_df.index)

    # ── VIX features ─────────────────────────────────────────────────────
    vix = macro.get("vix", pd.Series(dtype=float))
    vix_aligned = _align(vix)
    result["macro_vix"] = vix_aligned / 100.0  # Scale to ~0-1 range

    # VIX distance from 20-day SMA (mean-reversion signal)
    if not vix.empty:
        vix_sma20 = vix.rolling(20, min_periods=1).mean()
        result["macro_vix_sma20_dist"] = (
            _align(vix) - _align(vix_sma20)
        ) / (_align(vix_sma20) + 1e-10)
    else:
        result["macro_vix_sma20_dist"] = 0.0

    # VIX term structure: VIX / VIX3M — < 1 = contango (complacent), > 1 = backwardation (fear)
    vix3m = macro.get("vix3m", pd.Series(dtype=float))
    if not vix.empty and not vix3m.empty:
        result["macro_vix_term_structure"] = _align(vix) / (_align(vix3m) + 1e-10)
    else:
        result["macro_vix_term_structure"] = 1.0  # neutral default

    # ── Treasury yields ──────────────────────────────────────────────────
    tnx = macro.get("tnx", pd.Series(dtype=float))
    irx = macro.get("irx", pd.Series(dtype=float))

    result["macro_yield_10y"] = _align(tnx) / 100.0 if not tnx.empty else 0.0
    result["macro_yield_2y"] = _align(irx) / 100.0 if not irx.empty else 0.0

    # Yield curve slope (10Y - 2Y) — inversion signals recession
    y10 = _align(tnx) if not tnx.empty else np.zeros(n)
    y2 = _align(irx) if not irx.empty else np.zeros(n)
    result["macro_yield_curve"] = (y10 - y2) / 100.0

    # ── Credit spread (HYG - LQD return differential) ────────────────────
    hyg = macro.get("hyg", pd.Series(dtype=float))
    lqd = macro.get("lqd", pd.Series(dtype=float))
    if not hyg.empty and not lqd.empty:
        hyg_ret = hyg.pct_change(20, fill_method=None).reindex(daily_idx, method="ffill").values
        lqd_ret = lqd.pct_change(20, fill_method=None).reindex(daily_idx, method="ffill").values
        result["macro_credit_spread"] = np.nan_to_num(hyg_ret - lqd_ret, 0.0)
    else:
        result["macro_credit_spread"] = 0.0

    # ── Safe-haven / risk asset momentum (20-day returns) ────────────────
    for name, col in [("gld", "macro_gold_momentum"), ("uup", "macro_dxy_momentum"), ("uso", "macro_oil_momentum")]:
        series = macro.get(name, pd.Series(dtype=float))
        if not series.empty:
            mom = series.pct_change(20, fill_method=None)
            result[col] = _align(mom)
        else:
            result[col] = 0.0

    # ── Market breadth (RSP/SPY ratio change) ────────────────────────────
    rsp = macro.get("rsp", pd.Series(dtype=float))
    spy = macro.get("spy", pd.Series(dtype=float))
    if not rsp.empty and not spy.empty:
        # Align to common index first
        common = rsp.index.intersection(spy.index)
        ratio = rsp.reindex(common) / (spy.reindex(common) + 1e-10)
        ratio_chg = ratio.pct_change(20, fill_method=None)
        result["macro_breadth"] = ratio_chg.reindex(daily_idx, method="ffill").fillna(0).values
    else:
        result["macro_breadth"] = 0.0

    # ── S&P 500 features ─────────────────────────────────────────────────
    sp500 = macro.get("sp500", pd.Series(dtype=float))
    if not sp500.empty:
        sp_ret = sp500.pct_change(20, fill_method=None)
        sp_vol = sp500.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
        result["macro_sp500_return_20d"] = _align(sp_ret)
        result["macro_sp500_vol_20d"] = _align(sp_vol)
    else:
        result["macro_sp500_return_20d"] = 0.0
        result["macro_sp500_vol_20d"] = 0.0

    # ── Composite signals ────────────────────────────────────────────────
    # Fear/greed proxy: combine VIX level, credit spread, and breadth
    fg_vix = result["macro_vix"].values.copy()
    fg_credit = result["macro_credit_spread"].values.copy()
    fg_breadth = result["macro_breadth"].values.copy()
    # Higher VIX = more fear, wider credit spread = more fear,
    # negative breadth = more fear → combine as simple average (post-normalisation)
    result["macro_fear_greed_proxy"] = (
        -np.nan_to_num(fg_vix, nan=0.0) * 3  # VIX inversely related to greed
        + np.nan_to_num(fg_credit, nan=0.0)
        + np.nan_to_num(fg_breadth, nan=0.0)
    ) / 3.0

    # Risk-on / risk-off: gold momentum vs equity momentum
    gold_mom = result["macro_gold_momentum"].values.copy()
    sp_mom = result["macro_sp500_return_20d"].values.copy()
    result["macro_risk_on_off"] = np.nan_to_num(sp_mom, nan=0.0) - np.nan_to_num(gold_mom, nan=0.0)

    # ── Cleanup ──────────────────────────────────────────────────────────
    result = result.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    logger.info(f"  Macro features: {len(result.columns)} columns computed")
    return result


# ─── CLI self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.yahoo_data_loader import YahooDataLoader

    ticker = "AAPL"
    yahoo = YahooDataLoader()
    prices = yahoo.fetch_price_history(ticker, period="5y")

    print(f"\nFetched {len(prices)} price bars for {ticker}")

    macro = compute_macro_features(prices)

    print(f"\nMacro features shape: {macro.shape}")
    print(f"Columns: {list(macro.columns)}")
    print(f"\nLast row:\n{macro.iloc[-1]}")
    print(f"\nDescribe:\n{macro.describe()}")
    print("\n✅ Macro features test passed!")
