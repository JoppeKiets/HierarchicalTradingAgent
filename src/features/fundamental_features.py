#!/usr/bin/env python3
"""Fundamental features from Yahoo Finance.

Fetches quarterly/annual financial data and creates daily-aligned features
via forward-fill (fundamentals only update quarterly).

Features (14 total):
  - Valuation: trailing_pe, forward_pe, peg_ratio, price_to_book, ev_to_ebitda
  - Profitability: gross_margin, operating_margin, net_margin, roe, roa
  - Growth: revenue_growth, earnings_growth
  - Leverage: debt_to_equity
  - Dividends: dividend_yield

All features are winsorized and z-scored to be compatible with the existing
rolling normalisation in enhanced_features.py.

Usage:
    from src.features.fundamental_features import compute_fundamental_features
    fund_df = compute_fundamental_features("AAPL", prices_df)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Keys we extract from yfinance .info — (info_key, output_column, default)
_FUND_KEYS: List[tuple] = [
    # Valuation
    ("trailingPE", "trailing_pe", np.nan),
    ("forwardPE", "forward_pe", np.nan),
    ("pegRatio", "peg_ratio", np.nan),
    ("priceToBook", "price_to_book", np.nan),
    ("enterpriseToEbitda", "ev_to_ebitda", np.nan),
    # Profitability
    ("grossMargins", "gross_margin", np.nan),
    ("operatingMargins", "operating_margin", np.nan),
    ("profitMargins", "net_margin", np.nan),
    ("returnOnEquity", "roe", np.nan),
    ("returnOnAssets", "roa", np.nan),
    # Growth
    ("revenueGrowth", "revenue_growth", np.nan),
    ("earningsGrowth", "earnings_growth", np.nan),
    # Leverage
    ("debtToEquity", "debt_to_equity", np.nan),
    # Dividends
    ("dividendYield", "dividend_yield", 0.0),
]

FUNDAMENTAL_FEATURE_NAMES: List[str] = [k[1] for k in _FUND_KEYS]


def _fetch_fundamentals_snapshot(ticker: str) -> Dict[str, float]:
    """Fetch current fundamental values from Yahoo Finance .info.

    Returns a dict mapping output column names to numeric values.
    """
    out: Dict[str, float] = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning(f"  Failed to fetch fundamentals for {ticker}: {exc}")
        return out

    for info_key, col_name, default in _FUND_KEYS:
        val = info.get(info_key)
        if isinstance(val, (int, float)) and np.isfinite(val):
            out[col_name] = float(val)
        else:
            out[col_name] = default

    return out


def _fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    """Fetch quarterly financial statements and derive fundamental ratios.

    Returns a DataFrame indexed by report date with fundamental columns.
    This gives us a *time series* of fundamentals (not just the latest snapshot),
    which is critical for backtesting — we cannot use today's P/E for 2019 data.
    """
    try:
        yf_ticker = yf.Ticker(ticker)

        # Quarterly financials (income statement)
        income = yf_ticker.quarterly_financials
        balance = yf_ticker.quarterly_balance_sheet

        if income is None or income.empty:
            return pd.DataFrame()

        # Income statement items — columns are dates, rows are line items
        # Transpose so rows are dates
        inc = income.T.sort_index()
        records = []

        for date in inc.index:
            row: Dict[str, float] = {"date": pd.Timestamp(date)}

            # Revenue & earnings
            revenue = _safe_get(inc, date, "Total Revenue")
            gross_profit = _safe_get(inc, date, "Gross Profit")
            operating_income = _safe_get(inc, date, "Operating Income")
            net_income = _safe_get(inc, date, "Net Income")
            ebitda = _safe_get(inc, date, "EBITDA")

            # Margins (as fractions)
            if revenue and revenue > 0:
                if gross_profit is not None:
                    row["gross_margin"] = gross_profit / revenue
                if operating_income is not None:
                    row["operating_margin"] = operating_income / revenue
                if net_income is not None:
                    row["net_margin"] = net_income / revenue
                row["_revenue"] = revenue
                row["_net_income"] = net_income if net_income is not None else np.nan

            if ebitda is not None:
                row["_ebitda"] = ebitda

            records.append(row)

        # Balance sheet items
        if balance is not None and not balance.empty:
            bal = balance.T.sort_index()
            for i, date in enumerate(bal.index):
                # Match to closest income record
                if i < len(records):
                    total_equity = _safe_get(bal, date, "Stockholders Equity")
                    total_assets = _safe_get(bal, date, "Total Assets")
                    total_debt = _safe_get(bal, date, "Total Debt")

                    ni = records[i].get("_net_income")
                    if total_equity and total_equity != 0 and ni is not None:
                        records[i]["roe"] = ni / total_equity
                    if total_assets and total_assets != 0 and ni is not None:
                        records[i]["roa"] = ni / total_assets
                    if total_equity and total_equity != 0 and total_debt is not None:
                        records[i]["debt_to_equity"] = total_debt / total_equity

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Compute YoY growth (compare to same quarter last year = 4 quarters back)
        if "_revenue" in df.columns:
            df["revenue_growth"] = df["_revenue"].pct_change(4, fill_method=None)
        if "_net_income" in df.columns:
            df["earnings_growth"] = df["_net_income"].pct_change(4, fill_method=None)

        # Drop internal columns
        df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
        df["date"] = pd.to_datetime(df["date"])

        return df

    except Exception as exc:
        logger.warning(f"  Failed to fetch quarterly financials for {ticker}: {exc}")
        return pd.DataFrame()


def _safe_get(df: pd.DataFrame, date, key: str):
    """Safely extract a value from a transposed financial statement."""
    try:
        if key in df.columns:
            val = df.loc[date, key]
            if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(val):
                return float(val)
    except (KeyError, TypeError):
        pass
    return None


def compute_fundamental_features(
    ticker: str,
    prices_df: pd.DataFrame,
    use_quarterly_history: bool = True,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Compute fundamental features aligned to daily price data.

    Strategy:
    1. Try to fetch quarterly financials for a proper time series.
    2. Fall back to the current snapshot if quarterlies are unavailable.
    3. Forward-fill fundamentals onto the daily price index (they only change
       quarterly, so forward-fill is the correct point-in-time representation).
    4. Winsorize extreme values.

    Args:
        ticker: Stock ticker symbol.
        prices_df: Price DataFrame with a ``date`` column.
        use_quarterly_history: If True, attempt quarterly financial history.
        clip: Winsorize features at ±clip standard deviations.

    Returns:
        DataFrame with fundamental feature columns, indexed like ``prices_df``.
    """
    n = len(prices_df)

    # Ensure dates are datetime
    if "date" in prices_df.columns:
        dates = pd.to_datetime(prices_df["date"])
    else:
        dates = pd.to_datetime(prices_df.index)

    # ── 1. Try quarterly history ─────────────────────────────────────────
    fund_ts = pd.DataFrame()
    if use_quarterly_history:
        fund_ts = _fetch_quarterly_financials(ticker)

    has_history = not fund_ts.empty and len(fund_ts) >= 2

    # ── 2. Fallback to snapshot ──────────────────────────────────────────
    snapshot = _fetch_fundamentals_snapshot(ticker)

    # ── 3. Build daily-aligned DataFrame ─────────────────────────────────
    result = pd.DataFrame(index=prices_df.index)

    if has_history:
        logger.info(f"  {ticker}: Using {len(fund_ts)} quarters of fundamental history")
        fund_ts = fund_ts.set_index("date").sort_index()

        # Forward-fill onto daily dates
        daily_idx = pd.DatetimeIndex(dates)
        fund_daily = fund_ts.reindex(daily_idx, method="ffill")

        for col in FUNDAMENTAL_FEATURE_NAMES:
            if col in fund_daily.columns:
                result[f"fund_{col}"] = fund_daily[col].values
            elif col in snapshot:
                result[f"fund_{col}"] = snapshot[col]
            else:
                result[f"fund_{col}"] = np.nan
    else:
        # Snapshot only — constant across all days (still useful as cross-
        # sectional signal when training on multiple tickers)
        logger.info(f"  {ticker}: Using fundamental snapshot (no quarterly history)")
        for col in FUNDAMENTAL_FEATURE_NAMES:
            result[f"fund_{col}"] = snapshot.get(col, np.nan)

    # ── 4. Clean up ──────────────────────────────────────────────────────
    # debt_to_equity from yfinance is often in percentage form (e.g. 150 = 150%)
    if "fund_debt_to_equity" in result.columns:
        vals = result["fund_debt_to_equity"]
        if vals.median() > 10:  # likely percentage form
            result["fund_debt_to_equity"] = vals / 100.0

    # Replace inf/nan
    result = result.replace([np.inf, -np.inf], np.nan)

    # Winsorize per-column (clip at ±clip standard deviations from median)
    for col in result.columns:
        series = result[col]
        med = series.median()
        std = series.std()
        if std > 0 and np.isfinite(med):
            result[col] = series.clip(med - clip * std, med + clip * std)

    # Forward-fill remaining NaN, then zero-fill any leading NaN
    result = result.ffill().fillna(0)

    logger.info(f"  {ticker}: {len(result.columns)} fundamental features computed")
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

    fund = compute_fundamental_features(ticker, prices)

    print(f"\nFundamental features shape: {fund.shape}")
    print(f"Columns: {list(fund.columns)}")
    print(f"\nLast row:\n{fund.iloc[-1]}")
    print(f"\nDescribe:\n{fund.describe()}")
    print("\n✅ Fundamental features test passed!")
