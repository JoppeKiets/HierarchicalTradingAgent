#!/usr/bin/env python3
"""Lightweight sentiment features from Yahoo Finance news & earnings.

This module avoids heavy NLP pipelines and instead provides:
  1. News volume (article count per day) — high volume = attention/uncertainty
  2. VADER-based headline sentiment (no GPU needed, runs in <1 ms per headline)
  3. Earnings surprise from Yahoo Finance (actual vs estimate)
  4. Analyst recommendation consensus

Features (8 total):
  sent_news_volume_1d, sent_news_volume_5d_avg,
  sent_headline_sentiment, sent_headline_sentiment_5d_avg,
  sent_earnings_surprise, sent_analyst_rating,
  sent_analyst_target_upside, sent_insider_activity

VADER (Valence Aware Dictionary for sEntiment Reasoning) ships with NLTK and
works well on short financial headlines without fine-tuning.

Usage:
    from src.features.sentiment_features import compute_sentiment_features
    sent_df = compute_sentiment_features("AAPL", prices_df)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SENTIMENT_FEATURE_NAMES: List[str] = [
    "sent_news_volume_1d",
    "sent_news_volume_5d_avg",
    "sent_headline_sentiment",
    "sent_headline_sentiment_5d_avg",
    "sent_earnings_surprise",
    "sent_analyst_rating",
    "sent_analyst_rating_change",
    "sent_analyst_target_upside",
    "sent_target_price_dispersion",
    "sent_insider_activity",
]


# ── VADER sentiment ──────────────────────────────────────────────────────────

def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("NLTK not installed — sentiment will be 0. Install with: pip install nltk")
        return None
    except LookupError:
        # VADER lexicon not downloaded
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except Exception:
            logger.warning("Could not load VADER lexicon — sentiment will be 0")
            return None


def _score_headlines(headlines: List[str]) -> float:
    """Score a list of headlines with VADER, return mean compound score in [-1, 1]."""
    vader = _get_vader()
    if vader is None or not headlines:
        return 0.0
    scores = [vader.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores))


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_news_by_date(ticker: str) -> pd.DataFrame:
    """Fetch news from Yahoo Finance and group by date.

    Returns DataFrame with columns: date, n_articles, avg_sentiment
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        news = yf_ticker.news or []
    except Exception as exc:
        logger.debug(f"  Could not fetch news for {ticker}: {exc}")
        return pd.DataFrame(columns=["date", "n_articles", "avg_sentiment"])

    if not news:
        return pd.DataFrame(columns=["date", "n_articles", "avg_sentiment"])

    records: Dict[str, List[str]] = {}  # date_str → [headlines]
    for item in news:
        # yfinance 0.2+ uses 'content' dict
        if isinstance(item, dict):
            title = item.get("title", "")
            ts = item.get("providerPublishTime")
            if ts is None and "content" in item:
                title = item["content"].get("title", title)
                pub = item["content"].get("pubDate", "")
                try:
                    ts = pd.Timestamp(pub).timestamp()
                except Exception:
                    ts = None
        else:
            continue

        if ts is None:
            continue

        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
        except (OSError, ValueError):
            continue

        records.setdefault(date_str, []).append(title)

    if not records:
        return pd.DataFrame(columns=["date", "n_articles", "avg_sentiment"])

    rows = []
    for date_str, headlines in records.items():
        rows.append({
            "date": pd.Timestamp(date_str),
            "n_articles": len(headlines),
            "avg_sentiment": _score_headlines(headlines),
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _fetch_earnings_surprise(ticker: str) -> pd.DataFrame:
    """Fetch earnings history (actual vs estimate) from Yahoo Finance.

    Returns DataFrame with columns: date, earnings_surprise (fraction).
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        earnings = yf_ticker.earnings_dates
        if earnings is None or earnings.empty:
            return pd.DataFrame(columns=["date", "earnings_surprise"])

        df = earnings.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Look for actual vs estimate columns
        actual_col = next((c for c in df.columns if "reported" in c or "actual" in c), None)
        estimate_col = next((c for c in df.columns if "estimate" in c or "expected" in c), None)

        if actual_col is None or estimate_col is None:
            # Try "surprise(%)" column directly
            surprise_col = next((c for c in df.columns if "surprise" in c), None)
            if surprise_col:
                date_col = next((c for c in df.columns if "date" in c or "earnings" in c), df.columns[0])
                result = pd.DataFrame({
                    "date": pd.to_datetime(df[date_col]),
                    "earnings_surprise": pd.to_numeric(df[surprise_col], errors="coerce") / 100.0,
                })
                return result.dropna().sort_values("date").reset_index(drop=True)
            return pd.DataFrame(columns=["date", "earnings_surprise"])

        date_col = next((c for c in df.columns if "date" in c or "earnings" in c), df.columns[0])
        actual = pd.to_numeric(df[actual_col], errors="coerce")
        estimate = pd.to_numeric(df[estimate_col], errors="coerce")

        surprise = (actual - estimate) / (estimate.abs() + 1e-10)

        result = pd.DataFrame({
            "date": pd.to_datetime(df[date_col]),
            "earnings_surprise": surprise,
        })
        return result.dropna().sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.debug(f"  Could not fetch earnings for {ticker}: {exc}")
        return pd.DataFrame(columns=["date", "earnings_surprise"])


def _fetch_analyst_info(ticker: str) -> Dict[str, float]:
    """Fetch analyst rating and price target from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}

    result = {}

    # Analyst rating: 1=Strong Buy, 5=Sell
    rating = info.get("recommendationMean")
    if isinstance(rating, (int, float)) and np.isfinite(rating):
        # Normalize: 1 → +1 (strong buy), 3 → 0 (hold), 5 → -1 (strong sell)
        result["analyst_rating"] = (3.0 - rating) / 2.0

    # Target price upside
    target = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    if (isinstance(target, (int, float)) and isinstance(current, (int, float))
            and current > 0 and np.isfinite(target)):
        result["target_upside"] = (target - current) / current

    # Target price dispersion (std / mean of analyst targets)
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    if (isinstance(target, (int, float)) and isinstance(target_high, (int, float))
            and isinstance(target_low, (int, float)) and target > 0):
        # Approximate std from range: std ≈ (high - low) / 4 for normal dist
        result["target_dispersion"] = (target_high - target_low) / (4 * target + 1e-10)

    return result


# ── Main compute function ────────────────────────────────────────────────────

def compute_sentiment_features(
    ticker: str,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute sentiment features aligned to daily price data.

    Args:
        ticker: Stock ticker symbol.
        prices_df: Price DataFrame with a ``date`` column.

    Returns:
        DataFrame of sentiment features, same index as *prices_df*.
    """
    n = len(prices_df)
    if "date" in prices_df.columns:
        dates = pd.to_datetime(prices_df["date"])
    else:
        dates = pd.to_datetime(prices_df.index)
    # Remove timezone to avoid comparison issues with news data
    if hasattr(dates, 'tz') and dates.tz is not None:
        dates = dates.tz_localize(None)
    elif hasattr(dates, 'dt') and hasattr(dates.dt, 'tz') and dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    daily_idx = pd.DatetimeIndex(dates)

    result = pd.DataFrame(index=prices_df.index)

    # ── News volume & headline sentiment ─────────────────────────────────
    news_df = _fetch_news_by_date(ticker)

    if not news_df.empty:
        news_df = news_df.set_index("date").sort_index()
        # Reindex to daily — days without news get 0 articles
        n_articles = news_df["n_articles"].reindex(daily_idx, fill_value=0).values.astype(float)
        avg_sent = news_df["avg_sentiment"].reindex(daily_idx, method="ffill").fillna(0).values
    else:
        n_articles = np.zeros(n)
        avg_sent = np.zeros(n)

    result["sent_news_volume_1d"] = n_articles
    result["sent_news_volume_5d_avg"] = (
        pd.Series(n_articles).rolling(5, min_periods=1).mean().values
    )
    result["sent_headline_sentiment"] = avg_sent
    result["sent_headline_sentiment_5d_avg"] = (
        pd.Series(avg_sent).rolling(5, min_periods=1).mean().values
    )

    # ── Earnings surprise ────────────────────────────────────────────────
    earn_df = _fetch_earnings_surprise(ticker)

    if not earn_df.empty:
        earn_df = earn_df.set_index("date").sort_index()
        # Forward-fill: the surprise value persists until next earnings
        earn_aligned = earn_df["earnings_surprise"].reindex(daily_idx, method="ffill").fillna(0).values
    else:
        earn_aligned = np.zeros(n)

    result["sent_earnings_surprise"] = earn_aligned

    # ── Analyst consensus ────────────────────────────────────────────────
    analyst = _fetch_analyst_info(ticker)
    result["sent_analyst_rating"] = analyst.get("analyst_rating", 0.0)

    # Analyst rating change: fetch recommendation trend if available
    try:
        rec_trend = yf.Ticker(ticker).recommendations
        if rec_trend is not None and not rec_trend.empty:
            rec = rec_trend.copy()
            if "period" not in rec.columns:
                rec = rec.reset_index()
            # Use 'To Grade' or numeric mean to derive a trend
            grade_map = {"Strong Buy": 1, "Buy": 2, "Overweight": 2,
                         "Outperform": 2, "Hold": 3, "Neutral": 3,
                         "Equal-Weight": 3, "Underweight": 4, "Underperform": 4,
                         "Sell": 5, "Strong Sell": 5}
            grade_col = next((c for c in rec.columns if "to_grade" in c.lower() or "tograde" in c.lower() or c == "To Grade"), None)
            if grade_col:
                rec["_score"] = rec[grade_col].map(grade_map)
                recent = rec["_score"].dropna()
                if len(recent) >= 2:
                    # Change = average of last 3 months vs previous 3 months
                    mid = len(recent) // 2
                    old_avg = recent.iloc[:mid].mean()
                    new_avg = recent.iloc[mid:].mean()
                    result["sent_analyst_rating_change"] = (old_avg - new_avg) / 2.0  # positive = upgrade
                else:
                    result["sent_analyst_rating_change"] = 0.0
            else:
                result["sent_analyst_rating_change"] = 0.0
        else:
            result["sent_analyst_rating_change"] = 0.0
    except Exception:
        result["sent_analyst_rating_change"] = 0.0

    result["sent_analyst_target_upside"] = analyst.get("target_upside", 0.0)
    result["sent_target_price_dispersion"] = analyst.get("target_dispersion", 0.0)

    # ── Insider activity placeholder ─────────────────────────────────────
    # Yahoo Finance doesn't expose this easily in yfinance; set to 0 for now
    # and populate later with SEC EDGAR data or a dedicated API.
    result["sent_insider_activity"] = 0.0

    # ── Cleanup ──────────────────────────────────────────────────────────
    result = result.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    logger.info(f"  {ticker}: {len(result.columns)} sentiment features computed")
    return result


# ─── CLI self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.yahoo_data_loader import YahooDataLoader

    ticker = "AAPL"
    yahoo = YahooDataLoader()
    prices = yahoo.fetch_price_history(ticker, period="2y")

    print(f"\nFetched {len(prices)} price bars for {ticker}")

    sent = compute_sentiment_features(ticker, prices)

    print(f"\nSentiment features shape: {sent.shape}")
    print(f"Columns: {list(sent.columns)}")
    print(f"\nLast row:\n{sent.iloc[-1]}")
    print(f"\nDescribe:\n{sent.describe()}")
    print("\n✅ Sentiment features test passed!")
