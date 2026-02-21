"""Yahoo Finance data loader for prices, news, and fundamentals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


INTRADAY_INTERVALS: List[Tuple[str, str]] = [
    ("1m", "7d"),
    ("2m", "60d"),
    ("5m", "60d"),
    ("15m", "60d"),
    ("30m", "60d"),
    ("60m", "730d"),
]


@dataclass
class YahooNewsItem:
    title: str
    publisher: str
    link: str
    published_at: datetime
    summary: str = ""
    related_tickers: Optional[List[str]] = None

    def to_article(self, ticker: str) -> Dict:
        return {
            "title": self.title,
            "text": self.summary or self.title,
            "timestamp": self.published_at.isoformat(),
            "source": "yahoo_finance",
            "ticker": ticker,
            "publisher": self.publisher,
            "link": self.link,
            "related_tickers": self.related_tickers or [],
        }


class YahooDataLoader:
    """Wrapper around yfinance for prices, news, and fundamentals."""

    def fetch_price_history(
        self,
        ticker: str,
        interval: str = "1d",
        period: str = "2y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch price data and return standardized DataFrame.

        Columns: date, open, high, low, close, volume, adj_close
        """
        logger.info(f"Fetching {interval} prices for {ticker} (period={period})")

        data = yf.download(
            ticker,
            interval=interval,
            period=None if start else period,
            start=start,
            end=end,
            progress=False,
        )

        if data.empty:
            return pd.DataFrame()

        data = data.dropna().copy()
        
        # Handle multi-level columns from newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns (e.g., ('Close', 'AAPL') -> 'close')
            data.columns = [c[0].lower().replace(" ", "_") if isinstance(c, tuple) else c.lower().replace(" ", "_") for c in data.columns]
        else:
            data.columns = [c.lower().replace(" ", "_") for c in data.columns]

        if "adj_close" not in data.columns and "adjclose" in data.columns:
            data = data.rename(columns={"adjclose": "adj_close"})

        data = data.reset_index().rename(columns={"index": "date", "Date": "date"})
        data["date"] = pd.to_datetime(data["date"])

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in data.columns:
                data[col] = 0.0

        if "adj_close" not in data.columns:
            data["adj_close"] = data["close"]

        return data[["date", "open", "high", "low", "close", "volume", "adj_close"]]

    def fetch_best_intraday(self, ticker: str) -> Tuple[pd.DataFrame, str]:
        """Fetch the longest intraday series possible from Yahoo.

        Returns:
            (DataFrame, interval_used)
        """
        for interval, period in INTRADAY_INTERVALS:
            df = self.fetch_price_history(ticker, interval=interval, period=period)
            if not df.empty:
                return df, interval

        return pd.DataFrame(), ""

    def fetch_news(self, ticker: str, limit: int = 50) -> List[YahooNewsItem]:
        """Fetch recent news from Yahoo Finance via yfinance."""
        news_items: List[YahooNewsItem] = []
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as exc:
            logger.warning(f"Failed to fetch news for {ticker}: {exc}")
            return []

        for item in raw[:limit]:
            published_at = datetime.fromtimestamp(item.get("providerPublishTime", 0))
            news_items.append(
                YahooNewsItem(
                    title=item.get("title", ""),
                    publisher=item.get("publisher", ""),
                    link=item.get("link", ""),
                    published_at=published_at,
                    summary=item.get("summary", ""),
                    related_tickers=item.get("relatedTickers", []),
                )
            )

        return news_items

    def fetch_fundamentals(self, ticker: str) -> Dict[str, float]:
        """Fetch a small set of fundamentals/metadata from Yahoo.

        Returns a dict of numeric values only.
        """
        fundamentals: Dict[str, float] = {}
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info or {}
            fast = getattr(yf_ticker, "fast_info", {}) or {}
        except Exception as exc:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {exc}")
            return fundamentals

        numeric_keys = [
            "marketCap",
            "enterpriseValue",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToBook",
            "beta",
            "dividendYield",
            "profitMargins",
            "operatingMargins",
            "returnOnAssets",
            "returnOnEquity",
            "revenueGrowth",
            "earningsGrowth",
        ]

        for key in numeric_keys:
            val = info.get(key)
            if isinstance(val, (int, float)):
                fundamentals[key] = float(val)

        if isinstance(fast, dict):
            for key, val in fast.items():
                if isinstance(val, (int, float)):
                    fundamentals[f"fast_{key}"] = float(val)

        return fundamentals
