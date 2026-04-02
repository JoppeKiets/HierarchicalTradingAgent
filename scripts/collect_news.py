#!/usr/bin/env python3
"""Unified News Collector and Processor.

This script:
1.  Fetches news metadata from Yahoo Finance (via yfinance)
2.  Follows links to extract full text (if possible)
3.  Calculates Summaries (Lexrank/Textrank) if new
4.  Updates data/organized/{TICKER}/news_articles.csv
5.  Updates collection_metadata.json tracking

Dependencies:
    pip install yfinance beautifulsoup4 requests pandas
"""

import os
import time
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
import socket
from requests.exceptions import RequestException
import yfinance as yf
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def fetch_full_text(url: str) -> Optional[str]:
    """Extremely basic full-text scraper for Yahoo and common publishers."""
    soup = None
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            break
        except RequestException as e:
            if attempt == 0:
                time.sleep(1.0)
                continue
            logger.debug(f"Failed to fetch text from {url}: {e}")
            return None

    if soup is None:
        return None

    # Yahoo Finance specific text
    if "finance.yahoo.com" in url:
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
            text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            if text:
                return text

    # Generic fallback: longest div with many paragraphs
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40])
    return text if len(text) > 100 else None

_RATE_LIMIT_BACKOFF: float = 0.0  # module-level shared back-off seconds


def collect_ticker_news(ticker: str, data_dir: str = "data/organized",
                        per_ticker_delay: float = 0.5) -> int:
    """Collects news for a single ticker and saves to organized structure.

    Args:
        per_ticker_delay: Minimum seconds to sleep between tickers (avoids
            rate-limiting even at normal traffic levels).
    """
    global _RATE_LIMIT_BACKOFF

    ticker_dir = Path(data_dir) / ticker
    news_file = ticker_dir / "news_articles.csv"

    # Ensure directory exists but skip if it doesn't (we only collect for known tickers)
    if not ticker_dir.exists():
        return 0

    # Load existing news for deduplication
    existing_urls = set()
    if news_file.exists():
        try:
            df_old = pd.read_csv(news_file)
            if "Url" in df_old.columns:
                existing_urls = set(df_old["Url"].dropna().tolist())
        except Exception as e:
            logger.warning(f"Error reading {news_file}: {e}")

    # Respect any active back-off from a previous rate-limit hit
    if _RATE_LIMIT_BACKOFF > 0:
        logger.info(f"Rate-limit back-off: sleeping {_RATE_LIMIT_BACKOFF:.0f}s ...")
        time.sleep(_RATE_LIMIT_BACKOFF)
        _RATE_LIMIT_BACKOFF = 0.0
    else:
        time.sleep(per_ticker_delay)

    # Fetch from yfinance — handle rate-limits and transient network/DNS errors
    yf_news = []
    for attempt in range(5):
        try:
            yf_ticker = yf.Ticker(ticker)
            yf_news = yf_ticker.news or []
            break  # success
        except Exception as e:
            err = str(e)
            # Rate-limit detection (yfinance may raise an error containing 429/Too Many Requests)
            if "429" in err or "Too Many Requests" in err or "rate limit" in err.lower():
                backoff = 60 * (2 ** attempt)  # 60s, 120s, 240s, ...
                logger.warning(f"[{ticker}] Rate limited (attempt {attempt+1}), sleeping {backoff}s ...")
                # set a global backoff so subsequent tickers pause briefly
                _RATE_LIMIT_BACKOFF = backoff
                time.sleep(backoff)
                continue

            # Network/DNS/transient errors — retry with short exponential backoff
            if isinstance(e, (RequestException, socket.gaierror, OSError)) or "Could not resolve host" in err:
                backoff = 5 * (2 ** attempt)  # 5s, 10s, 20s, ...
                logger.warning(f"[{ticker}] Network error (attempt {attempt+1}), sleeping {backoff}s: {err}")
                time.sleep(backoff)
                continue

            # Non-retryable error — log and skip this ticker
            logger.error(f"Error calling yfinance for {ticker}: {e}")
            return 0
    else:
        logger.error(f"[{ticker}] Gave up after {attempt+1} retries")
        return 0

    new_articles = []
    for item in yf_news:
        # ---- Extract fields from old OR new yfinance format ----
        content = item.get("content") or {}

        # URL: new format nests under content.canonicalUrl.url or content.clickThroughUrl.url
        url = (
            item.get("link")
            or (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
        )
        if not url or url in existing_urls:
            continue

        # Title
        title = item.get("title") or content.get("title")

        # Publisher
        publisher = item.get("publisher") or (content.get("provider") or {}).get("displayName")

        # Date: old format uses Unix timestamp, new format uses ISO string
        ts = item.get("providerPublishTime")
        pub_date_str = content.get("pubDate")  # e.g. "2026-03-23T11:01:59Z"
        if ts:
            dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        elif pub_date_str:
            try:
                dt_str = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, AttributeError):
                dt_str = pub_date_str
        else:
            dt_str = None

        # ---- Get article text ----
        # 1) Try full-text scrape from the URL
        text = fetch_full_text(url)
        time.sleep(0.5)

        # 2) Fallback: use yfinance-provided description (strip HTML) or summary
        if not text:
            raw_desc = content.get("description", "")
            if raw_desc:
                text = BeautifulSoup(raw_desc, "html.parser").get_text(strip=True)
            if not text or len(text) < 80:
                text = content.get("summary", "") or text

        # Only store if we got meaningful text
        if text and len(text) >= 50:
            new_articles.append({
                "Date": dt_str,
                "Article_title": title,
                "Stock_symbol": ticker,
                "Url": url,
                "Publisher": publisher,
                "Author": None,
                "Article": text,
                "Lsa_summary": None,
                "Luhn_summary": None,
                "Textrank_summary": None,
                "Lexrank_summary": None,
                "fetched_at": datetime.now(timezone.utc).isoformat()
            })

    if new_articles:
        df_new = pd.DataFrame(new_articles)
        if news_file.exists():
            df_final = pd.concat([pd.read_csv(news_file), df_new], ignore_index=True)
            # Sort by date — use mixed format parsing to handle old ("... UTC") and new formats
            df_final["Date"] = pd.to_datetime(df_final["Date"], format="mixed", utc=True, errors="coerce")
            df_final = df_final.sort_values("Date", ascending=False).drop_duplicates(subset=["Url"])
            df_final.to_csv(news_file, index=False)
        else:
            df_new.to_csv(news_file, index=False)
        
        logger.info(f"[{ticker}] Added {len(new_articles)} new articles")
        return len(new_articles)
    
    return 0

def run_pipeline(tickers: List[str], max_tickers: int = None, continuous: bool = False,
                 interval: int = 3600, per_ticker_delay: float = 0.5):
    """Main loop for collection."""
    if max_tickers:
        tickers = tickers[:max_tickers]

    while True:
        total_added = 0
        start_time = time.time()

        logger.info(f"Starting news collection round for {len(tickers)} tickers "
                    f"(delay={per_ticker_delay:.1f}s/ticker) ...")
        for ticker in tickers:
            try:
                added = collect_ticker_news(ticker, per_ticker_delay=per_ticker_delay)
                total_added += added
            except Exception as e:
                logger.error(f"Failed ticker {ticker}: {e}")
                
        logger.info(f"Round finished. Total added: {total_added}")
        
        if not continuous:
            break
            
        elapsed = time.time() - start_time
        wait_time = max(0, interval - elapsed)
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.0f}s until next round...")
            time.sleep(wait_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to collect")
    parser.add_argument("--max-tickers", type=int, help="Limit number of tickers")
    parser.add_argument("--all", action="store_true", help="Collect for all tickers in data/organized")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous collection mode")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between collections in continuous mode")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to sleep between individual ticker requests (default: 0.5). "
                             "Increase to 1.0–2.0 if you see rate-limit errors.")
    args = parser.parse_args()

    target_tickers = []
    if args.tickers:
        target_tickers = args.tickers
    elif args.all:
        target_tickers = [d.name for d in Path("data/organized").iterdir() if d.is_dir()]

    if not target_tickers:
        print("No tickers specified. Use --tickers, --all, or --max-tickers.")
    else:
        run_pipeline(target_tickers, args.max_tickers, args.continuous, args.interval,
                     per_ticker_delay=args.delay)
