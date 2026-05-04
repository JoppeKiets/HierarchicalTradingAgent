#!/usr/bin/env python3
"""Backfill historical minute bars from Alpaca for all tickers.

Fetches 1-minute OHLCV bars and merges them into the existing parquet files
at data/minute_history/{TICKER}.parquet.

Usage:
    conda run -n trading311 python scripts/backfill_minute_alpaca.py \\
        --start 2020-01-01 --end 2026-01-25 \\
        --tickers-from data/organized \\
        --workers 4

    # Specific tickers only
    conda run -n trading311 python scripts/backfill_minute_alpaca.py \\
        --start 2020-01-01 --tickers AAPL MSFT GOOGL

    # Resume: skips tickers whose parquet already covers the requested range
    conda run -n trading311 python scripts/backfill_minute_alpaca.py \\
        --start 2020-01-01 --resume

Notes:
    - Alpaca free tier: ~200 req/min, each request can fetch up to 10,000 bars.
      For 1-min bars, one request covers ~26 trading days (10k / 390).
      A full year for one ticker needs ~10 requests.
    - Existing parquet data is preserved and merged (no duplicates).
    - Bars are stored with the same column schema as the yfinance collector:
        timestamp, open, high, low, close, volume
      (technical indicators are NOT pre-computed here; they are computed
       on-the-fly in _preprocess_minute_ticker in hierarchical_data.py)
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),  # stderr (visible in foreground)
        logging.FileHandler("logs/alpaca_backfill.log", mode="a"),  # always write to file
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINUTE_DIR = Path("data/minute_history")
DEFAULT_START = "2020-01-01"
# Alpaca: max bars per request (their documented limit)
ALPACA_MAX_BARS = 10_000
# Sleep between requests to stay under rate limits (free tier ~200 req/min)
REQUEST_DELAY = 0.35  # seconds


def make_client(api_key: str, secret_key: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_bars(
    client: StockHistoricalDataClient,
    ticker: str,
    start: datetime,
    end: datetime,
    request_delay: float = REQUEST_DELAY,
) -> pd.DataFrame:
    """Fetch all 1-min bars for ticker in [start, end) using pagination."""
    all_dfs = []
    cursor_start = start

    while cursor_start < end:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=cursor_start,
            end=end,
            limit=ALPACA_MAX_BARS,
        )
        try:
            bars = client.get_stock_bars(req)
            df = bars.df
        except Exception as e:
            logger.warning(f"  [{ticker}] fetch error ({cursor_start.date()}): {e}")
            time.sleep(5)
            break

        if df is None or df.empty:
            break

        # Reset multi-index: (symbol, timestamp) → just timestamp as column
        df = df.reset_index()
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        all_dfs.append(df)

        # Alpaca returns bars sorted ascending; check if we got a full page
        if len(df) < ALPACA_MAX_BARS:
            break  # Last page

        # Advance cursor past the last returned bar
        last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
        # Strip timezone so cursor stays tz-naive (matches start/end)
        last_naive = last_ts.to_pydatetime().replace(tzinfo=None)
        cursor_start = last_naive + timedelta(minutes=1)

        time.sleep(request_delay)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types to match the yfinance parquet schema."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Keep only the columns we need
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[keep]

    # Ensure timestamp is tz-aware UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def get_existing_range(parquet_path: Path):
    """Return (min_ts, max_ts) of existing parquet as tz-naive datetimes, or (None, None) if missing."""
    if not parquet_path.exists():
        return None, None
    try:
        df = pd.read_parquet(parquet_path, columns=["timestamp"])
        ts = pd.to_datetime(df["timestamp"])
        # Normalise: strip any timezone so comparisons with naive datetimes work
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        min_ts = ts.min().to_pydatetime()
        max_ts = ts.max().to_pydatetime()
        return min_ts, max_ts
    except Exception:
        return None, None


def merge_and_save(parquet_path: Path, new_df: pd.DataFrame):
    """Merge new_df with existing parquet (if any), dedup, and save."""
    if parquet_path.exists():
        try:
            existing = pd.read_parquet(parquet_path)
            # Drop any computed indicator columns — keep raw OHLCV only for the
            # new rows; existing rows keep their indicators intact
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"  Could not read existing parquet {parquet_path}: {e} — overwriting")
            combined = new_df
    else:
        combined = new_df

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, index=False)
    return len(combined)


def backfill_ticker(
    client: StockHistoricalDataClient,
    ticker: str,
    start: datetime,
    end: datetime,
    resume: bool,
    request_delay: float = REQUEST_DELAY,
) -> dict:
    """Backfill one ticker. Returns a status dict."""
    parquet_path = MINUTE_DIR / f"{ticker}.parquet"

    if resume:
        existing_min, existing_max = get_existing_range(parquet_path)
        if existing_min is not None:
            # existing_min/max are already tz-naive datetimes (from get_existing_range)
            # Already covers the entire requested range — skip
            if existing_min <= start + timedelta(days=5) and existing_max >= end - timedelta(days=5):
                return {"ticker": ticker, "status": "skipped", "rows": 0}
            # Partial coverage: only fetch the missing portion before existing data
            cutoff = existing_min - timedelta(minutes=1)
            if cutoff <= start:
                return {"ticker": ticker, "status": "skipped", "rows": 0}
            end = min(end, cutoff)

    logger.info(f"  [{ticker}] fetching {start.date()} → {end.date()}")
    new_df = fetch_bars(client, ticker, start, end, request_delay=request_delay)

    if new_df.empty:
        return {"ticker": ticker, "status": "no_data", "rows": 0}

    new_df = normalize_df(new_df)
    total_rows = merge_and_save(parquet_path, new_df)
    return {"ticker": ticker, "status": "ok", "rows": len(new_df), "total": total_rows}


def load_tickers_from_dir(organized_dir: str) -> List[str]:
    """Return sorted list of ticker names found as subdirectories."""
    return sorted(p.name for p in Path(organized_dir).iterdir() if p.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Backfill Alpaca minute bars")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to backfill")
    parser.add_argument(
        "--tickers-from",
        default="data/organized",
        help="Directory of ticker subdirs to use as ticker list (default: data/organized)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers whose parquet already covers the requested range",
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY,
                        help=f"Seconds to sleep between paginated requests (default: {REQUEST_DELAY})")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--secret-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ALPACA_API_KEY")
    secret_key = args.secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env or pass --api-key/--secret-key"
        )

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d")
        if args.end
        else datetime.combine(date.today() - timedelta(days=1), datetime.min.time())
    )

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = load_tickers_from_dir(args.tickers_from)

    logger.info(f"Backfilling {len(tickers)} tickers: {start_dt.date()} → {end_dt.date()}")
    logger.info(f"Workers: {args.workers} | Resume: {args.resume} | Delay: {args.delay}s")

    # Use one client per worker thread (thread-safe)
    def worker(ticker):
        c = make_client(api_key, secret_key)
        return backfill_ticker(c, ticker, start_dt, end_dt, args.resume, request_delay=args.delay)

    ok, skipped, no_data, errors = 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker = futures[fut]
            try:
                result = fut.result()
                status = result["status"]
                if status == "ok":
                    ok += 1
                    logger.info(
                        f"[{i}/{len(tickers)}] {ticker}: +{result['rows']:,} new rows "
                        f"(total {result.get('total', '?'):,})"
                    )
                elif status == "skipped":
                    skipped += 1
                    if i % 100 == 0:
                        logger.info(f"[{i}/{len(tickers)}] {ticker}: skipped (already covered)")
                else:
                    no_data += 1
                    logger.debug(f"[{i}/{len(tickers)}] {ticker}: no data returned")
            except Exception as e:
                errors += 1
                import traceback as _tb
                logger.error(f"[{i}/{len(tickers)}] {ticker}: exception — {e}\n{_tb.format_exc()}")

            # Throttle: small sleep between completions to avoid burst
            time.sleep(REQUEST_DELAY / args.workers)

    logger.info(
        f"\nDone. ok={ok} | skipped={skipped} | no_data={no_data} | errors={errors}"
    )


if __name__ == "__main__":
    main()
