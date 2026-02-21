#!/usr/bin/env python3
"""Update daily price history for all organized tickers.

Appends new rows to each ticker's price_history.csv from Yahoo Finance,
bringing the daily data up to date. This is needed because the existing
data in data/organized/ only goes up to ~2023-12-28 while minute data
covers 2025-2026.

Usage:
    # Update all tickers (incremental — only downloads missing dates)
    python scripts/update_daily_data.py

    # Force full re-download for specific tickers
    python scripts/update_daily_data.py --tickers AAPL MSFT --force

    # Dry run (check what needs updating, don't download)
    python scripts/update_daily_data.py --dry-run

    # Limit to first N tickers (for testing)
    python scripts/update_daily_data.py --max-tickers 20
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ORGANIZED_DIR = "data/organized"


def get_all_tickers(organized_dir: str = ORGANIZED_DIR) -> List[str]:
    """Get all tickers that have a price_history.csv."""
    tickers = []
    for d in sorted(Path(organized_dir).iterdir()):
        if d.is_dir() and (d / "price_history.csv").exists():
            tickers.append(d.name)
    return tickers


def get_last_date(ticker: str, organized_dir: str = ORGANIZED_DIR) -> Optional[pd.Timestamp]:
    """Get the last date in a ticker's price_history.csv."""
    path = Path(organized_dir) / ticker / "price_history.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df["date"].max()
    except Exception:
        return None


def update_ticker(
    ticker: str,
    organized_dir: str = ORGANIZED_DIR,
    force: bool = False,
) -> dict:
    """Update a single ticker's daily data.

    Returns dict with status info.
    """
    import yfinance as yf

    path = Path(organized_dir) / ticker / "price_history.csv"
    result = {"ticker": ticker, "status": "unknown", "new_rows": 0}

    if not path.exists():
        result["status"] = "no_file"
        return result

    try:
        existing = pd.read_csv(path)
    except Exception as e:
        result["status"] = f"read_error: {e}"
        return result

    # Normalize column names
    existing.columns = [c.strip().lower() for c in existing.columns]
    if "date" not in existing.columns:
        result["status"] = "no_date_column"
        return result

    # Detect existing date format BEFORE converting to datetime
    raw_date_sample = str(existing["date"].iloc[0]) if len(existing) > 0 else ""
    existing_has_timestamp = ":" in raw_date_sample  # e.g. "2020-01-02 00:00:00"

    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    last_date = existing["date"].max()

    if pd.isna(last_date):
        result["status"] = "no_valid_dates"
        return result

    # Check if update is needed
    today = pd.Timestamp.now().normalize()
    if not force and last_date >= today - pd.Timedelta(days=2):
        result["status"] = "up_to_date"
        return result

    # Download new data starting from day after last existing date
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    try:
        new_df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        result["status"] = f"download_error: {e}"
        return result

    if new_df.empty:
        result["status"] = "no_new_data"
        return result

    # Handle multi-level columns from yfinance
    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = [c[0].lower() for c in new_df.columns]
    else:
        new_df.columns = [c.lower() for c in new_df.columns]

    new_df = new_df.reset_index()

    # Normalize 'Date' → 'date'
    if "Date" in new_df.columns:
        new_df = new_df.rename(columns={"Date": "date"})
    if "date" not in new_df.columns:
        # Index might be the date
        new_df = new_df.reset_index()
        if "Date" in new_df.columns:
            new_df = new_df.rename(columns={"Date": "date"})

    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")
    new_df = new_df.dropna(subset=["date"])

    # Filter to only truly new dates
    new_df = new_df[new_df["date"] > last_date]

    if new_df.empty:
        result["status"] = "no_new_data"
        return result

    # Ensure columns match existing file
    # Map common column names
    col_map = {
        "adj close": "adj close",
        "adjclose": "adj close",
    }
    for old_name, new_name in col_map.items():
        if old_name in new_df.columns and new_name not in new_df.columns:
            new_df = new_df.rename(columns={old_name: new_name})

    # Keep only columns that exist in the original
    existing_cols = list(existing.columns)
    # Format dates consistently — match the existing file's date format
    # (existing_has_timestamp was detected from raw CSV before pd.to_datetime conversion)
    if existing_has_timestamp:
        new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        existing["date"] = existing["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
        existing["date"] = existing["date"].dt.strftime("%Y-%m-%d")

    # Select matching columns, fill missing with NaN
    for col in existing_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[existing_cols]

    # Append and sort chronologically
    combined = pd.concat([existing, new_df], ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(path, index=False)

    result["status"] = "updated"
    result["new_rows"] = len(new_df)
    result["new_range"] = f"{new_df['date'].iloc[0]} → {new_df['date'].iloc[-1]}"

    return result


def run_update(
    tickers: Optional[List[str]] = None,
    organized_dir: str = ORGANIZED_DIR,
    force: bool = False,
    dry_run: bool = False,
    max_tickers: int = 0,
    batch_pause: float = 0.5,
):
    """Update daily data for all (or specified) tickers."""
    if tickers is None:
        tickers = get_all_tickers(organized_dir)

    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    logger.info(f"Updating daily data for {len(tickers)} tickers")

    if dry_run:
        logger.info("DRY RUN — checking what needs updating")
        today = pd.Timestamp.now().normalize()
        needs_update = 0
        for ticker in tickers:
            last = get_last_date(ticker, organized_dir)
            if last is None:
                continue
            gap = (today - last).days
            if gap > 2:
                needs_update += 1
                if needs_update <= 20:
                    logger.info(f"  {ticker}: last={last.date()}, gap={gap} days")
        logger.info(f"\n{needs_update}/{len(tickers)} tickers need updating")
        return

    stats = {"updated": 0, "up_to_date": 0, "error": 0, "total_new_rows": 0}
    t0 = time.time()

    for i, ticker in enumerate(tickers):
        result = update_ticker(ticker, organized_dir, force=force)

        if result["status"] == "updated":
            stats["updated"] += 1
            stats["total_new_rows"] += result["new_rows"]
            logger.info(f"  [{i+1}/{len(tickers)}] {ticker}: "
                        f"+{result['new_rows']} rows ({result.get('new_range', '?')})")
        elif result["status"] == "up_to_date":
            stats["up_to_date"] += 1
        elif result["status"] == "no_new_data":
            stats["up_to_date"] += 1
        else:
            stats["error"] += 1
            if stats["error"] <= 10:
                logger.warning(f"  [{i+1}/{len(tickers)}] {ticker}: {result['status']}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate
            logger.info(f"  Progress: {i+1}/{len(tickers)} "
                        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # Rate limiting to avoid hitting Yahoo Finance too hard
        if result["status"] == "updated":
            time.sleep(batch_pause)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*50}")
    logger.info(f"Daily data update complete in {elapsed:.0f}s")
    logger.info(f"  Updated:      {stats['updated']}")
    logger.info(f"  Up to date:   {stats['up_to_date']}")
    logger.info(f"  Errors:       {stats['error']}")
    logger.info(f"  New rows:     {stats['total_new_rows']:,}")
    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Update daily price data from Yahoo Finance")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to update (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if recently updated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check what needs updating without downloading")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="Limit to first N tickers (for testing)")
    parser.add_argument("--batch-pause", type=float, default=0.5,
                        help="Pause between downloads in seconds")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_update(
        tickers=args.tickers,
        force=args.force,
        dry_run=args.dry_run,
        max_tickers=args.max_tickers,
        batch_pause=args.batch_pause,
    )


if __name__ == "__main__":
    main()
