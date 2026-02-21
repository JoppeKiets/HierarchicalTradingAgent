#!/usr/bin/env python3
"""Continuous minute data collector for building historical dataset.

Yahoo Finance only provides 8 days of minute data at a time.
This script collects and stores minute data incrementally, building up
a historical dataset over time.

Usage:
  # One-time collection for all tickers
  python collect_minute_data.py --tickers AAPL MSFT GOOGL

  # Continuous collection (run as cron job or systemd service)
  python collect_minute_data.py --tickers AAPL MSFT --continuous --interval 3600

  # Collect for top N liquid stocks
  python collect_minute_data.py --top-liquid 50

  # Check what data we have
  python collect_minute_data.py --status

Cron example (collect every 4 hours during market hours):
  0 9,13,17 * * 1-5 cd /path/to/tradingAgent && python collect_minute_data.py --top-liquid 100
"""

import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from src.minute_data_loader import MinuteDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MinuteDataCollector:
    """Collects and stores minute-level data incrementally."""
    
    def __init__(self, data_dir: str = "data/minute_history"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.loader = MinuteDataLoader()
        
        # Metadata file tracks collection history
        self.metadata_file = self.data_dir / "collection_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load collection metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"tickers": {}, "last_collection": None}
    
    def _save_metadata(self):
        """Save collection metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def collect_ticker(self, ticker: str, period: str = "8d") -> int:
        """Collect minute data for a single ticker.
        
        Returns:
            Number of new rows added
        """
        logger.info(f"Collecting minute data for {ticker}...")
        
        # Fetch latest data from Yahoo
        try:
            new_data = self.loader.fetch_minute_bars(ticker, period=period)
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return 0
        
        if new_data.empty:
            logger.warning(f"No minute data returned for {ticker}")
            return 0
        
        # Add technical indicators
        new_data = self.loader.add_technical_indicators(new_data)
        
        # Reset index to get timestamp as column
        new_data = new_data.reset_index()
        new_data.columns = ['timestamp'] + list(new_data.columns[1:])
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        
        # Load existing data if any
        ticker_file = self.data_dir / f"{ticker}.parquet"
        if ticker_file.exists():
            existing = pd.read_parquet(ticker_file)
            existing['timestamp'] = pd.to_datetime(existing['timestamp'])
            
            # Find new rows (by timestamp)
            existing_timestamps = set(existing['timestamp'])
            new_rows = new_data[~new_data['timestamp'].isin(existing_timestamps)]
            
            if len(new_rows) > 0:
                # Append new data
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                combined.to_parquet(ticker_file, index=False)
                n_new = len(new_rows)
            else:
                n_new = 0
        else:
            # First collection for this ticker
            new_data.to_parquet(ticker_file, index=False)
            n_new = len(new_data)
        
        # Update metadata
        if ticker not in self.metadata["tickers"]:
            self.metadata["tickers"][ticker] = {
                "first_collection": datetime.now().isoformat(),
                "collections": 0,
                "total_rows": 0,
            }
        
        self.metadata["tickers"][ticker]["last_collection"] = datetime.now().isoformat()
        self.metadata["tickers"][ticker]["collections"] += 1
        self.metadata["tickers"][ticker]["total_rows"] = self._get_ticker_row_count(ticker)
        self._save_metadata()
        
        logger.info(f"  {ticker}: +{n_new} new rows (total: {self.metadata['tickers'][ticker]['total_rows']})")
        return n_new
    
    def _get_ticker_row_count(self, ticker: str) -> int:
        """Get total row count for a ticker."""
        ticker_file = self.data_dir / f"{ticker}.parquet"
        if ticker_file.exists():
            return len(pd.read_parquet(ticker_file))
        return 0
    
    def collect_batch(self, tickers: List[str], delay: float = 1.0) -> Dict[str, int]:
        """Collect minute data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            delay: Delay between requests to avoid rate limiting
            
        Returns:
            Dict of {ticker: new_rows_added}
        """
        results = {}
        total_new = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING MINUTE DATA FOR {len(tickers)} TICKERS")
        logger.info(f"{'='*60}\n")
        
        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
            n_new = self.collect_ticker(ticker)
            results[ticker] = n_new
            total_new += n_new
            
            if delay > 0 and i < len(tickers) - 1:
                time.sleep(delay)
        
        self.metadata["last_collection"] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTION COMPLETE: +{total_new} total new rows")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def get_status(self) -> pd.DataFrame:
        """Get status of all collected data."""
        rows = []
        for ticker, info in self.metadata.get("tickers", {}).items():
            ticker_file = self.data_dir / f"{ticker}.parquet"
            if ticker_file.exists():
                df = pd.read_parquet(ticker_file)
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                days = (max_date - min_date).days
            else:
                min_date = max_date = None
                days = 0
            
            rows.append({
                "ticker": ticker,
                "total_rows": info.get("total_rows", 0),
                "collections": info.get("collections", 0),
                "first_collection": info.get("first_collection", ""),
                "last_collection": info.get("last_collection", ""),
                "data_start": str(min_date)[:10] if min_date else "",
                "data_end": str(max_date)[:10] if max_date else "",
                "days_span": days,
            })
        
        return pd.DataFrame(rows).sort_values("total_rows", ascending=False)
    
    def load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load collected minute data for a ticker."""
        ticker_file = self.data_dir / f"{ticker}.parquet"
        if ticker_file.exists():
            df = pd.read_parquet(ticker_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp')
        return None
    
    def get_combined_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Get combined minute data for multiple tickers."""
        if tickers is None:
            tickers = list(self.metadata.get("tickers", {}).keys())
        
        dfs = []
        for ticker in tickers:
            df = self.load_ticker_data(ticker)
            if df is not None:
                df['ticker'] = ticker
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


def get_top_liquid_tickers(n: int = 548) -> List[str]:
    """Get top N stocks for minute data collection.

    548 tickers selected from data/organized with >=2000 daily rows,
    diversified across all 11 GICS sectors, plus existing minute tickers.
    Seed=42 for reproducibility.
    """
    top_tickers = [
        # ── Mega-cap tech (original + expanded) ──────────────────────────
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
        "ORCL", "ADBE", "NFLX", "PYPL", "CSCO", "AVGO", "TXN", "QCOM", "IBM", "NOW",
        "ADI", "ADP", "ADSK", "AMAT", "DIOD", "EA", "KLAC", "LRCX", "MU", "MSI",
        "SWKS", "TER", "TYL", "COHR", "COHU", "CSPI", "IDCC", "KLIC",
        # ── Financial Services ───────────────────────────────────────────
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
        "PNC", "TFC", "COF", "BK", "STT", "FITB", "KEY", "RF", "CFG", "HBAN",
        "AFL", "AIG", "AON", "BAM", "BEN", "BOH", "BPOP", "CBSH", "CBU", "CFR",
        "CHCO", "CINF", "CMA", "CNA", "CSWC", "CTBI", "CVBF", "FCNCA", "FBP", "FBNC",
        "FFBC", "FHN", "FNB", "FULT", "GBCI", "GL", "INDB", "JEF", "L", "LNC",
        "MTB", "NTRS", "OFG", "ONB", "OPY", "PGR", "RJF", "TRMK", "TROW", "TRST",
        "TRV", "UBSI", "UFCS", "WAFD", "WASH", "WBS", "WSBC", "ZION", "AFG", "AJG",
        "ASB", "ASRV", "AROW", "BKSC", "BKTI", "BXMT", "CADE", "CTO", "DX", "FUND",
        "GV", "PEBK", "RDI", "STC", "VALU", "MMC", "BRO",
        # ── Healthcare ───────────────────────────────────────────────────
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "ISRG", "CVS", "CI", "HUM", "MCK", "CAH", "ZTS", "VRTX",
        "BAX", "BDX", "BIO", "CLDX", "CNMD", "EHC", "HAE", "HOLX", "HRTX", "LH",
        "MDT", "MMSI", "NHC", "OMI", "OSUR", "QDEL", "REGN", "RGEN", "SYK", "TECH",
        "THC", "UHS", "WST", "XRAY", "CYH", "ABEO", "ALKS", "ANIX", "BLFS", "BMRA",
        "VXRT", "XOMA",
        # ── Industrials ──────────────────────────────────────────────────
        "CAT", "DE", "HON", "UPS", "RTX", "BA", "LMT", "GE", "MMM", "UNP",
        "CMI", "DOV", "EMR", "ETN", "FDX", "GATX", "GGG", "GWW", "HEI", "HNI",
        "HUBB", "ITW", "J", "LUV", "MATX", "MEI", "MOD", "MTZ", "PNR", "ROL",
        "TKR", "TRN", "TRNS", "VMI", "WGO", "AIR", "AIT", "APOG", "ATRO", "AZZ",
        "CACI", "CRS", "CTS", "CW", "DCO", "DCI", "FLS", "FSS", "GFF", "HRB",
        "KMT", "MSA", "PATK", "PKOH", "TPC",
        # ── Consumer Cyclical ────────────────────────────────────────────
        "WMT", "HD", "PG", "COST", "NKE", "SBUX", "TGT",
        "LOW", "TJX", "BKNG", "MAR", "YUM", "CMG", "DG", "DLTR", "ROST", "ORLY",
        "DDS", "DIS", "F", "GPC", "GT", "HAS", "HELE", "HST", "HVT", "IAC",
        "LEN", "LZB", "MAT", "MCD", "MTCH", "PHM", "PVH", "VFC", "WHR", "WSM",
        "CBRL", "CAL", "CULP", "GCO", "LEE", "MCS", "WEN", "BSET",
        # ── Consumer Defensive ───────────────────────────────────────────
        "KO", "PEP", "CLX", "CPB", "CAG", "CHE", "CL", "EHC", "GIS", "K",
        "MKC", "MO", "SYY", "UVV", "COKE", "EDUC",
        # ── Communication Services ───────────────────────────────────────
        "CMCSA", "T", "TEVA", "TGNA", "NYT", "OMC", "IPG", "ERIC", "WPP",
        "ATNI", "CMTL", "ESCA", "IDC", "LRCX",
        # ── Energy ───────────────────────────────────────────────────────
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
        "APA", "CLF", "CDE", "EQT", "GFI", "GOLD", "HL", "HP", "MUR", "NBR",
        "NEM", "OII", "OKE", "RRC", "SU", "TDW", "USEG",
        # ── Real Estate ──────────────────────────────────────────────────
        "BDN", "BRT", "CUZ", "EGP", "FRT", "GTY", "HST", "IHT", "NNN", "OLP",
        "PSA", "UMH", "VNO", "FRPH", "PCH",
        # ── Utilities ────────────────────────────────────────────────────
        "AEP", "AWR", "AVA", "CMS", "CNP", "CWT", "DTE", "ED", "ES", "ETR",
        "EVRG", "LNT", "NEE", "NFG", "OGE", "PCG", "PNW", "SWX", "UGI", "WY",
        # ── Basic Materials ──────────────────────────────────────────────
        "APD", "AVY", "BHP", "CCK", "CMC", "DD", "ECL", "FMC", "FUL", "GSK",
        "IFF", "IP", "KWR", "LEG", "OLN", "ROG", "SCL", "VMC", "WDC",
        # ── Other / Mixed ────────────────────────────────────────────────
        "ARW", "ASH", "AVT", "B", "DLX", "EFX", "GLW", "GNTX", "GHC", "HPQ",
        "LEO", "NL", "PAYX", "RAND", "RGR", "RLI", "WDFC", "WWW",
        # ── Closed-end funds & international ADRs (keep for diversity) ───
        "ASA", "ASG", "BCV", "CEF", "CET", "CRF", "DNP", "EEA", "FAX", "FT",
        "GAB", "GAM", "HQH", "IAF", "JHI", "JHS", "JMM", "KF", "KTF", "MCI",
        "MMT", "MPV", "MXF", "NCA", "NNY", "NRT", "NUV", "PAI", "PCF", "PEO",
        "PMM", "PPT", "RVT", "TCI", "TMP", "TWN", "TY", "USA", "VBF",
        "BBVA", "BCE", "BP", "ERIC", "GFI", "GSK", "HMC", "IMO", "NVO", "SU",
        "TM", "TRP", "WPP",
        # ── Small-cap / micro-cap (high daily history) ───────────────────
        "AAME", "AB", "ABM", "ADX", "AEG", "AEM", "AGYS", "AIM", "ALCO", "ALOT",
        "ALX", "AMS", "AP", "ARL", "ARTW", "ASYS", "AXGN", "AXR", "BC", "BDL",
        "BELFA", "BH", "BKSC", "BRN", "CAMP", "CLFD", "CMU", "COHU", "CVR", "CVM",
        "CXE", "CXH", "DBD", "DGICB", "DJCO", "DXC", "DXR", "DYNT", "EBF", "ECF",
        "ELSE", "EML", "ESP", "FEIM", "FLXS", "FONR", "FRD", "GROW", "HCSG", "HE",
        "HOV", "ICCC", "IHT", "INTG", "JOB", "KELYA", "KEQU", "KOSS", "KTCC", "LEO",
        "LGL", "LXU", "LYTS", "MAYS", "MCY", "MSB", "MSEX", "MTR", "MTRN", "MYE",
        "NBN", "NC", "NEN", "NMI", "PHI", "PBT", "PDEX", "PW", "RAMP", "TPL",
        "TCCO", "TFX", "THMO", "TSI", "TV", "VLGEA", "VSH",
    ]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in top_tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:n]


def get_organized_tickers(data_dir: str = "data/organized") -> List[str]:
    """Get all tickers that have daily data in data/organized.

    This ensures we collect minute data for every ticker we could
    potentially train on, maximising the overlap between daily and
    minute datasets.
    """
    organized = Path(data_dir)
    if not organized.exists():
        return []
    tickers = sorted(
        d.name
        for d in organized.iterdir()
        if d.is_dir() and (d / "price_history.csv").exists()
    )
    return tickers


def run_continuous_collection(
    collector: MinuteDataCollector,
    tickers: List[str],
    interval_seconds: int = 3600,
    batch_size: int = 100,
):
    """Run continuous collection loop with batching for large ticker lists.

    When dealing with thousands of tickers the full round can take longer
    than *interval_seconds*.  We therefore collect in batches, rotating
    through the full list across iterations so that every ticker is
    eventually visited.
    """
    logger.info(f"Starting continuous collection every {interval_seconds}s")
    logger.info(f"Tickers: {len(tickers)} (batch_size={batch_size})")
    logger.info("Press Ctrl+C to stop\n")

    offset = 0

    while True:
        try:
            batch = tickers[offset : offset + batch_size]
            if not batch:
                offset = 0
                batch = tickers[offset : offset + batch_size]
            logger.info(f"Batch offset={offset}, size={len(batch)}")
            collector.collect_batch(batch, delay=0.5)
            offset += batch_size
            if offset >= len(tickers):
                offset = 0
            logger.info(f"Sleeping for {interval_seconds}s...")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopping continuous collection")
            break
        except Exception as e:
            logger.error(f"Error in collection loop: {e}")
            time.sleep(60)  # Wait a minute before retrying


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect minute-level stock data incrementally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect for specific tickers
  python collect_minute_data.py --tickers AAPL MSFT GOOGL

  # Collect for top 50 liquid stocks
  python collect_minute_data.py --top-liquid 50

  # Run continuously (every hour)
  python collect_minute_data.py --top-liquid 20 --continuous --interval 3600

  # Check collection status
  python collect_minute_data.py --status
        """
    )
    
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to collect")
    parser.add_argument("--top-liquid", type=int, help="Collect top N liquid stocks from built-in list")
    parser.add_argument("--all-organized", action="store_true",
                       help="Collect for ALL tickers that have daily data in data/organized")
    parser.add_argument("--data-dir", default="data/minute_history",
                       help="Directory to store minute data")
    parser.add_argument("--continuous", action="store_true",
                       help="Run in continuous collection mode")
    parser.add_argument("--interval", type=int, default=3600,
                       help="Seconds between collections in continuous mode")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Tickers per batch in continuous mode")
    parser.add_argument("--status", action="store_true",
                       help="Show collection status and exit")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between ticker requests (seconds)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    collector = MinuteDataCollector(data_dir=args.data_dir)
    
    # Status mode
    if args.status:
        status = collector.get_status()
        if len(status) == 0:
            print("No data collected yet.")
        else:
            print("\n" + "="*80)
            print("MINUTE DATA COLLECTION STATUS")
            print("="*80)
            print(f"\nData directory: {collector.data_dir}")
            print(f"Last collection: {collector.metadata.get('last_collection', 'Never')}")
            print(f"\nTickers: {len(status)}")
            print(f"Total rows: {status['total_rows'].sum():,}")
            print("\nPer-ticker breakdown:")
            print(status.to_string(index=False))
        return
    
    # Determine tickers
    if args.tickers:
        tickers = args.tickers
    elif args.all_organized:
        tickers = get_organized_tickers()
        logger.info(f"Collecting for ALL {len(tickers)} organized tickers")
    elif args.top_liquid:
        tickers = get_top_liquid_tickers(args.top_liquid)
    else:
        # Default: all organized tickers (maximises minute data coverage)
        tickers = get_organized_tickers()
        if not tickers:
            tickers = get_top_liquid_tickers(548)
        logger.info(f"Defaulting to {len(tickers)} tickers")
    
    # Run collection
    if args.continuous:
        run_continuous_collection(collector, tickers, args.interval,
                                  batch_size=args.batch_size)
    else:
        collector.collect_batch(tickers, delay=args.delay)
        
        # Print summary
        status = collector.get_status()
        if len(status) > 0:
            print(f"\nTotal minute data: {status['total_rows'].sum():,} rows across {len(status)} tickers")


if __name__ == "__main__":
    main()
