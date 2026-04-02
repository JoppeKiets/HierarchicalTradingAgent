"""Ticker metadata: sector, market cap, and other static info.

This module provides utilities to fetch and cache sector and market cap
information for tickers, which is used by the hierarchical ensemble model.
"""

import yfinance as yf
import pandas as pd
import json
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Sector & market-cap constants (inlined — no external dependency)
# ============================================================================

SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Communication Services", "Industrials", "Consumer Staples",
    "Energy", "Utilities", "Real Estate", "Materials", "Unknown",
]
SECTOR_TO_ID = {s: i for i, s in enumerate(SECTORS)}
NUM_SECTORS = len(SECTORS)

MARKET_CAP_BUCKETS = ["Mega", "Large", "Mid", "Small", "Micro", "Nano", "Unknown"]
MCAP_TO_ID = {b: i for i, b in enumerate(MARKET_CAP_BUCKETS)}
NUM_MCAP_BUCKETS = len(MARKET_CAP_BUCKETS)


def get_market_cap_bucket(market_cap: float) -> str:
    """Classify a market cap value into a named bucket."""
    if market_cap >= 200e9:
        return "Mega"
    elif market_cap >= 10e9:
        return "Large"
    elif market_cap >= 2e9:
        return "Mid"
    elif market_cap >= 300e6:
        return "Small"
    elif market_cap >= 50e6:
        return "Micro"
    elif market_cap > 0:
        return "Nano"
    return "Unknown"

# Cache file location
CACHE_DIR = Path("data/metadata")
CACHE_FILE = CACHE_DIR / "ticker_metadata.json"


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Dict:
    """Load cached metadata."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict):
    """Save metadata cache."""
    _ensure_cache_dir()
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def _normalize_sector(sector: Optional[str]) -> str:
    """Normalize sector name to match our categories."""
    if not sector:
        return 'Unknown'
    
    # Map common variations to our standard names
    sector_map = {
        'technology': 'Technology',
        'information technology': 'Technology',
        'healthcare': 'Healthcare',
        'health care': 'Healthcare',
        'financials': 'Financials',
        'financial services': 'Financials',
        'financial': 'Financials',
        'consumer cyclical': 'Consumer Discretionary',
        'consumer discretionary': 'Consumer Discretionary',
        'communication services': 'Communication Services',
        'telecommunications': 'Communication Services',
        'industrials': 'Industrials',
        'industrial': 'Industrials',
        'consumer defensive': 'Consumer Staples',
        'consumer staples': 'Consumer Staples',
        'energy': 'Energy',
        'utilities': 'Utilities',
        'real estate': 'Real Estate',
        'basic materials': 'Materials',
        'materials': 'Materials',
    }
    
    sector_lower = sector.lower().strip()
    return sector_map.get(sector_lower, 'Unknown')


def fetch_ticker_metadata(ticker: str, use_cache: bool = True) -> Dict:
    """Fetch sector and market cap metadata for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        use_cache: Whether to use/update cache
        
    Returns:
        Dict with keys: sector, sector_id, market_cap, mcap_bucket, mcap_id
    """
    cache = _load_cache() if use_cache else {}
    
    if ticker in cache and use_cache:
        cached = cache[ticker]
        # Ensure IDs are present (in case of old cache format)
        if 'sector_id' not in cached:
            cached['sector_id'] = SECTOR_TO_ID.get(cached.get('sector', 'Unknown'), SECTOR_TO_ID['Unknown'])
        if 'mcap_id' not in cached:
            cached['mcap_id'] = MCAP_TO_ID.get(cached.get('mcap_bucket', 'Unknown'), MCAP_TO_ID['Unknown'])
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract sector
        raw_sector = info.get('sector', None)
        sector = _normalize_sector(raw_sector)
        sector_id = SECTOR_TO_ID.get(sector, SECTOR_TO_ID['Unknown'])
        
        # Extract market cap
        market_cap = info.get('marketCap', 0) or 0
        mcap_bucket = get_market_cap_bucket(market_cap)
        mcap_id = MCAP_TO_ID.get(mcap_bucket, MCAP_TO_ID['Unknown'])
        
        metadata = {
            'sector': sector,
            'sector_id': sector_id,
            'market_cap': market_cap,
            'mcap_bucket': mcap_bucket,
            'mcap_id': mcap_id,
            'industry': info.get('industry', 'Unknown'),
            'name': info.get('shortName', ticker),
        }
        
        # Update cache
        if use_cache:
            cache[ticker] = metadata
            _save_cache(cache)
            
        logger.info(f"{ticker}: sector={sector}, mcap_bucket={mcap_bucket}")
        return metadata
        
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {ticker}: {e}")
        return {
            'sector': 'Unknown',
            'sector_id': SECTOR_TO_ID['Unknown'],
            'market_cap': 0,
            'mcap_bucket': 'Unknown',
            'mcap_id': MCAP_TO_ID['Unknown'],
            'industry': 'Unknown',
            'name': ticker,
        }


def get_batch_metadata(tickers: list, use_cache: bool = True) -> pd.DataFrame:
    """Fetch metadata for multiple tickers.
    
    Returns DataFrame with columns: ticker, sector, sector_id, market_cap, mcap_bucket, mcap_id
    """
    records = []
    for ticker in tickers:
        meta = fetch_ticker_metadata(ticker, use_cache)
        meta['ticker'] = ticker
        records.append(meta)
    
    return pd.DataFrame(records)


def get_sector_id(ticker: str) -> int:
    """Quick helper to get sector ID for a ticker."""
    meta = fetch_ticker_metadata(ticker)
    return meta['sector_id']


def get_mcap_id(ticker: str) -> int:
    """Quick helper to get market cap bucket ID for a ticker."""
    meta = fetch_ticker_metadata(ticker)
    return meta['mcap_id']


def get_ids(ticker: str) -> Tuple[int, int]:
    """Get both sector_id and mcap_id for a ticker."""
    meta = fetch_ticker_metadata(ticker)
    return meta['sector_id'], meta['mcap_id']


if __name__ == "__main__":
    # Test the metadata fetcher
    logging.basicConfig(level=logging.INFO)
    
    test_tickers = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'WMT', 'GOOGL', 'AMZN']
    
    print("=== Ticker Metadata Test ===\n")
    
    df = get_batch_metadata(test_tickers)
    print(df[['ticker', 'name', 'sector', 'sector_id', 'mcap_bucket', 'mcap_id']].to_string(index=False))
    
    print(f"\n\nSectors: {SECTORS}")
    print(f"Market Cap Buckets: {MARKET_CAP_BUCKETS}")
    
    print("\n✓ Metadata test complete!")
