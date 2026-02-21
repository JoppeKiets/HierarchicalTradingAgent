#!/usr/bin/env python3
"""Diversified ticker universe covering all major market sectors.

Provides 100+ liquid stocks across all GICS sectors for robust model training.
Focus on stocks with long trading history (10+ years) and high liquidity.
"""

from typing import List, Dict

# GICS Sector Classification
SECTORS = {
    "Technology": [
        # Large-cap tech
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO",
        "INTC", "AMD", "TXN", "QCOM", "IBM", "NOW", "INTU", "AMAT", "MU", "LRCX",
        # Mid-cap tech
        "SNPS", "CDNS", "KLAC", "MCHP", "ADI", "FTNT", "PANW",
    ],
    "Financials": [
        # Banks
        "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW",
        # Payment Networks
        "V", "MA",
        # Insurance
        "BRK-B", "AIG", "MET", "PRU", "ALL", "TRV", "AFL",
        # Asset Management
        "BLK", "BX", "KKR", "APO",
    ],
    "Healthcare": [
        # Pharma
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD",
        # Healthcare Equipment
        "UNH", "TMO", "ABT", "DHR", "MDT", "SYK", "BSX", "EW",
        # Biotech
        "VRTX", "REGN", "BIIB", "MRNA",
    ],
    "Consumer_Discretionary": [
        # Retail
        "AMZN", "HD", "LOW", "TJX", "ROST", "TGT", "COST",
        # Auto
        "TSLA", "GM", "F",
        # Entertainment
        "DIS", "NFLX", "CMCSA", "NKE", "SBUX", "MCD", "YUM",
    ],
    "Consumer_Staples": [
        "PG", "KO", "PEP", "WMT", "PM", "MO", "MDLZ", "CL", "KMB", "GIS",
        "K", "SJM", "CAG", "HSY",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
        "DVN", "HES", "FANG", "BKR",
    ],
    "Industrials": [
        # Aerospace & Defense
        "BA", "LMT", "RTX", "NOC", "GD", "GE",
        # Industrial Equipment
        "CAT", "DE", "HON", "MMM", "EMR", "ITW",
        # Transportation
        "UPS", "FDX", "UNP", "CSX", "NSC",
        # Other Industrials
        "WM", "RSG", "ROK", "ETN",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED",
    ],
    "Real_Estate": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    ],
    "Communication_Services": [
        "GOOG", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO",
    ],
}


def get_all_tickers() -> List[str]:
    """Get all tickers from all sectors (100+ stocks)."""
    all_tickers = []
    for sector_tickers in SECTORS.values():
        all_tickers.extend(sector_tickers)
    # Remove duplicates (e.g., GOOGL and GOOG)
    return list(dict.fromkeys(all_tickers))


def get_sector_tickers(sector: str) -> List[str]:
    """Get tickers for a specific sector."""
    return SECTORS.get(sector, [])


def get_diversified_sample(n: int = 50) -> List[str]:
    """Get a diversified sample of n tickers, balanced across sectors."""
    tickers = []
    sector_list = list(SECTORS.keys())
    n_sectors = len(sector_list)
    per_sector = max(n // n_sectors, 2)
    
    for sector in sector_list:
        sector_tickers = SECTORS[sector][:per_sector]
        tickers.extend(sector_tickers)
    
    return tickers[:n]


def get_liquid_large_caps(n: int = 30) -> List[str]:
    """Get the most liquid large-cap stocks across sectors."""
    # Top picks from each sector (most liquid)
    liquid = [
        # Tech (6)
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO",
        # Financials (4)
        "JPM", "BAC", "GS", "BRK-B",
        # Healthcare (4)
        "JNJ", "UNH", "PFE", "LLY",
        # Consumer Disc (4)
        "AMZN", "TSLA", "HD", "MCD",
        # Consumer Staples (3)
        "PG", "KO", "WMT",
        # Energy (3)
        "XOM", "CVX", "COP",
        # Industrials (3)
        "CAT", "HON", "UPS",
        # Utilities (2)
        "NEE", "DUK",
        # Materials (1)
        "LIN",
    ]
    return liquid[:n]


# Precomputed lists for convenience
TICKERS_100 = get_all_tickers()
TICKERS_50 = get_diversified_sample(50)
TICKERS_80 = get_diversified_sample(80)
TICKERS_30 = get_liquid_large_caps(30)

# Held-out evaluation tickers — NEVER use these for training.
# Spans diverse sectors to test generalization.
EVAL_HOLDOUT = [
    "V",       # Financials (payment network)
    "MA",      # Financials (payment network)
    "COST",    # Consumer Staples (retail)
    "NKE",     # Consumer Discretionary (apparel)
    "LLY",     # Healthcare (pharma)
    "AVGO",    # Technology (semiconductors)
    "UNP",     # Industrials (railroad)
    "DIS",     # Communication/Entertainment
    "NEM",     # Materials (gold mining)
    "PSA",     # Real Estate (storage)
]

# Training-safe tickers (excludes eval holdout)
TRAIN_TICKERS_90 = [t for t in TICKERS_100 if t not in set(EVAL_HOLDOUT)]

# Default for training
DEFAULT_TICKERS = TICKERS_80


if __name__ == "__main__":
    print(f"Total tickers available: {len(TICKERS_100)}")
    print(f"\nTickers by sector:")
    for sector, tickers in SECTORS.items():
        print(f"  {sector}: {len(tickers)} stocks")
    
    print(f"\nDiversified 50: {TICKERS_50}")
    print(f"\nLiquid 30: {TICKERS_30}")
