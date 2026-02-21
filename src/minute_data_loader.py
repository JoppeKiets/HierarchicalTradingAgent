"""Minute-level data loader with real-time versioning and technical indicators."""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange as ATRIndicator

logger = logging.getLogger(__name__)


class MinuteDataLoader:
    """Load and manage minute-level trading data."""
    
    def __init__(self, data_dir: str = "data/real-time"):
        """Initialize loader.
        
        Args:
            data_dir: Path to real-time data directory.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.today_dir = self.data_dir / self.today
        self.today_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_minute_bars(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "30d"
    ) -> pd.DataFrame:
        """Fetch minute-level bars from yfinance.
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL').
            start_date: Start date (YYYY-MM-DD), overrides period.
            end_date: End date (YYYY-MM-DD).
            period: Period if start_date not specified (e.g., '7d', '30d', '60d').
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex (minute-level)
        """
        logger.info(f"Fetching minute bars for {ticker}...")
        
        try:
            # Fetch using yfinance
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                period=period if start_date is None else None,
                interval="1m",
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data fetched for {ticker}")
                return pd.DataFrame()
            
            # Clean data
            data = data.dropna()
            
            # Handle multi-level columns from newer yfinance versions
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten: ('Close', 'AAPL') -> 'close'
                data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
            else:
                data.columns = [c.lower() for c in data.columns]
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"Missing column {col} for {ticker}")
                    return pd.DataFrame()
            
            data = data[required_cols]
            
            logger.info(f"Fetched {len(data)} minute bars for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def reconstruct_minute_data(
        self,
        ticker: str,
        date: str,
        close_price: float,
        volume: int
    ) -> pd.DataFrame:
        """
        Reconstruct minute-level data from daily close.
        
        Used for historical backtesting when minute data unavailable.
        Generates synthetic minute bars using random walk with daily close.
        
        Args:
            ticker: Stock ticker
            date: Trading date (YYYY-MM-DD)
            close_price: Daily close price
            volume: Daily volume
        
        Returns:
            DataFrame with 390 minute bars (09:30-16:00)
        """
        # 390 minutes in trading day (09:30-16:00)
        n_minutes = 390
        
        # Simulate minute bars converging to daily close
        np.random.seed(hash(ticker + date) % 2**32)  # Reproducible
        
        # Random walk to daily close
        returns = np.random.normal(0, 0.0005, n_minutes)
        returns[-1] = 0  # Ensure close price is exact
        
        minute_close = close_price * np.exp(np.cumsum(returns))
        minute_close[-1] = close_price  # Force exact daily close
        
        # Generate OHLC from closes
        minute_open = np.concatenate(([close_price * 0.995], minute_close[:-1]))
        minute_high = np.maximum(minute_open, minute_close) * 1.002
        minute_low = np.minimum(minute_open, minute_close) * 0.998
        minute_volume = volume // n_minutes  # Distribute daily volume
        
        # Create timestamps (09:30 to 16:00)
        market_open = datetime.strptime(date + " 09:30", "%Y-%m-%d %H:%M")
        timestamps = [market_open + timedelta(minutes=i) for i in range(n_minutes)]
        
        return pd.DataFrame({
            'open': minute_open,
            'high': minute_high,
            'low': minute_low,
            'close': minute_close,
            'volume': minute_volume
        }, index=timestamps)
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add technical indicators to minute bars.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to add. If None, adds standard set:
                       ['rsi', 'macd', 'ema', 'bb', 'atr', 'adx']
        
        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df
        
        if indicators is None:
            indicators = ['rsi', 'macd', 'ema', 'bb', 'atr', 'adx']
        
        df = df.copy()
        
        try:
            if 'rsi' in indicators:
                rsi = RSIIndicator(df['close'], window=14)
                df['rsi'] = rsi.rsi()
            
            if 'macd' in indicators:
                macd = MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_diff'] = macd.macd_diff()
            
            if 'ema' in indicators:
                for period in [5, 12, 26]:
                    ema = EMAIndicator(df['close'], window=period)
                    df[f'ema_{period}'] = ema.ema_indicator()
            
            if 'bb' in indicators:
                bb = BollingerBands(df['close'], window=20)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_mid'] = bb.bollinger_mavg()
            
            if 'atr' in indicators:
                atr = ATRIndicator(df['high'], df['low'], df['close'], window=14)
                df['atr'] = atr.average_true_range()
            
            if 'adx' in indicators:
                adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
                df['adx'] = adx.adx()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def save_minute_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        date: Optional[str] = None
    ) -> Path:
        """
        Save minute data to versioned directory.
        
        Args:
            ticker: Stock ticker
            df: DataFrame with minute bars
            date: Collection date (defaults to today)
        
        Returns:
            Path to saved file
        """
        if date is None:
            date = self.today
        
        ticker_dir = self.data_dir / date / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = ticker_dir / "minute_bars.parquet"
        df.to_parquet(output_path, compression='snappy')
        
        logger.info(f"Saved {len(df)} minute bars to {output_path}")
        return output_path
    
    def load_minute_data(
        self,
        ticker: str,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load minute data from versioned directory.
        
        Args:
            ticker: Stock ticker
            date: Collection date (defaults to latest)
        
        Returns:
            DataFrame with minute bars
        """
        if date is None:
            # Find latest date with data
            dates = sorted([d.name for d in self.data_dir.iterdir() 
                          if d.is_dir() and d.name != "latest"])
            if not dates:
                logger.warning(f"No minute data found for {ticker}")
                return pd.DataFrame()
            date = dates[-1]
        
        file_path = self.data_dir / date / ticker / "minute_bars.parquet"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} minute bars from {file_path}")
        return df
    
    def get_available_tickers(self, date: Optional[str] = None) -> List[str]:
        """
        Get list of tickers with available minute data.
        
        Args:
            date: Collection date (defaults to latest)
        
        Returns:
            List of ticker symbols
        """
        if date is None:
            dates = sorted([d.name for d in self.data_dir.iterdir() 
                          if d.is_dir() and d.name != "latest"])
            if not dates:
                return []
            date = dates[-1]
        
        date_dir = self.data_dir / date
        if not date_dir.exists():
            return []
        
        tickers = [d.name for d in date_dir.iterdir() if d.is_dir()]
        return sorted(tickers)
    
    def validate_data(self, df: pd.DataFrame, ticker: str) -> Dict[str, object]:
        """
        Validate minute data integrity.
        
        Args:
            df: DataFrame with minute bars
            ticker: Stock ticker (for logging)
        
        Returns:
            Dict with validation results
        """
        results = {
            'ticker': ticker,
            'valid': True,
            'issues': []
        }
        
        if df.empty:
            results['valid'] = False
            results['issues'].append("Empty DataFrame")
            return results
        
        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            results['valid'] = False
            results['issues'].append(f"Missing columns: {missing}")
        
        # Check for NaN values
        nan_count = df[required].isna().sum().sum()
        if nan_count > 0:
            results['valid'] = False
            results['issues'].append(f"{nan_count} NaN values found")
        
        # Check OHLC logic (high >= low, etc.)
        bad_logic = (df['high'] < df['low']).sum()
        if bad_logic > 0:
            results['valid'] = False
            results['issues'].append(f"{bad_logic} bars with high < low")
        
        # Check monotonic timestamps
        if not df.index.is_monotonic_increasing:
            results['valid'] = False
            results['issues'].append("Timestamps not monotonic")
        
        # Check minute spacing (allowing for market hours gaps)
        time_diffs = df.index.to_series().diff()
        expected_diff = timedelta(minutes=1)
        bad_spacing = (time_diffs != expected_diff).sum()
        
        if bad_spacing > 10:  # Allow some gaps (market hours)
            results['issues'].append(f"Non-minute spacing in {bad_spacing} bars")
        
        return results
    
    def create_metadata(
        self,
        ticker: str,
        date: Optional[str] = None,
        source: str = "yfinance",
        rows: int = 0
    ) -> Dict:
        """
        Create metadata for versioned data.
        
        Args:
            ticker: Stock ticker
            date: Collection date
            source: Data source
            rows: Number of rows
        
        Returns:
            Dict with metadata
        """
        if date is None:
            date = self.today
        
        metadata = {
            'ticker': ticker,
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'rows': rows,
            'granularity': '1m',
            'market': 'US',
            'hours': '09:30-16:00 ET'
        }
        
        return metadata
    
    def save_metadata(
        self,
        ticker: str,
        metadata: Dict,
        date: Optional[str] = None
    ) -> Path:
        """Save metadata file for versioned data."""
        if date is None:
            date = self.today
        
        ticker_dir = self.data_dir / date / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = ticker_dir / "metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_path}")
        return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Fetch and save minute data for AAPL
    loader = MinuteDataLoader()
    
    # Fetch last 30 days
    df = loader.fetch_minute_bars('AAPL', period='30d')
    
    # Add indicators
    df = loader.add_technical_indicators(df)
    
    # Validate
    validation = loader.validate_data(df, 'AAPL')
    print(f"Validation: {validation}")
    
    # Save
    if validation['valid']:
        loader.save_minute_data('AAPL', df)
        metadata = loader.create_metadata('AAPL', rows=len(df))
        loader.save_metadata('AAPL', metadata)
