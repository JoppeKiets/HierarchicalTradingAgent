"""Regime feature extractor for hierarchical ensemble.

Computes regime indicators that inform the meta-agent how to weight D vs M:
- Volatility regime (high/low/normal)
- VIX level and term structure
- Drawdown state
- Trend strength
- Market breadth proxies
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RegimeFeatureExtractor:
    """Extract regime features for meta-agent conditioning."""
    
    def __init__(
        self,
        vol_windows: Tuple[int, ...] = (20, 60, 120),
        trend_windows: Tuple[int, ...] = (20, 60, 120),
        drawdown_window: int = 252,
    ):
        self.vol_windows = vol_windows
        self.trend_windows = trend_windows
        self.drawdown_window = drawdown_window
    
    def compute_regime_features(
        self,
        prices_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute all regime features.
        
        Args:
            prices_df: Price data with 'date', 'close', 'volume' columns
            vix_df: Optional VIX data with 'date', 'close' columns
            
        Returns:
            DataFrame with regime features aligned to prices_df dates
        """
        df = prices_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        features = pd.DataFrame({'date': df['date']})
        
        # Returns
        df['return'] = df['close'].pct_change(fill_method=None)
        
        # Volatility features
        for w in self.vol_windows:
            features[f'vol_{w}d'] = df['return'].rolling(w).std() * np.sqrt(252)
        
        # Vol regime (current vol vs long-term)
        if len(self.vol_windows) >= 2:
            short_vol = features[f'vol_{self.vol_windows[0]}d']
            long_vol = features[f'vol_{self.vol_windows[-1]}d']
            features['vol_regime'] = short_vol / long_vol.clip(lower=0.01)
        
        # Trend strength (price vs moving averages)
        for w in self.trend_windows:
            ma = df['close'].rolling(w).mean()
            features[f'trend_{w}d'] = (df['close'] - ma) / ma.clip(lower=0.01)
        
        # Momentum
        for w in [20, 60, 120]:
            features[f'momentum_{w}d'] = df['close'] / df['close'].shift(w) - 1
        
        # Drawdown
        rolling_max = df['close'].rolling(self.drawdown_window, min_periods=1).max()
        features['drawdown'] = (df['close'] - rolling_max) / rolling_max
        features['drawdown_pct'] = features['drawdown'] * 100
        
        # Volume regime
        vol_ma = df['volume'].rolling(20).mean()
        features['volume_regime'] = df['volume'] / vol_ma.clip(lower=1)
        
        # Autocorrelation (mean-reversion vs momentum regime)
        features['autocorr_20d'] = df['return'].rolling(20).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        )
        
        # Skewness and kurtosis
        features['skew_60d'] = df['return'].rolling(60).skew()
        features['kurt_60d'] = df['return'].rolling(60).kurt()
        
        # VIX features if available
        if vix_df is not None:
            vix_df = vix_df.copy()
            vix_df['date'] = pd.to_datetime(vix_df['date'])
            vix_df = vix_df.rename(columns={'close': 'vix'})
            
            features = features.merge(vix_df[['date', 'vix']], on='date', how='left')
            features['vix'] = features['vix'].ffill().fillna(20)  # Default VIX
            
            # VIX regime (above/below 20)
            features['vix_regime'] = features['vix'] / 20
            
            # VIX change
            features['vix_change_5d'] = features['vix'].pct_change(5, fill_method=None)
        else:
            # Use realized vol as VIX proxy
            features['vix'] = features['vol_20d'] * 100  # Convert to VIX-like scale
            features['vix_regime'] = features['vix'] / 20
            features['vix_change_5d'] = features['vix'].pct_change(5, fill_method=None)
        
        # Fill NaN
        features = features.fillna(0)
        
        return features
    
    def get_regime_vector(
        self,
        features: pd.DataFrame,
        idx: int,
        vector_cols: Optional[list] = None,
    ) -> np.ndarray:
        """Get regime feature vector for a specific index.
        
        Default vector (8 features):
        - vol_regime, vix_regime, vix_change_5d
        - drawdown, trend_20d, momentum_20d
        - autocorr_20d, volume_regime
        """
        if vector_cols is None:
            vector_cols = [
                'vol_regime', 'vix_regime', 'vix_change_5d',
                'drawdown', 'trend_20d', 'momentum_20d',
                'autocorr_20d', 'volume_regime'
            ]
        
        # Ensure columns exist
        for col in vector_cols:
            if col not in features.columns:
                features[col] = 0
        
        return features.iloc[idx][vector_cols].values.astype(np.float32)
    
    def classify_regime(
        self,
        vol_regime: float,
        vix_level: float,
        trend: float,
        drawdown: float,
    ) -> str:
        """Classify current market regime for logging/debugging.
        
        Returns one of:
        - 'bull_low_vol': Strong uptrend, low volatility
        - 'bull_high_vol': Uptrend with elevated volatility
        - 'bear_low_vol': Downtrend, low volatility
        - 'bear_high_vol': Downtrend, high volatility (crisis)
        - 'sideways': No clear trend
        """
        high_vol = vix_level > 25 or vol_regime > 1.2
        uptrend = trend > 0.02
        downtrend = trend < -0.02
        deep_dd = drawdown < -0.10
        
        if deep_dd and high_vol:
            return 'bear_high_vol'
        elif downtrend and high_vol:
            return 'bear_high_vol'
        elif downtrend:
            return 'bear_low_vol'
        elif uptrend and high_vol:
            return 'bull_high_vol'
        elif uptrend:
            return 'bull_low_vol'
        else:
            return 'sideways'


def fetch_vix_data(
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch VIX data from Yahoo Finance.
    
    Returns:
        DataFrame with 'date', 'close' columns
    """
    try:
        import yfinance as yf
        
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        vix = yf.Ticker("^VIX")
        hist = vix.history(start=start_date, end=end_date)
        
        if len(hist) == 0:
            logger.warning("No VIX data fetched")
            return pd.DataFrame(columns=['date', 'close'])
        
        df = hist.reset_index()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'date': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        return df[['date', 'close']]
    
    except Exception as e:
        logger.error(f"Failed to fetch VIX data: {e}")
        return pd.DataFrame(columns=['date', 'close'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing RegimeFeatureExtractor...")
    
    # Create dummy price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices_df = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1_000_000, 10_000_000, 500),
    })
    
    # Extract features
    extractor = RegimeFeatureExtractor()
    features = extractor.compute_regime_features(prices_df)
    
    print(f"Feature columns: {list(features.columns)}")
    print(f"Feature shape: {features.shape}")
    print(f"\nSample features (last 5 rows):")
    print(features.tail())
    
    # Get regime vector
    vec = extractor.get_regime_vector(features, -1)
    print(f"\nRegime vector (latest): {vec}")
    print(f"Regime vector shape: {vec.shape}")
    
    # Classify regime
    latest = features.iloc[-1]
    regime = extractor.classify_regime(
        vol_regime=latest['vol_regime'],
        vix_level=latest['vix'],
        trend=latest['trend_20d'],
        drawdown=latest['drawdown'],
    )
    print(f"Current regime: {regime}")
    
    print("\n✓ RegimeFeatureExtractor test passed!")
