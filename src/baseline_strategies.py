#!/usr/bin/env python3
"""Baseline trading strategies for benchmarking.

Simple, well-understood strategies that any ML model MUST beat to prove value:
  1. Buy & Hold
  2. SMA Crossover (trend following)
  3. Mean Reversion (Bollinger Bands)
  4. Momentum (rate of change)
  5. Random (Monte Carlo baseline)
  6. Perfect Foresight (theoretical ceiling)

Each strategy takes a prices DataFrame and returns a signal array.
Signal values: [-1, +1] representing target position fraction.

Usage:
    from src.baseline_strategies import BuyAndHold, SMACrossover, ...
    strategy = SMACrossover(fast=20, slow=50)
    signals = strategy.generate(prices_df)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """Base class for all strategies."""
    
    name: str = "BaseStrategy"
    
    @abstractmethod
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        """Generate position signals from price data.
        
        Args:
            prices_df: DataFrame with at least 'close' column
            
        Returns:
            numpy array of position signals, same length as prices_df
            Values in [-1, +1] where +1 = fully long, -1 = fully short, 0 = flat
        """
        raise NotImplementedError


class BuyAndHold(BaseStrategy):
    """Always 100% long. The simplest possible benchmark."""
    
    name = "Buy & Hold"
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(prices_df))


class SMACrossover(BaseStrategy):
    """Trend-following: long when fast SMA > slow SMA, short otherwise."""
    
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow
        self.name = f"SMA({fast}/{slow})"
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        close = prices_df["close"].values
        n = len(close)
        signals = np.zeros(n)
        
        for i in range(self.slow, n):
            fast_sma = np.mean(close[i - self.fast + 1:i + 1])
            slow_sma = np.mean(close[i - self.slow + 1:i + 1])
            signals[i] = 1.0 if fast_sma > slow_sma else -0.5  # Asymmetric: more bullish bias
        
        return signals


class MACDStrategy(BaseStrategy):
    """MACD crossover: long when MACD > signal, short otherwise."""
    
    name = "MACD"
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def _ema(self, data: np.ndarray, span: int) -> np.ndarray:
        return pd.Series(data).ewm(span=span, adjust=False).mean().values
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        close = prices_df["close"].values
        
        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.signal_period)
        
        signals = np.where(macd_line > signal_line, 1.0, -0.5)
        # Warmup period
        signals[:self.slow] = 0.0
        
        return signals


class MeanReversion(BaseStrategy):
    """Mean reversion using Bollinger Bands.
    
    Buy when price drops below lower band (oversold),
    sell when price rises above upper band (overbought).
    """
    
    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = window
        self.n_std = n_std
        self.name = f"MeanRev(BB{window})"
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        close = prices_df["close"].values
        n = len(close)
        signals = np.zeros(n)
        
        for i in range(self.window, n):
            window_data = close[i - self.window + 1:i + 1]
            sma = np.mean(window_data)
            std = np.std(window_data)
            
            upper = sma + self.n_std * std
            lower = sma - self.n_std * std
            
            if std < 1e-10:
                signals[i] = 0.0
            else:
                # Z-score position
                z = (close[i] - sma) / std
                
                if close[i] < lower:
                    signals[i] = 1.0    # Buy oversold
                elif close[i] > upper:
                    signals[i] = -0.5   # Sell overbought
                elif abs(z) < 0.5:
                    signals[i] = 0.0    # Flat near mean
                else:
                    signals[i] = signals[i-1]  # Hold previous
        
        return signals


class MomentumStrategy(BaseStrategy):
    """Momentum / rate-of-change strategy.
    
    Long when N-day return is positive, short when negative.
    Position size scales with momentum strength.
    """
    
    def __init__(self, lookback: int = 20, scale: bool = True):
        self.lookback = lookback
        self.scale = scale
        self.name = f"Momentum({lookback}d)"
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        close = prices_df["close"].values
        n = len(close)
        signals = np.zeros(n)
        
        for i in range(self.lookback, n):
            roc = (close[i] - close[i - self.lookback]) / close[i - self.lookback]
            
            if self.scale:
                # Scale position by momentum strength (capped at ±1)
                signals[i] = np.clip(roc * 10, -1.0, 1.0)  # 10% move → full position
            else:
                signals[i] = 1.0 if roc > 0 else -0.5
        
        return signals


class RSIStrategy(BaseStrategy):
    """RSI-based strategy: buy oversold, sell overbought."""
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI({period})"
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        close = prices_df["close"].values
        n = len(close)
        signals = np.zeros(n)
        
        # Calculate RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        # Initial average
        if self.period < n:
            avg_gain[self.period] = np.mean(gain[1:self.period + 1])
            avg_loss[self.period] = np.mean(loss[1:self.period + 1])
        
            # Exponential smoothing
            for i in range(self.period + 1, n):
                avg_gain[i] = (avg_gain[i-1] * (self.period - 1) + gain[i]) / self.period
                avg_loss[i] = (avg_loss[i-1] * (self.period - 1) + loss[i]) / self.period
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss > 0, out=np.zeros(n))
        rsi = 100 - 100 / (1 + rs)
        
        # Generate signals
        for i in range(self.period + 1, n):
            if rsi[i] < self.oversold:
                signals[i] = 1.0
            elif rsi[i] > self.overbought:
                signals[i] = -0.5
            else:
                # Linear interpolation between oversold and overbought
                mid = (self.oversold + self.overbought) / 2
                if rsi[i] < mid:
                    signals[i] = 0.5  # Mild long
                else:
                    signals[i] = 0.0  # Flat
        
        return signals


class RandomStrategy(BaseStrategy):
    """Random position changes. Monte Carlo baseline."""
    
    def __init__(self, seed: int = 42, change_prob: float = 0.1):
        self.seed = seed
        self.change_prob = change_prob
        self.name = "Random"
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        n = len(prices_df)
        signals = np.zeros(n)
        
        current = 0.0
        for i in range(n):
            if rng.random() < self.change_prob:
                current = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            signals[i] = current
        
        return signals


class PerfectForesight(BaseStrategy):
    """Perfect foresight: knows future returns. Theoretical ceiling.
    
    This is CHEATING — used only to establish the upper bound
    of what's achievable with zero prediction error.
    """
    
    name = "Perfect Foresight"
    
    def __init__(self, horizon: int = 5):
        self.horizon = horizon
    
    def generate(self, prices_df: pd.DataFrame) -> np.ndarray:
        close = prices_df["close"].values
        n = len(close)
        signals = np.zeros(n)
        
        for i in range(n - self.horizon):
            future_return = (close[i + self.horizon] - close[i]) / close[i]
            # Scale position by expected return
            signals[i] = np.clip(future_return * 20, -1.0, 1.0)
        
        return signals


def get_all_baselines() -> list:
    """Return all baseline strategies for comparison."""
    return [
        BuyAndHold(),
        SMACrossover(fast=10, slow=30),
        SMACrossover(fast=20, slow=50),
        MACDStrategy(),
        MeanReversion(window=20, n_std=2.0),
        MomentumStrategy(lookback=20),
        MomentumStrategy(lookback=60),
        RSIStrategy(period=14),
        RandomStrategy(seed=42),
    ]


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.randn(n) * 0.001),
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    })
    
    print("Testing all baseline strategies on synthetic data:\n")
    for strategy in get_all_baselines():
        signals = strategy.generate(df)
        print(f"  {strategy.name:<25} signals range: [{signals.min():.2f}, {signals.max():.2f}], "
              f"mean: {signals.mean():.3f}, changes: {np.sum(np.diff(signals) != 0)}")
    
    # Also test perfect foresight
    pf = PerfectForesight(horizon=5)
    signals = pf.generate(df)
    print(f"  {pf.name:<25} signals range: [{signals.min():.2f}, {signals.max():.2f}], "
          f"mean: {signals.mean():.3f}, changes: {np.sum(np.diff(signals) != 0)}")
