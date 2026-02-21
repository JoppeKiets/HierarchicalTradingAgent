#!/usr/bin/env python3
"""Walk-forward backtester with realistic transaction costs.

Supports:
  - Walk-forward evaluation (no look-ahead bias)
  - Configurable slippage and commission models
  - Multiple metrics: Sharpe, Sortino, Calmar, max drawdown, hit rate
  - Strategy-agnostic: feed any signal generator
  - Per-trade P&L tracking
  - Equity curve generation

Usage:
    from src.backtester import Backtester, BacktestResult
    bt = Backtester(commission_bps=5, slippage_bps=5)
    result = bt.run(prices_df, signals)
    result.summary()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    holding_days: int


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Equity curve
    dates: List[str] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Trades
    trades: List[Trade] = field(default_factory=list)
    
    # Metadata
    initial_capital: float = 100_000.0
    ticker: str = ""
    strategy_name: str = ""
    
    def summary(self) -> Dict:
        """Compute summary statistics."""
        if len(self.equity_curve) < 2:
            return {"error": "insufficient data"}
        
        total_return = (self.equity_curve[-1] / self.equity_curve[0]) - 1
        
        # Annualized metrics
        n_days = len(self.equity_curve)
        n_years = n_days / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        
        daily_rets = self.daily_returns[np.isfinite(self.daily_returns)]
        if len(daily_rets) < 2:
            return {"total_return": total_return, "error": "insufficient returns"}
        
        ann_vol = np.std(daily_rets) * np.sqrt(252)
        
        # Sharpe (assuming 0 risk-free for simplicity, can parameterize)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0.0
        
        # Sortino
        downside = daily_rets[daily_rets < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 1 else 1e-10
        sortino = ann_return / downside_vol if downside_vol > 1e-10 else 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        max_dd = np.min(drawdown)
        
        # Calmar
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
        
        # Trade stats
        n_trades = len(self.trades)
        if n_trades > 0:
            winning = [t for t in self.trades if t.pnl > 0]
            losing = [t for t in self.trades if t.pnl <= 0]
            hit_rate = len(winning) / n_trades
            avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0.0
            avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0.0
            profit_factor = (sum(t.pnl for t in winning) / abs(sum(t.pnl for t in losing))
                           if losing and sum(t.pnl for t in losing) != 0 else float('inf'))
            total_commission = sum(t.commission for t in self.trades)
            total_slippage = sum(t.slippage for t in self.trades)
            avg_holding = np.mean([t.holding_days for t in self.trades])
        else:
            hit_rate = avg_win = avg_loss = profit_factor = 0.0
            total_commission = total_slippage = avg_holding = 0.0
        
        stats = {
            "strategy": self.strategy_name,
            "ticker": self.ticker,
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "n_trades": n_trades,
            "hit_rate": hit_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "avg_holding_days": avg_holding,
            "n_days": n_days,
        }
        return stats
    
    def print_summary(self):
        """Pretty-print the summary."""
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  BACKTEST: {s.get('strategy', '?')} on {s.get('ticker', '?')}")
        print(f"{'='*60}")
        print(f"  Total Return:      {s['total_return']:>10.2%}")
        print(f"  Ann. Return:       {s['ann_return']:>10.2%}")
        print(f"  Ann. Volatility:   {s['ann_volatility']:>10.2%}")
        print(f"  Sharpe Ratio:      {s['sharpe']:>10.3f}")
        print(f"  Sortino Ratio:     {s['sortino']:>10.3f}")
        print(f"  Max Drawdown:      {s['max_drawdown']:>10.2%}")
        print(f"  Calmar Ratio:      {s['calmar']:>10.3f}")
        print(f"  ---")
        print(f"  Trades:            {s['n_trades']:>10d}")
        print(f"  Hit Rate:          {s['hit_rate']:>10.2%}")
        print(f"  Avg Win:           {s['avg_win_pct']:>10.2%}")
        print(f"  Avg Loss:          {s['avg_loss_pct']:>10.2%}")
        print(f"  Profit Factor:     {s['profit_factor']:>10.3f}")
        print(f"  Total Commission:  ${s['total_commission']:>9.2f}")
        print(f"  Total Slippage:    ${s['total_slippage']:>9.2f}")
        print(f"  Avg Holding (d):   {s['avg_holding_days']:>10.1f}")
        print(f"{'='*60}")


class Backtester:
    """Walk-forward backtester with realistic costs.
    
    Signals should be a numpy array or pd.Series of target positions:
      - signal > 0: long position (fraction of capital)
      - signal < 0: short position
      - signal = 0: flat
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,     # 5 bps = 0.05% per trade
        slippage_bps: float = 5.0,       # 5 bps estimated slippage
        max_position: float = 1.0,       # max fraction of capital
        min_trade_size: float = 0.01,    # minimum position change to trigger trade
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10_000
        self.slippage_rate = slippage_bps / 10_000
        self.max_position = max_position
        self.min_trade_size = min_trade_size
    
    def run(
        self,
        prices_df: pd.DataFrame,
        signals: np.ndarray,
        strategy_name: str = "Strategy",
        ticker: str = "",
    ) -> BacktestResult:
        """Run backtest.
        
        Args:
            prices_df: DataFrame with at least 'close' column (and optionally 'date', 'open')
            signals: Array of target positions, same length as prices_df.
                     Values in [-max_position, max_position].
            strategy_name: Name for reporting
            ticker: Ticker symbol for reporting
            
        Returns:
            BacktestResult with equity curve, trades, etc.
        """
        close = prices_df["close"].values.astype(float)
        open_prices = prices_df["open"].values.astype(float) if "open" in prices_df.columns else close.copy()
        dates = (prices_df["date"].astype(str).values if "date" in prices_df.columns
                 else [str(i) for i in range(len(close))])
        
        n = len(close)
        assert len(signals) == n, f"Signal length {len(signals)} != price length {n}"
        
        # Clip signals
        signals = np.clip(signals, -self.max_position, self.max_position)
        
        # Track state
        equity = np.zeros(n)
        daily_returns = np.zeros(n)
        positions_arr = np.zeros(n)
        cash = self.initial_capital
        position_shares = 0.0
        current_position_frac = 0.0
        
        trades: List[Trade] = []
        trade_entry = None  # Track open trade
        
        for i in range(n):
            # Portfolio value at today's close
            port_value = cash + position_shares * close[i]
            
            if i == 0:
                equity[0] = port_value
                positions_arr[0] = 0.0
                # Execute first signal
                target_frac = signals[0]
                if abs(target_frac) > self.min_trade_size:
                    target_shares = (port_value * target_frac) / close[0]
                    trade_value = abs(target_shares * close[0])
                    commission = trade_value * self.commission_rate
                    slippage = trade_value * self.slippage_rate
                    cash -= target_shares * close[0] + commission + slippage
                    position_shares = target_shares
                    current_position_frac = target_frac
                    trade_entry = {
                        "date": dates[0], "price": close[0], 
                        "direction": 1 if target_frac > 0 else -1,
                        "size": abs(target_frac),
                        "commission": commission, "slippage": slippage,
                    }
                continue
            
            # Portfolio value at today's open (use open price for execution)
            port_value = cash + position_shares * close[i]
            equity[i] = port_value
            daily_returns[i] = (port_value / equity[i-1]) - 1 if equity[i-1] > 0 else 0
            positions_arr[i] = current_position_frac
            
            # Check if we need to rebalance
            target_frac = signals[i]
            delta = target_frac - current_position_frac
            
            if abs(delta) > self.min_trade_size:
                # Close existing trade (for trade tracking)
                if trade_entry is not None and abs(current_position_frac) > self.min_trade_size:
                    entry = trade_entry
                    exit_value = position_shares * close[i]
                    entry_value = entry["direction"] * entry["size"] * equity[0]  # approx
                    pnl = exit_value - abs(position_shares) * entry["price"] * np.sign(position_shares)
                    pnl_pct = pnl / max(port_value, 1)
                    
                    # Calculate holding days
                    try:
                        entry_idx = list(dates).index(entry["date"])
                        holding_days = i - entry_idx
                    except ValueError:
                        holding_days = 1
                    
                    trades.append(Trade(
                        entry_date=entry["date"],
                        exit_date=dates[i],
                        entry_price=entry["price"],
                        exit_price=close[i],
                        direction=entry["direction"],
                        size=entry["size"],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=entry["commission"],
                        slippage=entry["slippage"],
                        holding_days=holding_days,
                    ))
                
                # Execute new position
                exec_price = open_prices[i] if i < n else close[i]
                target_shares = (port_value * target_frac) / exec_price if exec_price > 0 else 0
                shares_delta = target_shares - position_shares
                trade_value = abs(shares_delta * exec_price)
                commission = trade_value * self.commission_rate
                slippage_cost = trade_value * self.slippage_rate
                
                cash -= shares_delta * exec_price + commission + slippage_cost
                position_shares = target_shares
                current_position_frac = target_frac
                
                # Track new trade entry
                if abs(target_frac) > self.min_trade_size:
                    trade_entry = {
                        "date": dates[i], "price": exec_price,
                        "direction": 1 if target_frac > 0 else -1,
                        "size": abs(target_frac),
                        "commission": commission, "slippage": slippage_cost,
                    }
                else:
                    trade_entry = None
        
        # Close final position
        if trade_entry is not None and abs(current_position_frac) > self.min_trade_size:
            entry = trade_entry
            pnl = position_shares * (close[-1] - entry["price"])
            pnl_pct = pnl / max(equity[-1], 1)
            try:
                entry_idx = list(dates).index(entry["date"])
                holding_days = n - 1 - entry_idx
            except ValueError:
                holding_days = 1
            trades.append(Trade(
                entry_date=entry["date"],
                exit_date=dates[-1],
                entry_price=entry["price"],
                exit_price=close[-1],
                direction=entry["direction"],
                size=entry["size"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=entry["commission"],
                slippage=entry["slippage"],
                holding_days=holding_days,
            ))
        
        return BacktestResult(
            dates=list(dates),
            equity_curve=equity,
            daily_returns=daily_returns,
            positions=positions_arr,
            trades=trades,
            initial_capital=self.initial_capital,
            ticker=ticker,
            strategy_name=strategy_name,
        )


def compare_strategies(results: List[BacktestResult]):
    """Compare multiple backtest results side by side."""
    print(f"\n{'='*90}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'='*90}")
    
    summaries = [r.summary() for r in results]
    
    headers = ["Strategy", "Return", "Sharpe", "Sortino", "MaxDD", "Trades", "HitRate", "PF"]
    fmt =     "{:<20} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8}"
    
    print(fmt.format(*headers))
    print("-" * 90)
    
    for s in summaries:
        print(fmt.format(
            s.get("strategy", "?")[:20],
            f"{s.get('total_return', 0):.1%}",
            f"{s.get('sharpe', 0):.3f}",
            f"{s.get('sortino', 0):.3f}",
            f"{s.get('max_drawdown', 0):.1%}",
            f"{s.get('n_trades', 0)}",
            f"{s.get('hit_rate', 0):.1%}",
            f"{s.get('profit_factor', 0):.2f}",
        ))
    
    # Find best strategy
    best_sharpe = max(summaries, key=lambda s: s.get("sharpe", -999))
    print(f"\n  Best Sharpe: {best_sharpe['strategy']} ({best_sharpe['sharpe']:.3f})")
    
    return summaries


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.randn(n) * 0.001),
        "close": prices,
    })
    
    # Buy-and-hold signal
    bh_signals = np.ones(n)
    
    # Random signal
    rand_signals = np.random.choice([-1, 0, 1], size=n) * 0.5
    
    bt = Backtester(initial_capital=100_000, commission_bps=5, slippage_bps=5)
    
    bh_result = bt.run(df, bh_signals, strategy_name="Buy & Hold", ticker="SYN")
    rand_result = bt.run(df, rand_signals, strategy_name="Random", ticker="SYN")
    
    bh_result.print_summary()
    rand_result.print_summary()
    compare_strategies([bh_result, rand_result])
