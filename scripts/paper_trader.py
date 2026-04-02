#!/usr/bin/env python3
"""Paper Trading Engine — runs the agent on live market data every trading day.

What this does
--------------
1. Fetches the latest daily OHLCV bars from Yahoo Finance (never the training cache).
2. Builds the live feature vector on-the-fly for each ticker.
3. Runs the full Screener → Analyst → Critic → Executor pipeline.
4. Records NEW positions (entries) from today's orders.
5. Marks EXISTING open positions as exited when stop-loss, take-profit,
   or max-hold-days is hit (using today's price).
6. Updates a JSON portfolio file and an append-only ledger.

Files written
-------------
  data/paper_trading/portfolio.json       — current open positions + equity curve
  data/paper_trading/ledger.jsonl         — every trade (entry + exit) in detail
  data/paper_trading/daily_snapshots/     — per-day portfolio snapshot
  data/paper_trading/price_cache/         — today's fetched prices (avoids re-fetching)

Usage
-----
  python scripts/paper_trader.py                          # normal daily run
  python scripts/paper_trader.py --dry-run                # fetch prices + run pipeline, no writes
  python scripts/paper_trader.py --force-exit-all         # close every position at market
  python scripts/paper_trader.py --status                 # print status without running pipeline
  python scripts/paper_trader.py --starting-capital 50000 # first-run capital (default $100k)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── make sure project root is on sys.path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CAPITAL       = 100_000.0   # Starting paper capital in USD
MAX_HOLD_DAYS         = 10          # Force-close a position after this many days
COMMISSION_PER_TRADE  = 0.0         # Set > 0 (e.g. 1.0) to simulate friction
SLIPPAGE_BPS          = 5          # basis points of slippage applied at entry/exit

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    direction: str            # "long" | "short"
    entry_price: float
    entry_date: str           # ISO date string
    shares: float             # fractional shares OK
    stop_loss: float
    take_profit: float
    atr: float
    position_size_pct: float
    predicted_return: float
    sub_model_agreement: float
    regime_label: str
    run_id: str
    # exit fields (None while open)
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None   # "stop_hit" | "tp_hit" | "time_exit" | "eod_close"

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    def pnl(self, current_price: float) -> float:
        """Unrealised P&L in dollars."""
        if self.direction == "long":
            return (current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - current_price) * self.shares

    def pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.direction == "long":
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


@dataclass
class Portfolio:
    starting_capital: float = DEFAULT_CAPITAL
    cash: float = DEFAULT_CAPITAL
    positions: List[Position] = field(default_factory=list)
    closed_trades: List[Position] = field(default_factory=list)
    # Daily equity snapshots  [{"date": "...", "equity": ...}, ...]
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # ── Persistence ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "starting_capital": self.starting_capital,
            "cash": self.cash,
            "positions": [asdict(p) for p in self.positions],
            "closed_trades": [asdict(p) for p in self.closed_trades],
            "equity_curve": self.equity_curve,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Portfolio":
        p = cls(
            starting_capital=d.get("starting_capital", DEFAULT_CAPITAL),
            cash=d.get("cash", DEFAULT_CAPITAL),
            equity_curve=d.get("equity_curve", []),
        )
        p.positions = [Position(**pos) for pos in d.get("positions", [])]
        p.closed_trades = [Position(**pos) for pos in d.get("closed_trades", [])]
        return p

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "Portfolio":
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ── Metrics ────────────────────────────────────────────────────────

    def total_equity(self, prices: Dict[str, float]) -> float:
        """Cash + mark-to-market value of all open positions."""
        equity = self.cash
        for pos in self.positions:
            price = prices.get(pos.ticker, pos.entry_price)
            if pos.direction == "long":
                equity += price * pos.shares
            else:
                # Short: we received entry_price * shares at entry (already in cash)
                # net value = entry_price*shares - current*shares (already deducted at entry)
                equity += (pos.entry_price - price) * pos.shares
        return equity

    def stats(self) -> Dict[str, Any]:
        closed = self.closed_trades
        if not closed:
            return {"total_trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                    "total_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0}
        pnls = [(p.exit_price - p.entry_price) * p.shares
                if p.direction == "long"
                else (p.entry_price - p.exit_price) * p.shares
                for p in closed if p.exit_price is not None]
        wins = sum(1 for v in pnls if v > 0)
        return {
            "total_trades":  len(pnls),
            "win_rate":      wins / len(pnls) if pnls else 0.0,
            "avg_return":    float(np.mean(pnls)) if pnls else 0.0,
            "total_pnl":     float(np.sum(pnls)),
            "best_trade":    float(max(pnls)) if pnls else 0.0,
            "worst_trade":   float(min(pnls)) if pnls else 0.0,
            "profit_factor": (sum(v for v in pnls if v > 0) /
                              max(1e-9, -sum(v for v in pnls if v < 0))),
        }


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_latest_prices(tickers: List[str], cache_dir: Path) -> Dict[str, float]:
    """
    Fetch today's closing price for every ticker.

    We use a simple file cache (one JSON per day) so that multiple intra-day
    calls don't hammer Yahoo and so the nightly systemd run always has prices
    even if the market is closed when it runs.
    """
    today = date.today().isoformat()
    cache_file = cache_dir / f"prices_{today}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        # Return only tickers we need; fetch missing ones
        missing = [t for t in tickers if t not in cached]
        if not missing:
            logger.info("Price cache hit for %s", today)
            return {t: cached[t] for t in tickers if t in cached}
    else:
        cached = {}
        missing = list(tickers)

    if missing:
        logger.info("Fetching %d prices from Yahoo Finance…", len(missing))
        try:
            raw = yf.download(
                missing,
                period="2d",       # grab 2 days so we always have at least one close
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if raw is not None and not raw.empty:
                close = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=missing[0])
                for ticker in missing:
                    if ticker in close.columns:
                        series = close[ticker].dropna()
                        if not series.empty:
                            cached[ticker] = float(series.iloc[-1])
        except Exception as exc:
            logger.warning("Yahoo batch fetch failed: %s", exc)
            # Fall back one by one
            for ticker in missing:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="2d")
                    if not hist.empty:
                        cached[ticker] = float(hist["Close"].iloc[-1])
                    time.sleep(0.1)
                except Exception as e2:
                    logger.warning("  Could not fetch %s: %s", ticker, e2)

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cached, f)

    return {t: cached[t] for t in tickers if t in cached}


# ---------------------------------------------------------------------------
# Stop / TP / time-exit evaluation
# ---------------------------------------------------------------------------

def _slippage(price: float, direction: str, entering: bool) -> float:
    """Apply SLIPPAGE_BPS of slippage."""
    bps = SLIPPAGE_BPS / 10_000
    if direction == "long":
        return price * (1 + bps) if entering else price * (1 - bps)
    else:
        return price * (1 - bps) if entering else price * (1 + bps)


def check_exits(
    positions: List[Position],
    prices: Dict[str, float],
    today: str,
) -> List[Position]:
    """
    Evaluate all open positions against today's price.
    Returns the same list with exit fields filled in where triggered.
    """
    for pos in positions:
        if not pos.is_open:
            continue
        price = prices.get(pos.ticker)
        if price is None:
            continue

        entry_date = date.fromisoformat(pos.entry_date)
        hold_days = (date.fromisoformat(today) - entry_date).days

        reason: Optional[str] = None

        if pos.direction == "long":
            if price <= pos.stop_loss:
                reason = "stop_hit"
                exit_px = _slippage(pos.stop_loss, "long", entering=False)
            elif price >= pos.take_profit:
                reason = "tp_hit"
                exit_px = _slippage(pos.take_profit, "long", entering=False)
            elif hold_days >= MAX_HOLD_DAYS:
                reason = "time_exit"
                exit_px = _slippage(price, "long", entering=False)
            else:
                exit_px = None
        else:  # short
            if price >= pos.stop_loss:
                reason = "stop_hit"
                exit_px = _slippage(pos.stop_loss, "short", entering=False)
            elif price <= pos.take_profit:
                reason = "tp_hit"
                exit_px = _slippage(pos.take_profit, "short", entering=False)
            elif hold_days >= MAX_HOLD_DAYS:
                reason = "time_exit"
                exit_px = _slippage(price, "short", entering=False)
            else:
                exit_px = None

        if reason and exit_px is not None:
            pos.exit_price = exit_px
            pos.exit_date = today
            pos.exit_reason = reason
            logger.info(
                "EXIT  %s %-5s  @ %.2f  (%s)  PnL $%.2f",
                pos.direction.upper(), pos.ticker, exit_px, reason,
                pos.pnl(exit_px),
            )

    return positions


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(model_path: str, top_n: int = 20, dry_run: bool = False):
    """Run the full Screener → Analyst → Critic → Executor pipeline.

    Returns a list of ExecutorOrder objects.
    """
    from agents.pipeline import SwingTradingPipeline
    from agents.state import TradingState

    logger.info("Running pipeline with model=%s", model_path)
    pipeline = SwingTradingPipeline(
        model_path=model_path,
        top_n=top_n,
        max_position_pct=0.05,
        min_agreement=0.50,
        min_predicted_return=0.001,
    )
    state = pipeline.run()
    logger.info(
        "Pipeline done: screened=%d approved=%d orders=%d",
        len(state.screened_tickers),
        len(state.approved_tickers),
        len(state.orders),
    )
    return state


# ---------------------------------------------------------------------------
# Core paper-trading loop
# ---------------------------------------------------------------------------

def paper_trade_day(
    portfolio: Portfolio,
    model_path: str,
    data_dir: Path,
    top_n: int = 20,
    dry_run: bool = False,
) -> Portfolio:
    today = date.today().isoformat()
    price_cache = data_dir / "price_cache"

    # ── 1. Get all tickers we need prices for ─────────────────────────
    open_tickers = [p.ticker for p in portfolio.positions if p.is_open]

    # Run pipeline to get today's signals
    state = run_pipeline(model_path, top_n=top_n, dry_run=dry_run)
    new_order_tickers = [o.ticker for o in state.orders]

    all_tickers = list(set(open_tickers + new_order_tickers))
    prices = fetch_latest_prices(all_tickers, price_cache)

    # ── 2. Check exits on existing open positions ─────────────────────
    portfolio.positions = check_exits(portfolio.positions, prices, today)

    # Return cash for closed positions
    newly_closed = [p for p in portfolio.positions if p.exit_date == today]
    for pos in newly_closed:
        if pos.exit_price is None:
            continue
        if pos.direction == "long":
            proceeds = pos.exit_price * pos.shares - COMMISSION_PER_TRADE
        else:
            # Short: we already deducted entry notional from cash at entry,
            # so we get back:  entry_notional - (exit - entry)*shares
            proceeds = pos.exit_price * pos.shares - COMMISSION_PER_TRADE
        portfolio.cash += proceeds
        logger.info(
            "  Cash after closing %s: $%.2f (+$%.2f)",
            pos.ticker, portfolio.cash, proceeds,
        )

    # Move closed positions out of open list
    portfolio.closed_trades.extend(newly_closed)
    portfolio.positions = [p for p in portfolio.positions if p.is_open]

    # ── 3. Open new positions ─────────────────────────────────────────
    existing_tickers = {p.ticker for p in portfolio.positions}

    for order in state.orders:
        # Skip if already holding this ticker
        if order.ticker in existing_tickers:
            logger.info("SKIP  %s — already in portfolio", order.ticker)
            continue
        price = prices.get(order.ticker)
        if price is None:
            logger.warning("SKIP  %s — no live price available", order.ticker)
            continue
        if price < 2.0:
            logger.info("SKIP  %s — price $%.2f below $2 floor", order.ticker, price)
            continue

        # How much capital to allocate?
        notional = portfolio.cash * order.position_size_pct
        if notional < 10.0:
            logger.info("SKIP  %s — insufficient cash (need $%.2f)", order.ticker, notional)
            continue
        if notional > portfolio.cash:
            notional = portfolio.cash * 0.9   # safety cap

        entry_px = _slippage(price, order.direction, entering=True)
        shares = notional / entry_px

        # Find the matching analyst report for meta-data
        report_map = {r.ticker: r for r in state.analyst_reports}
        report = report_map.get(order.ticker)

        raw_stop = order.stop_loss or (entry_px * 0.95 if order.direction == "long" else entry_px * 1.05)
        raw_tp   = order.take_profit or (entry_px * 1.10 if order.direction == "long" else entry_px * 0.90)

        # Sanity-check: stop must be on the correct side of entry, otherwise
        # recalculate using ATR or a fixed 5% fallback (stale price in CSV).
        if order.direction == "long" and raw_stop >= entry_px:
            logger.warning(
                "ADJUST %s: stop $%.2f >= entry $%.2f (stale data?) → recalculating",
                order.ticker, raw_stop, entry_px,
            )
            atr = order.atr or entry_px * 0.05
            raw_stop = entry_px - 2.0 * atr
            raw_tp   = entry_px + 3.0 * atr
        elif order.direction == "short" and raw_stop <= entry_px:
            logger.warning(
                "ADJUST %s: stop $%.2f <= entry $%.2f (stale data?) → recalculating",
                order.ticker, raw_stop, entry_px,
            )
            atr = order.atr or entry_px * 0.05
            raw_stop = entry_px + 2.0 * atr
            raw_tp   = entry_px - 3.0 * atr

        stop = raw_stop
        tp   = raw_tp

        pos = Position(
            ticker=order.ticker,
            direction=order.direction,
            entry_price=entry_px,
            entry_date=today,
            shares=shares,
            stop_loss=stop,
            take_profit=tp,
            atr=order.atr or 0.0,
            position_size_pct=order.position_size_pct,
            predicted_return=report.predicted_return if report else 0.0,
            sub_model_agreement=report.sub_model_agreement if report else 0.0,
            regime_label=order.regime_label or state.regime_label,
            run_id=state.run_id,
        )

        if not dry_run:
            # Deduct cost for longs; for shorts we receive proceeds
            if order.direction == "long":
                portfolio.cash -= entry_px * shares + COMMISSION_PER_TRADE
            else:
                portfolio.cash += entry_px * shares - COMMISSION_PER_TRADE
            portfolio.positions.append(pos)

        logger.info(
            "ENTER %s %-5s  @ $%.2f  shares=%.2f  stop=%.2f  tp=%.2f  size=%.1f%%",
            order.direction.upper(), order.ticker, entry_px, shares,
            stop, tp, order.position_size_pct * 100,
        )

    # ── 4. Snapshot equity ────────────────────────────────────────────
    equity = portfolio.total_equity(prices)
    snapshot = {
        "date": today,
        "equity": round(equity, 2),
        "cash": round(portfolio.cash, 2),
        "open_positions": len(portfolio.positions),
        "pnl_today": round(
            equity - (portfolio.equity_curve[-1]["equity"] if portfolio.equity_curve else portfolio.starting_capital),
            2,
        ),
    }
    portfolio.equity_curve.append(snapshot)

    logger.info(
        "Portfolio | equity=$%.2f  cash=$%.2f  open=%d  total_return=%.2f%%",
        equity, portfolio.cash,
        len(portfolio.positions),
        (equity / portfolio.starting_capital - 1) * 100,
    )
    return portfolio


# ---------------------------------------------------------------------------
# Ledger writer
# ---------------------------------------------------------------------------

def append_ledger(portfolio: Portfolio, ledger_path: Path) -> None:
    """Append all closed trades from today to the ledger JSONL."""
    today = date.today().isoformat()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        for pos in portfolio.closed_trades:
            if pos.exit_date == today:
                f.write(json.dumps(asdict(pos), default=str) + "\n")


def save_daily_snapshot(portfolio: Portfolio, snapshots_dir: Path) -> None:
    today = date.today().isoformat()
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snapshots_dir / f"snapshot_{today}.json"
    with open(snap_path, "w") as f:
        json.dump(portfolio.to_dict(), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Status printer
# ---------------------------------------------------------------------------

def print_status(portfolio: Portfolio, prices: Optional[Dict[str, float]] = None) -> None:
    prices = prices or {}
    equity = portfolio.total_equity(prices)
    total_return = (equity / portfolio.starting_capital - 1) * 100
    stats = portfolio.stats()

    print(f"\n{'═'*72}")
    print(f"  PAPER TRADING STATUS  —  {date.today().isoformat()}")
    print(f"{'═'*72}")
    print(f"  Starting capital : ${portfolio.starting_capital:>12,.2f}")
    print(f"  Current equity   : ${equity:>12,.2f}   ({total_return:+.2f}%)")
    print(f"  Cash available   : ${portfolio.cash:>12,.2f}")
    print(f"  Open positions   : {len(portfolio.positions)}")
    print(f"  Closed trades    : {stats['total_trades']}")
    print(f"  Win rate         : {stats['win_rate']*100:.1f}%")
    print(f"  Avg trade P&L    : ${stats['avg_return']:+.2f}")
    print(f"  Total realized   : ${stats['total_pnl']:+,.2f}")
    print(f"  Profit factor    : {stats.get('profit_factor', 0):.2f}")

    if portfolio.positions:
        print(f"\n  {'─'*68}")
        print(f"  {'TICKER':<8} {'DIR':<6} {'ENTRY':>8} {'NOW':>8} {'SHARES':>7} {'PnL $':>9} {'PnL %':>7} {'REGIME':<12}")
        print(f"  {'─'*68}")
        for pos in portfolio.positions:
            cur = prices.get(pos.ticker, pos.entry_price)
            pnl_d = pos.pnl(cur)
            pnl_p = pos.pnl_pct(cur) * 100
            print(
                f"  {pos.ticker:<8} {pos.direction.upper():<6} {pos.entry_price:>8.2f}"
                f" {cur:>8.2f} {pos.shares:>7.2f}"
                f" {pnl_d:>+9.2f} {pnl_p:>+6.1f}%  {pos.regime_label:<12}"
            )

    if portfolio.equity_curve:
        print(f"\n  Recent equity curve (last 5 days):")
        for snap in portfolio.equity_curve[-5:]:
            daily = snap.get("pnl_today", 0)
            print(f"    {snap['date']}  equity=${snap['equity']:>10,.2f}  daily={daily:>+8.2f}  positions={snap.get('open_positions',0)}")

    print(f"{'═'*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_latest_model() -> str:
    candidates = sorted(PROJECT_ROOT.glob("models/hierarchical_v*/forecaster_final.pt"))
    if not candidates:
        raise FileNotFoundError("No model found. Train first or pass --model.")
    return str(candidates[-1])


def parse_args():
    p = argparse.ArgumentParser(description="Paper Trading Engine")
    p.add_argument("--model", default=None, help="Path to forecaster_final.pt")
    p.add_argument("--top-n", type=int, default=20, help="Screener top-N")
    p.add_argument("--data-dir", default="data/paper_trading",
                   help="Directory for portfolio.json and ledger.jsonl")
    p.add_argument("--starting-capital", type=float, default=DEFAULT_CAPITAL,
                   help="Starting capital for a fresh portfolio (default $100k)")
    p.add_argument("--dry-run", action="store_true",
                   help="Run pipeline and show what would happen, write nothing")
    p.add_argument("--check-only", action="store_true",
                   help="Only evaluate stop/TP exits on open positions — skip the full pipeline. "
                        "Fast enough to run every 15 minutes during market hours.")
    p.add_argument("--force-exit-all", action="store_true",
                   help="Close every open position at today's market price")
    p.add_argument("--status", action="store_true",
                   help="Print portfolio status and exit (no pipeline run)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = data_dir / "portfolio.json"
    ledger_path    = data_dir / "ledger.jsonl"
    snapshots_dir  = data_dir / "daily_snapshots"

    # Load or initialise portfolio
    if portfolio_path.exists():
        portfolio = Portfolio.load(portfolio_path)
        logger.info("Loaded portfolio: equity≈$%.0f  open=%d  closed=%d",
                    portfolio.cash, len(portfolio.positions), len(portfolio.closed_trades))
    else:
        portfolio = Portfolio(
            starting_capital=args.starting_capital,
            cash=args.starting_capital,
        )
        logger.info("New portfolio created with $%.0f starting capital", args.starting_capital)

    # ── --status: just print and exit ─────────────────────────────────
    if args.status:
        all_tickers = [p.ticker for p in portfolio.positions]
        prices = fetch_latest_prices(all_tickers, data_dir / "price_cache") if all_tickers else {}
        print_status(portfolio, prices)
        return

    # ── --force-exit-all ──────────────────────────────────────────────
    if args.force_exit_all:
        tickers = [p.ticker for p in portfolio.positions if p.is_open]
        prices = fetch_latest_prices(tickers, data_dir / "price_cache")
        today = date.today().isoformat()
        for pos in portfolio.positions:
            if pos.is_open and pos.ticker in prices:
                px = _slippage(prices[pos.ticker], pos.direction, entering=False)
                pos.exit_price = px
                pos.exit_date = today
                pos.exit_reason = "eod_close"
                logger.info("FORCE-CLOSE %s @ $%.2f", pos.ticker, px)
                if pos.direction == "long":
                    portfolio.cash += px * pos.shares
                else:
                    portfolio.cash += (pos.entry_price - px) * pos.shares
        portfolio.closed_trades.extend([p for p in portfolio.positions if not p.is_open])
        portfolio.positions = [p for p in portfolio.positions if p.is_open]
        if not args.dry_run:
            portfolio.save(portfolio_path)
            append_ledger(portfolio, ledger_path)
        print_status(portfolio, prices)
        return

    # ── --check-only: evaluate exits without running the pipeline ──────
    if args.check_only:
        open_tickers = [p.ticker for p in portfolio.positions if p.is_open]
        if not open_tickers:
            logger.info("check-only: no open positions, nothing to do")
            return
        prices = fetch_latest_prices(open_tickers, data_dir / "price_cache")
        today = date.today().isoformat()
        portfolio.positions = check_exits(portfolio.positions, prices, today)
        newly_closed = [p for p in portfolio.positions if p.exit_date == today]
        for pos in newly_closed:
            if pos.exit_price is None:
                continue
            if pos.direction == "long":
                portfolio.cash += pos.exit_price * pos.shares - COMMISSION_PER_TRADE
            else:
                portfolio.cash += pos.exit_price * pos.shares - COMMISSION_PER_TRADE
        portfolio.closed_trades.extend(newly_closed)
        portfolio.positions = [p for p in portfolio.positions if p.is_open]
        if newly_closed and not args.dry_run:
            portfolio.save(portfolio_path)
            append_ledger(portfolio, ledger_path)
            logger.info("check-only: closed %d position(s), portfolio saved", len(newly_closed))
        else:
            logger.info("check-only: %d position(s) checked, none triggered", len(open_tickers))
        return

    # ── Normal daily run ──────────────────────────────────────────────
    model_path = args.model or find_latest_model()
    logger.info("Model: %s", model_path)

    portfolio = paper_trade_day(
        portfolio=portfolio,
        model_path=model_path,
        data_dir=data_dir,
        top_n=args.top_n,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        portfolio.save(portfolio_path)
        append_ledger(portfolio, ledger_path)
        save_daily_snapshot(portfolio, snapshots_dir)
        logger.info("Portfolio saved to %s", portfolio_path)

    # Always print status at the end
    all_tickers = [p.ticker for p in portfolio.positions]
    prices = fetch_latest_prices(all_tickers, data_dir / "price_cache") if all_tickers else {}
    print_status(portfolio, prices)


if __name__ == "__main__":
    main()
