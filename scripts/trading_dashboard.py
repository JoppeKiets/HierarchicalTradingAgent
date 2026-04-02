#!/usr/bin/env python3
"""Live trading dashboard — reads portfolio.json and ledger.jsonl.

Prints a colour-coded terminal view of:
  • Current portfolio equity vs benchmark (SPY)
  • Open positions with live unrealised P&L
  • Closed trade history with wins/losses highlighted
  • Equity curve (ASCII spark-line)
  • Summary statistics

Usage
-----
  python scripts/trading_dashboard.py                   # one-shot print
  python scripts/trading_dashboard.py --watch           # refresh every 60 s
  python scripts/trading_dashboard.py --data-dir data/paper_trading
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Colour helpers (pure ANSI — no third-party deps) ──────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"
GREY   = "\033[90m"
BG_DARK = "\033[40m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


def _pnl_color(val: float) -> str:
    if val > 0:
        return _c(f"{val:>+10.2f}", GREEN, BOLD)
    elif val < 0:
        return _c(f"{val:>+10.2f}", RED, BOLD)
    return _c(f"{val:>+10.2f}", WHITE)


def _pct_color(val: float) -> str:
    s = f"{val:>+7.2f}%"
    if val > 0:
        return _c(s, GREEN)
    elif val < 0:
        return _c(s, RED)
    return _c(s, WHITE)


# ── Spark-line ─────────────────────────────────────────────────────────────

_SPARKS = "▁▂▃▄▅▆▇█"

def sparkline(values: List[float], width: int = 40) -> str:
    if not values:
        return ""
    # Downsample to width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][-width:]
    lo, hi = min(sampled), max(sampled)
    if hi == lo:
        return _SPARKS[3] * len(sampled)
    chars = [_SPARKS[int((v - lo) / (hi - lo) * (len(_SPARKS) - 1))] for v in sampled]
    # Colour the last value
    last_idx = int((sampled[-1] - lo) / (hi - lo) * (len(_SPARKS) - 1))
    color = GREEN if sampled[-1] >= sampled[0] else RED
    return "".join(chars[:-1]) + _c(chars[-1], color, BOLD)


# ── Portfolio loading ──────────────────────────────────────────────────────

def load_portfolio(data_dir: Path) -> Optional[Dict[str, Any]]:
    path = data_dir / "portfolio.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_ledger(data_dir: Path) -> List[Dict[str, Any]]:
    path = data_dir / "ledger.jsonl"
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ── Live price fetch (lightweight, no pandas needed here) ─────────────────

def fetch_prices_simple(tickers: List[str], cache_dir: Path) -> Dict[str, float]:
    """Read from today's price cache if available, else call yfinance."""
    today = date.today().isoformat()
    cache_file = cache_dir / f"prices_{today}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if all(t in cached for t in tickers):
            return {t: cached[t] for t in tickers if t in cached}
    # Fetch missing
    missing = [t for t in tickers if not (cache_file.exists() and t in json.load(open(cache_file)))]
    if not missing:
        return {t: json.load(open(cache_file))[t] for t in tickers}
    try:
        import yfinance as yf
        raw = yf.download(missing, period="2d", interval="1d", progress=False, auto_adjust=True)
        prices: Dict[str, float] = {}
        if raw is not None and not raw.empty:
            close = raw["Close"] if "Close" in raw.columns else raw
            if hasattr(close, "columns"):
                for t in missing:
                    if t in close.columns:
                        s = close[t].dropna()
                        if not s.empty:
                            prices[t] = float(s.iloc[-1])
            else:
                s = close.dropna()
                if not s.empty:
                    prices[missing[0]] = float(s.values[-1])
        return prices
    except Exception:
        return {}


# ── SPY benchmark fetch ────────────────────────────────────────────────────

def fetch_spy_return(start_date: str, cache_dir: Path) -> float:
    """Total return of SPY from start_date to today."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start_date, interval="1d",
                          progress=False, auto_adjust=True)
        if spy is None or spy.empty:
            return 0.0
        close = spy["Close"].dropna()
        if len(close) < 2:
            return 0.0
        return float((close.iloc[-1] / close.iloc[0]) - 1) * 100
    except Exception:
        return 0.0


# ── Main display ───────────────────────────────────────────────────────────

def render_dashboard(data_dir: Path) -> None:
    portfolio_data = load_portfolio(data_dir)
    if portfolio_data is None:
        print(_c("  No portfolio found at " + str(data_dir / "portfolio.json"), YELLOW))
        print("  Run:  python scripts/paper_trader.py  to start\n")
        return

    ledger = load_ledger(data_dir)

    starting_capital = portfolio_data.get("starting_capital", 100_000.0)
    cash = portfolio_data.get("cash", starting_capital)
    open_positions = portfolio_data.get("positions", [])
    closed_trades  = portfolio_data.get("closed_trades", [])
    equity_curve   = portfolio_data.get("equity_curve", [])

    # ── Live prices for open positions ────────────────────────────────
    open_tickers = [p["ticker"] for p in open_positions]
    prices = fetch_prices_simple(open_tickers, data_dir / "price_cache") if open_tickers else {}

    # ── Compute current equity ────────────────────────────────────────
    equity = cash
    for pos in open_positions:
        cur = prices.get(pos["ticker"], pos["entry_price"])
        if pos["direction"] == "long":
            equity += cur * pos["shares"]
        else:
            equity += (pos["entry_price"] - cur) * pos["shares"]

    total_return_pct = (equity / starting_capital - 1) * 100
    total_pnl = equity - starting_capital

    # Benchmark
    if equity_curve:
        start_date = equity_curve[0]["date"]
        spy_return = fetch_spy_return(start_date, data_dir / "price_cache")
    else:
        spy_return = 0.0

    alpha = total_return_pct - spy_return

    # ── Win/loss stats ─────────────────────────────────────────────────
    realized_pnls = []
    for t in closed_trades:
        if t.get("exit_price") is not None:
            ep, xp, sh, d = t["entry_price"], t["exit_price"], t["shares"], t["direction"]
            pnl = (xp - ep) * sh if d == "long" else (ep - xp) * sh
            realized_pnls.append(pnl)

    wins  = sum(1 for v in realized_pnls if v > 0)
    total = len(realized_pnls)
    win_rate = wins / total if total else 0.0
    avg_win  = sum(v for v in realized_pnls if v > 0) / max(1, wins)
    avg_loss = sum(v for v in realized_pnls if v < 0) / max(1, total - wins)
    pf = sum(v for v in realized_pnls if v > 0) / max(1e-9, -sum(v for v in realized_pnls if v < 0))

    # Exit reason breakdown
    reasons: Dict[str, int] = {}
    for t in closed_trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    # ── Header ────────────────────────────────────────────────────────
    W = 72
    print()
    print(_c(f"{'═'*W}", CYAN))
    print(_c(f"  📈  PAPER TRADING DASHBOARD   —   {date.today().isoformat()}", CYAN, BOLD))
    print(_c(f"{'═'*W}", CYAN))

    # Equity summary
    print(f"\n  {'Starting capital':<24}  ${starting_capital:>12,.2f}")
    print(f"  {'Current equity':<24}  ${equity:>12,.2f}   {_pct_color(total_return_pct)}")
    print(f"  {'Realised P&L':<24}  {_pnl_color(sum(realized_pnls))}")
    print(f"  {'Unrealised P&L':<24}  {_pnl_color(equity - cash - sum(realized_pnls) if realized_pnls else equity - starting_capital)}")
    print(f"  {'Cash available':<24}  ${cash:>12,.2f}")
    spy_str = _pct_color(spy_return)
    alpha_str = _pct_color(alpha)
    print(f"  {'SPY since inception':<24}  {spy_str}   alpha {alpha_str}")

    # Equity sparkline
    if equity_curve:
        vals = [s["equity"] for s in equity_curve]
        print(f"\n  Equity  {sparkline(vals, width=50)}")
        print(f"  {'':8}  ${min(vals):,.0f} ←  → ${max(vals):,.0f}")

    # ── Stats ─────────────────────────────────────────────────────────
    print(f"\n  {_c('TRADE STATISTICS', BOLD)}")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Closed trades':<24}  {total}")
    win_str = _c(f"{win_rate*100:.1f}%", GREEN if win_rate >= 0.5 else RED)
    print(f"  {'Win rate':<24}  {win_str}   ({wins}W / {total-wins}L)")
    print(f"  {'Avg win':<24}  {_pnl_color(avg_win)}")
    print(f"  {'Avg loss':<24}  {_pnl_color(avg_loss)}")
    pf_str = _c(f"{pf:.2f}", GREEN if pf >= 1.0 else RED)
    print(f"  {'Profit factor':<24}  {pf_str}")
    if reasons:
        print(f"  {'Exit reasons':<24}  " +
              "  ".join(f"{_c(k, GREY)}={v}" for k, v in sorted(reasons.items())))

    # ── Open positions ────────────────────────────────────────────────
    print(f"\n  {_c('OPEN POSITIONS', BOLD)}  ({len(open_positions)})")
    if open_positions:
        print(f"  {'─'*(W-2)}")
        hdr = (f"  {'TICKER':<8} {'DIR':<6} {'ENTRY':>8} {'NOW':>8}"
               f" {'SHARES':>7} {'UNRL P&L':>10} {'%':>8}  {'STOP':>8}  {'TP':>8}  {'DAYS':>4}")
        print(_c(hdr, GREY))
        for pos in sorted(open_positions, key=lambda p: abs(
                (prices.get(p["ticker"], p["entry_price"]) - p["entry_price"]) / max(1, p["entry_price"])
            ), reverse=True):
            cur  = prices.get(pos["ticker"], pos["entry_price"])
            ep   = pos["entry_price"]
            sh   = pos["shares"]
            d    = pos["direction"]
            pnl_d = (cur - ep)*sh if d == "long" else (ep - cur)*sh
            pnl_p = ((cur - ep)/ep if d == "long" else (ep - cur)/ep) * 100
            days  = (date.today() - date.fromisoformat(pos["entry_date"])).days
            stop  = pos.get("stop_loss", 0)
            tp    = pos.get("take_profit", 0)
            pnl_str = _pnl_color(pnl_d)
            pct_str = _pct_color(pnl_p)
            dir_str = _c(d.upper(), GREEN if d == "long" else RED)
            print(
                f"  {pos['ticker']:<8} {dir_str:<14} {ep:>8.2f} {cur:>8.2f}"
                f" {sh:>7.2f} {pnl_str} {pct_str}  {stop:>8.2f}  {tp:>8.2f}  {days:>4}d"
            )
    else:
        print(f"  {_c('  No open positions.', GREY)}")

    # ── Recent closed trades ──────────────────────────────────────────
    recent = [t for t in closed_trades if t.get("exit_price") is not None][-10:]
    if recent:
        print(f"\n  {_c('RECENT CLOSED TRADES', BOLD)}  (last {len(recent)})")
        print(f"  {'─'*(W-2)}")
        hdr2 = (f"  {'TICKER':<8} {'DIR':<6} {'ENTRY':>8} {'EXIT':>8}"
                f" {'P&L $':>10} {'P&L %':>8}  {'REASON':<12} {'DATE'}")
        print(_c(hdr2, GREY))
        for t in reversed(recent):
            ep, xp, sh, d = t["entry_price"], t["exit_price"], t["shares"], t["direction"]
            pnl_d = (xp - ep)*sh if d == "long" else (ep - xp)*sh
            pnl_p = ((xp - ep)/ep if d == "long" else (ep - xp)/ep) * 100
            reason = t.get("exit_reason", "?")
            xdate  = t.get("exit_date", "?")
            dir_str = _c(d.upper(), GREEN if d == "long" else RED)
            reason_str = _c(reason, GREEN if reason == "tp_hit" else (RED if reason == "stop_hit" else YELLOW))
            print(
                f"  {t['ticker']:<8} {dir_str:<14} {ep:>8.2f} {xp:>8.2f}"
                f" {_pnl_color(pnl_d)} {_pct_color(pnl_p)}  {reason_str:<20} {xdate}"
            )

    print(f"\n{_c('═'*W, CYAN)}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Paper Trading Dashboard")
    p.add_argument("--data-dir", default="data/paper_trading",
                   help="Directory containing portfolio.json")
    p.add_argument("--watch", action="store_true",
                   help="Refresh every --interval seconds")
    p.add_argument("--interval", type=int, default=60,
                   help="Refresh interval in seconds (default 60)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.watch:
        try:
            while True:
                os.system("clear")
                render_dashboard(data_dir)
                print(f"  {_c(f'Next refresh in {args.interval}s  (Ctrl+C to quit)', GREY)}\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nBye!")
    else:
        render_dashboard(data_dir)


if __name__ == "__main__":
    main()
