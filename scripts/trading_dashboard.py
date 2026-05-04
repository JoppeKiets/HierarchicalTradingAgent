#!/usr/bin/env python3
"""Live trading dashboard — reads directly from the Alpaca paper account.

Shows:
  • Account equity vs starting capital / SPY benchmark
  • Open Alpaca positions with live unrealised P&L + ML signal metadata
  • Recent filled orders (last 20 closed trades)
  • Equity sparkline from local equity_curve.json

Usage
-----
  python scripts/trading_dashboard.py              # one-shot print
  python scripts/trading_dashboard.py --watch      # refresh every 60 s
  python scripts/trading_dashboard.py --interval 30
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

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m"
RED    = "\033[31m"; GREEN  = "\033[32m"; YELLOW = "\033[33m"
CYAN   = "\033[36m"; WHITE  = "\033[37m"; GREY   = "\033[90m"

def _c(text, *codes):
    return "".join(codes) + str(text) + RESET

def _pnl(val: float) -> str:
    s = f"{val:>+10.2f}"
    return _c(s, GREEN, BOLD) if val > 0 else (_c(s, RED, BOLD) if val < 0 else _c(s, WHITE))

def _pct(val: float) -> str:
    s = f"{val:>+7.2f}%"
    return _c(s, GREEN) if val > 0 else (_c(s, RED) if val < 0 else _c(s, WHITE))

_SPARKS = "▁▂▃▄▅▆▇█"

def sparkline(values: List[float], width: int = 50) -> str:
    if not values:
        return ""
    step    = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][-width:]
    lo, hi  = min(sampled), max(sampled)
    if hi == lo:
        return _SPARKS[3] * len(sampled)
    chars = [_SPARKS[int((v - lo) / (hi - lo) * (len(_SPARKS) - 1))] for v in sampled]
    color  = GREEN if sampled[-1] >= sampled[0] else RED
    return "".join(chars[:-1]) + _c(chars[-1], color, BOLD)


# ── Alpaca helpers ────────────────────────────────────────────────────────────

def _load_clients():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    return (TradingClient(api_key, secret_key, paper=True),
            StockHistoricalDataClient(api_key, secret_key))


def _alpaca_account(tc) -> Dict[str, float]:
    a = tc.get_account()
    return {
        "equity":        float(a.equity),
        "cash":          float(a.cash),
        "buying_power":  float(a.buying_power),
        "portfolio_value": float(a.portfolio_value),
    }


def _alpaca_positions(tc) -> List[Dict[str, Any]]:
    result = []
    for p in tc.get_all_positions():
        result.append({
            "ticker":          p.symbol,
            "qty":             float(p.qty),
            "side":            "long" if float(p.qty) > 0 else "short",
            "avg_entry_price": float(p.avg_entry_price),
            "current_price":   float(p.current_price),
            "unrealized_pl":   float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) * 100,
            "market_value":    float(p.market_value),
        })
    return result


def _alpaca_recent_orders(tc, limit: int = 20) -> List[Dict[str, Any]]:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    orders = tc.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit))
    result = []
    for o in orders:
        status_str = str(o.status).split(".")[-1].lower()
        if status_str not in ("filled", "partially_filled"):
            continue
        qty = float(getattr(o, "filled_qty", 0) or 0)
        avg = float(getattr(o, "filled_avg_price", 0) or 0)
        if qty == 0:
            continue
        result.append({
            "order_id":   str(o.id),
            "ticker":     o.symbol,
            "side":       str(o.side).split(".")[-1].lower(),
            "qty":        qty,
            "price":      avg,
            "filled_at":  str(getattr(o, "filled_at", ""))[:10],
            "type":       str(getattr(o, "order_type", "")).split(".")[-1].lower(),
        })
    return result


def _spy_return(start_date: str) -> float:
    """SPY total return from start_date to today (%)."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start_date, interval="1d",
                          progress=False, auto_adjust=True)
        if spy is None or spy.empty:
            return 0.0
        close = spy["Close"].dropna()
        return float((close.iloc[-1] / close.iloc[0] - 1) * 100) if len(close) >= 2 else 0.0
    except Exception:
        return 0.0


# ── ML metadata (local) ───────────────────────────────────────────────────────

def _load_ml_meta(data_dir: Path) -> Dict[str, Any]:
    path = data_dir / "ml_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_equity_curve(data_dir: Path) -> List[Dict[str, Any]]:
    path = data_dir / "equity_curve.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ── Main render ───────────────────────────────────────────────────────────────

def render_dashboard(data_dir: Path) -> None:
    try:
        tc, dc = _load_clients()
    except Exception as e:
        print(_c(f"\n  ✗ Could not connect to Alpaca: {e}\n", RED, BOLD))
        return

    try:
        acct      = _alpaca_account(tc)
        positions = _alpaca_positions(tc)
        orders    = _alpaca_recent_orders(tc)
    except Exception as e:
        print(_c(f"\n  ✗ Alpaca API error: {e}\n", RED, BOLD))
        return

    ml_meta    = _load_ml_meta(data_dir)
    curve      = _load_equity_curve(data_dir)
    starting   = float(ml_meta.get("_starting_capital", acct["equity"]))
    total_ret  = (acct["equity"] / starting - 1) * 100 if starting else 0.0

    # SPY benchmark
    spy_ret = 0.0
    if curve:
        spy_ret = _spy_return(curve[0]["date"])
    alpha = total_ret - spy_ret

    W = 72
    print()
    print(_c("═" * W, CYAN))
    print(_c(f"  📈  ALPACA PAPER TRADING DASHBOARD  —  {date.today().isoformat()}", CYAN, BOLD))
    print(_c("═" * W, CYAN))

    # ── Account summary ────────────────────────────────────────────────────
    print(f"\n  {'Starting capital':<26}  ${starting:>12,.2f}")
    print(f"  {'Current equity':<26}  ${acct['equity']:>12,.2f}   {_pct(total_ret)}")
    print(f"  {'Cash':<26}  ${acct['cash']:>12,.2f}")
    print(f"  {'Buying power':<26}  ${acct['buying_power']:>12,.2f}")
    print(f"  {'SPY since inception':<26}  {_pct(spy_ret)}   alpha {_pct(alpha)}")

    # Equity sparkline
    if curve:
        vals = [s["equity"] for s in curve]
        print(f"\n  Equity  {sparkline(vals, width=50)}")
        print(f"          ${min(vals):,.0f} ← → ${max(vals):,.0f}")
        print(f"\n  Recent snapshots (last 5 days):")
        for snap in curve[-5:]:
            d = snap.get("pnl_today", 0)
            print(f"    {snap['date']}  equity=${snap['equity']:>10,.2f}  "
                  f"daily={d:>+8.2f}  pos={snap.get('open_positions', 0)}")

    # ── Open positions ─────────────────────────────────────────────────────
    print(f"\n  {_c('OPEN POSITIONS', BOLD)}  ({len(positions)})")
    if positions:
        print(f"  {'─'*(W-2)}")
        hdr = (f"  {'TICKER':<8} {'SIDE':<6} {'ENTRY':>8} {'NOW':>8}"
               f" {'QTY':>8} {'UNRL P&L':>10} {'%':>8}  {'STOP':>8}  {'TP':>8}  REGIME")
        print(_c(hdr, GREY))
        for pos in sorted(positions, key=lambda p: abs(p["unrealized_plpc"]), reverse=True):
            meta   = ml_meta.get(pos["ticker"], {})
            regime = meta.get("regime_label", "")
            stop   = meta.get("stop_loss", 0)
            tp     = meta.get("take_profit", 0)
            ep_date = meta.get("entry_date", "")
            dir_str = _c(pos["side"].upper(), GREEN if pos["side"] == "long" else RED)
            print(f"  {pos['ticker']:<8} {dir_str:<14}"
                  f" {pos['avg_entry_price']:>8.2f} {pos['current_price']:>8.2f}"
                  f" {pos['qty']:>8.2f} {_pnl(pos['unrealized_pl'])} {_pct(pos['unrealized_plpc'])}"
                  f"  {stop:>8.2f}  {tp:>8.2f}  {regime}")
    else:
        print(f"  {_c('  No open positions.', GREY)}")

    # ── Recent orders (closed) ─────────────────────────────────────────────
    if orders:
        print(f"\n  {_c('RECENT ORDERS', BOLD)}  (last {len(orders)})")
        print(f"  {'─'*(W-2)}")
        hdr2 = f"  {'TICKER':<8} {'SIDE':<6} {'PRICE':>8} {'QTY':>8}  {'TYPE':<16} {'DATE'}"
        print(_c(hdr2, GREY))
        for o in orders[:20]:
            side_str = _c(o["side"].upper(), GREEN if o["side"] == "buy" else RED)
            otype    = _c(o["type"], YELLOW if "stop" in o["type"] else GREY)
            print(f"  {o['ticker']:<8} {side_str:<14} {o['price']:>8.2f} {o['qty']:>8.2f}"
                  f"  {otype:<24} {o['filled_at']}")

    print(f"\n{_c('═'*W, CYAN)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Alpaca Paper Trading Dashboard")
    p.add_argument("--data-dir",  default="data/paper_trading")
    p.add_argument("--watch",     action="store_true", help="Auto-refresh")
    p.add_argument("--interval",  type=int, default=60, help="Refresh interval (s)")
    args = p.parse_args()
    data_dir = Path(args.data_dir)

    if args.watch:
        try:
            while True:
                os.system("clear")
                render_dashboard(data_dir)
                print(f"  {_c(f'Refreshing every {args.interval}s  (Ctrl+C to quit)', GREY)}\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nBye!")
    else:
        render_dashboard(data_dir)


if __name__ == "__main__":
    main()
