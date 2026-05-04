#!/usr/bin/env python3
"""Paper Trading Engine — Alpaca paper trading backend.

What this does
--------------
1. Runs the ML pipeline (Screener → Analyst → Critic → Executor).
2. Fetches live prices from the Alpaca market data API.
3. Submits bracket orders (built-in stop-loss + take-profit) to the Alpaca paper account.
4. Syncs open positions and account equity back from Alpaca.
5. Saves ML metadata (predicted return, regime, run_id, etc.) locally in JSON
   because Alpaca doesn't know about our model signals.

Files written
-------------
  data/paper_trading/ml_metadata.json    — ML metadata keyed by ticker
  data/paper_trading/equity_curve.json   — daily equity snapshots
  data/paper_trading/ledger.jsonl        — filled/closed order records

Usage
-----
  python scripts/paper_trader.py                   # normal daily run
  python scripts/paper_trader.py --dry-run         # show orders, submit nothing
  python scripts/paper_trader.py --status          # print Alpaca account status
  python scripts/paper_trader.py --sync            # sync Alpaca state to local files
  python scripts/paper_trader.py --force-exit-all  # liquidate all positions
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpaca client factory
# ---------------------------------------------------------------------------

def _load_alpaca_clients():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    trading = TradingClient(api_key, secret_key, paper=True)
    data    = StockHistoricalDataClient(api_key, secret_key)
    return trading, data


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_latest_prices(tickers: List[str], data_client) -> Dict[str, float]:
    """Fetch latest mid-price for each ticker via Alpaca data API.
    Tries latest quote (real-time mid) first; falls back to latest bar close."""
    if not tickers:
        return {}
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest
    prices: Dict[str, float] = {}

    # Latest quote (real-time mid-price)
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=tickers)
        quotes = data_client.get_stock_latest_quote(req)
        for sym, q in quotes.items():
            ask = float(getattr(q, "ask_price", 0) or 0)
            bid = float(getattr(q, "bid_price", 0) or 0)
            if ask > 0 and bid > 0:
                prices[sym] = round((ask + bid) / 2.0, 4)
            elif ask > 0:
                prices[sym] = ask
            elif bid > 0:
                prices[sym] = bid
    except Exception as e:
        logger.warning("Quote fetch failed: %s — trying latest bar", e)

    # Fall back to latest bar close for anything still missing
    missing = [t for t in tickers if t not in prices]
    if missing:
        try:
            req2 = StockLatestBarRequest(symbol_or_symbols=missing)
            bars = data_client.get_stock_latest_bar(req2)
            for sym, bar in bars.items():
                prices[sym] = float(bar.close)
        except Exception as e2:
            logger.warning("Bar fallback failed for %d tickers: %s", len(missing), e2)

    return prices


# ---------------------------------------------------------------------------
# ML metadata  (local JSON — Alpaca doesn't know our model signals)
# ---------------------------------------------------------------------------

_ML_META_FILE    = "ml_metadata.json"
_EQUITY_CURVE_FILE = "equity_curve.json"
_LEDGER_FILE     = "ledger.jsonl"


def load_ml_metadata(data_dir: Path) -> Dict[str, Any]:
    path = data_dir / _ML_META_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_ml_metadata(data_dir: Path, meta: Dict[str, Any]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / _ML_META_FILE, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_equity_curve(data_dir: Path) -> List[Dict[str, Any]]:
    path = data_dir / _EQUITY_CURVE_FILE
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def append_equity_snapshot(data_dir: Path, equity: float, cash: float, n_positions: int) -> None:
    curve = load_equity_curve(data_dir)
    today = date.today().isoformat()
    if curve and curve[-1]["date"] == today:
        curve[-1].update({"equity": round(equity, 2), "cash": round(cash, 2),
                          "open_positions": n_positions})
    else:
        pnl_today = round(equity - curve[-1]["equity"], 2) if curve else 0.0
        curve.append({"date": today, "equity": round(equity, 2),
                      "cash": round(cash, 2), "open_positions": n_positions,
                      "pnl_today": pnl_today})
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / _EQUITY_CURVE_FILE, "w") as f:
        json.dump(curve, f, indent=2)


def append_ledger(data_dir: Path, record: Dict[str, Any]) -> None:
    path = data_dir / _LEDGER_FILE
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Alpaca account / position helpers
# ---------------------------------------------------------------------------

def get_account_summary(trading_client) -> Dict[str, float]:
    acct = trading_client.get_account()
    return {
        "equity":          float(acct.equity),
        "cash":            float(acct.cash),
        "buying_power":    float(acct.buying_power),
        "portfolio_value": float(acct.portfolio_value),
        "daytrade_count":  int(getattr(acct, "daytrade_count", 0)),
    }


def get_open_positions(trading_client) -> List[Dict[str, Any]]:
    result = []
    for pos in trading_client.get_all_positions():
        result.append({
            "ticker":          pos.symbol,
            "qty":             float(pos.qty),
            "side":            "long" if float(pos.qty) > 0 else "short",
            "avg_entry_price": float(pos.avg_entry_price),
            "current_price":   float(pos.current_price),
            "unrealized_pl":   float(pos.unrealized_pl),
            "unrealized_plpc": float(pos.unrealized_plpc) * 100,
            "market_value":    float(pos.market_value),
        })
    return result


def get_recent_orders(trading_client, limit: int = 50) -> List[Dict[str, Any]]:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    orders = trading_client.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.CLOSED, limit=limit
    ))
    result = []
    for o in orders:
        status_str = str(o.status).split(".")[-1].lower()
        if status_str not in ("filled", "partially_filled"):
            continue
        filled_qty = float(getattr(o, "filled_qty", 0) or 0)
        filled_avg = float(getattr(o, "filled_avg_price", 0) or 0)
        if filled_qty == 0:
            continue
        result.append({
            "order_id":    str(o.id),
            "ticker":      o.symbol,
            "side":        str(o.side).split(".")[-1].lower(),
            "qty":         filled_qty,
            "filled_price": filled_avg,
            "filled_at":   str(getattr(o, "filled_at", "")),
            "order_type":  str(getattr(o, "order_type", "")).split(".")[-1].lower(),
        })
    return result


# ---------------------------------------------------------------------------
# Order submission  (bracket = SL + TP built into Alpaca)
# ---------------------------------------------------------------------------

def submit_bracket_order(
    trading_client,
    ticker: str,
    notional: float,
    direction: str,
    stop_price: float,
    take_profit_price: float,
    dry_run: bool = False,
) -> Optional[Any]:
    """Submit a market bracket order to Alpaca paper account.

    Alpaca enforces:  for LONG  → stop < entry < take_profit
                      for SHORT → take_profit < entry < stop
    """
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    side = OrderSide.BUY if direction == "long" else OrderSide.SELL
    stop_price        = round(stop_price, 2)
    take_profit_price = round(take_profit_price, 2)

    order_req = MarketOrderRequest(
        symbol=ticker,
        notional=round(notional, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_price),
    )

    if dry_run:
        logger.info("[DRY-RUN] %s %-6s  notional=$%.2f  stop=%.2f  tp=%.2f",
                    direction.upper(), ticker, notional, stop_price, take_profit_price)
        return None

    try:
        order = trading_client.submit_order(order_req)
        logger.info("ORDER  %s %-6s  notional=$%.2f  stop=%.2f  tp=%.2f  id=%s",
                    direction.upper(), ticker, notional, stop_price, take_profit_price, order.id)
        return order
    except Exception as e:
        logger.error("Order failed for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(model_path: str, top_n: int = 20):
    from agents.pipeline import SwingTradingPipeline
    logger.info("Running pipeline with model=%s", model_path)
    pipeline = SwingTradingPipeline(
        model_path=model_path,
        top_n=top_n,
        max_position_pct=0.05,
        min_agreement=0.50,
        min_predicted_return=0.001,
    )
    state = pipeline.run()
    logger.info("Pipeline: screened=%d  approved=%d  orders=%d",
                len(state.screened_tickers), len(state.approved_tickers), len(state.orders))
    return state


# ---------------------------------------------------------------------------
# Status printer
# ---------------------------------------------------------------------------

def print_status(trading_client, ml_meta: Dict[str, Any]) -> None:
    acct      = get_account_summary(trading_client)
    positions = get_open_positions(trading_client)
    starting  = float(ml_meta.get("_starting_capital", acct["equity"]))
    total_ret = (acct["equity"] / starting - 1) * 100 if starting else 0.0
    sign = "+" if total_ret >= 0 else ""

    print(f"\n{'═'*72}")
    print(f"  ALPACA PAPER TRADING STATUS  —  {date.today().isoformat()}")
    print(f"{'═'*72}")
    print(f"  Equity           : ${acct['equity']:>12,.2f}   ({sign}{total_ret:.2f}%)")
    print(f"  Cash             : ${acct['cash']:>12,.2f}")
    print(f"  Buying power     : ${acct['buying_power']:>12,.2f}")
    print(f"  Open positions   : {len(positions)}")

    if positions:
        print(f"\n  {'─'*68}")
        print(f"  {'TICKER':<8} {'SIDE':<6} {'ENTRY':>8} {'NOW':>8} {'QTY':>8} {'UNRL P&L':>10} {'%':>8}  REGIME")
        print(f"  {'─'*68}")
        for pos in positions:
            meta   = ml_meta.get(pos["ticker"], {})
            regime = meta.get("regime_label", "")
            sign_p = "+" if pos["unrealized_pl"] >= 0 else ""
            sign_r = "+" if pos["unrealized_plpc"] >= 0 else ""
            print(f"  {pos['ticker']:<8} {pos['side'].upper():<6}"
                  f" {pos['avg_entry_price']:>8.2f} {pos['current_price']:>8.2f}"
                  f" {pos['qty']:>8.2f} {sign_p}{pos['unrealized_pl']:>9.2f}"
                  f" {sign_r}{pos['unrealized_plpc']:>7.2f}%  {regime}")

    print(f"{'═'*72}\n")


# ---------------------------------------------------------------------------
# Core daily run
# ---------------------------------------------------------------------------

def paper_trade_day(
    trading_client,
    data_client,
    data_dir: Path,
    model_path: str,
    top_n: int = 20,
    dry_run: bool = False,
) -> None:
    ml_meta = load_ml_metadata(data_dir)

    # ── 1. Run ML pipeline ─────────────────────────────────────────────
    state = run_pipeline(model_path, top_n=top_n)

    # ── 2. Current Alpaca state ────────────────────────────────────────
    acct  = get_account_summary(trading_client)
    held  = {p["ticker"] for p in get_open_positions(trading_client)}

    # ── 3. Fetch prices for new candidates ────────────────────────────
    new_tickers = [o.ticker for o in state.orders if o.ticker not in held]
    prices = fetch_latest_prices(new_tickers, data_client) if new_tickers else {}

    # ── 4. Submit bracket orders for new signals ───────────────────────
    report_map = {r.ticker: r for r in state.analyst_reports}

    for order in state.orders:
        if order.ticker in held:
            logger.info("SKIP  %s — already held in Alpaca", order.ticker)
            continue

        price = prices.get(order.ticker)
        if not price or price < 2.0:
            logger.warning("SKIP  %s — no price (%.2f)", order.ticker, price or 0)
            continue

        notional = acct["buying_power"] * order.position_size_pct
        if notional < 10.0:
            logger.info("SKIP  %s — notional $%.2f too small", order.ticker, notional)
            continue

        atr = order.atr or price * 0.03
        if order.direction == "long":
            raw_stop = order.stop_loss   or (price - 2.0 * atr)
            raw_tp   = order.take_profit or (price + 3.0 * atr)
            if raw_stop >= price:
                raw_stop = price * 0.95
            if raw_tp <= price:
                raw_tp = price * 1.10
        else:
            raw_stop = order.stop_loss   or (price + 2.0 * atr)
            raw_tp   = order.take_profit or (price - 3.0 * atr)
            if raw_stop <= price:
                raw_stop = price * 1.05
            if raw_tp >= price:
                raw_tp = price * 0.90

        submitted = submit_bracket_order(
            trading_client, order.ticker, notional,
            order.direction, raw_stop, raw_tp, dry_run=dry_run,
        )

        if submitted or dry_run:
            report = report_map.get(order.ticker)
            ml_meta[order.ticker] = {
                "direction":           order.direction,
                "predicted_return":    report.predicted_return if report else 0.0,
                "sub_model_agreement": report.sub_model_agreement if report else 0.0,
                "regime_label":        order.regime_label or getattr(state, "regime_label", ""),
                "run_id":              getattr(state, "run_id", ""),
                "entry_date":          date.today().isoformat(),
                "stop_loss":           raw_stop,
                "take_profit":         raw_tp,
                "notional":            notional,
                "alpaca_order_id":     str(submitted.id) if submitted else "dry-run",
            }

    # ── 5. Record equity snapshot ──────────────────────────────────────
    acct_now = get_account_summary(trading_client)
    if "_starting_capital" not in ml_meta:
        ml_meta["_starting_capital"] = acct_now["equity"]

    if not dry_run:
        n_pos = len(get_open_positions(trading_client))
        save_ml_metadata(data_dir, ml_meta)
        append_equity_snapshot(data_dir, acct_now["equity"], acct_now["cash"], n_pos)

    logger.info("Done | equity=$%.2f  cash=$%.2f  buying_power=$%.2f",
                acct_now["equity"], acct_now["cash"], acct_now["buying_power"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_latest_model() -> str:
    candidates = sorted(PROJECT_ROOT.glob("models/hierarchical_v*/forecaster_final.pt"))
    if not candidates:
        raise FileNotFoundError("No model found. Train first or pass --model.")
    return str(candidates[-1])


def parse_args():
    p = argparse.ArgumentParser(description="Paper Trading Engine (Alpaca)")
    p.add_argument("--model",   default=None, help="Path to forecaster_final.pt")
    p.add_argument("--top-n",   type=int, default=20)
    p.add_argument("--data-dir", default="data/paper_trading")
    p.add_argument("--dry-run", action="store_true",
                   help="Run pipeline, print orders, submit nothing to Alpaca")
    p.add_argument("--status",  action="store_true",
                   help="Print Alpaca account status and exit")
    p.add_argument("--sync",    action="store_true",
                   help="Sync Alpaca account state to local equity_curve.json")
    p.add_argument("--force-exit-all", action="store_true",
                   help="Liquidate all Alpaca positions immediately")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    trading_client, data_client = _load_alpaca_clients()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ml_meta = load_ml_metadata(data_dir)

    if args.status:
        print_status(trading_client, ml_meta)
        return

    if args.sync:
        acct  = get_account_summary(trading_client)
        n_pos = len(get_open_positions(trading_client))
        append_equity_snapshot(data_dir, acct["equity"], acct["cash"], n_pos)
        save_ml_metadata(data_dir, ml_meta)
        logger.info("Synced: equity=$%.2f  positions=%d", acct["equity"], n_pos)
        print_status(trading_client, ml_meta)
        return

    if args.force_exit_all:
        if not args.dry_run:
            logger.info("Liquidating all Alpaca positions…")
            trading_client.close_all_positions(cancel_orders=True)
            logger.info("Done — all positions closed.")
        else:
            logger.info("[DRY-RUN] Would call close_all_positions(cancel_orders=True)")
        print_status(trading_client, ml_meta)
        return

    model_path = args.model or find_latest_model()
    logger.info("Model: %s", model_path)
    paper_trade_day(
        trading_client=trading_client,
        data_client=data_client,
        data_dir=data_dir,
        model_path=model_path,
        top_n=args.top_n,
        dry_run=args.dry_run,
    )
    print_status(trading_client, ml_meta)


if __name__ == "__main__":
    main()
