#!/usr/bin/env python3
"""CLI entry point for the Swing-Trading Multi-Agent pipeline.

Usage
-----
    python run_swing_pipeline.py                            # defaults
    python run_swing_pipeline.py --model models/hierarchical_v8/forecaster_final.pt --top-n 30

The pipeline runs 4 agents in order:
  1) Screener  – rank all tickers
  2) Analyst   – enrich with features & agreement scores
  3) Critic    – gate by model confidence & risk
  4) Executor  – compute stops, sizes, final orders
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from agents.pipeline import SwingTradingPipeline
from agents.state import TradingState


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def find_latest_model() -> str:
    """Auto-discover the most recent model checkpoint."""
    model_root = Path("models")
    candidates = sorted(model_root.glob("hierarchical_v*/forecaster_final.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No hierarchical model checkpoints found under models/. "
            "Train first with train_hierarchical.py or specify --model."
        )
    return str(candidates[-1])


def print_orders(state: TradingState):
    """Pretty-print the final orders to stdout."""
    if not state.orders:
        print("\n⚠  No orders generated (model confidence may be COLD or no tickers passed critic).")
        return

    print(f"\n{'='*80}")
    print(f"  SWING TRADING ORDERS  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Regime: {state.regime_label}   |   Model Confidence: {state.model_confidence.value}")
    print(f"{'='*80}\n")

    header = f"  {'#':>3}  {'Dir':>5}  {'Ticker':<6}  {'Price':>9}  {'Size%':>6}  {'Stop':>9}  {'TP':>9}  {'ATR':>8}  {'Risk':<6}"
    print(header)
    print(f"  {'─'*76}")

    for i, o in enumerate(state.orders, 1):
        print(
            f"  {i:>3}  {o.direction.upper():>5}  {o.ticker:<6}"
            f"  {o.current_price or 0:>9.2f}"
            f"  {o.position_size_pct*100:>5.2f}%"
            f"  {o.stop_loss or 0:>9.2f}"
            f"  {o.take_profit or 0:>9.2f}"
            f"  {o.atr or 0:>8.4f}"
            f"  {o.risk_level.value:<6}"
        )

    print(f"\n  Total orders: {len(state.orders)}")
    total_exposure = sum(o.position_size_pct for o in state.orders)
    print(f"  Total portfolio exposure: {total_exposure*100:.2f}%\n")


def save_results(state: TradingState, output_dir: str = "results/swing_orders"):
    """Persist the orders to a JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = out / f"orders_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "regime_label": state.regime_label,
        "model_confidence": state.model_confidence.value if state.model_confidence else None,
        "regime_features": state.regime_features,
        "metadata": state.metadata,
        "orders": [],
    }
    for o in state.orders:
        payload["orders"].append({
            "ticker": o.ticker,
            "direction": o.direction,
            "position_size_pct": o.position_size_pct,
            "stop_loss": o.stop_loss,
            "take_profit": o.take_profit,
            "atr": o.atr,
            "current_price": o.current_price,
            "risk_level": o.risk_level.value,
            "regime_label": o.regime_label,
        })

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Orders saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Swing-Trading Multi-Agent Pipeline"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to forecaster_final.pt  (default: auto-detect latest)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top tickers to pass from Screener to Analyst",
    )
    parser.add_argument(
        "--max-position", type=float, default=0.05,
        help="Max position size as fraction of portfolio (default: 0.05)",
    )
    parser.add_argument(
        "--min-agreement", type=float, default=0.50,
        help="Minimum sub-model agreement to pass Critic (default: 0.50)",
    )
    parser.add_argument(
        "--min-return", type=float, default=0.001,
        help="Minimum predicted return to pass Critic (default: 0.001)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max tickers to run inference on (0=all, useful for quick CPU tests)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save orders to results/swing_orders/",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    model_path = args.model or find_latest_model()
    logging.info("Using model: %s", model_path)

    pipeline = SwingTradingPipeline(
        model_path=model_path,
        top_n=args.top_n,
        max_position_pct=args.max_position,
        min_agreement=args.min_agreement,
        min_predicted_return=args.min_return,
        max_tickers=args.limit,
    )

    state = pipeline.run()
    print_orders(state)

    if args.save:
        save_results(state)


if __name__ == "__main__":
    main()
