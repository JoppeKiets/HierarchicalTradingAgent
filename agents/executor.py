"""Executor Agent — builds concrete trade orders with stops and sizing.

Pipeline step 4/4.

Writes every decision (entry, size, stop, TP) to a structured trade journal
that becomes ground truth for later feedback loops.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from agents.base import BaseAgent
from agents.state import (
    TradingState,
    CriticAssessment,
    ExecutorOrder,
    RiskLevel,
    TradeJournalEntry,
)
from agents.tools.executor_tools import (
    GetCurrentRegimeTool,
    ClassifyRegimeTool,
    ComputeATRStopTool,
    PositionSizerTool,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Trade Journal — structured paper-trade logging
# ============================================================================

class TradeJournal:
    """Structured paper-trade logger.

    Writes every Executor decision (entry, size, stop, TP) to both
    JSON and CSV files.  This becomes ground truth for later feedback
    loops (e.g. comparing predicted vs actual return, stop-loss hit rates,
    regime-specific P&L).

    Files:
        {journal_dir}/trade_journal.jsonl    — append-only, one JSON per line
        {journal_dir}/trade_journal.csv      — append-only CSV for easy analysis
        {journal_dir}/runs/{run_id}.json     — per-run snapshot
    """

    def __init__(self, journal_dir: str = "data/trade_journal"):
        self.journal_dir = journal_dir
        self.runs_dir = os.path.join(journal_dir, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)

        self.jsonl_path = os.path.join(journal_dir, "trade_journal.jsonl")
        self.csv_path = os.path.join(journal_dir, "trade_journal.csv")

        # Initialize CSV with header if it doesn't exist
        if not os.path.exists(self.csv_path):
            self._write_csv_header()

    def _write_csv_header(self):
        """Write CSV header row."""
        headers = [
            "timestamp", "run_id", "ticker", "direction", "entry_price",
            "position_size_pct", "stop_loss", "take_profit", "atr",
            "risk_reward_ratio", "risk_level", "regime_label",
            "model_confidence", "predicted_return", "sub_model_agreement",
            "sub_model_spread", "regime_confidence_score",
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_entries(
        self, entries: List[TradeJournalEntry], run_id: str = ""
    ):
        """Append trade journal entries to JSONL and CSV files.

        Also writes a per-run snapshot JSON for easy lookup.
        """
        if not entries:
            return

        # Append to JSONL (one JSON object per line)
        with open(self.jsonl_path, "a") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry), default=str) + "\n")

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for entry in entries:
                writer.writerow([
                    entry.timestamp,
                    entry.run_id,
                    entry.ticker,
                    entry.direction,
                    entry.entry_price,
                    entry.position_size_pct,
                    entry.stop_loss,
                    entry.take_profit,
                    entry.atr,
                    entry.risk_reward_ratio,
                    entry.risk_level,
                    entry.regime_label,
                    entry.model_confidence,
                    entry.predicted_return,
                    entry.sub_model_agreement,
                    entry.sub_model_spread,
                    entry.regime_confidence_score,
                ])

        # Per-run snapshot
        if run_id:
            run_path = os.path.join(self.runs_dir, f"{run_id}.json")
            with open(run_path, "w") as f:
                json.dump(
                    {"run_id": run_id, "n_orders": len(entries),
                     "entries": [asdict(e) for e in entries]},
                    f, indent=2, default=str,
                )

        logger.info(
            "TradeJournal | Logged %d entries → %s", len(entries), self.jsonl_path
        )


class ExecutorAgent(BaseAgent):
    """Build ExecutorOrders for every approved ticker.

    Also writes every decision to a structured trade journal for
    feedback loop ground truth.
    """

    name = "executor"

    def __init__(
        self,
        organized_dir: str = "data/organized",
        max_position_pct: float = 0.05,
        journal_dir: str = "data/trade_journal",
        min_price: float = 2.0,
    ):
        super().__init__()
        self.max_position_pct = max_position_pct
        self.min_price = min_price
        self.journal = TradeJournal(journal_dir=journal_dir)
        self.register_tool(GetCurrentRegimeTool(organized_dir=organized_dir))
        self.register_tool(ClassifyRegimeTool())
        self.register_tool(
            ComputeATRStopTool(organized_dir=organized_dir)
        )
        self.register_tool(PositionSizerTool())

    def _run(self, state: TradingState) -> TradingState:
        regime_tool = self.tools["get_current_regime"]
        classify_tool = self.tools["classify_regime"]
        atr_tool = self.tools["compute_atr_stop"]
        sizer_tool = self.tools["position_sizer"]

        # Fetch regime once for the whole batch
        regime_features = regime_tool()
        regime_label = classify_tool(regime_features=regime_features)
        state.regime_label = regime_label
        state.regime_features = regime_features
        logger.info("Executor | Current regime: %s", regime_label)

        approved = [a for a in state.critic_assessments if a.approved]

        # Build lookup for analyst reports (for predicted_return, agreement)
        report_map = {r.ticker: r for r in state.analyst_reports}
        # Build lookup for critic assessments (for regime confidence)
        assessment_map = {a.ticker: a for a in state.critic_assessments}

        now = datetime.now(timezone.utc)
        run_id = state.run_id or now.strftime("%Y%m%d_%H%M%S")

        orders: List[ExecutorOrder] = []
        journal_entries: List[TradeJournalEntry] = []

        for assessment in approved:
            ticker = assessment.ticker
            report = report_map.get(ticker)
            if report is None:
                continue

            direction = "long" if report.predicted_return > 0 else "short"
            atr_info = atr_tool(ticker=ticker, direction=direction)
            current_price = atr_info.get("current_price")
            if current_price is None:
                logger.info("Executor | SKIP %s no current price available", ticker)
                continue
            if current_price < self.min_price:
                logger.info(
                    "Executor | SKIP %s low price %.2f < min_price %.2f",
                    ticker,
                    current_price,
                    self.min_price,
                )
                continue

            position_pct = sizer_tool(
                regime_label=regime_label,
                model_confidence=state.model_confidence.value,
                sub_model_agreement=report.sub_model_agreement,
                max_position_pct=self.max_position_pct,
            )

            order = ExecutorOrder(
                ticker=ticker,
                direction=direction,
                position_size_pct=position_pct,
                stop_loss=atr_info.get("stop_loss"),
                take_profit=atr_info.get("take_profit"),
                atr=atr_info.get("atr"),
                current_price=atr_info.get("current_price"),
                risk_level=assessment.risk_level,
                regime_label=regime_label,
            )
            orders.append(order)

            # ── Build trade journal entry ─────────────────────────────
            rr_ratio = atr_info.get("risk_reward_ratio")
            critic_assess = assessment_map.get(ticker)

            journal_entry = TradeJournalEntry(
                timestamp=now.isoformat(),
                run_id=run_id,
                ticker=ticker,
                direction=direction,
                entry_price=atr_info.get("current_price"),
                position_size_pct=position_pct,
                stop_loss=atr_info.get("stop_loss"),
                take_profit=atr_info.get("take_profit"),
                atr=atr_info.get("atr"),
                risk_reward_ratio=rr_ratio,
                risk_level=assessment.risk_level.value,
                regime_label=regime_label,
                model_confidence=state.model_confidence.value,
                predicted_return=report.predicted_return,
                sub_model_agreement=report.sub_model_agreement,
                sub_model_spread=critic_assess.sub_model_spread if critic_assess else 0.0,
                regime_confidence_score=critic_assess.regime_confidence_score if critic_assess else 1.0,
                regime_features=regime_features,
                approved_reason="; ".join(assessment.reasons) if assessment.reasons else "all checks passed",
            )
            journal_entries.append(journal_entry)

            logger.info(
                "Executor | ORDER %s %s  size=%.2f%%  stop=%.2f  tp=%.2f  atr=%.4f",
                direction.upper(),
                ticker,
                position_pct * 100,
                order.stop_loss or 0.0,
                order.take_profit or 0.0,
                order.atr or 0.0,
            )

        state.orders = orders
        state.trade_journal = journal_entries

        # Write journal to disk
        self.journal.log_entries(journal_entries, run_id=run_id)

        return state
