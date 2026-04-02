"""SwingTradingPipeline — sequential orchestrator for all 4 agents."""

from __future__ import annotations

import logging
import time
from typing import Optional

from agents.state import TradingState
from agents.screener import ScreenerAgent
from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.executor import ExecutorAgent

logger = logging.getLogger(__name__)


class SwingTradingPipeline:
    """Run Screener → Analyst → Critic → Executor in sequence.

    Example
    -------
    >>> pipeline = SwingTradingPipeline(model_path="models/hierarchical_v7/forecaster_final.pt")
    >>> state = pipeline.run()
    >>> for order in state.orders:
    ...     print(order)
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str = "data/feature_cache",
        organized_dir: str = "data/organized",
        results_dir: str = "results/hierarchical_eval",
        top_n: int = 20,
        max_position_pct: float = 0.05,
        min_agreement: float = 0.50,
        min_predicted_return: float = 0.001,
        max_tickers: int = 0,
        journal_dir: str = "data/trade_journal",
        weight_log_dir: str = "data/critic_weights",
    ):
        self.screener = ScreenerAgent(
            model_path=model_path,
            cache_dir=cache_dir,
            top_n=top_n,
            max_tickers=max_tickers,
        )
        self.analyst = AnalystAgent(
            cache_dir=cache_dir,
            organized_dir=organized_dir,
        )
        self.critic = CriticAgent(
            results_dir=results_dir,
            min_agreement=min_agreement,
            min_predicted_return=min_predicted_return,
            weight_log_dir=weight_log_dir,
        )
        self.executor = ExecutorAgent(
            organized_dir=organized_dir,
            max_position_pct=max_position_pct,
            journal_dir=journal_dir,
        )

    def run(self, state: Optional[TradingState] = None) -> TradingState:
        """Execute the full pipeline and return the final TradingState."""
        if state is None:
            state = TradingState()

        t0 = time.time()
        agents = [
            ("Screener", self.screener),
            ("Analyst", self.analyst),
            ("Critic", self.critic),
            ("Executor", self.executor),
        ]

        for label, agent in agents:
            step_t0 = time.time()
            logger.info("━━━ %s Agent starting ━━━", label)
            state = agent.run(state)
            elapsed = time.time() - step_t0
            logger.info(
                "━━━ %s Agent done (%.1fs) ━━━\n", label, elapsed
            )

        assert state is not None  # agent.run() always returns state
        total = time.time() - t0
        state.metadata["pipeline_elapsed_seconds"] = round(total, 2)

        # Summary log
        n_screened = len(state.screened_tickers)
        n_approved = sum(1 for a in state.critic_assessments if a.approved)
        n_orders = len(state.orders)
        logger.info(
            "Pipeline complete in %.1fs  |  screened=%d  approved=%d  orders=%d",
            total, n_screened, n_approved, n_orders,
        )

        return state
