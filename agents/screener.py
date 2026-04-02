"""Screener Agent — ranks all tickers and returns a shortlist.

Pipeline step 1/4.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, cast

from agents.base import BaseAgent
from agents.state import TradingState, TickerPrediction
from agents.tools.screener_tools import RankAllTickersTool

logger = logging.getLogger(__name__)


class ScreenerAgent(BaseAgent):
    """Run hierarchical model on all cached tickers, keep the top-N."""

    name = "screener"

    def __init__(
        self,
        model_path: str,
        cache_dir: str = "data/feature_cache",
        top_n: int = 20,
        max_tickers: int = 0,
    ):
        super().__init__()
        self.top_n = top_n
        self.register_tool(
            RankAllTickersTool(model_path=model_path, cache_dir=cache_dir, max_tickers=max_tickers)
        )

    def _run(self, state: TradingState) -> TradingState:
        rank_tool = cast(RankAllTickersTool, self.tools["rank_all_tickers"])
        all_preds: List[TickerPrediction] = rank_tool()

        logger.info(
            "Screener evaluated %d tickers. Top predicted return: %.4f (%s)",
            len(all_preds),
            all_preds[0].predicted_return if all_preds else 0.0,
            all_preds[0].ticker if all_preds else "N/A",
        )

        # Keep top-N by predicted return (list is already sorted desc)
        state.screened_tickers = all_preds[: self.top_n]

        # Also store a summary for downstream agents
        state.metadata["screener_total_tickers"] = len(all_preds)
        state.metadata["screener_top_n"] = self.top_n

        return state
