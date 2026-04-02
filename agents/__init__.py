"""Multi-Agent Swing Trading Framework.

Agents:
    ScreenerAgent  — ranks tickers, selects watchlist
    AnalystAgent   — technical + news analysis per ticker
    CriticAgent    — validates model reliability
    ExecutorAgent  — regime-based position sizing + stops

Pipeline:
    SwingTradingPipeline — sequential orchestrator
"""

from agents.state import TradingState
from agents.pipeline import SwingTradingPipeline
