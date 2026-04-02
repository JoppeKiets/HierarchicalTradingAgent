"""Shared state flowing through the multi-agent swing trading pipeline.

Each agent reads upstream fields and writes its own section.
The state is a mutable dataclass — agents modify it in place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelConfidence(Enum):
    HOT = "hot"          # Recent IC > threshold
    WARM = "warm"        # Marginal
    COLD = "cold"        # Model is unreliable right now


@dataclass
class TickerPrediction:
    """Output from the Screener for a single ticker."""
    ticker: str
    predicted_return: float
    lstm_d_pred: float
    tft_d_pred: float
    lstm_m_pred: float
    tft_m_pred: float
    # Extended sub-model predictions (v10+: tcn_d, gnn, fund_mlp, news, etc.)
    extra_preds: Dict[str, float] = field(default_factory=dict)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    has_minute_data: bool = False
    rank: int = 0


@dataclass
class AnalystReport:
    """Output from the Analyst for a single ticker."""
    ticker: str
    predicted_return: float = 0.0
    sub_model_agreement: float = 0.0
    sentiment_score: float = 0.0
    has_news: bool = False
    key_features: Dict[str, float] = field(default_factory=dict)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    # Feature importance feedback — populated by AnalyzeFeatureImportanceTool
    suspicious_features: List[str] = field(default_factory=list)
    supportive_features: List[str] = field(default_factory=list)


@dataclass
class CriticAssessment:
    """Output from the Critic for a single ticker."""
    ticker: str
    approved: bool = True
    risk_level: "RiskLevel" = RiskLevel.MEDIUM
    reasons: List[str] = field(default_factory=list)
    model_confidence: ModelConfidence = ModelConfidence.WARM
    regime_confidence_score: float = 1.0  # Per-regime confidence (0-1), used as sample weight
    sub_model_spread: float = 0.0         # Spread of sub-model predictions


@dataclass
class ExecutorOrder:
    """Output from the Executor for a single ticker."""
    ticker: str
    direction: str = "long"              # "long" | "short" | "skip"
    position_size_pct: float = 0.0       # % of portfolio
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None
    current_price: Optional[float] = None
    risk_level: "RiskLevel" = RiskLevel.MEDIUM
    regime_label: str = ""


@dataclass
class TradeJournalEntry:
    """Structured paper-trade log entry written by the Executor.

    Captures every decision (entry, size, stop, TP) so it becomes
    ground truth for later feedback loops.
    """
    timestamp: str = ""                  # ISO-8601
    run_id: str = ""
    ticker: str = ""
    direction: str = ""                  # "long" | "short"
    entry_price: Optional[float] = None
    position_size_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    risk_level: str = ""
    regime_label: str = ""
    model_confidence: str = ""
    predicted_return: float = 0.0
    sub_model_agreement: float = 0.0
    sub_model_spread: float = 0.0
    regime_confidence_score: float = 1.0
    regime_features: Dict[str, float] = field(default_factory=dict)
    approved_reason: str = ""            # Why critic approved
    # Outcome fields — filled in by feedback loop later
    actual_return: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    exit_reason: Optional[str] = None    # "stop_hit" | "tp_hit" | "time_exit" | "manual"


@dataclass
class TradingState:
    """Complete state passed through the 4-agent pipeline.

    Flow:
        Screener → writes `screened_tickers`
        Analyst  → reads `screened_tickers`, writes `analyst_reports`
        Critic   → reads `analyst_reports`, writes `critic_assessments`
        Executor → reads all above, writes `orders`
    """

    # ── Pipeline metadata ─────────────────────────────────────────────
    run_id: str = ""
    timestamp: float = field(default_factory=time.time)
    model_path: str = ""
    cache_dir: str = "data/feature_cache"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Screener output ───────────────────────────────────────────────
    screened_tickers: List[TickerPrediction] = field(default_factory=list)

    # ── Analyst output ────────────────────────────────────────────────
    analyst_reports: List[AnalystReport] = field(default_factory=list)

    # ── Critic output ─────────────────────────────────────────────
    model_confidence: ModelConfidence = ModelConfidence.WARM
    critic_assessments: List[CriticAssessment] = field(default_factory=list)
    critic_sample_weights: Dict[str, float] = field(default_factory=dict)
    # ^ Maps regime_label → sample weight (higher = model weaker in that regime)

    # ── Executor output ───────────────────────────────────────────
    regime_label: str = ""
    regime_features: Dict[str, float] = field(default_factory=dict)
    orders: List[ExecutorOrder] = field(default_factory=list)
    trade_journal: List["TradeJournalEntry"] = field(default_factory=list)

    # ── Error tracking ────────────────────────────────────────────────
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def approved_tickers(self) -> List[str]:
        """Tickers that passed the Critic's filter."""
        return [
            a.ticker
            for a in self.critic_assessments
            if a.approved
        ]

    def summary(self) -> Dict[str, Any]:
        """Quick summary for logging."""
        return {
            "run_id": self.run_id,
            "n_screened": len(self.screened_tickers),
            "n_analyst_reports": len(self.analyst_reports),
            "n_approved": len(self.approved_tickers),
            "n_orders": len(self.orders),
            "n_errors": len(self.errors),
        }
