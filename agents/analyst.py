"""Analyst Agent — enriches each screened ticker with features & agreement scores.

Pipeline step 2/4.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from agents.base import BaseAgent
from agents.state import TradingState, AnalystReport
from agents.tools.analyst_tools import (
    ReadTechnicalFeaturesTool,
    ReadNewsSentimentTool,
    ComputeSubModelAgreementTool,
    AnalyzeFeatureImportanceTool,
)
from agents.feedback.feature_feedback import FeatureTrustTracker

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """For each screened ticker build an AnalystReport.

    If a ``FeatureTrustTracker`` is provided (or auto-created), the agent
    logs per-run feature assessments so trust scores accumulate over weeks.
    Features with a trust score below ``drop_threshold`` are flagged for
    automatic down-weighting or removal in future training cycles.
    """

    name = "analyst"

    def __init__(
        self,
        cache_dir: str = "data/feature_cache",
        organized_dir: str = "data/organized",
        feedback_dir: str = "data/feature_feedback",
        feature_tracker: Optional[FeatureTrustTracker] = None,
    ):
        super().__init__()
        self.register_tool(ReadTechnicalFeaturesTool(cache_dir=cache_dir))
        self.register_tool(ReadNewsSentimentTool(organized_dir=organized_dir))
        self.register_tool(ComputeSubModelAgreementTool())
        self.register_tool(AnalyzeFeatureImportanceTool(cache_dir=cache_dir))

        # Feature trust tracker — accumulates across runs
        self.feature_tracker = feature_tracker or FeatureTrustTracker(
            feedback_dir=feedback_dir,
        )

    def _run(self, state: TradingState) -> TradingState:
        tech_tool = self.tools["read_technical_features"]
        news_tool = self.tools["read_news_sentiment"]
        agree_tool = self.tools["compute_sub_model_agreement"]
        importance_tool = self.tools["analyze_feature_importance"]

        # Check which features are already distrusted
        dropped_features = self.feature_tracker.get_dropped_features()
        if dropped_features:
            logger.info(
                "Analyst | %d features below trust threshold: %s",
                len(dropped_features),
                dropped_features[:5],
            )

        reports = []
        for pred in state.screened_tickers:
            tech = tech_tool(ticker=pred.ticker)
            news = news_tool(ticker=pred.ticker)
            agreement = agree_tool(
                lstm_d_pred=pred.lstm_d_pred,
                tft_d_pred=pred.tft_d_pred,
                lstm_m_pred=pred.lstm_m_pred,
                tft_m_pred=pred.tft_m_pred,
                **pred.extra_preds,
            )

            # ── Feature importance analysis ──────────────────────────
            importance = importance_tool(
                ticker=pred.ticker,
                predicted_return=pred.predicted_return,
            )
            suspicious = importance.get("suspicious", [])
            supportive = importance.get("supportive", [])
            all_feats = importance.get("all_features", list(tech.keys()))

            # Record assessments in the tracker (persisted across runs)
            self.feature_tracker.record_run(
                suspicious_features=suspicious,
                supportive_features=supportive,
                all_features=all_feats,
                ticker=pred.ticker,
                run_id=state.run_id,
            )

            report = AnalystReport(
                ticker=pred.ticker,
                predicted_return=pred.predicted_return,
                sub_model_agreement=agreement["overall_agreement"],
                sentiment_score=news.get("sentiment_score", 0.0),
                has_news=news.get("available", False),
                key_features={
                    **tech,
                    "direction_agreement": agreement.get("direction_agreement", 0.5),
                    "magnitude_agreement": agreement.get("magnitude_agreement", 0.5),
                    "mean_prediction": agreement.get("mean_prediction", 0.0),
                    "std_prediction": agreement.get("std_prediction", 0.0),
                    "n_active_models": agreement.get("n_active_models", 0.0),
                    "n_nonzero_models": agreement.get("n_nonzero_models", 0.0),
                    "sign_balance": agreement.get("sign_balance", 0.0),
                },
                attention_weights=pred.attention_weights,
                suspicious_features=suspicious,
                supportive_features=supportive,
            )
            reports.append(report)

            logger.debug(
                "Analyst | %s: pred=%.4f  agreement=%.2f  news=%.2f  "
                "suspicious=%d  supportive=%d",
                pred.ticker,
                pred.predicted_return,
                agreement["overall_agreement"],
                news.get("sentiment_score", 0.0),
                len(suspicious),
                len(supportive),
            )

        state.analyst_reports = reports

        # Persist updated trust scores to disk
        self.feature_tracker.save()
        trust_summary = self.feature_tracker.summary()
        logger.info(
            "Analyst | Feature trust summary: %d features tracked, "
            "%d dropped, trust range [%.2f, %.2f]",
            trust_summary.get("n_features", 0),
            trust_summary.get("n_dropped", 0),
            trust_summary.get("trust_min", 0),
            trust_summary.get("trust_max", 0),
        )

        return state
