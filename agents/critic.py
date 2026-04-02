"""Critic Agent — applies risk filters and model-confidence gates.

Pipeline step 3/4.

Also computes per-regime confidence scores that can be used as sample
weights in the next training cycle (upweight regimes where the model is
weakest).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agents.base import BaseAgent
from agents.state import (
    TradingState,
    AnalystReport,
    CriticAssessment,
    ModelConfidence,
    RiskLevel,
)
from agents.tools.critic_tools import (
    RecentModelPerformanceTool,
    ClassifyModelConfidenceTool,
    PerRegimePerformanceTool,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Regime confidence → sample weight mapping
# ============================================================================

def compute_regime_sample_weights(
    per_regime_metrics: Dict[str, Dict[str, float]],
    base_weight: float = 1.0,
    max_weight: float = 5.0,
) -> Dict[str, float]:
    """Convert per-regime performance metrics into training sample weights.

    Strategy: regimes where the model performs *worst* (lowest IC, lowest
    directional accuracy) get the *highest* weight so the next training
    cycle focuses on them.

    Args:
        per_regime_metrics: {regime_label: {"ic": ..., "dir_acc": ..., "n_samples": ...}}
        base_weight: minimum weight for the best-performing regime
        max_weight: cap to prevent extreme outliers

    Returns:
        {regime_label: weight}  where higher = model is weaker there
    """
    if not per_regime_metrics:
        return {}

    # Compute a "weakness score" per regime: lower IC + lower DA = weaker
    weakness_scores: Dict[str, float] = {}
    for regime, m in per_regime_metrics.items():
        ic = m.get("ic", 0.0)
        da = m.get("directional_accuracy", 0.5)
        # Combine: invert both (1 - da) and (1 - |ic|) so higher = weaker
        weakness = (1.0 - da) + (1.0 - min(abs(ic), 1.0))
        weakness_scores[regime] = weakness

    if not weakness_scores:
        return {}

    # Normalize to [base_weight, max_weight]
    min_w = min(weakness_scores.values())
    max_w = max(weakness_scores.values())
    spread = max_w - min_w if max_w > min_w else 1.0

    weights = {}
    for regime, w in weakness_scores.items():
        normalized = (w - min_w) / spread  # 0..1
        weights[regime] = base_weight + normalized * (max_weight - base_weight)

    return weights


class CriticAgent(BaseAgent):
    """Gate candidates by model confidence + agreement threshold.

    Also logs per-regime confidence scores to disk for sample weighting.
    """

    name = "critic"

    def __init__(
        self,
        results_dir: str = "results/hierarchical_eval",
        min_agreement: float = 0.5,
        min_predicted_return: float = 0.001,
        weight_log_dir: str = "data/critic_weights",
    ):
        super().__init__()
        self.min_agreement = min_agreement
        self.min_predicted_return = min_predicted_return
        self.weight_log_dir = weight_log_dir
        self.register_tool(RecentModelPerformanceTool(results_dir=results_dir))
        self.register_tool(ClassifyModelConfidenceTool())
        self.register_tool(PerRegimePerformanceTool(results_dir=results_dir))

    def _run(self, state: TradingState) -> TradingState:
        perf_tool = self.tools["recent_model_performance"]
        conf_tool = self.tools["classify_model_confidence"]
        regime_tool = self.tools["per_regime_performance"]

        perf = perf_tool()
        model_confidence: ModelConfidence = conf_tool(
            ic=perf.get("ic", 0.0),
            directional_accuracy=perf.get("directional_accuracy", 0.5),
            available=perf.get("available", False),
        )

        logger.info(
            "Critic | Model confidence: %s  IC=%.4f  DA=%.2f%%",
            model_confidence.value,
            perf.get("ic", 0.0),
            perf.get("directional_accuracy", 0.5) * 100,
        )

        state.model_confidence = model_confidence

        # ── Per-regime confidence scoring ────────────────────────────
        per_regime_metrics = regime_tool()
        regime_weights = compute_regime_sample_weights(per_regime_metrics)
        state.critic_sample_weights = regime_weights

        if regime_weights:
            logger.info("Critic | Per-regime sample weights:")
            for regime, weight in sorted(regime_weights.items()):
                rm = per_regime_metrics.get(regime, {})
                logger.info(
                    "  %-18s  weight=%.2f  (IC=%.4f, DA=%.2f%%, n=%d)",
                    regime, weight,
                    rm.get("ic", 0.0),
                    rm.get("directional_accuracy", 0.5) * 100,
                    rm.get("n_samples", 0),
                )

        # Persist to disk for the training pipeline to pick up
        self._save_weight_log(regime_weights, per_regime_metrics, perf)

        # ── Assess each ticker ───────────────────────────────────────
        assessments: List[CriticAssessment] = []
        for report in state.analyst_reports:
            approved, reasons = self._assess(report, model_confidence)
            risk = self._classify_risk(report, model_confidence)

            # Per-regime confidence for this ticker's prediction
            # (will be populated once we know the regime in Executor;
            #  for now use global model confidence as proxy)
            regime_conf = self._regime_confidence_score(model_confidence, report)

            assessment = CriticAssessment(
                ticker=report.ticker,
                approved=approved,
                risk_level=risk,
                reasons=reasons,
                model_confidence=model_confidence,
                regime_confidence_score=regime_conf,
                sub_model_spread=report.key_features.get("std_prediction", 0.0),
            )
            assessments.append(assessment)

            if approved:
                logger.info(
                    "Critic | APPROVED %s — risk=%s  pred=%.4f  agree=%.2f  regime_conf=%.2f",
                    report.ticker, risk.value,
                    report.predicted_return, report.sub_model_agreement,
                    regime_conf,
                )
            else:
                logger.debug(
                    "Critic | REJECTED %s — %s",
                    report.ticker, "; ".join(reasons),
                )

        state.critic_assessments = assessments
        return state

    # ── helpers ──────────────────────────────────────────────

    def _regime_confidence_score(
        self,
        confidence: ModelConfidence,
        report: AnalystReport,
    ) -> float:
        """Compute a 0-1 confidence score combining model state + agreement."""
        conf_map = {ModelConfidence.HOT: 1.0, ModelConfidence.WARM: 0.6, ModelConfidence.COLD: 0.2}
        base = conf_map.get(confidence, 0.5)
        # Blend with sub-model agreement
        return base * (0.5 + 0.5 * report.sub_model_agreement)

    def _save_weight_log(
        self,
        weights: Dict[str, float],
        per_regime: Dict[str, Dict[str, float]],
        global_perf: Dict[str, Any],
    ):
        """Persist regime weights to disk for next training cycle."""
        os.makedirs(self.weight_log_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        log_entry = {
            "timestamp": timestamp,
            "global_performance": {
                "ic": global_perf.get("ic", 0.0),
                "directional_accuracy": global_perf.get("directional_accuracy", 0.5),
            },
            "per_regime_metrics": per_regime,
            "sample_weights": weights,
        }

        # Write timestamped log
        log_path = os.path.join(self.weight_log_dir, f"weights_{timestamp}.json")
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

        # Also write a "latest" symlink-style file for easy pickup
        latest_path = os.path.join(self.weight_log_dir, "latest_weights.json")
        with open(latest_path, "w") as f:
            json.dump(log_entry, f, indent=2)

        logger.info("Critic | Regime weights saved → %s", log_path)

    def _assess(
        self,
        report: AnalystReport,
        confidence: ModelConfidence,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        n_active_models = int(report.key_features.get("n_active_models", 0.0))
        direction_agreement = float(
            report.key_features.get("direction_agreement", report.sub_model_agreement)
        )
        std_prediction = float(report.key_features.get("std_prediction", 0.0))

        if confidence == ModelConfidence.COLD:
            reasons.append("Model confidence is COLD – full stop.")
            return False, reasons

        if n_active_models > 0 and n_active_models < 3:
            reasons.append(
                f"Only {n_active_models} active sub-models; insufficient modality confirmation"
            )

        if abs(report.predicted_return) < self.min_predicted_return:
            reasons.append(
                f"Predicted return {report.predicted_return:.5f} below threshold "
                f"{self.min_predicted_return}"
            )

        if report.sub_model_agreement < self.min_agreement:
            reasons.append(
                f"Sub-model agreement {report.sub_model_agreement:.2f} "
                f"below threshold {self.min_agreement}"
            )

        if direction_agreement < 0.55:
            reasons.append(
                f"Directional agreement {direction_agreement:.2f} below safety floor 0.55"
            )

        if std_prediction < 1e-6 and confidence != ModelConfidence.HOT:
            reasons.append("Sub-model spread is near-zero under non-HOT confidence")

        return len(reasons) == 0, reasons

    def _classify_risk(
        self,
        report: AnalystReport,
        confidence: ModelConfidence,
    ) -> RiskLevel:
        direction_agreement = float(
            report.key_features.get("direction_agreement", report.sub_model_agreement)
        )
        if confidence == ModelConfidence.HOT and report.sub_model_agreement > 0.7:
            return RiskLevel.LOW
        elif (
            confidence == ModelConfidence.COLD
            or report.sub_model_agreement < 0.4
            or direction_agreement < 0.55
        ):
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM
