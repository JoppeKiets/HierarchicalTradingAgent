"""Tools that wrap regime feature reading and position sizing.

Reuses:
  - _build_regime_dataframe() from src/hierarchical_data.py
  - REGIME_FEATURE_NAMES from src/hierarchical_data.py
  - RegimeFeatureExtractor.classify_regime() from src/regime_features.py
  - Price data from data/organized/{TICKER}/price_history.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base import BaseTool


class GetCurrentRegimeTool(BaseTool):
    """Read the latest 8 regime features (SPY ret, VIX, TLT, GLD, etc.).

    Wraps _build_regime_dataframe() which is already used in training
    and in collect_predictions().
    """

    name = "get_current_regime"
    description = "Get the latest market regime features (VIX, vol, trend, correlations)."

    def __init__(self, organized_dir: str = "data/organized", **kwargs):
        super().__init__(**kwargs)
        self.organized_dir = organized_dir

    def lazy_init(self):
        self._initialized = True

    def execute(self, **kwargs) -> Dict[str, float]:
        from src.hierarchical_data import (
            HierarchicalDataConfig,
            _build_regime_dataframe,
            REGIME_FEATURE_NAMES,
        )

        # Reset the module-level cache so we get fresh data
        import src.hierarchical_data as hd
        hd._REGIME_CACHE = None

        cfg = HierarchicalDataConfig(organized_dir=self.organized_dir)
        regime_df = _build_regime_dataframe(cfg)

        if regime_df.empty:
            return {name: 0.0 for name in REGIME_FEATURE_NAMES}

        latest = regime_df.iloc[-1]
        result = {}
        for name in REGIME_FEATURE_NAMES:
            val = latest.get(name, 0.0)
            result[name] = float(val) if np.isfinite(val) else 0.0

        return result


class ClassifyRegimeTool(BaseTool):
    """Classify the current market regime into a human-readable label.

    Wraps RegimeFeatureExtractor.classify_regime() from src/regime_features.py.
    """

    name = "classify_regime"
    description = "Classify market regime (bull/bear × high/low vol, sideways)."

    def lazy_init(self):
        self._initialized = True

    def execute(self, regime_features: Dict[str, float], **kwargs) -> str:
        from src.regime_features import RegimeFeatureExtractor

        extractor = RegimeFeatureExtractor()
        return extractor.classify_regime(
            vol_regime=regime_features.get("regime_spy_vol20", 0.0),
            vix_level=regime_features.get("regime_vix_level", 0.0),
            trend=regime_features.get("regime_spy_ret5", 0.0),
            drawdown=regime_features.get("regime_spy_breadth", 0.0),
        )


class ComputeATRStopTool(BaseTool):
    """Compute ATR-based stop-loss and take-profit levels from price_history.csv."""

    name = "compute_atr_stop"
    description = "Compute ATR-based stop-loss for a ticker."

    def __init__(self, organized_dir: str = "data/organized", **kwargs):
        super().__init__(**kwargs)
        self.organized_dir = organized_dir

    def lazy_init(self):
        self._initialized = True

    def execute(
        self,
        ticker: str,
        atr_multiplier: float = 2.0,
        atr_period: int = 14,
        direction: str = "long",
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        import pandas as pd

        price_path = Path(self.organized_dir) / ticker / "price_history.csv"
        if not price_path.exists():
            return {"stop_loss": None, "take_profit": None, "atr": None, "current_price": None}

        try:
            df = pd.read_csv(price_path)
        except Exception:
            return {"stop_loss": None, "take_profit": None, "atr": None, "current_price": None}

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        if len(df) < atr_period + 1:
            return {"stop_loss": None, "take_profit": None, "atr": None, "current_price": None}

        high: np.ndarray = df["high"].to_numpy()
        low: np.ndarray = df["low"].to_numpy()
        close: np.ndarray = df["close"].to_numpy()

        # True Range computation
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = float(np.mean(tr[-atr_period:]))
        current_price = float(close[-1])

        if direction == "long":
            stop_loss = current_price - atr_multiplier * atr
            take_profit = current_price + (atr_multiplier * 1.5) * atr
        else:
            stop_loss = current_price + atr_multiplier * atr
            take_profit = current_price - (atr_multiplier * 1.5) * atr

        return {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "atr": round(atr, 4),
            "current_price": round(current_price, 2),
            "risk_reward_ratio": round(1.5, 2),
        }


class PositionSizerTool(BaseTool):
    """Determine position size based on regime, model confidence, and agreement."""

    name = "position_sizer"
    description = "Compute position size as a % of portfolio."

    def lazy_init(self):
        self._initialized = True

    def execute(
        self,
        regime_label: str,
        model_confidence: str,       # "hot", "warm", "cold"
        sub_model_agreement: float,  # 0.0-1.0
        max_position_pct: float = 0.05,
        **kwargs,
    ) -> float:
        """Returns position size as fraction of portfolio (0.0 to max_position_pct)."""
        size = max_position_pct

        # Scale down for adverse regimes
        regime_scale = {
            "bull_low_vol": 1.0,
            "bull_high_vol": 0.7,
            "sideways": 0.5,
            "bear_low_vol": 0.3,
            "bear_high_vol": 0.2,
        }
        size *= regime_scale.get(regime_label, 0.5)

        # Scale down for cold model
        confidence_scale = {"hot": 1.0, "warm": 0.6, "cold": 0.2}
        size *= confidence_scale.get(model_confidence, 0.5)

        # Scale by sub-model agreement (0.5 to 1.0 range)
        size *= (0.5 + 0.5 * sub_model_agreement)

        return round(min(size, max_position_pct), 4)
