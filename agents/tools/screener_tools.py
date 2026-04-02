"""Tools that wrap scripts/predict.py inference logic.

Reuses:
  - HierarchicalForecaster.load()
  - generate_predictions() from scripts/predict.py
  - load_latest_features() from scripts/predict.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base import BaseTool
from agents.state import TickerPrediction


class RankAllTickersTool(BaseTool):
    """Run the full HierarchicalForecaster on all cached tickers and rank them.

    Wraps the logic in scripts/predict.py:
      - Loads the forecaster checkpoint once (lazy_init)
      - Discovers tickers from feature_cache/daily/
      - Calls generate_predictions()
      - Returns List[TickerPrediction]
    """

    name = "rank_all_tickers"
    description = "Rank all tickers by predicted next-day return using the full hierarchical model."

    def __init__(self, model_path: str, cache_dir: str = "data/feature_cache", max_tickers: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.max_tickers = max_tickers  # 0 = no limit
        self.forecaster = None
        self.device = None

    def lazy_init(self):
        import torch
        from src.hierarchical_models import HierarchicalForecaster

        self.device = torch.device(self.device_str)
        self.forecaster = HierarchicalForecaster.load(
            self.model_path, device=self.device_str
        )
        self.forecaster.eval()
        logger.info(
            "HierarchicalForecaster loaded ← %s  sub-models=%s",
            self.model_path, self.forecaster.sub_model_names,
        )
        self._initialized = True

    def execute(self, **kwargs) -> List[TickerPrediction]:
        """Returns ranked list of TickerPrediction (descending by predicted return)."""
        from src.hierarchical_data import HierarchicalDataConfig
        from scripts.predict import generate_predictions

        assert self.forecaster is not None, "lazy_init() must run first"
        assert self.device is not None, "lazy_init() must run first"

        cfg = HierarchicalDataConfig(cache_dir=self.cache_dir)

        # Discover tickers from cache
        cache_daily = Path(cfg.cache_dir) / "daily"
        tickers = sorted(set(
            f.stem.replace("_features", "")
            for f in cache_daily.glob("*_features.npy")
        ))

        if self.max_tickers > 0:
            tickers = tickers[:self.max_tickers]
            logger.info("Limiting to %d tickers (--limit)", len(tickers))

        raw_preds = generate_predictions(self.forecaster, tickers, cfg, self.device)

        results = []
        for p in raw_preds:
            # Collect any extra sub-model preds beyond the core 4
            extra_preds = {
                k: v for k, v in p.items()
                if k.endswith("_pred") and k not in
                   ("lstm_d_pred", "tft_d_pred", "lstm_m_pred", "tft_m_pred")
                and isinstance(v, (int, float))
            }
            results.append(TickerPrediction(
                ticker=p["ticker"],
                predicted_return=p["predicted_return"],
                lstm_d_pred=p["lstm_d_pred"],
                tft_d_pred=p["tft_d_pred"],
                lstm_m_pred=p.get("lstm_m_pred", 0.0),
                tft_m_pred=p.get("tft_m_pred", 0.0),
                extra_preds=extra_preds,
                attention_weights=p.get("attention_weights", {}),
                has_minute_data=p.get("has_minute_data", False),
                rank=p.get("rank", 0),
            ))

        return results
