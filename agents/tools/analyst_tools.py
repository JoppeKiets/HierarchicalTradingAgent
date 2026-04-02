"""Tools that read technical features and news sentiment.

Reuses:
  - Memory-mapped .npy feature caches from data/feature_cache/
  - data/feature_cache/metadata.json for feature names
  - News articles from data/organized/{TICKER}/news_articles.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base import BaseTool


class ReadTechnicalFeaturesTool(BaseTool):
    """Load the most recent cached daily features for a ticker.

    Reads the memory-mapped .npy files the pipeline already produces.
    Returns a dict of feature_name -> latest_value.
    """

    name = "read_technical_features"
    description = "Read latest daily technical features for a ticker from the feature cache."

    def __init__(self, cache_dir: str = "data/feature_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self._feature_names: Optional[List[str]] = None

    def lazy_init(self):
        meta_path = Path(self.cache_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            # Get feature names from the first ticker in the daily section
            daily = meta.get("daily", {})
            if daily:
                first_ticker = next(iter(daily))
                self._feature_names = daily[first_ticker].get("feature_names", [])
            else:
                self._feature_names = []
        else:
            self._feature_names = []
        self._initialized = True

    def execute(self, ticker: str, **kwargs) -> Dict[str, float]:
        feat_path = Path(self.cache_dir) / "daily" / f"{ticker}_features.npy"
        if not feat_path.exists():
            return {}

        # Memory-mapped read — near-zero RAM cost
        features = np.load(str(feat_path), mmap_mode="r")
        latest = features[-1]  # Last row = most recent day

        result = {}
        feat_names = self._feature_names or []
        for i, val in enumerate(latest):
            name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
            result[name] = float(val) if np.isfinite(val) else 0.0

        return result


class ReadNewsSentimentTool(BaseTool):
    """Aggregate recent news sentiment for a ticker.

    Reads from data/organized/{TICKER}/news_articles.csv if it exists.
    """

    name = "read_news_sentiment"
    description = "Read aggregated news sentiment for a ticker."

    def __init__(self, organized_dir: str = "data/organized", **kwargs):
        super().__init__(**kwargs)
        self.organized_dir = organized_dir

    def lazy_init(self):
        self._initialized = True

    def execute(self, ticker: str, lookback_days: int = 7, **kwargs) -> Dict[str, Any]:
        import pandas as pd

        news_path = Path(self.organized_dir) / ticker / "news_articles.csv"
        if not news_path.exists():
            return {"available": False, "sentiment_score": 0.0, "n_articles": 0}

        try:
            df = pd.read_csv(news_path, parse_dates=["Date"])
        except Exception:
            return {"available": False, "sentiment_score": 0.0, "n_articles": 0}

        if df.empty:
            return {"available": False, "sentiment_score": 0.0, "n_articles": 0}

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        # Handle timezone-naive dates
        if df["Date"].dt.tz is None:  # type: ignore[union-attr]
            cutoff = cutoff.tz_localize(None)
        recent = df[df["Date"] >= cutoff]

        sentiment_score = 0.0
        if "sentiment_compound" in recent.columns and len(recent) > 0:
            sentiment_score = float(recent["sentiment_compound"].mean())
        elif "sentiment_positive" in recent.columns and len(recent) > 0:
            pos = recent["sentiment_positive"].mean()
            neg = recent.get("sentiment_negative", recent["sentiment_positive"] * 0).mean()
            sentiment_score = float(pos - neg)

        return {
            "available": len(recent) > 0,
            "sentiment_score": sentiment_score,
            "n_articles": len(recent),
            "total_articles": len(df),
        }


class ComputeSubModelAgreementTool(BaseTool):
    """Measure how much the sub-models agree on direction and magnitude.

    Accepts any number of sub-model predictions via keyword arguments
    whose names end in ``_pred`` (e.g. lstm_d_pred=…, tft_d_pred=…,
    tcn_d_pred=…, gnn_pred=…, fund_mlp_pred=…).
    Legacy positional style (lstm_d=…, tft_d=…, lstm_m=…, tft_m=…)
    is still accepted for backwards compatibility.
    """

    name = "compute_sub_model_agreement"
    description = "Compute agreement score across all active sub-model predictions."

    def lazy_init(self):
        self._initialized = True

    def execute(
        self,
        lstm_d: float = 0.0, tft_d: float = 0.0,
        lstm_m: float = 0.0, tft_m: float = 0.0,
        **kwargs,
    ) -> Dict[str, float]:
        eps = 1e-6
        # Collect all *_pred kwargs (new-style) plus legacy positional args
        named_preds: Dict[str, float] = {}
        for k, v in kwargs.items():
            if k.endswith("_pred") and isinstance(v, (int, float)):
                named_preds[k] = float(v)
        # Legacy: add the 4 positional args only if no *_pred kwargs found
        if not named_preds:
            named_preds = {
                "lstm_d": lstm_d, "tft_d": tft_d,
                "lstm_m": lstm_m, "tft_m": tft_m,
            }
        # Filter out near-zero placeholders if we have real data
        nonzero = {k: v for k, v in named_preds.items() if abs(v) > eps}
        active = nonzero if len(nonzero) >= 2 else named_preds
        preds = np.array(list(active.values()))
        signs = np.sign(preds)
        nonzero_signs = signs[np.abs(preds) > eps]

        # Direction agreement: fraction of non-zero models agreeing with majority side
        if len(nonzero_signs) == 0:
            direction_agreement = 0.5
        else:
            pos = int(np.sum(nonzero_signs > 0))
            neg = int(np.sum(nonzero_signs < 0))
            majority = max(pos, neg)
            direction_agreement = float(majority / max(len(nonzero_signs), 1))

        # Magnitude agreement: 1 - normalized spread
        pred_range = np.ptp(preds)  # max - min
        mean_abs = np.mean(np.abs(preds))
        magnitude_agreement = 1.0 - (pred_range / (mean_abs + 1e-8))
        magnitude_agreement = float(np.clip(magnitude_agreement, 0.0, 1.0))

        # Overall agreement: direction is more important than magnitude spread
        overall_agreement = 0.65 * direction_agreement + 0.35 * magnitude_agreement
        sign_balance = 0.0
        if len(nonzero_signs) > 0:
            sign_balance = float(abs(np.sum(nonzero_signs)) / len(nonzero_signs))

        return {
            "direction_agreement": direction_agreement,
            "magnitude_agreement": magnitude_agreement,
            "overall_agreement": float(np.clip(overall_agreement, 0.0, 1.0)),
            "mean_prediction": float(np.mean(preds)),
            "std_prediction": float(np.std(preds)),
            "n_active_models": float(len(active)),
            "n_nonzero_models": float(len(nonzero)),
            "sign_balance": sign_balance,
        }


class AnalyzeFeatureImportanceTool(BaseTool):
    """Flag features as suspicious or supportive based on statistical criteria.

    Suspicious features:
      - Extreme z-score (|z| > 3) relative to a rolling window
      - High NaN / zero rate (> 50 % of recent window)
      - Contradicts the predicted return direction (negative correlation
        between feature trend and prediction sign)

    Supportive features:
      - Moderate z-score aligned with prediction direction
      - Low NaN rate and stable variance
      - Trend direction agrees with prediction

    The Analyst calls this per-ticker.  Results are aggregated by the
    FeatureTrustTracker across runs/weeks to build long-term trust scores.
    """

    name = "analyze_feature_importance"
    description = (
        "Analyze daily feature vectors for a ticker and flag "
        "suspicious/supportive features."
    )

    def __init__(self, cache_dir: str = "data/feature_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self._feature_names: Optional[List[str]] = None

    def lazy_init(self):
        meta_path = Path(self.cache_dir) / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            daily = meta.get("daily", {})
            if daily:
                first_ticker = next(iter(daily))
                self._feature_names = daily[first_ticker].get("feature_names", [])
            else:
                self._feature_names = []
        else:
            self._feature_names = []
        self._initialized = True

    def execute(
        self,
        ticker: str,
        predicted_return: float = 0.0,
        lookback: int = 60,
        zscore_threshold: float = 3.0,
        nan_threshold: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze feature health and alignment with the prediction.

        Args:
            ticker:            Ticker symbol
            predicted_return:  The ensemble's predicted return (used to check alignment)
            lookback:          Number of recent days to compute statistics over
            zscore_threshold:  |z| above this → suspicious
            nan_threshold:     NaN/zero fraction above this → suspicious

        Returns:
            dict with 'suspicious', 'supportive', 'all_features',
            and per-feature 'details'.
        """
        feat_path = Path(self.cache_dir) / "daily" / f"{ticker}_features.npy"
        if not feat_path.exists():
            return {
                "suspicious": [],
                "supportive": [],
                "all_features": [],
                "details": {},
            }

        features = np.load(str(feat_path), mmap_mode="r")
        n_rows, n_cols = features.shape
        window = features[-lookback:] if n_rows >= lookback else features

        feat_names = self._feature_names or []
        suspicious: List[str] = []
        supportive: List[str] = []
        all_feature_names: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        pred_sign = np.sign(predicted_return) if predicted_return != 0 else 0.0

        for col_idx in range(n_cols):
            name = feat_names[col_idx] if col_idx < len(feat_names) else f"feat_{col_idx}"
            all_feature_names.append(name)

            col = window[:, col_idx].astype(np.float64)

            # --- NaN / zero rate ---
            finite_mask = np.isfinite(col)
            nan_rate = 1.0 - float(finite_mask.mean())
            zero_rate = float((col[finite_mask] == 0).mean()) if finite_mask.any() else 1.0
            bad_rate = nan_rate + zero_rate * 0.5  # zeros are half as bad as NaNs

            # --- Z-score of latest value ---
            clean = col[finite_mask]
            if len(clean) >= 5:
                mu = float(clean.mean())
                sigma = float(clean.std())
                latest = float(col[-1]) if np.isfinite(col[-1]) else mu
                zscore = (latest - mu) / (sigma + 1e-10)
            else:
                zscore = 0.0
                mu = 0.0
                sigma = 0.0
                latest = 0.0

            # --- Trend direction (simple 5-day slope sign) ---
            if len(clean) >= 10:
                recent_5 = clean[-5:]
                older_5 = clean[-10:-5]
                trend_sign = float(np.sign(recent_5.mean() - older_5.mean()))
            else:
                trend_sign = 0.0

            # --- Classification ---
            is_suspicious = False
            is_supportive = False
            reasons: List[str] = []

            # Extreme z-score
            if abs(zscore) > zscore_threshold:
                is_suspicious = True
                reasons.append(f"extreme_zscore({zscore:.1f})")

            # High NaN/zero rate
            if bad_rate > nan_threshold:
                is_suspicious = True
                reasons.append(f"high_bad_rate({bad_rate:.2f})")

            # Trend contradicts prediction
            if pred_sign != 0 and trend_sign != 0:
                if trend_sign != pred_sign and abs(zscore) > 1.0:
                    is_suspicious = True
                    reasons.append("trend_contradicts_prediction")
                elif not is_suspicious and trend_sign == pred_sign and abs(zscore) > 0.5:
                    is_supportive = True
                    reasons.append("trend_supports_prediction")

            # Low variance + aligned → supportive
            if not is_suspicious and bad_rate < 0.1 and sigma > 1e-8:
                cv = sigma / (abs(mu) + 1e-8)
                if cv < 1.0 and abs(zscore) < 1.5:
                    is_supportive = True
                    reasons.append("stable_feature")

            if is_suspicious:
                suspicious.append(name)
            elif is_supportive:
                supportive.append(name)

            details[name] = {
                "zscore": round(zscore, 3),
                "nan_rate": round(nan_rate, 3),
                "bad_rate": round(bad_rate, 3),
                "trend_sign": trend_sign,
                "latest_value": round(latest, 6),
                "mean": round(mu, 6),
                "std": round(sigma, 6),
                "suspicious": is_suspicious,
                "supportive": is_supportive,
                "reasons": reasons,
            }

        return {
            "suspicious": suspicious,
            "supportive": supportive,
            "all_features": all_feature_names,
            "details": details,
        }
