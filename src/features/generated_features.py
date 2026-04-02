"""Auto-generated feature formulas promoted by Analyst ablations.

This file is managed by `agents.feedback.auto_feature_engineer`.
If no features are promoted yet, `GENERATED_FEATURE_NAMES` stays empty.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

GENERATED_FEATURE_NAMES: List[str] = []


def compute_generated_features(base_features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=base_features.index)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype("float32")
