"""Feature modules for the trading agent.

Each module exposes a ``compute_*`` function that takes raw data and returns
a pd.DataFrame of features aligned to the price index.
"""

from src.features.fundamental_features import compute_fundamental_features
from src.features.macro_features import compute_macro_features
from src.features.sentiment_features import compute_sentiment_features

__all__ = [
    "compute_fundamental_features",
    "compute_macro_features",
    "compute_sentiment_features",
]
