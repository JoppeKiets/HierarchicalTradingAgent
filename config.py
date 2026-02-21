"""Configuration for trading agent pipeline.

Central hub for all training configurations.

Usage:
  python train_hierarchical.py
  python train_hierarchical.py --config massive
"""

from dataclasses import dataclass
from typing import List, Optional

# ─── ACTIVE CONFIGURATION ──────────────────────────────────────────────────────
# Change this to switch between training modes
ACTIVE_CONFIG = "massive"  # Options: "quick", "standard", "production", "massive"


# ─── CONFIGURATION CLASSES ─────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Base training configuration."""

    # Data parameters
    seq_len: int = 60
    forecast_horizon: int = 5
    period: str = "10y"
    n_train_tickers: int = 80
    use_small_caps: bool = False
    small_cap_count: int = 0

    # Model architecture
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2

    # Training parameters
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500

    # Regularization
    augment_noise: float = 0.02
    augment_prob: float = 0.3
    early_stop_patience: int = 20
    gradient_clip: float = 1.0

    # Evaluation
    val_split: float = 0.2
    feature_selection: bool = True

    # Output
    model_name: str = "tft_model"

    @property
    def total_tickers(self) -> int:
        """Total tickers including small-caps."""
        return self.n_train_tickers + self.small_cap_count


class QuickConfig(TrainingConfig):
    """Fast training for quick iteration/debugging."""
    def __init__(self):
        super().__init__(
            seq_len=60,
            period="5y",
            n_train_tickers=20,
            hidden_dim=128,
            n_heads=4,
            n_layers=2,
            epochs=50,
            batch_size=128,
            learning_rate=3e-4,
            early_stop_patience=15,
            model_name="tft_quick",
        )


class StandardConfig(TrainingConfig):
    """Standard production-ready training (default)."""
    def __init__(self):
        super().__init__(
            seq_len=60,
            period="10y",
            n_train_tickers=90,
            hidden_dim=256,
            n_heads=4,
            n_layers=2,
            epochs=120,
            batch_size=256,
            learning_rate=2e-4,
            early_stop_patience=20,
            model_name="tft_standard",
        )


class ProductionConfig(TrainingConfig):
    """Larger production model with extended data."""
    def __init__(self):
        super().__init__(
            seq_len=120,
            period="12y",
            n_train_tickers=90,
            hidden_dim=384,
            n_heads=6,
            n_layers=3,
            epochs=140,
            batch_size=384,
            learning_rate=2e-4,
            early_stop_patience=25,
            model_name="tft_production",
        )


class MassiveConfig(TrainingConfig):
    """Ultra-aggressive massive model: 10x larger, extensive data."""
    def __init__(self):
        super().__init__(
            seq_len=240,
            period="15y",
            n_train_tickers=90,
            use_small_caps=True,
            small_cap_count=60,
            hidden_dim=512,
            n_heads=8,
            n_layers=4,
            epochs=150,
            batch_size=512,
            learning_rate=2e-4,
            early_stop_patience=25,
            augment_noise=0.025,
            augment_prob=0.35,
            model_name="tft_massive",
        )


# ─── GET ACTIVE CONFIG ──────────────────────────────────────────────────────
def get_config(config_name: Optional[str] = None) -> TrainingConfig:
    """Get configuration by name or use ACTIVE_CONFIG."""
    name = config_name or ACTIVE_CONFIG
    configs = {
        "quick": QuickConfig(),
        "standard": StandardConfig(),
        "production": ProductionConfig(),
        "massive": MassiveConfig(),
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Options: {list(configs.keys())}")
    return configs[name]
