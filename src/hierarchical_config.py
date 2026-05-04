"""Configuration classes for hierarchical forecaster training."""

import logging
import os

logger = logging.getLogger(__name__)


def _check_shm_available() -> bool:
    """Return True if /dev/shm is available and writable (needed for num_workers>0)."""
    import tempfile
    shm_path = "/dev/shm"
    if not os.path.isdir(shm_path):
        return False
    try:
        with tempfile.TemporaryFile(dir=shm_path):
            pass
        return True
    except (OSError, IOError):
        return False


class TrainConfig:
    """Training hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 1e-5
    lr_meta: float = 5e-4
    lr_finetune: float = 1e-5

    # Per-model LR overrides (v10)
    lr_tft: float = 1e-4            # TFT-specific LR (defaults to 1e-4)
    lr_lstm: float = 2e-4           # reduced LR for LSTM (LSTM_D)
    lr_minute: float = 0.0          # 0 = use default lr; set to e.g. 1e-4 for minute models
    weight_decay_minute: float = 0.0  # 0 = use default; higher for minute (regularize)

    # Warmup (v10) — linear warmup before cosine decay
    warmup_steps: int = 500         # linear warmup steps (useful for TFT attention/VSN stability)

    # Per-model grad clip (v10)
    grad_clip_tft: float = 0.0      # 0 = use default grad_clip; e.g. 0.5

    scheduler: str = "cosine"

    epochs_phase1: int = 30       # early stopping (patience=20) will trigger before 30
    epochs_news: int = 30           # Phase 1.5: news encoder
    epochs_phase2: int = 30
    epochs_phase3: int = 30
    epochs_phase4: int = 10

    patience: int = 15
    min_delta: float = 1e-6
    # early_stop_metric: "ic" tracks val IC (better for regression),
    # "loss" tracks val loss (original behaviour)
    early_stop_metric: str = "ic"
    ic_min_delta: float = 1e-4  # min IC improvement to reset patience

    batch_size_daily: int = 32    # base daily batch
    batch_size_minute: int = 32   # base minute batch
    batch_size_meta: int = 64
    batch_size_news: int = 64     # News model batch size

    # Per-model batch size overrides (0 = use default above)
    batch_size_lstm_d: int = 0
    batch_size_tft_d: int = 8     # TFT is memory-heavy, reduce only TFT
    batch_size_lstm_m: int = 0
    batch_size_tft_m: int = 8     # TFT is memory-heavy, reduce only TFT

    loss_fn: str = "huber"
    rank_loss_weight: float = 0.5   # Weight of pairwise ranking loss (0 = disabled, 0.5 = 50/50 blend)

    # Per-model Huber delta overrides (allows widening quadratic region)
    huber_delta_lstm_d: float = 0.05
    huber_delta_tft_d: float = 0.05

    # Per-model weight decay overrides (increase regularization for sensitive models)
    weight_decay_lstm: float = 1e-4
    weight_decay_tft: float = 1e-4

    grad_clip: float = 1.0
    num_workers: int = 0  # OPTIMIZED: set to 0 to save memory (was 4); data loading overhead can exceed benefits
    
    # Memory optimization
    use_gradient_checkpointing: bool = True  # OPTIMIZED: enabled by default
    use_amp: bool = True  # Automatic Mixed Precision — leverage Tensor Cores on RTX 5080
    log_interval: int = 100
    eval_interval: int = 1

    output_dir: str = "models/hierarchical"
    log_dir: str = "logs/hierarchical"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Auto-fallback: multi-process data loading requires /dev/shm shared
        # memory. If it is unavailable (e.g. some container environments),
        # silently drop to single-process loading to avoid RuntimeError.
        if self.num_workers > 0 and not _check_shm_available():
            logger.warning(
                f"num_workers={self.num_workers} requested but /dev/shm is not "
                "available — falling back to num_workers=0 (single-process loading)."
            )
            self.num_workers = 0

    def reduce_memory(self):
        """Reduce memory footprint for low-memory systems."""
        self.batch_size_daily = max(8, self.batch_size_daily // 2)
        self.batch_size_minute = max(4, self.batch_size_minute // 2)
        self.batch_size_meta = max(16, self.batch_size_meta // 2)
        self.batch_size_tft_d = max(4, self.batch_size_tft_d // 2)
        self.batch_size_tft_m = max(4, self.batch_size_tft_m // 2)
        self.num_workers = 0
        self.use_amp = True
        self.use_gradient_checkpointing = True
        logger.warning(f"Memory optimization enabled: daily={self.batch_size_daily}, "
                      f"minute={self.batch_size_minute}, meta={self.batch_size_meta}, "
                      f"tft_d={self.batch_size_tft_d}, tft_m={self.batch_size_tft_m}")
        return self
