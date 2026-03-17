#!/usr/bin/env python3
"""Regression-adapted models for hierarchical forecasting.

Contains:
  - RegressionLSTM: LSTM with regression head + embedding output
  - RegressionTFT:  Thin wrapper that creates TFT in regression mode (n_classes=0)
  - MetaMLP:        Small neural network combining 4 sub-model predictions + regime
  - HierarchicalForecaster: Orchestrates all 5 models

All sub-models output:
    prediction:  (batch,)  scalar predicted return
    embedding:   (batch, 64)  latent representation passed to meta model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from src.models.temporal_fusion_transformer import TFTConfig, TemporalFusionTransformer
from src.news_encoder import NewsEncoder, NewsEncoderConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HierarchicalModelConfig:
    """Configuration for the 5-model hierarchical forecaster."""

    # Daily models
    daily_input_dim: int = 51       # Features from enhanced_features.py
    daily_seq_len: int = 720        # ~3 years
    daily_hidden_dim: int = 128
    daily_n_layers: int = 2
    daily_n_heads: int = 4          # For TFT
    daily_dropout: float = 0.15

    # Minute models
    minute_input_dim: int = 18      # Features from minute data
    minute_seq_len: int = 780       # ~2 trading days
    minute_hidden_dim: int = 128
    minute_n_layers: int = 2
    minute_n_heads: int = 4
    minute_dropout: float = 0.15

    # News model
    news_input_dim: int = 774       # 768 FinBERT + 6 sentiment features
    news_seq_len: int = 720         # Same as daily
    news_hidden_dim: int = 128
    news_n_layers: int = 2
    news_dropout: float = 0.15
    use_news_model: bool = True     # Toggle news sub-model

    # Shared
    embedding_dim: int = 64         # Embedding size for all sub-models

    # Meta MLP
    regime_dim: int = 8             # vol_regime, vix_regime, etc.
    meta_hidden_dim: int = 128
    meta_n_layers: int = 3
    meta_dropout: float = 0.2
    n_sub_models: int = 5           # 4 price models + 1 news model (or 4 if no news)


# ============================================================================
# Regression LSTM
# ============================================================================

class RegressionLSTM(nn.Module):
    """LSTM adapted for return regression.

    Outputs:
        prediction:  (batch,)    predicted return
        embedding:   (batch, E)  latent representation for meta model
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Embedding head  (→ meta model input)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )

        # Regression head (→ scalar prediction)
        # Single linear avoids dead-neuron collapse; residual keeps gradient flow.
        self.regression_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.regression_act  = nn.GELU()
        self.regression_drop = nn.Dropout(dropout)
        self.regression_out  = nn.Linear(hidden_dim // 2, 1)
        # Residual shortcut: project hidden_dim → 1 directly
        self.regression_skip = nn.Linear(hidden_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(p)       # Orthogonal init for recurrent weights
            elif "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(p, gain=0.5)  # Smaller gain for input weights
            elif "lstm" in name and "bias" in name:
                nn.init.zeros_(p)
                # Set forget gate bias to 1.0 for long sequences
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)
            elif "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        self.lstm.flatten_parameters()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            dict with 'prediction' (batch,) and 'embedding' (batch, E)
        """
        # LayerNorm across feature dim
        x = self.input_norm(x)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden_dim)

        embedding = self.embedding_head(last_hidden)

        # Residual regression head: skip connection prevents full collapse
        h = self.regression_drop(self.regression_act(self.regression_proj(last_hidden)))
        prediction = (self.regression_out(h) + self.regression_skip(last_hidden)).squeeze(-1)

        return {"prediction": prediction, "embedding": embedding}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Regression TFT (wrapper)
# ============================================================================

class RegressionTFT(nn.Module):
    """Thin wrapper around TemporalFusionTransformer in regression mode.

    Sets n_classes=0 so the action_head outputs a scalar.
    Exposes the same interface as RegressionLSTM for uniformity.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.config = TFTConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            n_classes=0,               # Regression mode
            embedding_dim=embedding_dim,
            seq_len=seq_len,
            dropout=dropout,
        )
        self.tft = TemporalFusionTransformer(self.config)

    def forward(
        self,
        x: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            dict with 'prediction' (batch,), 'embedding' (batch, E),
            'feature_weights', 'attention_weights'
        """
        out = self.tft(x, static_context)
        return {
            "prediction": out["action_logits"].squeeze(-1),   # (batch,)
            "embedding": out["embedding"],                     # (batch, E)
            "feature_weights": out["feature_weights"],
            "attention_weights": out["attention_weights"],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Meta MLP  (combines 4 sub-model outputs + regime)
# ============================================================================

class MetaMLP(nn.Module):
    """Small neural network that fuses sub-model predictions into
    a single next-day return forecast.

    Input vector (per sample):
      - N scalar predictions     (LSTM_D, TFT_D, LSTM_M, TFT_M, [News])
      - N × embedding_dim        (64 × N)
      - regime_dim               (8)
    Total default (5 models) = 5 + 1 + 320 + 8 = 334
    Total default (4 models) = 4 + 1 + 256 + 8 = 269

    The network learns:
      - How to weight daily vs minute vs news predictions
      - When to trust which model (regime-dependent)
      - Non-linear interactions between model signals
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        regime_dim: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
        n_sub_models: int = 5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.regime_dim = regime_dim
        self.n_sub_models = n_sub_models

        # N raw preds + 1 weighted pred + N embs + regime
        input_dim = n_sub_models + 1 + n_sub_models * embedding_dim + regime_dim

        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(prev_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = out_dim

        self.trunk = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev_dim, 1)

        # Interpretable attention: learns D-vs-M-vs-News weighting from regime
        self.attention = nn.Sequential(
            nn.Linear(regime_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_sub_models),  # weights for N sub-models
            nn.Softmax(dim=-1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        predictions: torch.Tensor,          # (batch, 4)
        embeddings: List[torch.Tensor],      # list of 4 × (batch, E)
        regime: torch.Tensor,                # (batch, regime_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with 'prediction' (batch,) and 'attention_weights' (batch, 4)
        """
        emb_cat = torch.cat(embeddings, dim=-1)      # (batch, 4*E)

        # Regime-conditioned attention: weight the 4 sub-model predictions
        attn_weights = self.attention(regime)          # (batch, 4)
        weighted_preds = (predictions * attn_weights).sum(dim=-1, keepdim=True)  # (batch, 1)

        # Feed everything (including the weighted prediction) into the trunk
        combined = torch.cat([predictions, weighted_preds, emb_cat, regime], dim=-1)

        hidden = self.trunk(combined)
        prediction = self.output_head(hidden).squeeze(-1)

        return {
            "prediction": prediction,
            "attention_weights": attn_weights,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# HierarchicalForecaster: orchestrates all 5 models
# ============================================================================

class HierarchicalForecaster(nn.Module):
    """Complete hierarchical forecaster with up to 6 models.

    Models:
        1. lstm_d   — RegressionLSTM on daily data
        2. tft_d    — RegressionTFT  on daily data
        3. lstm_m   — RegressionLSTM on minute data
        4. tft_m    — RegressionTFT  on minute data
        5. news     — NewsEncoder on FinBERT embeddings (optional)
        6. meta     — MetaMLP combining all sub-models

    Training phases:
        Phase 1:   Train lstm_d and tft_d on daily data
        Phase 1.5: Train news encoder on news embeddings (if enabled)
        Phase 2:   Train lstm_m and tft_m on minute data
        Phase 3:   Freeze sub-models, train meta on combined predictions
        Phase 4:   (Optional) Fine-tune all jointly with small LR
    """

    def __init__(self, cfg: HierarchicalModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- Daily models ---
        self.lstm_d = RegressionLSTM(
            input_dim=cfg.daily_input_dim,
            hidden_dim=cfg.daily_hidden_dim,
            num_layers=cfg.daily_n_layers,
            dropout=cfg.daily_dropout,
            embedding_dim=cfg.embedding_dim,
        )
        self.tft_d = RegressionTFT(
            input_dim=cfg.daily_input_dim,
            seq_len=cfg.daily_seq_len,
            hidden_dim=cfg.daily_hidden_dim,
            n_heads=cfg.daily_n_heads,
            n_layers=cfg.daily_n_layers,
            dropout=cfg.daily_dropout,
            embedding_dim=cfg.embedding_dim,
        )

        # --- Minute models ---
        self.lstm_m = RegressionLSTM(
            input_dim=cfg.minute_input_dim,
            hidden_dim=cfg.minute_hidden_dim,
            num_layers=cfg.minute_n_layers,
            dropout=cfg.minute_dropout,
            embedding_dim=cfg.embedding_dim,
        )
        self.tft_m = RegressionTFT(
            input_dim=cfg.minute_input_dim,
            seq_len=cfg.minute_seq_len,
            hidden_dim=cfg.minute_hidden_dim,
            n_heads=cfg.minute_n_heads,
            n_layers=cfg.minute_n_layers,
            dropout=cfg.minute_dropout,
            embedding_dim=cfg.embedding_dim,
        )

        # --- News model (optional) ---
        self.use_news = cfg.use_news_model
        if self.use_news:
            news_cfg = NewsEncoderConfig(
                finbert_dim=768,
                sentiment_dim=6,
                input_dim=cfg.news_input_dim,
                proj_dim=cfg.news_hidden_dim,
                hidden_dim=cfg.news_hidden_dim,
                n_layers=cfg.news_n_layers,
                dropout=cfg.news_dropout,
                seq_len=cfg.news_seq_len,
                embedding_dim=cfg.embedding_dim,
            )
            self.news = NewsEncoder(news_cfg)
            n_sub = 5
        else:
            self.news = None
            n_sub = 4

        # --- Meta model ---
        self.meta = MetaMLP(
            embedding_dim=cfg.embedding_dim,
            regime_dim=cfg.regime_dim,
            hidden_dim=cfg.meta_hidden_dim,
            n_layers=cfg.meta_n_layers,
            dropout=cfg.meta_dropout,
            n_sub_models=n_sub,
        )

        self._log_params()

    def _log_params(self):
        parts = {
            "lstm_d": self.lstm_d.count_parameters(),
            "tft_d": self.tft_d.count_parameters(),
            "lstm_m": self.lstm_m.count_parameters(),
            "tft_m": self.tft_m.count_parameters(),
            "meta": self.meta.count_parameters(),
        }
        if self.use_news and self.news is not None:
            parts["news"] = self.news.count_parameters()
        total = sum(parts.values())
        logger.info("Hierarchical Forecaster parameter count:")
        for name, count in parts.items():
            logger.info(f"  {name:8s}: {count:>10,}")
        logger.info(f"  {'TOTAL':8s}: {total:>10,}")

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------
    def freeze_sub_models(self):
        """Freeze all sub-models (for meta training)."""
        models = [self.lstm_d, self.tft_d, self.lstm_m, self.tft_m]
        if self.use_news and self.news is not None:
            models.append(self.news)
        for m in models:
            for p in m.parameters():
                p.requires_grad = False
        logger.info(f"Sub-models frozen for meta training ({len(models)} models)")

    def unfreeze_all(self):
        """Unfreeze everything (for joint fine-tuning)."""
        for p in self.parameters():
            p.requires_grad = True
        logger.info("All models unfrozen for joint fine-tuning")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        daily_x: torch.Tensor,             # (batch, daily_seq, daily_feat)
        minute_x: torch.Tensor,            # (batch, minute_seq, minute_feat)
        regime: torch.Tensor,              # (batch, regime_dim)
        news_x: Optional[torch.Tensor] = None,  # (batch, news_seq, news_feat) or None
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through all models.

        Returns dict with:
            prediction:         (batch,) — meta model output
            sub_predictions:    (batch, N) — [lstm_d, tft_d, lstm_m, tft_m, (news)]
            attention_weights:  (batch, N) — learned sub-model weights
        """
        # Sub-model forward passes
        d1 = self.lstm_d(daily_x)
        d2 = self.tft_d(daily_x)
        m1 = self.lstm_m(minute_x)
        m2 = self.tft_m(minute_x)

        pred_list = [d1["prediction"], d2["prediction"],
                     m1["prediction"], m2["prediction"]]
        emb_list = [d1["embedding"], d2["embedding"],
                    m1["embedding"], m2["embedding"]]

        result = {
            "lstm_d": d1,
            "tft_d": d2,
            "lstm_m": m1,
            "tft_m": m2,
        }

        # Optional news model
        if self.use_news and self.news is not None and news_x is not None:
            n1 = self.news(news_x)
            pred_list.append(n1["prediction"])
            emb_list.append(n1["embedding"])
            result["news"] = n1

        preds = torch.stack(pred_list, dim=-1)  # (batch, N)

        meta_out = self.meta(preds, emb_list, regime)

        result.update({
            "prediction": meta_out["prediction"],
            "sub_predictions": preds,
            "attention_weights": meta_out["attention_weights"],
        })

        return result

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        save_dict = {
            "cfg": self.cfg,
            "lstm_d": self.lstm_d.state_dict(),
            "tft_d": self.tft_d.state_dict(),
            "lstm_m": self.lstm_m.state_dict(),
            "tft_m": self.tft_m.state_dict(),
            "meta": self.meta.state_dict(),
        }
        if self.use_news and self.news is not None:
            save_dict["news"] = self.news.state_dict()
        torch.save(save_dict, path)
        logger.info(f"HierarchicalForecaster saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HierarchicalForecaster":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["cfg"])
        model.lstm_d.load_state_dict(ckpt["lstm_d"])
        model.tft_d.load_state_dict(ckpt["tft_d"])
        model.lstm_m.load_state_dict(ckpt["lstm_m"])
        model.tft_m.load_state_dict(ckpt["tft_m"])
        model.meta.load_state_dict(ckpt["meta"])
        if "news" in ckpt and model.use_news and model.news is not None:
            model.news.load_state_dict(ckpt["news"])
        model.to(device)
        logger.info(f"HierarchicalForecaster loaded ← {path}")
        return model


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing HierarchicalForecaster...\n")

    # Test with news model enabled
    cfg = HierarchicalModelConfig(use_news_model=True)
    model = HierarchicalForecaster(cfg)

    B = 4
    daily_x = torch.randn(B, cfg.daily_seq_len, cfg.daily_input_dim)
    minute_x = torch.randn(B, cfg.minute_seq_len, cfg.minute_input_dim)
    regime = torch.randn(B, cfg.regime_dim)
    news_x = torch.randn(B, cfg.news_seq_len, cfg.news_input_dim)

    out = model(daily_x, minute_x, regime, news_x=news_x)

    print(f"Meta prediction:   {out['prediction'].shape}")
    print(f"Sub predictions:   {out['sub_predictions'].shape}")
    print(f"Attention weights: {out['attention_weights']}")
    print(f"LSTM_D embedding:  {out['lstm_d']['embedding'].shape}")
    print(f"TFT_D embedding:   {out['tft_d']['embedding'].shape}")
    print(f"News embedding:    {out['news']['embedding'].shape}")

    # Test save/load roundtrip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model.save(f.name)
        loaded = HierarchicalForecaster.load(f.name)
        out2 = loaded(daily_x, minute_x, regime, news_x=news_x)
        diff = (out["prediction"] - out2["prediction"]).abs().max().item()
        print(f"\nSave/load roundtrip max diff: {diff:.2e}")
        os.unlink(f.name)

    # Test without news model
    cfg2 = HierarchicalModelConfig(use_news_model=False)
    model2 = HierarchicalForecaster(cfg2)
    out3 = model2(daily_x, minute_x, regime)
    print(f"\nNo-news sub predictions: {out3['sub_predictions'].shape}")
    print(f"No-news attention weights: {out3['attention_weights'].shape}")

    print("\n✅ HierarchicalForecaster test passed!")
