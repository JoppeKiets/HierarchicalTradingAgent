#!/usr/bin/env python3
"""News Encoder sub-model for the hierarchical forecaster.

Architecture:
    Per day: Pre-computed FinBERT embeddings (768-dim per article, mean-pooled per day)
    → Linear projection (768 → hidden_dim)
    → Temporal LSTM over daily news embedding sequence
    → Regression head → scalar prediction + 64-dim embedding

This is the 5th sub-model in the hierarchy, alongside LSTM_D, TFT_D, LSTM_M, TFT_M.
It contributes its own prediction and embedding to the MetaMLP.

The model operates on pre-computed embeddings (from scripts/preprocess_news_embeddings.py)
so there is no FinBERT inference during training — only lightweight LSTM computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsEncoderConfig:
    """Configuration for the News Encoder sub-model."""
    # Input
    finbert_dim: int = 768           # FinBERT [CLS] embedding dimension
    sentiment_dim: int = 6           # Sentiment features per day
    input_dim: int = 774             # finbert_dim + sentiment_dim

    # Projection
    proj_dim: int = 128              # Project 774 → 128 before LSTM

    # Temporal model
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.15

    # Sequence
    seq_len: int = 720               # Same as daily seq_len (match daily models)

    # Output
    embedding_dim: int = 64          # Must match other sub-models


class NewsEncoder(nn.Module):
    """LSTM-based news encoder operating on pre-computed FinBERT embeddings.

    Input: (batch, seq_len, input_dim)  where input_dim = 774
        - First 768 dims: FinBERT [CLS] mean embedding per day
        - Last 6 dims: [pos_mean, neg_mean, neu_mean, compound_mean, compound_std, news_count]
        - Days with no news: all zeros (handled by the has_news mask features)

    The model adds two learned features:
        - has_news: binary indicator (1 if any news that day, 0 otherwise)
        - news_recency: days since last news article (normalized)

    Output: dict with 'prediction' (batch,) and 'embedding' (batch, 64)
    """

    def __init__(self, cfg: NewsEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Project FinBERT embedding + sentiment → smaller dim
        # +2 for has_news and news_recency features
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim + 2, cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.input_norm = nn.LayerNorm(cfg.proj_dim)

        self.lstm = nn.LSTM(
            input_size=cfg.proj_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0,
            batch_first=True,
        )

        # Embedding head (→ meta model input)
        self.embedding_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
            nn.Tanh(),
        )

        # Regression head (→ scalar prediction)
        self.regression_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2)
        self.regression_act = nn.GELU()
        self.regression_drop = nn.Dropout(cfg.dropout)
        self.regression_out = nn.Linear(cfg.hidden_dim // 2, 1)
        self.regression_skip = nn.Linear(cfg.hidden_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(p, gain=0.5)
            elif "lstm" in name and "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)  # forget gate bias
            elif "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        self.lstm.flatten_parameters()

    def forward(self, x: torch.Tensor, news_coverage: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
                where input_dim = 774 (768 FinBERT + 6 sentiment)
                Days with no news are all zeros.

        Returns:
            dict with 'prediction' (batch,) and 'embedding' (batch, E)
        """
        # Compute has_news indicator and news_recency
        # has_news = 1 if any feature is non-zero for that day
        has_news = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).float()  # (B, T, 1)

        # news_recency: how many days since last news (normalized by seq_len)
        # For each position, count backward to last non-zero day
        recency = self._compute_recency(has_news)  # (B, T, 1)

        # Concatenate extra features
        x_augmented = torch.cat([x, has_news, recency], dim=-1)  # (B, T, input_dim+2)

        # Project down
        x_proj = self.input_proj(x_augmented)  # (B, T, proj_dim)
        x_proj = self.input_norm(x_proj)

        # Temporal encoding
        lstm_out, _ = self.lstm(x_proj)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Outputs
        embedding = self.embedding_head(last_hidden)

        # Optionally gate the embedding by a coverage/confidence scalar so the
        # meta-MLP can learn to down-weight zero-filled sequences.
        if news_coverage is not None:
            # news_coverage expected shape: (B,) or (B,1) with values in [0,1]
            cov = news_coverage.squeeze(-1) if news_coverage.dim() > 1 else news_coverage
            coverage_gate = cov.unsqueeze(-1).to(embedding.device)
            embedding = embedding * coverage_gate

        h = self.regression_drop(self.regression_act(self.regression_proj(last_hidden)))
        prediction = (self.regression_out(h) + self.regression_skip(last_hidden)).squeeze(-1)

        # Scale prediction toward zero when coverage is low so zero-filled
        # sequences contribute minimally to the loss.
        if news_coverage is not None:
            prediction = prediction * cov.to(prediction.device)

        return {"prediction": prediction, "embedding": embedding}

    @staticmethod
    def _compute_recency(has_news: torch.Tensor) -> torch.Tensor:
        """Compute normalized recency (days since last news).

        Args:
            has_news: (batch, seq_len, 1) binary indicator

        Returns:
            (batch, seq_len, 1) normalized recency in [0, 1]
        """
        B, T, _ = has_news.shape
        recency = torch.zeros_like(has_news)
        seq_len = float(T)

        for t in range(T):
            if t == 0:
                recency[:, t, 0] = (1.0 - has_news[:, t, 0]) * (1.0 / seq_len)
            else:
                # If has news today, recency = 0; else increment from yesterday
                recency[:, t, 0] = torch.where(
                    has_news[:, t, 0] > 0.5,
                    torch.zeros_like(recency[:, t, 0]),
                    recency[:, t - 1, 0] + 1.0 / seq_len,
                )
        return recency.clamp(0, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = NewsEncoderConfig()
    model = NewsEncoder(cfg)
    print(f"NewsEncoder parameters: {model.count_parameters():,}")

    B = 4
    x = torch.randn(B, cfg.seq_len, cfg.input_dim)
    # Simulate some days with no news (zeros)
    x[:, ::3, :] = 0.0

    out = model(x)
    print(f"Prediction shape: {out['prediction'].shape}")
    print(f"Embedding shape:  {out['embedding'].shape}")
    print("✅ NewsEncoder test passed!")
