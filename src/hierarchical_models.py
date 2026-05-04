#!/usr/bin/env python3
"""Regression-adapted models for hierarchical forecasting.

Contains:
  - RegressionLSTM: LSTM with regression head + embedding output
  - RegressionTFT:  Thin wrapper that creates TFT in regression mode (n_classes=0)
  - RegressionTCN:  Temporal Convolutional Network (dilated causal convolutions)
  - FundamentalMLP: Tabular model on quarterly financial features
  - SectorGNN:      Graph Attention Network on cross-sectional stock graph
  - MetaMLP:        Small neural network combining N sub-model predictions + regime
  - HierarchicalForecaster: Orchestrates all sub-models via a registry (nn.ModuleDict)

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
# Utility helpers
# ============================================================================

def _reset_parameters(module: nn.Module):
    """Re-initialize a single module's parameters with standard defaults.

    Used to recover from NaN/Inf weights that indicate a diverged sub-model.
    Supports Linear, LayerNorm, Embedding, LSTM, GRU, Conv1d, and modules
    with bare nn.Parameter attributes (e.g. attention vectors).
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, p in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Fallback: re-init any bare nn.Parameter on this module (e.g. GAT attn vectors)
    for name, param in module.named_parameters(recurse=False):
        if param.isnan().any() or param.isinf().any():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.normal_(param.data, std=0.02)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HierarchicalModelConfig:
    """Configuration for the hierarchical forecaster.

    Sub-models are toggled with `use_*` flags.  `n_sub_models` is computed
    automatically from the flags when left at 0.
    """

    # Daily models
    daily_input_dim: int = 51       # Features from enhanced_features.py
    daily_seq_len: int = 720        # ~3 years
    daily_hidden_dim: int = 128     # LSTM hidden dim (reduced for faster training)
    daily_tft_hidden_dim: int = 64  # TFT hidden dim (reduced)
    daily_n_layers: int = 2
    daily_n_heads: int = 2          # For TFT
    daily_dropout: float = 0.15
    daily_lstm_dropout: float = 0.05

    # Minute models
    minute_input_dim: int = 18      # Features from minute data
    minute_seq_len: int = 780       # ~2 trading days
    minute_hidden_dim: int = 128    # LSTM hidden dim (reduced)
    minute_tft_hidden_dim: int = 64  # TFT hidden dim (reduced)
    minute_n_layers: int = 2
    minute_n_heads: int = 2
    minute_dropout: float = 0.15
    minute_lstm_dropout: float = 0.05

    # News model
    news_input_dim: int = 774       # 768 FinBERT + 6 sentiment features
    news_seq_len: int = 720         # Same as daily
    news_hidden_dim: int = 128
    news_n_layers: int = 2
    news_dropout: float = 0.15
    use_news_model: bool = True     # Toggle news sub-model
    use_minute_models: bool = True  # Toggle minute sub-models (lstm_m, tft_m)

    # TCN (daily) — dilated causal 1D convolutions on daily data
    use_tcn_d: bool = False
    tcn_d_n_filters: int = 64       # Channels per conv layer
    tcn_d_n_layers: int = 8         # 8 layers × kernel=3 → receptive field 1021 (covers 720)
    tcn_d_kernel_size: int = 3
    tcn_d_dropout: float = 0.15

    # FundamentalMLP — tabular model on quarterly financials
    use_fund_mlp: bool = False
    fund_input_dim: int = 14        # From fundamental_features.py
    fund_hidden_dim: int = 64
    fund_n_layers: int = 3
    fund_dropout: float = 0.2

    # Sector GNN — graph neural network on cross-sectional snapshots
    use_gnn_features: bool = False  # If True, use GNN embeddings as auxiliary input
    gnn_feature_dim: int = 64       # Dimensionality of GNN embedding to append

    # Shared
    embedding_dim: int = 32         # Embedding size for all sub-models (reduced)

    # Meta MLP
    regime_dim: int = 8             # vol_regime, vix_regime, etc.
    meta_hidden_dim: int = 128
    meta_n_layers: int = 2
    meta_dropout: float = 0.2
    n_sub_models: int = 0           # 0 = auto-compute from use_* flags

    def count_sub_models(self) -> int:
        """Count active sub-models from toggle flags."""
        n = 2  # lstm_d, tft_d are always present
        if self.use_minute_models:
            n += 2  # lstm_m, tft_m
        if self.use_news_model:
            n += 1
        if self.use_tcn_d:
            n += 1
        if self.use_fund_mlp:
            n += 1
        return n

    def __post_init__(self):
        if self.n_sub_models == 0:
            self.n_sub_models = self.count_sub_models()


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
# Regression TCN (Temporal Convolutional Network)
# ============================================================================

class _CausalConv1dBlock(nn.Module):
    """Single causal dilated conv block with residual connection."""

    def __init__(self, n_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: left-pad only
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)
        self.drop = nn.Dropout(dropout)
        self.padding = padding  # how many right-side values to chop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        residual = x
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # causal trim
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)

        return out + residual  # residual connection


class RegressionTCN(nn.Module):
    """Temporal Convolutional Network for time-series regression.

    Uses dilated causal convolutions with exponentially growing dilation
    factors to cover the full sequence length.  Receptive field =
    1 + 2 * (kernel-1) * sum(2^i for i in range(n_layers)).

    Outputs same interface as RegressionLSTM:
        prediction:  (batch,)     scalar predicted return
        embedding:   (batch, E)   latent representation for meta model
    """

    def __init__(
        self,
        input_dim: int,
        n_filters: int = 64,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.15,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, n_filters, 1)  # project features to channels
        self.blocks = nn.ModuleList([
            _CausalConv1dBlock(n_filters, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.embedding_head = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters, embedding_dim),
            nn.Tanh(),
        )
        self.regression_proj = nn.Linear(n_filters, n_filters // 2)
        self.regression_act = nn.GELU()
        self.regression_drop = nn.Dropout(dropout)
        self.regression_out = nn.Linear(n_filters // 2, 1)
        self.regression_skip = nn.Linear(n_filters, 1, bias=False)

        rf = 1 + 2 * (kernel_size - 1) * sum(2**i for i in range(n_layers))
        logger.info(f"  TCN: {n_layers} layers, k={kernel_size}, "
                    f"receptive field={rf}, filters={n_filters}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (batch, seq_len, input_dim)"""
        # Conv1d expects (batch, channels, seq_len)
        h = self.input_proj(x.transpose(1, 2))  # (B, n_filters, seq)
        for block in self.blocks:
            h = block(h)
        last = h[:, :, -1]  # (B, n_filters) — last timestep

        embedding = self.embedding_head(last)
        proj = self.regression_drop(self.regression_act(self.regression_proj(last)))
        prediction = (self.regression_out(proj) + self.regression_skip(last)).squeeze(-1)

        return {"prediction": prediction, "embedding": embedding}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Fundamental MLP (tabular model on quarterly financials)
# ============================================================================

class FundamentalMLP(nn.Module):
    """Simple feed-forward network on fundamental/financial features.

    Unlike the sequential models, this operates on a *snapshot* of the
    latest fundamental data (forward-filled quarterly values) rather
    than a time series.  Input is (batch, fund_input_dim).

    Outputs:
        prediction:  (batch,)     scalar predicted return
        embedding:   (batch, E)   latent representation for meta model
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        layers = []
        prev = input_dim
        for i in range(n_layers):
            out = hidden_dim if i < n_layers - 1 else hidden_dim
            layers.extend([nn.Linear(prev, out), nn.GELU(), nn.Dropout(dropout)])
            prev = out
        self.trunk = nn.Sequential(*layers)

        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )
        self.regression_out = nn.Linear(hidden_dim, 1)
        self.regression_skip = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (batch, input_dim) — flat fundamental features snapshot."""
        x_norm = self.input_norm(x)
        h = self.trunk(x_norm)  # (B, hidden_dim)

        embedding = self.embedding_head(h)
        prediction = (self.regression_out(h) + self.regression_skip(x_norm)).squeeze(-1)

        return {"prediction": prediction, "embedding": embedding}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Sector GNN (Graph Neural Network over sector-based stock graph)
# ============================================================================

class _MultiHeadGraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer (GAT-style with multiple heads).

    Each head learns independent attention patterns. Outputs are concatenated
    and projected back to ``out_dim``, maintaining dimensionality.

    Computes attention-weighted message passing on a sparse graph defined
    by ``edge_index``.  Pure PyTorch — no ``torch_geometric`` dependency.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0, f"out_dim ({out_dim}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim
        self.dropout = dropout

        # Separate transformation per head
        self.W_heads = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False)
            for _ in range(n_heads)
        ])

        # Attention vector per head: a^T [Wh_i || Wh_j]
        self.attn_heads = nn.ParameterList([
            nn.Parameter(torch.empty(2 * self.head_dim))
            for _ in range(n_heads)
        ])
        for attn in self.attn_heads:
            nn.init.xavier_uniform_(attn.unsqueeze(0))

        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

        # Project concatenated heads back to out_dim
        self.out_proj = nn.Linear(n_heads * self.head_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,            # (N, in_dim)
        edge_index: torch.Tensor,    # (2, E)  long
        mask: Optional[torch.Tensor] = None,  # (N,)  bool — valid nodes
    ) -> torch.Tensor:
        """Returns (N, out_dim)."""
        N = x.size(0)
        src, dst = edge_index  # each (E,)

        head_outputs = []
        for head_idx in range(self.n_heads):
            # Transform input with this head's weights
            h_head = self.W_heads[head_idx](x)  # (N, head_dim)

            # Concatenate transformed features of source and destination
            h_src = h_head[src]  # (E, head_dim)
            h_dst = h_head[dst]  # (E, head_dim)
            attn_input = torch.cat([h_src, h_dst], dim=-1)  # (E, 2*head_dim)
            e = self.leaky(attn_input @ self.attn_heads[head_idx])  # (E,)

            # Masked softmax over neighbours (per destination node)
            if mask is not None:
                mask_dev = mask.to(x.device)
                invalid_src = ~mask_dev[src]
                e = e.masked_fill(invalid_src, float("-inf"))

            # Softmax per destination node using scatter
            e_max = torch.zeros(N, device=x.device)
            e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
            e_stable = e - e_max[dst]
            e_stable = e_stable.clamp(max=20.0)
            alpha = torch.exp(e_stable)

            if mask is not None:
                alpha = alpha.masked_fill(invalid_src, 0.0)

            alpha_sum = torch.zeros(N, device=x.device)
            alpha_sum.scatter_add_(0, dst, alpha)
            alpha_norm = alpha / (alpha_sum[dst] + 1e-8)
            alpha_norm = self.drop(alpha_norm)

            # Aggregate messages for this head
            msg = alpha_norm.unsqueeze(-1) * h_src  # (E, head_dim)
            head_out = torch.zeros(N, self.head_dim, device=x.device)
            head_out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

            head_outputs.append(head_out)

        # Concatenate all heads and project back to out_dim
        out_concat = torch.cat(head_outputs, dim=-1)  # (N, n_heads * head_dim)
        out = self.out_proj(out_concat)  # (N, out_dim)

        return out


# ============================================================================
# GNNSubModel: GNN as a regression sub-model (like LSTM, TFT)
# ============================================================================

class GNNSubModel(nn.Module):
    """Graph neural network sub-model that operates on correlation-based stock graphs.
    
    For each batch of stock sequences, builds a correlation graph and applies
    graph attention to produce predictions and embeddings.
    
    Input: (batch, seq_len, input_dim) - time series for a batch of stocks
    Output: 
        prediction: (batch,) - scalar prediction
        embedding: (batch, embedding_dim) - learned representation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.15,
        embedding_dim: int = 64,
        correlation_threshold: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.correlation_threshold = correlation_threshold
        
        # Project sequence to hidden dimension
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(
                _MultiHeadGraphAttentionLayer(
                    hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output heads
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def _compute_correlation_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edges from correlation matrix.
        
        Args:
            x: (batch, seq_len, hidden_dim) - projected sequences
        
        Returns:
            edge_index: (2, num_edges) - edge list for fully connected batch
        """
        batch_size = x.shape[0]
        
        # Compute mean correlation across batch
        # Flatten to (batch * seq_len, hidden_dim)
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Compute pairwise correlations between nodes
        # For simplicity, use cosine similarity on sequence means
        x_mean = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Compute cosine similarity
        x_norm = torch.nn.functional.normalize(x_mean, p=2, dim=-1)
        similarity = torch.mm(x_norm, x_norm.t())  # (batch, batch)
        
        # Create edges for high-correlation pairs
        edges = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and similarity[i, j] > self.correlation_threshold:
                    edges.append([i, j])
        
        if len(edges) == 0:
            # If no edges found, create a simple chain graph
            edges = [[i, (i + 1) % batch_size] for i in range(batch_size)]
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=x.device).t()
        return edge_index
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) time series
        
        Returns:
            {
                "prediction": (batch,),
                "embedding": (batch, embedding_dim),
            }
        """
        batch_size = x.shape[0]
        
        # Project sequence: (batch, seq_len, input_dim) → (batch, seq_len, hidden_dim)
        h = self.seq_proj(x)
        
        # Pool to node level: (batch, hidden_dim)
        h = h.mean(dim=1)
        
        # Create batch graph edges
        edge_index = self._compute_correlation_edges(h.unsqueeze(1))
        
        # Create mask for valid nodes (all nodes in this case)
        mask = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        
        # Apply graph attention
        h_node = h
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h_new = gat(h_node, edge_index, mask=mask)
            h_node = norm(h_node + h_new)  # residual
        
        # Generate outputs
        embedding = self.embedding_head(h_node)
        prediction = self.prediction_head(h_node).squeeze(-1)
        
        return {
            "prediction": prediction,
            "embedding": embedding,
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SectorGNN(nn.Module):
    """Graph neural network on sector-based stock graph.

    Operates on *cross-sectional snapshots*: all stocks at one trading date.
    Each node is a stock; edges connect stocks in the same sector.

    Architecture:
        Input projection → 2 GAT layers with residual → per-node heads
        (embedding + regression).

    The model receives ``node_features`` (N, F) — the daily feature vector
    for each stock — plus ``edge_index`` (sector graph) and ``mask``.

    Outputs (for each masked node):
        prediction:  (B,)     — where B = sum(mask)
        embedding:   (B, E)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.15,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(
                _MultiHeadGraphAttentionLayer(
                    hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )
        self.regression_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.regression_act = nn.GELU()
        self.regression_drop = nn.Dropout(dropout)
        self.regression_out = nn.Linear(hidden_dim // 2, 1)
        self.regression_skip = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,    # (N, F)
        edge_index: torch.Tensor,        # (2, E)
        mask: torch.Tensor,              # (N,) bool
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over the full graph.

        Returns predictions/embeddings only for masked (valid) nodes.

        Returns:
            prediction: (B,)   where B = mask.sum()
            embedding:  (B, E)
        """
        h = self.input_proj(node_features)  # (N, hidden)

        mask = mask.to(node_features.device)
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h_new = gat(h, edge_index, mask=mask)
            h = norm(h + h_new)  # residual + LayerNorm

        # Select only valid (masked) nodes
        h_valid = h[mask]  # (B, hidden)

        embedding = self.embedding_head(h_valid)
        proj = self.regression_drop(self.regression_act(self.regression_proj(h_valid)))
        prediction = (self.regression_out(proj) + self.regression_skip(h_valid)).squeeze(-1)

        return {"prediction": prediction, "embedding": embedding}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Meta MLP  (combines sub-model outputs + regime)
# ============================================================================

class MetaMLP(nn.Module):
    """Small neural network that fuses sub-model predictions into
    a single next-day return forecast.

    Input vector (per sample):
      - N scalar predictions     (LSTM_D, TFT_D, LSTM_M, TFT_M, [News])
      - N × embedding_dim        (64 × N)
      - regime_dim               (8)
      - 2 disagreement features  (pred_variance, pred_spread)
    Total default (5 models) = 5 + 1 + 320 + 8 + 2 = 336
    Total default (4 models) = 4 + 1 + 256 + 8 + 2 = 271

    The disagreement features capture sub-model prediction spread:
      - pred_variance: variance of the N sub-model predictions
      - pred_spread:   (max - min) / (mean_abs + eps)
    High disagreement → low confidence → meta learns smaller positions.

    The network learns:
      - How to weight daily vs minute vs news predictions
      - When to trust which model (regime-dependent)
      - Non-linear interactions between model signals
      - How to scale down when sub-models disagree
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

        # N raw preds + 1 weighted pred + N embs + regime + 2 disagreement features
        self.disagreement_dim = 2  # pred_variance, pred_spread
        input_dim = n_sub_models + 1 + n_sub_models * embedding_dim + regime_dim + self.disagreement_dim

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

    def apply_attention_bias(self, bias: torch.Tensor):
        """Shift the attention network's pre-softmax logit bias.

        This initializes the attention distribution with a prior from
        previous walk-forward windows.  Positive values increase the
        weight for a sub-model; negative values decrease it.

        Args:
            bias:  (n_sub_models,) or smaller tensor of logit offsets.
                   If smaller than n_sub_models, only the first len(bias) elements are applied.
                   Values are *added* to the existing output-layer bias.
        """
        # self.attention is: Linear(R→32) → GELU → Linear(32→N) → Softmax
        # The last Linear before Softmax is at index 2.
        attn_output_layer = self.attention[2]
        assert isinstance(attn_output_layer, nn.Linear), (
            f"Expected nn.Linear at attention[2], got {type(attn_output_layer)}"
        )
        
        # Handle cases where bias might have fewer elements than n_sub_models
        # (e.g., from a previous model with fewer sub-models)
        if bias.shape[0] > self.n_sub_models:
            logger.warning(
                f"Bias has {bias.shape[0]} elements but forecaster has {self.n_sub_models} sub-models. "
                f"Truncating bias."
            )
            bias = bias[:self.n_sub_models]
        elif bias.shape[0] < self.n_sub_models:
            logger.warning(
                f"Bias has {bias.shape[0]} elements but forecaster has {self.n_sub_models} sub-models. "
                f"Padding bias with zeros."
            )
            # Pad with zeros for new sub-models
            padding = torch.zeros(self.n_sub_models - bias.shape[0], device=bias.device, dtype=bias.dtype)
            bias = torch.cat([bias, padding], dim=0)
        
        with torch.no_grad():
            attn_output_layer.bias.add_(bias.to(attn_output_layer.bias.device))

        logger.info(
            "MetaMLP attention bias applied: %s",
            {f"sub_{i}": round(float(bias[i]), 3) for i in range(len(bias))},
        )

    def forward(
        self,
        predictions: torch.Tensor,          # (batch, N)
        embeddings: List[torch.Tensor],      # list of N × (batch, E)
        regime: torch.Tensor,                # (batch, regime_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with 'prediction' (batch,), 'attention_weights' (batch, N),
            and 'disagreement' (batch, 2)
        """
        emb_cat = torch.cat(embeddings, dim=-1)      # (batch, N*E)

        # Regime-conditioned attention: weight the N sub-model predictions
        attn_weights = self.attention(regime)          # (batch, N)
        weighted_preds = (predictions * attn_weights).sum(dim=-1, keepdim=True)  # (batch, 1)

        # ── Sub-model disagreement features ─────────────────────────
        pred_variance = predictions.var(dim=-1, keepdim=True)      # (batch, 1)
        pred_mean_abs = predictions.abs().mean(dim=-1, keepdim=True) + 1e-8
        pred_spread = (predictions.max(dim=-1, keepdim=True).values
                       - predictions.min(dim=-1, keepdim=True).values) / pred_mean_abs  # (batch, 1)
        disagreement = torch.cat([pred_variance, pred_spread], dim=-1)  # (batch, 2)

        # Feed everything (including disagreement) into the trunk
        combined = torch.cat([predictions, weighted_preds, emb_cat, regime, disagreement], dim=-1)

        hidden = self.trunk(combined)
        prediction = self.output_head(hidden).squeeze(-1)

        return {
            "prediction": prediction,
            "attention_weights": attn_weights,
            "disagreement": disagreement,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FusionMLP: Combines daily and minute meta predictions
# ============================================================================

class FusionMLP(nn.Module):
    """Lightweight network that fuses daily and minute model predictions.
    
    Takes predictions from separate daily and minute meta models and combines
    them into a single forecast using regime-aware fusion.
    
    Input:
      - daily_pred: (batch,) scalar prediction from daily meta
      - minute_pred: (batch,) scalar prediction from minute meta
      - regime: (batch, regime_dim) regime encoding
    
    Output:
      - (batch,) fused prediction
    """
    
    def __init__(
        self,
        regime_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.regime_dim = regime_dim
        
        # Input: daily_pred (1) + minute_pred (1) + regime (R)
        input_dim = 2 + regime_dim
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for name, p in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name:
                nn.init.normal_(p, 0.0, 0.01)
    
    def forward(
        self,
        daily_pred: torch.Tensor,  # (batch,)
        minute_pred: torch.Tensor,  # (batch,)
        regime: torch.Tensor,  # (batch, regime_dim)
    ) -> torch.Tensor:
        """Fuse daily and minute predictions.
        
        Args:
            daily_pred: (batch,) predictions from daily meta model
            minute_pred: (batch,) predictions from minute meta model
            regime: (batch, regime_dim) regime encoding
        
        Returns:
            (batch,) fused predictions
        """
        # Stack predictions
        preds = torch.stack([daily_pred, minute_pred], dim=-1)  # (batch, 2)
        
        # Concatenate with regime
        combined = torch.cat([preds, regime], dim=-1)  # (batch, 2 + R)
        
        # Forward through network
        hidden = self.trunk(combined)
        output = self.output_head(hidden).squeeze(-1)  # (batch,)
        
        return output
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# HierarchicalForecaster: orchestrates all 5 models
# ============================================================================

class HierarchicalForecaster(nn.Module):
    """Complete hierarchical forecaster with a dynamic set of sub-models.

    Sub-models are stored in ``self.sub_models`` (an ``nn.ModuleDict``).
    Every sub-model must output ``{"prediction": (B,), "embedding": (B, E)}``.
    The ``meta`` module learns a regime-conditioned weighting over them.

    The canonical sub-model order is stored in ``self.sub_model_names``
    (a plain Python list) so that predictions/embeddings are always
    stacked in a deterministic order.

    Core (always present):
        lstm_d, tft_d, lstm_m, tft_m

    Optional (toggled via config flags):
        news     — cfg.use_news_model
        tcn_d    — cfg.use_tcn_d
        fund_mlp — cfg.use_fund_mlp
        gnn      — cfg.use_gnn

    Training phases:
        Phase 1:   Train daily models (lstm_d, tft_d, tcn_d)
        Phase 1.5: Train news encoder (if enabled)
        Phase 1.6: Train FundamentalMLP (if enabled)
        Phase 1.7: Train SectorGNN (if enabled)
        Phase 2:   Train minute models (lstm_m, tft_m)
        Phase 3:   Freeze sub-models, train meta on combined predictions
        Phase 4:   (Optional) Fine-tune all jointly with small LR
    """

    # Maps sub-model names to the *data modality* they consume.
    # Used by the training loop to pick the right DataLoader.
    MODALITY = {
        "lstm_d": "daily",
        "tft_d": "daily",
        "tcn_d": "daily",
        "lstm_m": "minute",
        "tft_m": "minute",
        "news": "news",
        "fund_mlp": "fundamental",
        "gnn": "graph",
    }

    # Training phase for each sub-model (used in run_pipeline)
    PHASE = {
        "lstm_d": 1,
        "tft_d": 1,
        "tcn_d": 1,
        "lstm_m": 2,
        "tft_m": 2,
        "news": 1,  # Phase 1.5 in practice
        "fund_mlp": 1,
        "gnn": 1,  # Phase 1.7 in practice
    }

    def __init__(self, cfg: HierarchicalModelConfig):
        super().__init__()
        self.cfg = cfg

        # Build sub-models into an ordered ModuleDict
        subs: Dict[str, nn.Module] = {}
        names: List[str] = []

        # --- Daily models (always present) ---
        subs["lstm_d"] = RegressionLSTM(
            input_dim=cfg.daily_input_dim,
            hidden_dim=cfg.daily_hidden_dim,
            num_layers=cfg.daily_n_layers,
            dropout=cfg.daily_lstm_dropout,
            embedding_dim=cfg.embedding_dim,
        )
        names.append("lstm_d")

        subs["tft_d"] = RegressionTFT(
            input_dim=cfg.daily_input_dim,
            seq_len=cfg.daily_seq_len,
            hidden_dim=cfg.daily_tft_hidden_dim,  # Use TFT-optimized dim, not LSTM dim
            n_heads=cfg.daily_n_heads,
            n_layers=cfg.daily_n_layers,
            dropout=cfg.daily_dropout,
            embedding_dim=cfg.embedding_dim,
        )
        names.append("tft_d")

        # --- TCN_D (optional) ---
        if cfg.use_tcn_d:
            subs["tcn_d"] = RegressionTCN(
                input_dim=cfg.daily_input_dim,
                n_filters=cfg.tcn_d_n_filters,
                n_layers=cfg.tcn_d_n_layers,
                kernel_size=cfg.tcn_d_kernel_size,
                dropout=cfg.tcn_d_dropout,
                embedding_dim=cfg.embedding_dim,
            )
            names.append("tcn_d")

        # --- Minute models (conditionally present) ---
        if cfg.use_minute_models:
            subs["lstm_m"] = RegressionLSTM(
                input_dim=cfg.minute_input_dim,
                hidden_dim=cfg.minute_hidden_dim,
                num_layers=cfg.minute_n_layers,
                dropout=cfg.minute_lstm_dropout,
                embedding_dim=cfg.embedding_dim,
            )
            names.append("lstm_m")

            subs["tft_m"] = RegressionTFT(
                input_dim=cfg.minute_input_dim,
                seq_len=cfg.minute_seq_len,
                hidden_dim=cfg.minute_tft_hidden_dim,  # Use TFT-optimized dim, not LSTM dim
                n_heads=cfg.minute_n_heads,
                n_layers=cfg.minute_n_layers,
                dropout=cfg.minute_dropout,
                embedding_dim=cfg.embedding_dim,
            )
            names.append("tft_m")

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
            subs["news"] = NewsEncoder(news_cfg)
            names.append("news")

        # --- Fundamental MLP (optional) ---
        if cfg.use_fund_mlp:
            subs["fund_mlp"] = FundamentalMLP(
                input_dim=cfg.fund_input_dim,
                hidden_dim=cfg.fund_hidden_dim,
                n_layers=cfg.fund_n_layers,
                dropout=cfg.fund_dropout,
                embedding_dim=cfg.embedding_dim,
            )
            names.append("fund_mlp")


        # GNN is no longer a sub-model. If using GNN features, input dims must be updated.
        self.use_gnn_features = cfg.use_gnn_features
        self.gnn_feature_dim = cfg.gnn_feature_dim
        if self.use_gnn_features:
            # Increase input dims for daily/minute models to accept GNN features
            for name in ["lstm_d", "tft_d", "tcn_d", "lstm_m", "tft_m"]:
                if name in subs:
                    model = subs[name]
                    # RegressionLSTM/RegressionTCN have direct .input_dim
                    if hasattr(model, "input_dim"):
                        model.input_dim += self.gnn_feature_dim
                    # RegressionTFT stores it in .config.input_dim
                    elif hasattr(model, "config") and hasattr(model.config, "input_dim"):
                        model.config.input_dim += self.gnn_feature_dim


        self.sub_models = nn.ModuleDict(subs)
        # Remove GNN from sub_model_names if present
        self.sub_model_names: List[str] = [n for n in names if n != "gnn"]

        # Ensure n_sub_models is consistent
        n_sub = len(self.sub_model_names)
        cfg.n_sub_models = n_sub

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

    # ------------------------------------------------------------------
    # Convenience accessors (backward compatibility)
    # ------------------------------------------------------------------
    # These properties let existing code keep using `forecaster.lstm_d` etc.
    # They are thin proxies into self.sub_models.

    @property
    def lstm_d(self) -> nn.Module:
        return self.sub_models["lstm_d"]

    @property
    def tft_d(self) -> nn.Module:
        return self.sub_models["tft_d"]

    @property
    def lstm_m(self) -> nn.Module:
        return self.sub_models["lstm_m"]

    @property
    def tft_m(self) -> nn.Module:
        return self.sub_models["tft_m"]

    @property
    def news(self) -> Optional[nn.Module]:
        return self.sub_models.get("news")

    @property
    def tcn_d(self) -> Optional[nn.Module]:
        return self.sub_models.get("tcn_d")

    @property
    def fund_mlp(self) -> Optional[nn.Module]:
        return self.sub_models.get("fund_mlp")


    # GNN is no longer a sub-model; embeddings must be provided as auxiliary input.

    def _log_params(self):
        parts = {}
        for name in self.sub_model_names:
            parts[name] = self.sub_models[name].count_parameters()
        parts["meta"] = self.meta.count_parameters()
        total = sum(parts.values())
        logger.info("Hierarchical Forecaster parameter count:")
        for name, count in parts.items():
            logger.info(f"  {name:12s}: {count:>10,}")
        logger.info(f"  {'TOTAL':12s}: {total:>10,}")

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------
    def freeze_sub_models(self):
        """Freeze all sub-models (for meta training)."""
        for name, m in self.sub_models.items():
            for p in m.parameters():
                p.requires_grad = False
        logger.info(f"Sub-models frozen for meta training ({len(self.sub_models)} models)")

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
        news_x: Optional[torch.Tensor] = None,
        fund_x: Optional[torch.Tensor] = None,
        graph_x: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through all registered sub-models + meta.

        Args:
            graph_x: Optional dict with keys 'node_features' (N, F),
                      'edge_index' (2, E), 'mask' (N,) for the GNN.
                      Because GNN is cross-sectional, its predictions are
                      pre-indexed to match batch B before stacking.

        Returns dict with:
            prediction:         (batch,) — meta model output
            sub_predictions:    (batch, N) — ordered sub-model predictions
            attention_weights:  (batch, N) — learned sub-model weights
            <name>:             per-sub-model output dict
        """
        # Map modalities to their tensor inputs
        modality_inputs = {
            "daily": daily_x,
            "minute": minute_x,
            "news": news_x,
            "fundamental": fund_x,
        }

        pred_list: List[torch.Tensor] = []
        emb_list: List[torch.Tensor] = []
        result: Dict[str, torch.Tensor] = {}

        B = daily_x.shape[0]

        for name in self.sub_model_names:
            modality = self.MODALITY[name]

            # GNN has a special call signature
            if modality == "graph":
                if graph_x is not None:
                    out = self.sub_models[name](
                        graph_x["node_features"],
                        graph_x["edge_index"],
                        graph_x["mask"],
                    )
                    # GNN returns variable-length (mask.sum(),) — caller must
                    # ensure this equals B or pre-pad.  For simplicity, pad/trim.
                    gnn_pred = out["prediction"]
                    gnn_emb = out["embedding"]
                    if gnn_pred.shape[0] != B:
                        # Zero-fill or trim to match batch
                        gnn_pred_full = torch.zeros(B, device=daily_x.device)
                        gnn_emb_full = torch.zeros(B, self.cfg.embedding_dim, device=daily_x.device)
                        n = min(gnn_pred.shape[0], B)
                        gnn_pred_full[:n] = gnn_pred[:n]
                        gnn_emb_full[:n] = gnn_emb[:n]
                        gnn_pred = gnn_pred_full
                        gnn_emb = gnn_emb_full
                    pred_list.append(gnn_pred)
                    emb_list.append(gnn_emb)
                    result[name] = out
                else:
                    pred_list.append(torch.zeros(B, device=daily_x.device))
                    emb_list.append(torch.zeros(B, self.cfg.embedding_dim, device=daily_x.device))
                continue

            x = modality_inputs.get(modality)
            if x is None:
                # Optional modality not provided — skip with zeros
                pred_list.append(torch.zeros(B, device=daily_x.device))
                emb_list.append(torch.zeros(B, self.cfg.embedding_dim, device=daily_x.device))
                continue
            out = self.sub_models[name](x)
            pred_list.append(out["prediction"])
            emb_list.append(out["embedding"])
            result[name] = out

        preds = torch.stack(pred_list, dim=-1)  # (batch, N)
        meta_out = self.meta(preds, emb_list, regime)

        result.update({
            "prediction": meta_out["prediction"],
            "sub_predictions": preds,
            "attention_weights": meta_out["attention_weights"],
            "disagreement": meta_out["disagreement"],  # (batch, 2): variance + spread
        })
        return result

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        save_dict: Dict = {
            "cfg": self.cfg,
            "sub_model_names": self.sub_model_names,
            "meta": self.meta.state_dict(),
        }
        for name in self.sub_model_names:
            save_dict[name] = self.sub_models[name].state_dict()
        torch.save(save_dict, path)
        logger.info(f"HierarchicalForecaster saved → {path}")

    @classmethod
    def load(
        cls, path: str, device: str = "cpu",
        override_cfg: Optional["HierarchicalModelConfig"] = None,
    ) -> "HierarchicalForecaster":
        """Load a saved forecaster from disk.

        Args:
            path:         Checkpoint file path.
            device:       Target device.
            override_cfg: If provided, build the model from *this* config
                          instead of the checkpoint's saved config.  Weights
                          for sub-models that exist in the checkpoint are
                          loaded; new sub-models start with random init.
                          This enables "resume old checkpoint → add new
                          sub-models" workflows.

        The attribute ``model.loaded_sub_models`` (a set of str) records
        which sub-models received real checkpoint weights.  Sub-models NOT
        in this set are freshly initialized and need training.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        saved_cfg = ckpt["cfg"]

        # Backward compatibility: ensure all new config fields exist
        for attr, default in [
            ('use_tcn_d', False), ('use_fund_mlp', False), ('use_gnn', False),
            ('tcn_d_n_filters', 64), ('tcn_d_n_layers', 8),
            ('tcn_d_kernel_size', 3), ('tcn_d_dropout', 0.15),
            ('fund_input_dim', 14), ('fund_hidden_dim', 64),
            ('fund_n_layers', 3), ('fund_dropout', 0.2),
            ('gnn_input_dim', 51), ('gnn_hidden_dim', 64),
            ('gnn_n_layers', 4), ('gnn_n_heads', 4), ('gnn_dropout', 0.15),
        ]:
            if not hasattr(saved_cfg, attr):
                setattr(saved_cfg, attr, default)

        saved_names = set(ckpt.get("sub_model_names") or [])
        if not saved_names:
            # Legacy: infer from top-level keys
            saved_names = {"lstm_d", "tft_d", "lstm_m", "tft_m"}
            if "news" in ckpt:
                saved_names.add("news")

        if override_cfg is not None:
            # Use the caller's config (wider model), but copy dimension
            # fields from the checkpoint so matching sub-models' shapes
            # are compatible.
            cfg = override_cfg
            for dim_attr in [
                "daily_input_dim", "daily_hidden_dim", "daily_n_layers",
                "daily_n_heads", "daily_dropout", "daily_seq_len",
                "minute_input_dim", "minute_hidden_dim", "minute_n_layers",
                "minute_n_heads", "minute_dropout", "minute_seq_len",
                "news_input_dim", "news_hidden_dim", "news_n_layers",
                "news_dropout", "news_seq_len",
                "embedding_dim", "regime_dim",
            ]:
                if hasattr(saved_cfg, dim_attr):
                    setattr(cfg, dim_attr, getattr(saved_cfg, dim_attr))
            cfg.n_sub_models = 0  # force recount
            cfg.__post_init__()
        else:
            cfg = saved_cfg
            # Reconcile use_* flags to match checkpoint contents exactly
            cfg.use_news_model = "news" in saved_names
            cfg.use_tcn_d = "tcn_d" in saved_names
            cfg.use_fund_mlp = "fund_mlp" in saved_names
            cfg.use_gnn = "gnn" in saved_names
            cfg.n_sub_models = 0
            cfg.__post_init__()

        model = cls(cfg)

        # Load sub-model weights where available
        loaded: set = set()
        for name in model.sub_model_names:
            if name in ckpt and name in saved_names:
                try:
                    model.sub_models[name].load_state_dict(ckpt[name])
                    # Detect NaN/Inf in loaded weights and re-initialize if corrupt
                    params = list(model.sub_models[name].parameters())
                    corrupt = any(
                        p.isnan().any().item() or p.isinf().any().item() for p in params
                    )
                    if corrupt:
                        logger.warning(
                            f"  Sub-model '{name}' has NaN/Inf weights in checkpoint "
                            f"(likely diverged during training) \u2014 re-initializing with fresh weights"
                        )
                        model.sub_models[name].apply(_reset_parameters)
                    else:
                        loaded.add(name)
                except RuntimeError as e:
                    logger.warning(f"  Sub-model '{name}' shape mismatch — random init  ({e})")
            else:
                logger.info(f"  Sub-model '{name}' not in checkpoint — random init (needs training)")

        model.loaded_sub_models = loaded  # track what was loaded

        # Meta model: only load if shapes match (n_sub_models may differ)
        try:
            model.meta.load_state_dict(ckpt["meta"])
            # Detect NaN/Inf in meta weights
            meta_corrupt = any(
                p.isnan().any().item() or p.isinf().any().item()
                for p in model.meta.parameters()
            )
            if meta_corrupt:
                logger.warning(
                    "  Meta model has NaN/Inf weights in checkpoint "
                    "(likely diverged during Phase 3) \u2014 re-initializing with fresh weights"
                )
                model.meta.apply(_reset_parameters)
            else:
                loaded.add("meta")
        except RuntimeError:
            logger.info("  Meta model shape mismatch (new sub-models added) — random init")

        model.to(device)
        new_models = set(model.sub_model_names) - loaded
        logger.info(
            f"HierarchicalForecaster loaded ← {path}\n"
            f"  Sub-models: {model.sub_model_names}\n"
            f"  Loaded from checkpoint: {sorted(loaded)}\n"
            f"  Need training (random init): {sorted(new_models) if new_models else 'none'}"
        )
        return model


# ============================================================================
# DailyForecaster: Daily-only pipeline (LSTM_D, TFT_D, News, DailyMeta)
# ============================================================================

class DailyForecaster(nn.Module):
    """Forecaster that contains only daily models and daily meta.
    
    Contains:
      - lstm_d, tft_d, tcn_d, gnn_d (always)
      - daily_meta: MetaMLP that fuses 4 daily model outputs
    
    Does NOT contain minute models or news.
    """
    
    def __init__(self, cfg: HierarchicalModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Daily models: 4 sub-models
        self.lstm_d = RegressionLSTM(
            input_dim=cfg.daily_input_dim,
            hidden_dim=cfg.daily_hidden_dim,
            num_layers=cfg.daily_n_layers,
            dropout=cfg.daily_lstm_dropout,
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
        self.tcn_d = RegressionTCN(
            input_dim=cfg.daily_input_dim,
            n_filters=cfg.daily_hidden_dim,
            n_layers=3,
            kernel_size=3,
            dropout=cfg.daily_dropout,
            embedding_dim=cfg.embedding_dim,
        )
        self.gnn_d = GNNSubModel(
            input_dim=cfg.daily_input_dim,
            hidden_dim=cfg.daily_hidden_dim,
            n_heads=cfg.daily_n_heads,
            embedding_dim=cfg.embedding_dim,
        )
        
        # Daily meta model (4 sub-models)
        self.daily_meta = MetaMLP(
            embedding_dim=cfg.embedding_dim,
            regime_dim=cfg.regime_dim,
            hidden_dim=cfg.meta_hidden_dim,
            n_layers=cfg.meta_n_layers,
            dropout=cfg.meta_dropout,
            n_sub_models=4,
        )
    
    def forward(
        self,
        daily_x: torch.Tensor,
        regime: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for daily models.
        
        Args:
            daily_x: (batch, seq_len, daily_input_dim)
            regime: (batch, regime_dim)
        
        Returns:
            {
                "prediction": (batch,) final prediction from daily_meta,
                "lstm_d_pred": (batch,) lstm_d prediction,
                "tft_d_pred": (batch,) tft_d prediction,
                "tcn_d_pred": (batch,) tcn_d prediction,
                "gnn_d_pred": (batch,) gnn_d prediction,
                "lstm_d_emb": (batch, E) lstm_d embedding,
                "tft_d_emb": (batch, E) tft_d embedding,
                "tcn_d_emb": (batch, E) tcn_d embedding,
                "gnn_d_emb": (batch, E) gnn_d embedding,
            }
        """
        # Daily models
        lstm_d_out = self.lstm_d(daily_x)
        tft_d_out = self.tft_d(daily_x)
        tcn_d_out = self.tcn_d(daily_x)
        gnn_d_out = self.gnn_d(daily_x)
        
        predictions = [
            lstm_d_out["prediction"],
            tft_d_out["prediction"],
            tcn_d_out["prediction"],
            gnn_d_out["prediction"],
        ]
        embeddings = [
            lstm_d_out["embedding"],
            tft_d_out["embedding"],
            tcn_d_out["embedding"],
            gnn_d_out["embedding"],
        ]
        
        # Stack for meta model
        preds_stacked = torch.stack(predictions, dim=-1)  # (batch, 4)
        
        # Daily meta
        meta_out = self.daily_meta(preds_stacked, embeddings, regime)
        
        return {
            "prediction": meta_out["prediction"],
            "lstm_d_pred": lstm_d_out["prediction"],
            "tft_d_pred": tft_d_out["prediction"],
            "tcn_d_pred": tcn_d_out["prediction"],
            "gnn_d_pred": gnn_d_out["prediction"],
            "lstm_d_emb": lstm_d_out["embedding"],
            "tft_d_emb": tft_d_out["embedding"],
            "tcn_d_emb": tcn_d_out["embedding"],
            "gnn_d_emb": gnn_d_out["embedding"],
            "attention_weights": meta_out["attention_weights"],
        }


# ============================================================================
# MinuteForecaster: Minute-only pipeline (LSTM_M, TFT_M, TCN_M, GNN_M, News)
# ============================================================================

class MinuteForecaster(nn.Module):
    """Forecaster that contains only minute models and minute meta.
    
    Contains:
      - lstm_m, tft_m, tcn_m, gnn_m (always)
      - news (optional, toggled by config)
      - minute_meta: MetaMLP that fuses minute model outputs
    
    Does NOT contain daily models.
    """
    
    def __init__(self, cfg: HierarchicalModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Minute models: 4 sub-models
        self.lstm_m = RegressionLSTM(
            input_dim=cfg.minute_input_dim,
            hidden_dim=cfg.minute_hidden_dim,
            num_layers=cfg.minute_n_layers,
            dropout=cfg.minute_lstm_dropout,
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
        self.tcn_m = RegressionTCN(
            input_dim=cfg.minute_input_dim,
            n_filters=cfg.minute_hidden_dim,
            n_layers=3,
            kernel_size=3,
            dropout=cfg.minute_dropout,
            embedding_dim=cfg.embedding_dim,
        )
        self.gnn_m = GNNSubModel(
            input_dim=cfg.minute_input_dim,
            hidden_dim=cfg.minute_hidden_dim,
            n_heads=cfg.minute_n_heads,
            embedding_dim=cfg.embedding_dim,
        )
        
        # News encoder (optional for minute)
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
            n_sub_minute = 5  # lstm_m, tft_m, tcn_m, gnn_m, news
        else:
            n_sub_minute = 4  # lstm_m, tft_m, tcn_m, gnn_m
        
        # Minute meta model
        self.minute_meta = MetaMLP(
            embedding_dim=cfg.embedding_dim,
            regime_dim=cfg.regime_dim,
            hidden_dim=cfg.meta_hidden_dim,
            n_layers=cfg.meta_n_layers,
            dropout=cfg.meta_dropout,
            n_sub_models=n_sub_minute,
        )
    
    def forward(
        self,
        minute_x: torch.Tensor,
        regime: torch.Tensor,
        news_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for minute models.
        
        Args:
            minute_x: (batch, seq_len, minute_input_dim)
            regime: (batch, regime_dim)
            news_x: (batch, news_seq_len, news_input_dim) if use_news=True
        
        Returns:
            {
                "prediction": (batch,) final prediction from minute_meta,
                "lstm_m_pred": (batch,) lstm_m prediction,
                "tft_m_pred": (batch,) tft_m prediction,
                "tcn_m_pred": (batch,) tcn_m prediction,
                "gnn_m_pred": (batch,) gnn_m prediction,
                "news_pred": (batch,) news prediction (if use_news=True),
                ... embeddings ...
            }
        """
        # Minute models
        lstm_m_out = self.lstm_m(minute_x)
        tft_m_out = self.tft_m(minute_x)
        tcn_m_out = self.tcn_m(minute_x)
        gnn_m_out = self.gnn_m(minute_x)
        
        predictions = [
            lstm_m_out["prediction"],
            tft_m_out["prediction"],
            tcn_m_out["prediction"],
            gnn_m_out["prediction"],
        ]
        embeddings = [
            lstm_m_out["embedding"],
            tft_m_out["embedding"],
            tcn_m_out["embedding"],
            gnn_m_out["embedding"],
        ]
        
        # News (optional)
        if self.use_news and news_x is not None:
            news_out = self.news(news_x)
            predictions.append(news_out["prediction"])
            embeddings.append(news_out["embedding"])
        else:
            # Zero-fill if no news
            E = self.cfg.embedding_dim
            if self.use_news:
                predictions.append(torch.zeros(minute_x.shape[0], device=minute_x.device, dtype=minute_x.dtype))
                embeddings.append(torch.zeros(minute_x.shape[0], E, device=minute_x.device, dtype=minute_x.dtype))
        
        # Stack for meta model
        preds_stacked = torch.stack(predictions, dim=-1)  # (batch, n_sub)
        
        # Minute meta
        meta_out = self.minute_meta(preds_stacked, embeddings, regime)
        
        result = {
            "prediction": meta_out["prediction"],
            "lstm_m_pred": lstm_m_out["prediction"],
            "tft_m_pred": tft_m_out["prediction"],
            "tcn_m_pred": tcn_m_out["prediction"],
            "gnn_m_pred": gnn_m_out["prediction"],
            "lstm_m_emb": lstm_m_out["embedding"],
            "tft_m_emb": tft_m_out["embedding"],
            "tcn_m_emb": tcn_m_out["embedding"],
            "gnn_m_emb": gnn_m_out["embedding"],
            "attention_weights": meta_out["attention_weights"],
        }
        
        if self.use_news:
            result["news_pred"] = predictions[4] if len(predictions) > 4 else None
            result["news_emb"] = embeddings[4] if len(embeddings) > 4 else None
        
        return result


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing HierarchicalForecaster...\n")

    # Test with news model enabled
    cfg = HierarchicalModelConfig(use_news_model=True, n_sub_models=0)
    model = HierarchicalForecaster(cfg)
    print(f"Sub-models: {model.sub_model_names}")

    B = 4
    daily_x = torch.randn(B, cfg.daily_seq_len, cfg.daily_input_dim)
    minute_x = torch.randn(B, cfg.minute_seq_len, cfg.minute_input_dim)
    regime = torch.randn(B, cfg.regime_dim)
    news_x = torch.randn(B, cfg.news_seq_len, cfg.news_input_dim)

    out = model(daily_x, minute_x, regime, news_x=news_x)

    print(f"Meta prediction:   {out['prediction'].shape}")
    print(f"Sub predictions:   {out['sub_predictions'].shape}")
    print(f"Attention weights: {out['attention_weights']}")
    print(f"Disagreement:      {out['disagreement'].shape} (variance + spread)")
    print(f"LSTM_D embedding:  {out['lstm_d']['embedding'].shape}")

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
    cfg2 = HierarchicalModelConfig(use_news_model=False, n_sub_models=0)
    model2 = HierarchicalForecaster(cfg2)
    out3 = model2(daily_x, minute_x, regime)
    print(f"\nNo-news sub predictions: {out3['sub_predictions'].shape}")
    print(f"No-news sub-models: {model2.sub_model_names}")

    # Test with TCN_D + FundamentalMLP
    cfg3 = HierarchicalModelConfig(
        use_news_model=False, use_tcn_d=True, use_fund_mlp=True, n_sub_models=0,
    )
    model3 = HierarchicalForecaster(cfg3)
    fund_x = torch.randn(B, cfg3.fund_input_dim)
    out4 = model3(daily_x, minute_x, regime, fund_x=fund_x)
    print(f"\nTCN+Fund sub-models: {model3.sub_model_names}")
    print(f"TCN+Fund predictions: {out4['sub_predictions'].shape}")
    print(f"TCN+Fund attention:   {out4['attention_weights'].shape}")

    # Test with ALL models
    cfg4 = HierarchicalModelConfig(
        use_news_model=True, use_tcn_d=True, use_fund_mlp=True, use_gnn=True,
        n_sub_models=0,
    )
    model4 = HierarchicalForecaster(cfg4)

    # Build a small fake graph for GNN testing
    N_tickers = 10
    graph_x = {
        "node_features": torch.randn(N_tickers, cfg4.gnn_input_dim),
        "edge_index": torch.tensor([[0,1,2,3,0,1,2,3,4,5,6,7,8,9],
                                     [1,0,3,2,0,1,2,3,4,5,6,7,8,9]], dtype=torch.long),
        "mask": torch.ones(N_tickers, dtype=torch.bool),
    }
    # GNN returns N_tickers predictions, but forward() pads/trims to B
    out5 = model4(daily_x, minute_x, regime, news_x=news_x, fund_x=fund_x,
                  graph_x=graph_x)
    print(f"\nAll models (incl GNN): {model4.sub_model_names}")
    print(f"All predictions: {out5['sub_predictions'].shape}")
    print(f"All attention:   {out5['attention_weights'].shape}")

    # Test SectorGNN standalone
    gnn_model = SectorGNN(input_dim=51, hidden_dim=64, n_layers=4, n_heads=4)
    N = 20
    gnn_x = torch.randn(N, 51)
    gnn_edge = torch.tensor([[0,1,2,3,4,5] + list(range(N)),
                              [1,0,3,2,5,4] + list(range(N))], dtype=torch.long)
    gnn_mask = torch.ones(N, dtype=torch.bool)
    gnn_mask[15:] = False  # mask out 5 nodes
    gnn_out = gnn_model(gnn_x, gnn_edge, gnn_mask)
    print(f"\nGNN standalone: pred={gnn_out['prediction'].shape} (expected {gnn_mask.sum().item()}), "
          f"emb={gnn_out['embedding'].shape}")
    assert gnn_out["prediction"].shape[0] == gnn_mask.sum().item()
    print(f"GNN params: {gnn_model.count_parameters():,}")

    # Test save/load with GNN
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model4.save(f.name)
        loaded4 = HierarchicalForecaster.load(f.name)
        print(f"Loaded model sub-models: {loaded4.sub_model_names}")
        assert "gnn" in loaded4.sub_model_names
        os.unlink(f.name)

    print("\n✅ HierarchicalForecaster test passed!")
