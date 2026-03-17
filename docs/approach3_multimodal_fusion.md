# Approach 3: Multimodal Fusion — Injecting News Directly into TFT

## Overview

Approach 3 fuses news embeddings **directly into the Temporal Fusion Transformer's
Variable Selection Network (VSN)**, making news a first-class input alongside
price/volume features. Instead of being a separate sub-model whose prediction
and embedding feed the MetaMLP (Approach 2), the news signal is injected
*inside* the TFT itself, enabling the attention mechanism to learn cross-modal
interactions between price and text at every time step.

This is the most architecturally invasive of the three approaches but offers
the deepest integration: the TFT can learn that, for example, a sudden spike
in negative news sentiment *combined with* an earnings gap-down deserves
different attention than the same gap-down without news.

---

## Architecture Changes

### Current TFT Input (before fusion)

```
daily_features: (B, T, 51)
    ↓
[ Variable Selection Network ]  →  per-feature importance weights
    ↓
[ LSTM Encoder ]
    ↓
[ Multi-Head Self-Attention ]
    ↓
[ Gated Residual Network ]
    ↓
prediction (scalar) + embedding (64-dim)
```

### Proposed TFT Input (after fusion)

```
daily_features: (B, T, 51)  ─┐
                              ├─→ [Concat / Projection] → (B, T, D_fused)
news_features:  (B, T, D_news) ─┘
    ↓
[ Variable Selection Network ]  →  per-feature importance weights
    ↓                              (now includes news features)
[ LSTM Encoder ]
    ↓
[ Multi-Head Self-Attention ]
    ↓
[ Gated Residual Network ]
    ↓
prediction (scalar) + embedding (64-dim)
```

The key insight: the VSN already performs **learned feature selection** —
it can now learn which news features matter and when, in context of the
price features.

---

## Detailed Implementation Plan

### Step 1: News Feature Projection

The raw FinBERT embeddings (768-dim per day) are too large to concatenate
directly with the 51 daily features. We need a projection layer:

```python
# In a new class: MultimodalTFT(RegressionTFT)

class NewsProjection(nn.Module):
    """Project daily news embeddings to match price feature scale."""

    def __init__(self, news_dim=774, proj_dim=32, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(news_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, news: torch.Tensor) -> torch.Tensor:
        # news: (B, T, 774) → (B, T, 32)
        return self.proj(news)
```

This projects the 774-dim news vector (768 FinBERT + 6 sentiment) down to
32 dimensions per day. The fused input to the VSN becomes (B, T, 51 + 32 = 83).

**Design choice: projection dimension (32)**

- Too small (8–16): loses nuance in FinBERT embeddings
- Too large (64–128): dominates the price features, unstable training
- Sweet spot (24–48): enough capacity without overwhelming price signal
- The projection is learned end-to-end, so the network decides what to keep

### Step 2: Modify the TFT's Variable Selection Network

The VSN in `TemporalFusionTransformer` currently processes `input_dim` features.
After fusion, `input_dim` becomes `original_dim + proj_dim`:

**File: `src/models/temporal_fusion_transformer.py`**

```python
# Current TFTConfig
@dataclass
class TFTConfig:
    input_dim: int = 51      # ← this needs to accept 83 (51 + 32)
    hidden_dim: int = 128
    ...
```

The VSN uses `input_dim` to create per-feature GRNs (Gated Residual Networks).
With 83 features instead of 51, the VSN will create 83 GRNs:

- Features 0–50: price/volume/technical indicators (as before)
- Features 51–82: projected news embedding dimensions

The VSN's softmax-weighted feature selection will now span both modalities,
letting the model learn:
- "Today's news projection dim 7 is highly relevant for this prediction"
- "When VIX is high (regime), pay more attention to news features 20–25"

**No structural changes to the VSN code are needed** — just pass a larger
`input_dim`. The TFT architecture is already feature-agnostic.

### Step 3: Create the MultimodalTFT Wrapper

```python
class MultimodalTFT(nn.Module):
    """TFT with fused news input via projection + VSN."""

    def __init__(
        self,
        price_input_dim: int = 51,
        news_input_dim: int = 774,
        news_proj_dim: int = 32,
        seq_len: int = 720,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.news_proj = NewsProjection(news_input_dim, news_proj_dim, dropout)

        # The TFT receives the fused input
        fused_dim = price_input_dim + news_proj_dim
        self.tft = RegressionTFT(
            input_dim=fused_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )

    def forward(self, price_x, news_x):
        # price_x: (B, T, 51)
        # news_x:  (B, T, 774)
        news_proj = self.news_proj(news_x)    # (B, T, 32)
        fused = torch.cat([price_x, news_proj], dim=-1)  # (B, T, 83)
        return self.tft(fused)
```

### Step 4: Integration into HierarchicalForecaster

Two options for integrating MultimodalTFT:

**Option A: Replace TFT_D entirely**

```python
# In HierarchicalForecaster.__init__:
if use_multimodal_tft:
    self.tft_d = MultimodalTFT(
        price_input_dim=cfg.daily_input_dim,
        news_input_dim=cfg.news_input_dim,
        ...
    )
    # TFT_D now requires news_x in forward pass
```

- Pros: clean, no extra model
- Cons: TFT_D can no longer be trained independently in Phase 1
  (it needs news data from Phase 1.5)

**Option B: Add MultimodalTFT as a 6th model alongside TFT_D** (recommended)

```python
# In HierarchicalForecaster.__init__:
self.tft_d = RegressionTFT(...)    # Trained in Phase 1 (no news)
self.mm_tft = MultimodalTFT(...)   # Trained in Phase 1.5 (with news)
```

- Pros: TFT_D still trains independently; MM_TFT adds a new perspective
- Cons: more parameters, MetaMLP needs n_sub_models=6
- This is compatible with Approach 2 — you'd have 6 sub-models feeding MetaMLP

**Recommendation: Option B** — it's additive, doesn't break existing training,
and gives the MetaMLP more diverse signals.

### Step 5: Training Phase Changes

```
Phase 0:    Preprocess (features + FinBERT embeddings)
Phase 1:    Train LSTM_D + TFT_D on daily data (unchanged)
Phase 1.5a: Train NewsEncoder on news embeddings (Approach 2, unchanged)
Phase 1.5b: Train MultimodalTFT on daily + news data (NEW)
Phase 2:    Train LSTM_M + TFT_M on minute data (unchanged)
Phase 3:    Train MetaMLP on frozen sub-model outputs (6 models now)
Phase 4:    Joint fine-tuning (all 6 + MetaMLP)
```

Phase 1.5b is new: the MultimodalTFT needs both daily price features and
news embeddings aligned at training time. This uses the same `LazyNewsDataset`
but returns both price and news sequences per sample.

### Step 6: New Aligned DataLoader

A new dataset class that yields both modalities per sample:

```python
class LazyMultimodalDataset(Dataset):
    """Yields (price_seq, news_seq, target, date, ticker) tuples.

    Loads from both daily cache and news cache, aligned by (ticker, row_index).
    """

    def __getitem__(self, idx):
        ticker, row = self.index[idx]

        price_feat = np.load(f"daily/{ticker}_features.npy", mmap_mode="r")
        news_feat  = np.load(f"news_sequences/{ticker}_features.npy", mmap_mode="r")
        target     = np.load(f"daily/{ticker}_targets.npy", mmap_mode="r")

        price_seq = price_feat[row - seq_len : row]   # (T, 51)
        news_seq  = news_feat[row - seq_len : row]    # (T, 774)
        tgt = float(target[row])

        return (torch.from_numpy(price_seq),
                torch.from_numpy(news_seq),
                torch.tensor(tgt), ordinal_date, ticker)
```

This is straightforward because `preprocess_news_ticker()` (from Approach 2)
already aligns news sequences 1:1 with daily rows.

---

## What Changes from Approach 2 (Current Implementation)

### Files to Modify

| File | Change |
|------|--------|
| `src/models/temporal_fusion_transformer.py` | No changes needed (input_dim is already configurable) |
| `src/hierarchical_models.py` | Add `MultimodalTFT` class, add it to `HierarchicalForecaster`, update `MetaMLP` to `n_sub_models=6` |
| `src/news_data.py` | Add `LazyMultimodalDataset` + `create_multimodal_dataloaders()` |
| `train_hierarchical.py` | Add Phase 1.5b, update Meta/Joint to handle 6 models |

### Files That Don't Change

| File | Why |
|------|-----|
| `src/news_encoder.py` | Still used as-is (Approach 2 is kept) |
| `scripts/preprocess_news_embeddings.py` | Same FinBERT preprocessing |
| `src/enhanced_features.py` | Price features unchanged |
| `src/hierarchical_data.py` | Daily/minute pipelines unchanged |

### New Files

| File | Purpose |
|------|---------|
| `src/multimodal_tft.py` | `NewsProjection` + `MultimodalTFT` wrapper |

---

## Advantages over Approach 2

1. **Cross-modal attention**: The TFT's self-attention can learn temporal
   patterns *across* price and news features simultaneously. Approach 2
   treats news as an isolated time series.

2. **Feature selection**: The VSN explicitly learns which news embedding
   dimensions are informative for each prediction, and this selection is
   conditioned on the current price context.

3. **Temporal alignment**: Price and news features are processed through
   the same LSTM encoder, so the model can learn lead/lag relationships
   (e.g., news on day T-2 affecting price on day T).

4. **Interpretability**: The VSN's feature importance weights show exactly
   which news dimensions matter and when — you can visualize this.

## Disadvantages / Risks

1. **Increased complexity**: The TFT now has 83 input features instead of
   51, increasing the number of GRNs in the VSN and training time.

2. **Potential overfitting**: With more features, especially noisy FinBERT
   projections, the model may overfit. Strong regularization is needed:
   - Higher dropout on the news projection (0.2–0.3)
   - Feature dropout (randomly zero out news dims during training)
   - Warmup: train on price-only for N epochs, then add news (curriculum)

3. **Training difficulty**: End-to-end training with both modalities from
   the start can be unstable. The recommended mitigation:
   - Pre-train the news projection layer (using the NewsEncoder's weights)
   - Freeze the news projection for the first 10–20 epochs
   - Then unfreeze and train end-to-end

4. **Harder to debug**: When the model performs poorly, it's harder to
   tell if the issue is in the price features, news features, or their
   interaction. The separate Approach 2 models are easier to diagnose.

---

## Recommended Implementation Order

1. **Implement and validate Approach 2 first** (already done) — this gives
   a baseline for the NewsEncoder and proves the news data pipeline works.

2. **Create `MultimodalTFT`** as a standalone class in `src/multimodal_tft.py`.
   Test it independently with synthetic data.

3. **Add `LazyMultimodalDataset`** to `src/news_data.py`. Verify alignment
   with a few tickers.

4. **Add MultimodalTFT to HierarchicalForecaster** as a 6th model (Option B).
   Update MetaMLP to `n_sub_models=6`.

5. **Add Phase 1.5b** to `train_hierarchical.py`. Consider initializing the
   news projection from the Approach 2 NewsEncoder's `input_proj` weights.

6. **Run ablation**: Compare
   - Approach 2 only (5 models: LSTM_D, TFT_D, LSTM_M, TFT_M, News)
   - Approach 3 only (6 models: LSTM_D, TFT_D, MM_TFT, LSTM_M, TFT_M, News)
   - Both combined (6 models)
   to see if the fusion adds value over the separate NewsEncoder.

---

## Advanced Extensions (Future)

### Cross-Attention Instead of Concatenation

Instead of projecting news to 32-dim and concatenating with price features,
use cross-attention between price and news modalities:

```
price_features → Q (queries)
news_embeddings → K, V (keys, values)
CrossAttention(Q, K, V) → news-informed price features
```

This is closer to how modern multimodal transformers work (e.g., Flamingo,
BLIP-2) and avoids the information bottleneck of the 32-dim projection.

### Gated Fusion

Add a learned gate that controls how much news signal to mix in:

```python
gate = sigmoid(W_price @ price_features + W_news @ news_proj + bias)
fused = gate * news_proj + (1 - gate) * zero_news
final_input = concat(price_features, fused)
```

This lets the model learn to ignore news when it's not informative
(e.g., no news days, or stale news).

### Minute-Level News Fusion

If intraday news timestamps are available, the same fusion can be applied
to the minute models, allowing ultra-fast reaction to breaking news.
This requires:
- Intraday news collection (see Yahoo Finance collector)
- Minute-level alignment of news embeddings
- A `MultimodalMinuteTFT` variant

---

## Configuration Defaults

```python
@dataclass
class MultimodalTFTConfig:
    # News projection
    news_input_dim: int = 774
    news_proj_dim: int = 32       # Projected news features per timestep
    news_proj_dropout: float = 0.25

    # Fused TFT
    fused_input_dim: int = 83     # 51 price + 32 news
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    embedding_dim: int = 64
    seq_len: int = 720

    # Training
    warmup_epochs: int = 10       # Epochs with frozen news projection
    feature_dropout: float = 0.1  # Random feature masking rate
```

---

## Summary

| Aspect | Approach 2 (Current) | Approach 3 (This Doc) |
|--------|----------------------|----------------------|
| News model | Separate LSTM encoder | Fused into TFT's VSN |
| Integration | MetaMLP level | Feature level |
| Cross-modal learning | No (isolated) | Yes (shared attention) |
| Training complexity | Lower | Higher |
| Interpretability | Medium | High (VSN weights) |
| Risk of overfitting | Lower | Higher |
| Implementation effort | ✅ Done | ~2-3 days |
| Recommended order | First | Second |
