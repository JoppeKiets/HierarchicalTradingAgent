# Hierarchical Trading Agent

A hierarchical stock return forecaster that combines daily and minute-level models with market regime features. The system trains 5 neural networks in phases — two LSTMs, two Temporal Fusion Transformers, and a Meta MLP that learns when to trust each sub-model based on market conditions.

## Architecture

```
                    ┌─────────────┐
                    │  Meta MLP   │ ── final prediction
                    │  (Phase 3)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     4 predictions   4 embeddings   8 regime
       (scalars)      (64-dim)      features
              │            │
    ┌─────────┴─────────┐  │
    │                   │  │
┌───┴───┐  ┌───┴───┐  ┌┴──┴──┐  ┌───┴───┐
│LSTM_D │  │ TFT_D │  │LSTM_M│  │ TFT_M │
│Phase 1│  │Phase 1│  │Phase 2│  │Phase 2│
└───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
    │          │          │           │
  Daily      Daily      Minute     Minute
  (720d)     (720d)     (780 bar)  (780 bar)
```

| Model | Type | Input | Purpose |
|-------|------|-------|---------|
| LSTM_D | LSTM + regression head | Daily OHLCV (720-day sequences, ~51 features) | Long-term daily patterns |
| TFT_D | Temporal Fusion Transformer | Daily OHLCV (720-day sequences, ~51 features) | Interpretable attention-based daily features |
| LSTM_M | LSTM + regression head | Minute bars (780-bar sequences, ~23 features) | Intraday patterns |
| TFT_M | Temporal Fusion Transformer | Minute bars (780-bar sequences, ~23 features) | Interpretable attention-based minute features |
| MetaMLP | Feed-forward network | 4 predictions + 4 embeddings + 8 regime features | Regime-conditioned ensemble combination |

### Training Phases

1. **Phase 0 — Preprocess**: Compute features for all tickers, save as memory-mapped `.npy` files.
2. **Phase 1 — Daily models**: Train LSTM_D and TFT_D on daily data (up to 100 epochs each).
3. **Phase 2 — Minute models**: Train LSTM_M and TFT_M on minute data (up to 100 epochs each).
4. **Phase 3 — Meta model**: Freeze sub-models, train MetaMLP on their combined outputs + regime features.
5. **Phase 4 — Joint fine-tuning** *(optional)*: Unfreeze all 5 models, fine-tune end-to-end with a small learning rate.

### Features

**Daily features (~51 dims):** Multi-horizon returns, realized/Parkinson/Garman-Klass volatility, SMA crossovers, ADX, RSI, MACD, Stochastic, Williams %R, OBV, VWAP ratio, volume momentum, market microstructure (gap, range, body ratio), calendar encoding, lagged news sentiment.

**Minute features (~23 dims):** 1-min returns, log-volume, spread, VWAP proxy, RSI, MACD, Bollinger Bands, ATR, ADX, EMA deviations, time-of-day sin/cos encoding, lagged news sentiment.

**Regime features (8 dims):** SPY 5-day return, SPY 20-day realized vol, VIX proxy level/change, TLT return, GLD return, SPY breadth (distance from 50-day SMA), SPY-TLT rolling correlation. All rolling z-scored and lagged 1 day.

### Key Design Decisions

- **Huber loss** (δ=0.01) instead of MSE — robust to fat-tailed return distributions.
- **IC-based early stopping** — tracks Information Coefficient (Pearson correlation between predicted and actual returns) rather than raw loss.
- **Temporal splitting** — data is split by time (70/15/15) not by ticker, preventing future leakage.
- **Per-bar minute targets** — each minute bar predicts its own forward return (`close[i+390] / close[i] - 1`), not a duplicated next-day return, so the model must learn actual intraday patterns.
- **Lazy memory-mapped datasets** — `.npy` files are memory-mapped at training time, giving near-zero RAM overhead regardless of dataset size.
- **Meta trains on val split** — the MetaMLP trains on data unseen by the sub-models (val split) and validates on the test split, preventing overfitting to the sub-models' training distribution.

---

## Replicating the Dataset

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥8 GB VRAM (16 GB recommended)
- ~50 GB disk space for full dataset + feature cache
- Internet connection for Yahoo Finance API

### 1. Clone and Install

```bash
git clone https://github.com/JoppeKiets/HierarchicalTradingAgent.git
cd HierarchicalTradingAgent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Daily Data

The daily data lives in `data/organized/{TICKER}/price_history.csv`. Each file contains OHLCV columns with a `date` column. You need to populate this directory for every ticker you want to train on.

**Download from Yahoo Finance:**

```bash
# Downloads full price history for all tickers in the built-in universe (~7,600+).
# Incremental — if a ticker already has data, only fetches missing dates.
python scripts/update_daily_data.py

# Force full re-download for specific tickers
python scripts/update_daily_data.py --tickers AAPL MSFT GOOGL --force

# Dry run to see what needs updating
python scripts/update_daily_data.py --dry-run
```

**Expected format** for `data/organized/{TICKER}/price_history.csv`:

```csv
date,open,high,low,close,adj close,volume
2026-02-13 00:00:00,262.01,262.23,255.45,255.78,,56229900
2026-02-12 00:00:00,275.59,275.72,260.18,261.73,,81077200
...
```

> **Note:** `adj close` can be empty/NaN for recent dates — the code falls back to `close` automatically.

### 3. Collect Minute Data

Yahoo Finance only provides ~7 days of 1-minute data at a time, so minute data must be collected **incrementally over time**. The longer you run the collector, the more minute history you accumulate.

**One-time collection:**

```bash
# Collect latest ~7 days of minute data for all tickers with daily data
python collect_minute_data.py --all-organized

# Or for specific tickers only
python collect_minute_data.py --tickers AAPL MSFT GOOGL AMZN

# Check what you have
python collect_minute_data.py --status
```

**Continuous collection (recommended):**

Set up a systemd service that collects minute data every hour, rotating through all tickers:

```bash
# Install and start the collector service
sudo bash infra/setup_collector_service.sh
sudo systemctl enable minute-collector
sudo systemctl start minute-collector

# Check status
sudo systemctl status minute-collector

# View logs
tail -f logs/minute_collector.log
```

The service is configured in `infra/minute-collector.service`. By default it collects all tickers in `data/organized/` in batches of 500 per hour.

> **Important:** Minute data accumulates over weeks and months. With only a few weeks of data, the minute models have very limited training signal. The daily models work well from day one; the minute models become increasingly useful after ~2–3 months of collection.

**Minute data is stored as:** `data/minute_history/{TICKER}.parquet` with columns: `timestamp, open, high, low, close, volume, rsi, macd, macd_signal, macd_diff, ema_5, ema_12, ema_26, bb_upper, bb_lower, bb_mid, atr, adx`.

### 4. (Optional) News Data

If you have news articles, place them at `data/organized/{TICKER}/news_articles.csv` with columns:

```csv
Date,Article_title,Article,Stock_symbol
2023-12-16 22:00:00 UTC,My 6 Largest Portfolio Holdings...,After an absolute disaster...,AAPL
```

The pipeline computes lightweight lexicon-based sentiment features (lagged by 1 day). If no news file exists for a ticker, the news features are zero-filled.

---

## Training

### Quick Start

```bash
# Full pipeline: preprocess → train daily → train minute → train meta → evaluate
python train_hierarchical.py --phase 0 1 2 3 --output-dir models/my_run --force-preprocess
```

This will:
1. Discover all tickers with sufficient daily (≥750 rows) and minute (≥780 bars) data
2. Compute and cache features to `data/feature_cache/`
3. Train LSTM_D + TFT_D on daily data
4. Train LSTM_M + TFT_M on minute data
5. Train the Meta MLP on frozen sub-model outputs
6. Evaluate all models on the test set
7. Save the final model to `models/my_run/forecaster_final.pt`

### Command-Line Options

```bash
python train_hierarchical.py \
  --phase 0 1 2 3           # Phases to run (0=preprocess, 1=daily, 2=minute, 3=meta, 4=finetune)
  --output-dir models/v1    # Where to save checkpoints
  --resume models/v1/checkpoint_phase2.pt  # Resume from checkpoint
  --epochs 100              # Override epoch count for all phases
  --batch-size 32           # Override batch size
  --lr 3e-4                 # Override learning rate
  --daily-seq-len 720       # Daily lookback window (trading days)
  --minute-seq-len 780      # Minute lookback window (bars)
  --daily-stride 5          # Stride for daily sequence sampling
  --minute-stride 30        # Stride for minute sequence sampling
  --split-mode temporal     # "temporal" (split by time) or "ticker" (split by company)
  --loss huber              # "huber" or "mse"
  --patience 15             # Early stopping patience (epochs)
  --early-stop-metric ic    # "ic" (Information Coefficient) or "loss"
  --num-workers 4           # DataLoader workers
  --force-preprocess        # Recompute features even if cache exists
  --low-memory              # Halve batch sizes, enable AMP
```

### Run Individual Phases

```bash
# Preprocess only (useful for debugging features)
python train_hierarchical.py --phase 0 --force-preprocess

# Train only daily models (skip minute + meta)
python train_hierarchical.py --phase 1 --output-dir models/daily_only

# Train meta model (requires phase 1 + 2 checkpoints)
python train_hierarchical.py --phase 3 --resume models/my_run/checkpoint_phase2.pt

# Optional joint fine-tuning
python train_hierarchical.py --phase 4 --resume models/my_run/checkpoint_phase3.pt
```

---

## Closed-Loop Self-Improvement (Fully Autonomous)

The repository includes an end-to-end autonomous loop script at
`scripts/closed_loop_self_improvement.py` that runs:

1. **Deploy agents** with current model (`run_swing_pipeline.py`)
2. **Collect outcomes** in `data/trade_journal/trade_journal.jsonl`
   (fills `actual_return`, `exit_price`, `exit_timestamp` after an N-day horizon)
3. **Analyst auto feature engineering**:
  - proposes feature combinations,
  - auto-generates feature code (`src/features/generated_features.py`),
  - runs ablation IC tests,
  - promotes only features that improve IC
4. **Generate Critic training report** with:
   - weak regimes,
   - unreliable tickers,
   - regime-feature drift signals
5. **Auto-trigger retraining**:
  - normal case: selective retrain (`--phase 3 4`)
  - if new features were promoted: full retrain (`--phase 0 1 2 3 4 --force-preprocess`)
6. **Redeploy** the newly trained model automatically

### One-cycle run

```bash
python scripts/closed_loop_self_improvement.py \
  --model models/hierarchical_v10/forecaster_final.pt \
  --iterations 1 \
  --outcome-horizon-days 5 \
  --feature-min-ic-improvement 0.001 \
  --top-n 20 \
  --min-agreement 0.50 \
  --min-return 0.001
```

### Daemon-style multi-cycle run (no human in loop)

```bash
python scripts/closed_loop_self_improvement.py \
  --model models/hierarchical_v10/forecaster_final.pt \
  --iterations 30 \
  --sleep-hours 24 \
  --outcome-horizon-days 5
```

### Outputs

- Training reports: `data/closed_loop/reports/latest_training_report.json`
- Timestamped reports: `data/closed_loop/reports/training_report_*.json`
- Feature-engineering reports: `data/feature_feedback/generated_feature_reports/latest_feature_engineering.json`
- Accepted generated features: `data/feature_feedback/accepted_generated_features.json`
- Generated feature code: `src/features/generated_features.py`
- New retrained models: `models/closed_loop/selective_*/forecaster_final.pt`

### Useful feature-engineering flags

- `--enable-feature-engineering` / `--no-enable-feature-engineering`
- `--feature-min-ic-improvement` (default `0.001`)
- `--feature-max-candidates` (default `5`)
- `--feature-max-tickers-sample` (default `200`)
- `--feature-max-rows-per-ticker` (default `250`)


### Background Training

For long runs, use `nohup` so training survives terminal disconnection:

```bash
nohup python train_hierarchical.py \
  --phase 0 1 2 3 \
  --output-dir models/my_run \
  --force-preprocess \
  > logs/my_run_nohup.log 2>&1 &

# Monitor progress
tail -f logs/my_run_nohup.log
```

### Training Time Estimates

On an RTX 5080 (16 GB VRAM) with ~600 tickers:

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 0 (preprocess) | ~10 min | One-time; cached to disk |
| Phase 1 (daily models) | ~20–30 hours | Two models × ~100 epochs × 22K batches |
| Phase 2 (minute models) | ~1–3 hours | Depends on amount of minute data |
| Phase 3 (meta model) | ~15 min | Small network, few epochs |
| **Total** | **~24–34 hours** | |

---

## Evaluation

```bash
# Evaluate a trained model against baselines
python scripts/evaluate_hierarchical.py --model models/my_run/forecaster_final.pt

# Generate top-K stock predictions
python scripts/predict.py --model models/my_run/forecaster_final.pt --top-k 20

# Walk-forward validation
python scripts/walk_forward_hierarchical.py --model models/my_run/forecaster_final.pt
```

### Key Metrics

- **IC** (Information Coefficient): Pearson correlation between predicted and actual returns. IC > 0.05 is considered meaningful for daily return prediction.
- **Rank IC**: Spearman rank correlation — more robust to outliers.
- **Directional Accuracy**: Fraction of correct up/down predictions. > 0.52 is useful.

---

## Project Layout

```
train_hierarchical.py          # Main training entry point
collect_minute_data.py         # Minute data collection service
requirements.txt               # Python dependencies

infra/
  minute-collector.service     # systemd service — minute data collector
  news-collector.service       # systemd service — news collector
  news-embedding-batch.service # systemd service — FinBERT batch encoder
  news-embedding-batch.timer   # Timer — weekly Sunday 2 AM
  paper-trader.service         # systemd service — daily signal generation
  paper-trader.timer           # Timer — Mon–Fri 09:35 ET
  paper-trader-intraday.service# systemd service — intraday stop/TP checker
  paper-trader-intraday.timer  # Timer — every 15 min during market hours
  setup_collector_service.sh   # Install minute-collector service
  setup_news_collector_service.sh  # Install news-collector service
  setup_news_embedding_batch_service.sh  # Install FinBERT batch service
  setup_paper_trader.sh        # Install all paper-trader units

src/
  hierarchical_data.py         # Data loading, feature caching, lazy datasets
  hierarchical_models.py       # LSTM, TFT, MetaMLP, HierarchicalForecaster
  enhanced_features.py         # 50+ daily feature computations
  regime_features.py           # Market regime extraction
  yahoo_data_loader.py         # Yahoo Finance price/news data
  minute_data_loader.py        # Minute-level data + technical indicators
  backtester.py                # Backtesting engine
  baseline_strategies.py       # Buy-and-hold, momentum, etc.
  ticker_universe.py           # Train/eval ticker splits
  ticker_metadata.py           # Sector/industry metadata
  features/
    fundamental_features.py    # P/E, margins, leverage
    macro_features.py          # VIX, yields, credit spreads
    sentiment_features.py      # News sentiment, analyst ratings
  models/
    temporal_fusion_transformer.py  # TFT implementation

scripts/
  run_full_pipeline.sh         # End-to-end pipeline script
  update_daily_data.py         # Incremental daily data update from Yahoo Finance
  evaluate_hierarchical.py     # Model evaluation + baseline comparison
  walk_forward_hierarchical.py # Walk-forward validation
  predict.py                   # Generate predictions from trained model
  plot_training_curves.py      # Visualize training loss/IC curves
  plot_results.py              # Diagnostic plots (auto-generated after training)
  data_quality_check_v2.py     # Feature cache quality checks

data/
  organized/{TICKER}/          # Daily data per ticker
    price_history.csv          #   OHLCV daily bars
    news_articles.csv          #   News articles (optional)
    metadata.json              #   Sector, description
  minute_history/              # Minute data (parquet files)
    {TICKER}.parquet           #   1-min OHLCV + technical indicators
    collection_metadata.json   #   Collection tracking
  feature_cache/               # Preprocessed features (generated by Phase 0)
    daily/{TICKER}_*.npy       #   Memory-mapped daily features/targets/dates
    minute/{TICKER}_*.npy      #   Memory-mapped minute features/targets/dates
    metadata.json

models/                        # Trained model checkpoints
logs/                          # Training logs
```

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch, pandas, numpy, yfinance, scikit-learn, scipy, ta (technical analysis), pyarrow, beautifulsoup4, matplotlib.
