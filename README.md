# Trading Agent

Hierarchical stock forecaster combining daily and minute-level models with market regime features.

## Architecture

5-model hierarchical system trained in phases:

| Model | Data | Purpose |
|-------|------|---------|
| LSTM_D | Daily (720-day sequences) | Long-term patterns |
| TFT_D | Daily (720-day sequences) | Attention-based daily features |
| LSTM_M | Minute (780-bar sequences) | Intraday patterns |
| TFT_M | Minute (780-bar sequences) | Attention-based minute features |
| MetaMLP | Sub-model outputs + regime | Ensemble combination |

## Quick Start

```bash
# Update daily data
python scripts/update_daily_data.py

# Full pipeline (preprocess → train → evaluate)
bash scripts/run_full_pipeline.sh

# Quick test run
bash scripts/run_full_pipeline.sh --quick

# Train directly
python train_hierarchical.py --phase 0 1 2 3

# Evaluate
python scripts/evaluate_hierarchical.py --model models/hierarchical_v3/forecaster_final.pt

# Predict
python scripts/predict.py --model models/hierarchical_v3/forecaster_final.pt --top-k 20
```

## Configuration

Edit `config.py` to switch presets:

```python
ACTIVE_CONFIG = "massive"  # quick | standard | production | massive
```

Or override on the command line:

```bash
python train_hierarchical.py --config massive --epochs 50
```

## Project Layout

```
train_hierarchical.py       # Main training entry point
config.py                   # Training configuration presets
collect_minute_data.py      # Minute data collection service

src/
  hierarchical_data.py      # Data loading, feature caching, lazy datasets
  hierarchical_models.py    # LSTM, TFT, MetaMLP model definitions
  enhanced_features.py      # 50+ daily feature computations
  regime_features.py        # Market regime extraction
  yahoo_data_loader.py      # Yahoo Finance price data
  minute_data_loader.py     # Minute-level data loading
  backtester.py             # Backtesting engine
  baseline_strategies.py    # Buy-and-hold, momentum, etc.
  ticker_universe.py        # Train/eval ticker splits
  ticker_metadata.py        # Sector/industry metadata
  features/
    fundamental_features.py # P/E, margins, leverage, etc.
    macro_features.py       # VIX, yields, credit spreads, etc.
    sentiment_features.py   # News sentiment, analyst ratings, earnings
  models/
    temporal_fusion_transformer.py

scripts/
  run_full_pipeline.sh      # End-to-end pipeline
  update_daily_data.py      # Incremental daily data update
  evaluate_hierarchical.py  # Model evaluation + baselines
  walk_forward_hierarchical.py  # Walk-forward validation
  predict.py                # Generate predictions from trained model
  plot_training_curves.py   # Visualize training loss/IC
  data_quality_check_v2.py  # Feature cache quality checks

data/                       # Price history, feature cache, minute data
models/                     # Trained model checkpoints
results/                    # Evaluation outputs
logs/                       # Training logs
```

## Requirements

```bash
pip install -r requirements.txt
```
