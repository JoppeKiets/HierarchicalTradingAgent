#!/usr/bin/env bash
# Runs the daily news collection and processing pipeline.
#
# This script orchestrates:
#   1. News collection (Yahoo Finance)
#   2. Article summarization (abstractive)
#   3. FinBERT encoding (optional, default: skip for speed)
#   4. Sequence alignment (for training)
#
# Usage:
#   # Collect and process for specific tickers (no embeddings, fast)
#   ./scripts/daily_news_pipeline.sh AAPL MSFT
#
#   # Collect and process for all tickers with news
#   ./scripts/daily_news_pipeline.sh --all
#
#   # Collect, process, AND encode to FinBERT (slow, ~2-3 hours)
#   ./scripts/daily_news_pipeline.sh --all --with-embeddings
#
#   # Continuous collection (runs indefinitely, restarts every 4 hours)
#   ./scripts/daily_news_pipeline.sh --all --continuous --interval 14400 --with-embeddings
#
# Environment:
#   - Uses CUDA GPU if available (RTX 5080)
#   - Batch size: 128 for FinBERT (adjust for your GPU)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Parse arguments
ENCODE_EMBEDDINGS=false
ARGS=""

if [[ "$1" == "--all" ]]; then
    ARGS="--all"
    shift
elif [[ "$1" == "--tickers" ]]; then
    shift
    # consume tickers until we hit another flag
    TICKERS=""
    while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
      TICKERS="$TICKERS $1"
      shift
    done
    ARGS="--tickers $TICKERS"
fi

# Extract --with-embeddings flag if present
EXTRA_ARGS=""
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--with-embeddings" ]]; then
        ENCODE_EMBEDDINGS=true
        shift
    else
        EXTRA_ARGS="$EXTRA_ARGS $1"
        shift
    fi
done

echo "========================================================================="
echo "NEWS DATA COLLECTION AND PROCESSING PIPELINE"
echo "========================================================================="
echo ""
echo "Configuration:"
echo "  Tickers:        $ARGS"
echo "  Encode to FinBERT: $ENCODE_EMBEDDINGS"
echo "  Extra args:     $EXTRA_ARGS"
echo ""

echo "--- STEP 1: Collecting news via yfinance ---"
python scripts/collect_news.py $ARGS $EXTRA_ARGS

echo ""
echo "--- STEP 2: Running Summarization and Sequence Alignment ---"
# We add --force to ensure we re-align if new news was added.
python scripts/process_collected_news.py $ARGS $EXTRA_ARGS --force

if [[ "$ENCODE_EMBEDDINGS" == true ]]; then
    echo ""
    echo "--- STEP 3: Encoding to FinBERT embeddings (this will take 2-3 hours) ---"
    echo "Starting FinBERT batch encoding..."
    python scripts/preprocess_news_embeddings.py --batch-size 128 --device cuda --force
    echo "FinBERT encoding complete!"
else
    echo ""
    echo "--- STEP 3: FinBERT encoding SKIPPED (use --with-embeddings to enable) ---"
fi

echo ""
echo "========================================================================="
echo "News pipeline complete!"
echo ""
if [[ "$ENCODE_EMBEDDINGS" == false ]]; then
    echo "Note: FinBERT embeddings were NOT computed (add --with-embeddings to enable)"
    echo "      New tickers need embeddings before using --use-news flag in training"
fi
echo "========================================================================="
