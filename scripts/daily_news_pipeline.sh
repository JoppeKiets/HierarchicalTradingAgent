#!/usr/bin/env bash
# Runs the daily news collection and processing pipeline.
#
# Usage:
#   ./scripts/daily_news_pipeline.sh AAPL MSFT
#   ./scripts/daily_news_pipeline.sh --all

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

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

# Pass through any remaining arguments (e.g., --continuous --interval)
EXTRA_ARGS="$@"

echo "--- STEP 1: Collecting news via yfinance ---"
python scripts/collect_news.py $ARGS $EXTRA_ARGS

echo "--- STEP 2: Running Summarization, FinBERT, and Sequence Alignment ---"
# We add --force to ensure we re-align if new news was added.
python scripts/process_collected_news.py $ARGS $EXTRA_ARGS --force

echo "News pipeline complete! New news features are ready for inference/training."
