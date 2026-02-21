#!/bin/bash
# Quick memory test: run phase 1+2 only with new memory settings

set -e

PYTHON=".venv/bin/python"
OUTPUT_DIR="models/hierarchical_test"
LOG_DIR="logs/hierarchical_test"

echo "Memory-optimized training test"
echo "================================"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Run phases 1-2 only (fastest test)
$PYTHON train_hierarchical.py \
    --phase 1 2 \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --patience 2 \
    2>&1 | tee "$LOG_DIR/test.log"

echo ""
echo "================================"
echo "Test completed successfully!"
echo "Check: $LOG_DIR/test.log"
echo ""
echo "If this ran without OOM, you can now run:"
echo "  python train_hierarchical.py --phase 0 1 2 3 --output-dir models/hierarchical_v2 --force-preprocess"
