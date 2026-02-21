#!/bin/bash
# Full pipeline: preprocess → train → evaluate → walk-forward
# Run after daily data update is complete.
#
# Usage:
#   bash scripts/run_full_pipeline.sh                    # Full pipeline
#   bash scripts/run_full_pipeline.sh --skip-preprocess  # Skip phase 0
#   bash scripts/run_full_pipeline.sh --quick            # Quick test run

set -e

PYTHON=".venv/bin/python"
OUTPUT_DIR="models/hierarchical_v2"
LOG_DIR="logs/hierarchical_v2"
EPOCHS=50
QUICK=false
SKIP_PREPROCESS=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            EPOCHS=5
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================="
echo "  Hierarchical Trading Agent Full Pipeline"
echo "============================================="
echo "Output dir: $OUTPUT_DIR"
echo "Epochs:     $EPOCHS"
echo "Quick mode: $QUICK"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"
echo "Log file: $LOGFILE"
echo ""

# Step 1: Preprocess
if [ "$SKIP_PREPROCESS" = false ]; then
    echo "[1/5] Preprocessing features (force rebuild)..."
    $PYTHON train_hierarchical.py \
        --phase 0 \
        --output-dir "$OUTPUT_DIR" \
        --force-preprocess \
        2>&1 | tee -a "$LOGFILE"
    echo "[1/5] Preprocessing complete ✓"
    echo ""
fi

# Step 2: Train all phases
echo "[2/5] Training phases 1-4..."
if [ "$QUICK" = true ]; then
    $PYTHON train_hierarchical.py \
        --phase 1 2 3 4 \
        --epochs $EPOCHS \
        --output-dir "$OUTPUT_DIR" \
        --patience 3 \
        --num-workers 2 \
        2>&1 | tee -a "$LOGFILE"
else
    $PYTHON train_hierarchical.py \
        --phase 1 2 3 4 \
        --output-dir "$OUTPUT_DIR" \
        --num-workers 4 \
        2>&1 | tee -a "$LOGFILE"
fi
echo "[2/5] Training complete ✓"
echo ""

# Step 3: Evaluate
echo "[3/5] Evaluating model..."
$PYTHON scripts/evaluate_hierarchical.py \
    --model "$OUTPUT_DIR/forecaster_final.pt" \
    --output-dir "results/hierarchical_evaluation" \
    --baselines \
    2>&1 | tee -a "$LOGFILE"
echo "[3/5] Evaluation complete ✓"
echo ""

# Step 4: Walk-forward validation
echo "[4/5] Walk-forward validation..."
if [ "$QUICK" = true ]; then
    $PYTHON scripts/walk_forward_hierarchical.py \
        --n-windows 3 \
        --epochs 3 \
        --output-dir "results/walk_forward_hierarchical" \
        2>&1 | tee -a "$LOGFILE"
else
    $PYTHON scripts/walk_forward_hierarchical.py \
        --n-windows 5 \
        --epochs 20 \
        --output-dir "results/walk_forward_hierarchical" \
        2>&1 | tee -a "$LOGFILE"
fi
echo "[4/5] Walk-forward complete ✓"
echo ""

# Step 5: Plot training curves
echo "[5/6] Plotting training curves..."
$PYTHON scripts/plot_training_curves.py \
    --model-dir "$OUTPUT_DIR" \
    --save \
    2>&1 | tee -a "$LOGFILE"
echo "[5/6] Plot complete ✓"
echo ""

# Step 6: Generate predictions
echo "[6/6] Generating stock predictions..."
$PYTHON scripts/predict.py \
    --model "$OUTPUT_DIR/forecaster_final.pt" \
    --top-k 30 \
    --output "results/latest_predictions.json" \
    2>&1 | tee -a "$LOGFILE"
echo "[6/6] Predictions complete ✓"
echo ""

echo "============================================="
echo "  Pipeline complete!"
echo "  Results:      results/hierarchical_evaluation/"
echo "  Walk-forward: results/walk_forward_hierarchical/"
echo "  Predictions:  results/latest_predictions.json"
echo "  Curves:       $OUTPUT_DIR/training_curves.png"
echo "  Model:        $OUTPUT_DIR/forecaster_final.pt"
echo "  Log:          $LOGFILE"
echo "============================================="
