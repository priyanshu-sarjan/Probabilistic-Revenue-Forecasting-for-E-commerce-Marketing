#!/usr/bin/env bash
set -euo pipefail

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=== Step 1: Generating Features from Marketing & Supply Data ==="
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet

echo "=== Step 2: Running Probabilistic Forecasts & Causal AI ==="
python src/predict.py \
    --features features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "Done! Predictions written to $OUTPUT_PATH"
