#!/bin/bash
# Run model inference on MATH-500 dataset

set -e

# Default values
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
DATASET="${DATASET:-HuggingFaceH4/MATH-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUTPUT="${OUTPUT:-outputs/math500_inference/qwen3-0.8b-math500-cot.jsonl}"
LIMIT="${LIMIT:-}"

# Build command
CMD="uv run probing-reflection inference"
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "=== Running Inference ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Batch Size: $BATCH_SIZE"
echo "Max Tokens: $MAX_TOKENS"
echo "Output: $OUTPUT"
echo ""

# Run inference
$CMD

echo ""
echo "Inference complete. Results saved to: $OUTPUT"
