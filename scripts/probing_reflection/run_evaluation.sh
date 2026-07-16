#!/bin/bash
# Evaluate model outputs using LLM judge

set -e

# Check required argument
if [ -z "$1" ]; then
    echo "Usage: $0 <jsonl_path> [options]"
    echo ""
    echo "Arguments:"
    echo "  jsonl_path    Path to JSONL file containing model outputs"
    echo ""
    echo "Options (set as environment variables):"
    echo "  MODEL         Judge model name (default: Qwen/Qwen3.5-27B)"
    echo "  BATCH_SIZE    Batch size for evaluation (default: 8)"
    echo "  THRESHOLD     Confidence threshold (default: 0.7)"
    echo "  OUTPUT        Output report path (optional)"
    exit 1
fi

JSONL_PATH="$1"

# Default values
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
THRESHOLD="${THRESHOLD:-0.7}"
OUTPUT="${OUTPUT:-}"

# Build command
CMD="uv run probing-reflection evaluate $JSONL_PATH"
CMD="$CMD --model $MODEL"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --confidence-threshold $THRESHOLD"

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

echo "=== Running Evaluation ==="
echo "Input: $JSONL_PATH"
echo "Judge Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Confidence Threshold: $THRESHOLD"
echo ""

# Run evaluation
$CMD

echo ""
echo "Evaluation complete."
