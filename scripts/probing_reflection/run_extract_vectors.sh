#!/bin/bash
# Extract steering vectors from model activations

set -e

# Check required argument
if [ -z "$1" ]; then
    echo "Usage: $0 <input_jsonl> --layers <layers> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_jsonl   Path to input JSONL file with model outputs"
    echo ""
    echo "Required Options:"
    echo "  --layers      Layer indices, comma-separated (e.g., '10,15,20')"
    echo ""
    echo "Optional (set as environment variables):"
    echo "  MODEL         Model name (default: Qwen/Qwen2.5-0.5B)"
    echo "  OUTPUT        Output .pt file path (default: steering_vectors.pt)"
    echo "  MIN_SAMPLES   Minimum samples in R/N sets (default: 10)"
    echo "  BATCH_SIZE    Batch size for extraction (default: 4)"
    exit 1
fi

INPUT_PATH="$1"
shift

# Parse layers argument
LAYERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers|-l)
            LAYERS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$LAYERS" ]; then
    echo "Error: --layers argument is required"
    echo "Example: $0 input.jsonl --layers 10,15,20"
    exit 1
fi

# Default values
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
OUTPUT="${OUTPUT:-steering_vectors.pt}"
MIN_SAMPLES="${MIN_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "=== Extracting Steering Vectors ==="
echo "Input: $INPUT_PATH"
echo "Model: $MODEL"
echo "Layers: $LAYERS"
echo "Output: $OUTPUT"
echo "Min Samples: $MIN_SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Run extraction
uv run probing-reflection extract-vectors \
    --input "$INPUT_PATH" \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --output "$OUTPUT" \
    --min-samples "$MIN_SAMPLES" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "Extraction complete."
echo "Vectors saved to: $OUTPUT"
