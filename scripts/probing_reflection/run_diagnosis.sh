#!/bin/bash
# Diagnose reflection tokens in model outputs

set -e

# Check required argument
if [ -z "$1" ]; then
    echo "Usage: $0 <input_jsonl> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_jsonl   Path to input JSONL file containing model outputs"
    echo ""
    echo "Options (set as environment variables):"
    echo "  MODEL         Judge model name (default: Qwen/Qwen3.5-27B)"
    echo "  OUTPUT_DIR    Output directory (default: outputs/reflection_analysis/)"
    exit 1
fi

INPUT_PATH="$1"

# Default values
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reflection_analysis/}"

echo "=== Running Reflection Diagnosis ==="
echo "Input: $INPUT_PATH"
echo "Judge Model: $MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run diagnosis
uv run probing-reflection reflection-diagnose \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL"

echo ""
echo "Diagnosis complete."
echo "Results saved to: $OUTPUT_DIR"
