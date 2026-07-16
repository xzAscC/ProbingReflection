#!/bin/bash
# Run linear probe training on model activations

set -e

# Check required argument
if [ -z "$1" ]; then
    echo "Usage: $0 <input_jsonl> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_jsonl   Path to input JSONL file with reflection analysis"
    echo ""
    echo "Options (set as environment variables):"
    echo "  MODEL         Model name (required)"
    echo "  LAYERS        Layer indices, comma-separated (required)"
    echo "  TEST_SIZE     Test set fraction (default: 0.2)"
    echo "  OUTPUT_DIR    Output directory (default: outputs/linear_probe/)"
    exit 1
fi

INPUT_PATH="$1"

# Check required environment variables
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is required"
    echo "Example: MODEL=Qwen/Qwen2.5-0.5B $0 input.jsonl"
    exit 1
fi

if [ -z "$LAYERS" ]; then
    echo "Error: LAYERS environment variable is required"
    echo "Example: LAYERS='10,15,20' MODEL=Qwen/Qwen2.5-0.5B $0 input.jsonl"
    exit 1
fi

# Default values
TEST_SIZE="${TEST_SIZE:-0.2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/linear_probe/}"

echo "=== Running Linear Probe Training ==="
echo "Input: $INPUT_PATH"
echo "Model: $MODEL"
echo "Layers: $LAYERS"
echo "Test Size: $TEST_SIZE"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run linear probe without interpolating values into Python source
uv run python -c '
import sys
from probing_reflection.linear_probe import run_linear_probe
from probing_reflection.types import LinearProbeConfig

input_path, model_name, layers, test_size, output_dir = sys.argv[1:]
layer_indices = tuple(int(value.strip()) for value in layers.split(","))

config = LinearProbeConfig(
    input_path=input_path,
    model_name=model_name,
    layer_indices=layer_indices,
    test_size=float(test_size),
    output_dir=output_dir,
)

result = run_linear_probe(config)

print("\n=== Linear Probe Results ===")
for metric in result["metrics"]:
    print(
        "Layer {}: Accuracy={:.2%} (train={}, test={})".format(
            metric["layer_index"],
            metric["accuracy"],
            metric["train_samples"],
            metric["test_samples"],
        )
    )
print(f"\nResults saved to: {output_dir}")
' "$INPUT_PATH" "$MODEL" "$LAYERS" "$TEST_SIZE" "$OUTPUT_DIR"

echo ""
echo "Linear probe training complete."
