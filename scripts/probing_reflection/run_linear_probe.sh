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

# Convert layers to Python tuple format
LAYER_TUPLE=$(echo "$LAYERS" | sed 's/\([^,]*\)/\1/g' | tr ',' ' ')

# Run linear probe using Python directly
uv run python -c "
import json
from probing_reflection.linear_probe import run_linear_probe
from probing_reflection.types import LinearProbeConfig

layer_indices = tuple(int(x.strip()) for x in '$LAYERS'.split(','))

config = LinearProbeConfig(
    input_path='$INPUT_PATH',
    model_name='$MODEL',
    layer_indices=layer_indices,
    test_size=$TEST_SIZE,
    output_dir='$OUTPUT_DIR',
)

result = run_linear_probe(config)

print('\\n=== Linear Probe Results ===')
for metric in result['metrics']:
    print(f\"Layer {metric['layer_index']}: Accuracy={metric['accuracy']:.2%} (train={metric['train_samples']}, test={metric['test_samples']})\")
print(f\"\\nResults saved to: $OUTPUT_DIR\")
"

echo ""
echo "Linear probe training complete."
