#!/bin/bash
# Run steering inference with steering vectors

set -e

# Check required arguments
if [ -z "$1" ] || [ -z "$VECTOR_PATH" ]; then
    echo "Usage: VECTOR_PATH=<path> $0 <dataset_name> [options]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name  Dataset name (e.g., HuggingFaceH4/MATH-500)"
    echo ""
    echo "Required Environment Variables:"
    echo "  VECTOR_PATH   Path to steering vectors .pt file"
    echo ""
    echo "Options (set as environment variables):"
    echo "  MODEL         Model name (default: Qwen/Qwen2.5-32B)"
    echo "  COEFFICIENT   Steering coefficient (default: 1.0)"
    echo "  LAYERS        Layer indices to apply steering (optional, uses all from vector)"
    echo "  OUTPUT        Output JSONL path (required)"
    echo "  LIMIT         Limit number of samples (optional)"
    echo "  BATCH_SIZE    Batch size (default: 1)"
    echo "  MAX_TOKENS    Max new tokens (default: 512)"
    echo "  DATASET_CONFIG Dataset config name (optional, for datasets like GPQA)"
    exit 1
fi

DATASET="$1"

# Check required environment variables
if [ -z "$OUTPUT" ]; then
    echo "Error: OUTPUT environment variable is required"
    echo "Example: OUTPUT=outputs/steered_results.jsonl $0 HuggingFaceH4/MATH-500"
    exit 1
fi

# Default values
MODEL="${MODEL:-Qwen/Qwen2.5-32B}"
COEFFICIENT="${COEFFICIENT:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
LIMIT="${LIMIT:-}"
LAYERS="${LAYERS:-}"

echo "=== Running Steering Inference ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Steering Vector: $VECTOR_PATH"
echo "Coefficient: $COEFFICIENT"
echo "Output: $OUTPUT"
echo "Batch Size: $BATCH_SIZE"
echo "Max Tokens: $MAX_TOKENS"
if [ -n "$LAYERS" ]; then
    echo "Layers: $LAYERS"
fi
if [ -n "$LIMIT" ]; then
    echo "Limit: $LIMIT"
fi
echo ""

# Run steering inference without interpolating values into Python source
uv run python -c '
import sys
from probing_reflection.steering_inference import run_steering_inference
from probing_reflection.types import SteeringInferenceConfig

(
    model_name,
    vector_path,
    layers,
    coefficient,
    dataset_name,
    dataset_config,
    batch_size,
    max_tokens,
    output_path,
    limit,
) = sys.argv[1:]

config = SteeringInferenceConfig(
    model_name=model_name,
    steering_vector_path=vector_path,
    layer_indices=tuple(int(value.strip()) for value in layers.split(",") if value.strip()),
    coefficient=float(coefficient),
    dataset_name=dataset_name,
    dataset_config=dataset_config or None,
    batch_size=int(batch_size),
    max_new_tokens=int(max_tokens),
    output_path=output_path,
    limit=int(limit) if limit else None,
)

result = run_steering_inference(config)
print(f"Steering inference complete. Results saved to: {result}")
' "$MODEL" "$VECTOR_PATH" "$LAYERS" "$COEFFICIENT" "$DATASET" "$DATASET_CONFIG" \
  "$BATCH_SIZE" "$MAX_TOKENS" "$OUTPUT" "$LIMIT"

echo ""
echo "Steering inference complete."
echo "Results saved to: $OUTPUT"
