#!/usr/bin/env bash
# Full test script - runs all models for complete validation
# Usage: ./scripts/full_test.sh

set -e

echo "=== Full Test (All Models) ==="
echo ""

# All models to test (comma-separated)
export MODELS="${MODELS:-Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B}"

echo "Models: $MODELS"
echo ""

# Install dependencies
echo "[1/4] Installing dependencies..."
uv sync

# Lint check
echo "[2/4] Running lint check..."
uv run ruff check src/ tests/

# Format check
echo "[3/4] Running format check..."
uv run ruff format --check src/ tests/

# Type check
echo "[4/4] Running type check..."
uv run mypy src/

# Run tests for each model
echo "[5/5] Running tests for all models..."
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
for model in "${MODEL_ARRAY[@]}"; do
    echo ""
    echo "--- Testing with model: $model ---"
    export MODEL_NAME="$model"
    uv run pytest -v
done

echo ""
echo "=== Full Test Complete ==="
