#!/usr/bin/env bash
# Quick validation script - uses small model for fast iteration
# Usage: ./scripts/quick_test.sh

set -e

echo "=== Quick Validation (Small Model) ==="
echo ""

# Small model for quick testing
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"

echo "Model: $MODEL_NAME"
echo ""

# Install dependencies
echo "[1/5] Installing dependencies..."
uv sync

# Lint check
echo "[2/5] Running lint check..."
uv run ruff check src/ tests/

# Format check
echo "[3/5] Running format check..."
uv run ruff format --check src/ tests/

# Type check
echo "[4/5] Running type check..."
uv run mypy src/

# Run tests
echo "[5/5] Running tests..."
uv run pytest -v

echo ""
echo "=== Quick Validation Complete ==="
