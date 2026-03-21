#!/usr/bin/env python
"""Run steering experiments on benchmarks.

This script orchestrates running steering inference across multiple
datasets and conditions, with proper memory management between runs.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

from probing_reflection.steering_inference import (
    get_output_path,
    load_steering_vectors,
    run_steering_inference,
)
from probing_reflection.types import SteeringInferenceConfig

DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "aime": ("AI-MO/aimo-validation-aime", None, "train"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
}

CONDITIONS = {
    "baseline": 0.0,
    "positive": 1.0,
    "negative": -1.0,
}


def get_memory_usage() -> str:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CPU mode"


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run steering experiments on benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["math500", "aime", "gpqa", "all"],
        default="all",
        help="Dataset to run experiments on (default: all)",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "positive", "negative", "all"],
        default="all",
        help="Steering condition (default: all)",
    )
    parser.add_argument(
        "--steering-vector-path",
        required=True,
        help="Path to steering vectors .pt file",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-32B",
        help="Model name (default: Qwen/Qwen2.5-32B)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples per dataset",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/steering_experiments",
        help="Base directory for outputs (default: outputs/steering_experiments)",
    )
    args = parser.parse_args()

    steering_path = Path(args.steering_vector_path)
    if not steering_path.exists():
        print(f"Error: Steering vector file not found: {steering_path}")
        sys.exit(1)

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    conditions = list(CONDITIONS.keys()) if args.condition == "all" else [args.condition]

    print(f"Loading steering vectors from: {steering_path}")
    vectors = load_steering_vectors(steering_path)
    layer_indices = tuple(sorted(vectors.keys()))
    print(f"Found steering vectors for layers: {layer_indices}")
    print(f"Datasets: {datasets}")
    print(f"Conditions: {conditions}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit if args.limit else 'none'}")
    print()

    total_runs = len(datasets) * len(conditions)
    current_run = 0

    for dataset in datasets:
        for condition in conditions:
            current_run += 1
            print(f"\n{'=' * 60}")
            print(f"[{current_run}/{total_runs}] Running {dataset} / {condition}")
            print(f"Memory before: {get_memory_usage()}")
            print("=" * 60)

            dataset_name, dataset_config, split = DATASETS[dataset]
            coefficient = CONDITIONS[condition]

            output_path = get_output_path(dataset, condition, args.output_dir)

            config = SteeringInferenceConfig(
                model_name=args.model,
                steering_vector_path=str(steering_path),
                layer_indices=layer_indices,
                coefficient=coefficient,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                batch_size=1,
                max_new_tokens=args.max_tokens,
                output_path=str(output_path),
                limit=args.limit,
            )

            try:
                result_path = run_steering_inference(config)
                print(f"Results saved to: {result_path}")
            except Exception as e:
                print(f"Error running {dataset}/{condition}: {e}")
                raise

            print(f"Memory after: {get_memory_usage()}")
            print("Cleaning up memory...")
            cleanup_memory()
            print(f"Memory after cleanup: {get_memory_usage()}")

    print(f"\n{'=' * 60}")
    print(f"All experiments completed! Total runs: {total_runs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
