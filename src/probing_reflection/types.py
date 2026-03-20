"""Core type definitions for ProbingReflection.

This module defines the foundational types used throughout the project
for LLM probing and steering experiments. All configuration types are
immutable (frozen dataclasses) to ensure reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict


@dataclass(frozen=True)
class ProbingConfig:
    """Immutable configuration for probing experiments.

    Attributes:
        model_name: Name or path of the model to probe.
        layer_indices: Tuple of layer indices to extract activations from.
            Using tuple (not list) ensures hashability.
    """

    model_name: str = ""
    layer_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable configuration for inference experiments.

    Attributes:
        model_name: Name or path of the model for inference.
        dataset_name: Name of the dataset to run inference on.
        batch_size: Batch size for inference.
        max_new_tokens: Maximum number of new tokens to generate.
        output_path: Path to save inference results.
    """

    model_name: str = "Qwen/Qwen3.5-0.8B"
    dataset_name: str = "HuggingFaceH4/MATH-500"
    batch_size: int = 8
    max_new_tokens: int = 256
    output_path: str = "outputs/math500_inference/qwen3-0.8b-math500-cot.jsonl"


@dataclass(frozen=True)
class ReflectionResult:
    """Immutable result of a reflection analysis.

    Stores the outcome of analyzing a sample for self-reflection patterns.

    Attributes:
        sample_id: Unique identifier for the analyzed sample.
        reflection_score: Computed reflection score in [0.0, 1.0].
        metadata: Additional metadata about the analysis (source, model, etc.).
    """

    sample_id: str
    reflection_score: float
    metadata: MappingProxyType[str, str] = field(default_factory=lambda: MappingProxyType({}))


class ContrastivePair(TypedDict):
    """Structured representation of contrastive examples.

    Used for training steering vectors and probes. Each pair consists
    of a positive example (exhibiting target behavior) and a negative
    example (lacking target behavior).

    Attributes:
        positive: Example text exhibiting the target behavior.
        negative: Example text lacking the target behavior.
    """

    positive: str
    negative: str
