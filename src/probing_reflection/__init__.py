"""ProbingReflection - Probing and Modulating Self-Reflection in Language Models.

This package provides tools for investigating self-reflection in Large Language
Models through probing and steering techniques.

Public API:
    ProbingConfig: Configuration for probing experiments
    InferenceConfig: Configuration for inference experiments
    ReflectionResult: Result container for reflection analysis
    ContrastivePair: TypedDict for contrastive example pairs
"""

from probing_reflection.types import (
    ContrastivePair,
    InferenceConfig,
    ProbingConfig,
    ReflectionResult,
)

__all__ = [
    "ContrastivePair",
    "InferenceConfig",
    "ProbingConfig",
    "ReflectionResult",
]

__version__ = "0.1.0"
