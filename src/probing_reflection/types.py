"""Core type definitions for ProbingReflection.

This module defines the foundational types used throughout the project
for LLM probing and steering experiments. All configuration types are
immutable (frozen dataclasses) to ensure reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NotRequired, TypedDict

from torch import Tensor


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


class JudgeVerdict(TypedDict):
    """Verdict from a judge model evaluating answer equivalence.

    Represents the output of an LLM judge that determines whether
    a model's extracted answer matches a reference answer.

    Attributes:
        explanation: Explanation of the judge's reasoning.
        equivalent: Whether the extracted answer is equivalent to the reference.
        confidence: Confidence score in [0.0, 1.0] for the verdict.
    """

    explanation: str
    equivalent: bool
    confidence: float


class EvaluationResult(TypedDict):
    """Result of evaluating a single sample against a reference.

    Stores all information about the evaluation of one problem,
    including the judge's verdict and metadata about the sample.

    Attributes:
        problem_id: Unique identifier for the problem.
        extracted_answer: Answer extracted from the model output, or None if extraction failed.
        reference_answer: The ground truth reference answer.
        is_correct: Whether the extracted answer was judged correct.
        judge_explanation: Explanation from the judge model.
        confidence: Confidence score in [0.0, 1.0] for the evaluation.
        subject: Optional subject category (e.g., "algebra", "geometry").
        level: Optional difficulty level of the problem.
    """

    problem_id: str
    extracted_answer: str | None
    reference_answer: str
    is_correct: bool
    judge_explanation: str
    confidence: float
    subject: NotRequired[str | None]
    level: NotRequired[int | None]


class EvaluationReport(TypedDict):
    """Aggregated report of evaluation results across all samples.

    Provides summary statistics and detailed results for analysis
    of model performance on an evaluation dataset.

    Attributes:
        overall_accuracy: Fraction of correct answers across all samples.
        total_samples: Total number of samples evaluated.
        correct_count: Number of correctly answered samples.
        per_subject_accuracy: Accuracy broken down by subject category.
        per_level_accuracy: Accuracy broken down by difficulty level.
        results: List of individual evaluation results.
    """

    overall_accuracy: float
    total_samples: int
    correct_count: int
    per_subject_accuracy: dict[str, float]
    per_level_accuracy: dict[str, float]
    results: list[EvaluationResult]


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable configuration for evaluation experiments.

    Attributes:
        judge_model_name: Name or path of the judge model.
        batch_size: Batch size for evaluation.
        confidence_threshold: Minimum confidence to accept a verdict.
        output_file: Optional path to save evaluation results.
    """

    judge_model_name: str = "Qwen/Qwen3.5-27B"
    batch_size: int = 8
    confidence_threshold: float = 0.7
    output_file: str | None = None


@dataclass(frozen=True)
class ReflectionDiagnosisConfig:
    """Immutable configuration for reflection diagnosis experiments.

    Attributes:
        input_path: Path to the input JSONL file containing samples.
        output_dir: Directory to save diagnosis results.
        model_name: Name or path of the model for token analysis.
        batch_size: Batch size for processing samples.
        max_retries: Maximum number of retries for failed API calls.
    """

    input_path: str = ""
    output_dir: str = "outputs/reflection_diagnosis/"
    model_name: str = "Qwen/Qwen3.5-27B"
    batch_size: int = 1
    max_retries: int = 3


@dataclass(frozen=True)
class ExtractVectorsConfig:
    """Immutable configuration for steering vector extraction.

    Attributes:
        input_path: Path to the input data file.
        model_name: Name or path of the model for extraction.
        layer_indices: Tuple of layer indices to extract vectors from.
        output_path: Path to save the extracted steering vectors.
        min_samples: Minimum number of samples required for extraction.
        batch_size: Batch size for processing during extraction.
    """

    input_path: str = ""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layer_indices: tuple[int, ...] = ()
    output_path: str = "steering_vectors.pt"
    min_samples: int = 10
    batch_size: int = 4


class ReflectionToken(TypedDict):
    """Structured representation of a reflection token in model output.

    Represents a single instance of self-reflection language detected
    in the model's generated text.

    Attributes:
        text: The actual token text (e.g., "Wait", "Actually").
        category: Category of reflection (e.g., "correction", "verification").
        context: Surrounding context where the token appeared.
        confidence: Confidence score in [0.0, 1.0] for the detection.
    """

    text: str
    category: str
    context: str
    confidence: float


class SampleWithReflection(TypedDict):
    """Sample data augmented with reflection analysis results.

    Extends the base sample structure with reflection-specific metrics
    and detected tokens.

    Attributes:
        problem_id: Unique identifier for the problem.
        problem: The problem text or question.
        generated: The model's generated response.
        reference_answer: The ground truth reference answer.
        subject: Optional subject category (e.g., "algebra", "geometry").
        level: Optional difficulty level of the problem.
        reflection_tokens: List of detected reflection tokens.
        reflection_count: Total count of reflection tokens found.
        reflection_density: Ratio of reflection tokens to total tokens.
    """

    problem_id: str
    problem: str
    generated: str
    reference_answer: str
    subject: NotRequired[str | None]
    level: NotRequired[int | None]
    reflection_tokens: list[ReflectionToken]
    reflection_count: int
    reflection_density: float


class ReflectionAnalysisReport(TypedDict):
    """Aggregated report of reflection analysis across all samples.

    Provides comprehensive statistics on reflection patterns detected
    in a dataset of model outputs.

    Attributes:
        total_samples: Total number of samples analyzed.
        total_tokens: Total reflection tokens detected across all samples.
        avg_tokens_per_sample: Average reflection tokens per sample.
        overall_density: Average reflection density across all samples.
        token_frequency: Frequency count of each unique token text.
        category_distribution: Distribution of tokens by category.
        per_subject_stats: Reflection statistics broken down by subject.
        per_level_stats: Reflection statistics broken down by difficulty level.
        processing_errors: Number of samples that failed processing.
    """

    total_samples: int
    total_tokens: int
    avg_tokens_per_sample: float
    overall_density: float
    token_frequency: dict[str, int]
    category_distribution: dict[str, int]
    per_subject_stats: dict[str, dict[str, float | int]]
    per_level_stats: dict[str, dict[str, float | int]]
    processing_errors: int


class SteeringVectorResult(TypedDict):
    """Result of steering vector extraction.

    Attributes:
        vectors: Mapping of layer index to steering vector tensor.
        metadata: Metadata about the extraction (model name, sample counts, etc.).
    """

    vectors: dict[int, Tensor]
    metadata: dict[str, str | int | tuple[int, ...]]
