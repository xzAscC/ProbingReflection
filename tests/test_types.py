"""Tests for core type definitions.

These tests verify the foundational types used throughout
the ProbingReflection research project.
"""

import pytest

from probing_reflection.types import (
    ContrastivePair,
    ExtractVectorsConfig,
    ProbingConfig,
    ReflectionResult,
    SteeringVectorResult,
)


class TestProbingConfig:
    """Tests for ProbingConfig dataclass."""

    def test_default_initialization(self) -> None:
        """ProbingConfig should have sensible defaults."""
        config = ProbingConfig()
        assert config.model_name == ""
        assert config.layer_indices == ()

    def test_custom_initialization(self) -> None:
        """ProbingConfig should accept custom values."""
        config = ProbingConfig(
            model_name="gpt2-small",
            layer_indices=(0, 6, 11),
        )
        assert config.model_name == "gpt2-small"
        assert config.layer_indices == (0, 6, 11)

    def test_is_frozen(self) -> None:
        """ProbingConfig should be immutable (frozen)."""
        config = ProbingConfig(model_name="test")
        with pytest.raises(AttributeError):
            config.model_name = "changed"  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """ProbingConfig should be hashable (for use in sets/dicts)."""
        config = ProbingConfig(model_name="test")
        assert hash(config) is not None
        # Can be used in a set
        configs = {config, ProbingConfig(model_name="test")}
        assert len(configs) == 1  # Same config hashes equal


class TestReflectionResult:
    """Tests for ReflectionResult dataclass."""

    def test_basic_creation(self) -> None:
        """ReflectionResult should store analysis results."""
        result = ReflectionResult(
            sample_id="sample-001",
            reflection_score=0.85,
        )
        assert result.sample_id == "sample-001"
        assert result.reflection_score == 0.85

    def test_with_metadata(self) -> None:
        """ReflectionResult should accept optional metadata."""
        result = ReflectionResult(
            sample_id="sample-002",
            reflection_score=0.42,
            metadata={"source": "experiment-1", "model": "gpt2"},
        )
        assert result.metadata["source"] == "experiment-1"
        assert result.metadata["model"] == "gpt2"

    def test_default_metadata(self) -> None:
        """ReflectionResult should default to empty metadata."""
        result = ReflectionResult(
            sample_id="sample-003",
            reflection_score=0.0,
        )
        assert result.metadata == {}

    def test_score_bounds_validation(self) -> None:
        """ReflectionResult should accept valid scores."""
        # These should not raise
        ReflectionResult(sample_id="test", reflection_score=0.0)
        ReflectionResult(sample_id="test", reflection_score=1.0)
        ReflectionResult(sample_id="test", reflection_score=0.5)


class TestContrastivePair:
    """Tests for ContrastivePair TypedDict."""

    def test_structure(self) -> None:
        """ContrastivePair should have positive/negative keys."""
        pair: ContrastivePair = {
            "positive": "I love this product!",
            "negative": "I hate this product!",
        }
        assert pair["positive"] == "I love this product!"
        assert pair["negative"] == "I hate this product!"

    def test_used_in_list(self) -> None:
        """ContrastivePair should work in collections."""
        pairs: list[ContrastivePair] = [
            {"positive": "good", "negative": "bad"},
            {"positive": "happy", "negative": "sad"},
        ]
        assert len(pairs) == 2


class TestExtractVectorsConfig:
    """Tests for ExtractVectorsConfig dataclass."""

    def test_default_initialization(self) -> None:
        """ExtractVectorsConfig should have sensible defaults."""
        config = ExtractVectorsConfig()
        assert config.input_path == ""
        assert config.model_name == "Qwen/Qwen2.5-0.5B"
        assert config.layer_indices == ()
        assert config.output_path == "steering_vectors.pt"
        assert config.min_samples == 10
        assert config.batch_size == 4

    def test_custom_initialization(self) -> None:
        """ExtractVectorsConfig should accept custom values."""
        config = ExtractVectorsConfig(
            input_path="data/samples.jsonl",
            model_name="gpt2-small",
            layer_indices=(0, 6, 11),
            output_path="vectors.pt",
            min_samples=20,
            batch_size=8,
        )
        assert config.input_path == "data/samples.jsonl"
        assert config.model_name == "gpt2-small"
        assert config.layer_indices == (0, 6, 11)
        assert config.output_path == "vectors.pt"
        assert config.min_samples == 20
        assert config.batch_size == 8

    def test_is_frozen(self) -> None:
        """ExtractVectorsConfig should be immutable (frozen)."""
        config = ExtractVectorsConfig()
        with pytest.raises(AttributeError):
            config.min_samples = 20  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """ExtractVectorsConfig should be hashable (for use in sets/dicts)."""
        config = ExtractVectorsConfig(model_name="test")
        assert hash(config) is not None
        configs = {config, ExtractVectorsConfig(model_name="test")}
        assert len(configs) == 1


class TestSteeringVectorResult:
    """Tests for SteeringVectorResult TypedDict."""

    def test_structure(self) -> None:
        """SteeringVectorResult should have vectors and metadata keys."""
        from torch import Tensor

        result: SteeringVectorResult = {
            "vectors": {0: Tensor([1.0, 2.0, 3.0])},
            "metadata": {"model_name": "test", "layer_count": 5, "layer_indices": (0, 1, 2)},
        }
        assert "vectors" in result
        assert "metadata" in result
        assert 0 in result["vectors"]
        assert result["metadata"]["model_name"] == "test"

    def test_metadata_types(self) -> None:
        """SteeringVectorResult metadata should accept str, int, and tuple."""
        from torch import Tensor

        result: SteeringVectorResult = {
            "vectors": {0: Tensor([1.0])},
            "metadata": {
                "str_value": "test",
                "int_value": 42,
                "tuple_value": (1, 2, 3),
            },
        }
        assert isinstance(result["metadata"]["str_value"], str)
        assert isinstance(result["metadata"]["int_value"], int)
        assert isinstance(result["metadata"]["tuple_value"], tuple)
