"""Tests for dataset adapters.

These tests verify the unified dataset loading functions for
evaluation datasets (MATH-500, AIME, GPQA Diamond).
"""

from __future__ import annotations

import pytest

from probing_reflection.dataset_adapters import (
    DatasetSample,
    load_aime,
    load_gpqa_diamond,
    load_math500,
)


@pytest.fixture(autouse=True)
def mock_huggingface_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    math500 = [
        {
            "unique_id": f"math-{index}",
            "problem": f"Problem {index}",
            "answer": str(index),
            "subject": "algebra",
            "level": 1,
        }
        for index in range(500)
    ]
    aime: list[dict[str, object]] = [
        {"id": f"aime-{index}", "problem": f"AIME {index}", "answer": str(index)}
        for index in range(90)
    ]
    gpqa: list[dict[str, object]] = [
        {
            "Question": f"Science question {index}",
            "Choice 1": "choice one",
            "Choice 2": "choice two",
            "Choice 3": "choice three",
            "Choice 4": "choice four",
            "Correct Answer": "choice one",
        }
        for index in range(198)
    ]

    def fake_load_dataset(
        path: str,
        config_name: str | None = None,
        *,
        split: str,
    ) -> list[dict[str, object]]:
        del config_name, split
        if path == "HuggingFaceH4/MATH-500":
            return math500
        if path == "AI-MO/aimo-validation-aime":
            return aime
        if path == "Idavidrein/gpqa":
            return gpqa
        raise AssertionError(f"Unexpected dataset: {path}")

    monkeypatch.setattr("probing_reflection.dataset_adapters.load_dataset", fake_load_dataset)


class TestLoadMath500:
    """Tests for load_math500 function."""

    def test_returns_500_samples(self) -> None:
        """MATH-500 should return exactly 500 samples."""
        samples = load_math500()
        assert len(samples) == 500

    def test_required_fields_present(self) -> None:
        """Each MATH-500 sample should have required fields."""
        samples = load_math500()
        for sample in samples[:10]:  # Check first 10 for efficiency
            assert "problem_id" in sample
            assert "problem" in sample
            assert "reference_answer" in sample

    def test_field_types(self) -> None:
        """MATH-500 sample fields should have correct types."""
        samples = load_math500()
        for sample in samples[:10]:
            assert isinstance(sample["problem_id"], str)
            assert isinstance(sample["problem"], str)
            assert isinstance(sample["reference_answer"], str)
            assert sample["problem_id"]  # Non-empty
            assert sample["problem"]  # Non-empty
            assert sample["reference_answer"]  # Non-empty


class TestLoadAime:
    """Tests for load_aime function."""

    def test_returns_90_samples(self) -> None:
        """AIME should return exactly 90 samples."""
        samples = load_aime()
        assert len(samples) == 90

    def test_required_fields_present(self) -> None:
        """Each AIME sample should have required fields."""
        samples = load_aime()
        for sample in samples[:10]:
            assert sample["problem_id"]
            assert sample["problem"]
            assert sample["reference_answer"]

    def test_field_types(self) -> None:
        """AIME sample fields should have correct types."""
        samples = load_aime()
        for sample in samples[:10]:
            assert isinstance(sample["problem_id"], str)
            assert isinstance(sample["problem"], str)
            assert isinstance(sample["reference_answer"], str)


class TestLoadGpqaDiamond:
    """Tests for load_gpqa_diamond function."""

    def test_returns_198_samples(self) -> None:
        """GPQA Diamond should return exactly 198 samples."""
        samples = load_gpqa_diamond()
        assert len(samples) == 198

    def test_includes_choices_in_problem(self) -> None:
        """GPQA Diamond problems should include answer choices A/B/C/D."""
        samples = load_gpqa_diamond()
        for sample in samples[:10]:
            problem = sample["problem"]
            assert "A)" in problem, f"Missing choice A in: {problem[:100]}"
            assert "B)" in problem, f"Missing choice B in: {problem[:100]}"
            assert "C)" in problem, f"Missing choice C in: {problem[:100]}"
            assert "D)" in problem, f"Missing choice D in: {problem[:100]}"

    def test_required_fields_present(self) -> None:
        """Each GPQA Diamond sample should have required fields."""
        samples = load_gpqa_diamond()
        for sample in samples[:10]:
            assert sample["problem_id"]
            assert sample["problem"]
            assert sample["reference_answer"]

    def test_field_types(self) -> None:
        """GPQA Diamond sample fields should have correct types."""
        samples = load_gpqa_diamond()
        for sample in samples[:10]:
            assert isinstance(sample["problem_id"], str)
            assert isinstance(sample["problem"], str)
            assert isinstance(sample["reference_answer"], str)


class TestDatasetSampleType:
    """Tests for DatasetSample TypedDict."""

    def test_create_valid_sample(self) -> None:
        """Should be able to create a valid DatasetSample dict."""
        sample: DatasetSample = {
            "problem_id": "test-001",
            "problem": "What is 2 + 2?",
            "reference_answer": "4",
        }
        assert sample["problem_id"] == "test-001"
        assert sample["problem"] == "What is 2 + 2?"
        assert sample["reference_answer"] == "4"

    def test_sample_with_optional_fields(self) -> None:
        """DatasetSample should accept optional subject and level fields."""
        sample: DatasetSample = {
            "problem_id": "test-002",
            "problem": "Solve for x: 2x = 4",
            "reference_answer": "2",
            "subject": "algebra",
            "level": 1,
        }
        assert sample["subject"] == "algebra"
        assert sample["level"] == 1

    def test_samples_in_list(self) -> None:
        """DatasetSample should work in collections."""
        samples: list[DatasetSample] = [
            {"problem_id": "1", "problem": "Q1", "reference_answer": "A1"},
            {"problem_id": "2", "problem": "Q2", "reference_answer": "A2"},
        ]
        assert len(samples) == 2
        assert samples[0]["problem_id"] == "1"
        assert samples[1]["problem_id"] == "2"
