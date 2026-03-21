"""Tests for reflection diagnosis utilities.

These tests verify the reflection diagnosis pipeline components including
prompt building, token detection, and statistics aggregation.

TDD RED PHASE: Tests for unimplemented functions are marked to fail/skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from probing_reflection.reflection_diagnosis import (
    REFLECTION_TAXONOMY,
    build_diagnosis_prompt,
    diagnose_all,
    diagnose_sample,
    ensure_output_dir,
)
from probing_reflection.types import ReflectionDiagnosisConfig, ReflectionToken

# ============================================================================
# MOCK JUDGE FOR TESTING
# ============================================================================


class MockReflectionJudge:
    """Mock judge that detects reflection tokens via keyword matching."""

    def __init__(self) -> None:
        self._reflection_keywords = {
            "wait",
            "hmm",
            "actually",
            "let me think",
            "reconsider",
            "however",
            "verify",
            "check",
            "alternatively",
            "on the other hand",
            "double-check",
        }

    def judge(self, text: str) -> list[ReflectionToken]:
        """Return reflection tokens found in text via keyword matching."""
        text_lower = text.lower()
        tokens: list[ReflectionToken] = []

        for keyword in self._reflection_keywords:
            if keyword in text_lower:
                tokens.append(
                    ReflectionToken(
                        text=keyword,
                        category="detected",
                        context=text[:50],
                        confidence=0.8,
                    )
                )

        return tokens


# ============================================================================
# TEST FIXTURES
# ============================================================================

SAMPLE_WITH_REFLECTION: dict[str, Any] = {
    "problem_id": "test_001",
    "problem": "What is 6 times 7?",
    "generated": "Wait, let me reconsider. Actually, the answer is 42.",
    "reference_answer": "42",
    "subject": "Arithmetic",
    "level": 1,
}

SAMPLE_WITHOUT_REFLECTION: dict[str, Any] = {
    "problem_id": "test_002",
    "problem": "What is 2 + 2?",
    "generated": "Adding 2 and 2 gives 4.",
    "reference_answer": "4",
}

SAMPLE_WITH_MULTIPLE_REFLECTIONS: dict[str, Any] = {
    "problem_id": "test_003",
    "problem": "Solve for x: 2x = 8",
    "generated": (
        "Hmm, let me think about this. First, I need to isolate x. "
        "However, I should verify my answer. x = 4. Let me check: 2 * 4 = 8. Yes!"
    ),
    "reference_answer": "4",
    "subject": "Algebra",
    "level": 2,
}

SAMPLE_EMPTY_TEXT: dict[str, Any] = {
    "problem_id": "test_empty",
    "problem": "Test problem",
    "generated": "",
    "reference_answer": "answer",
}

SAMPLE_COMPLEX_TOKENS: dict[str, Any] = {
    "problem_id": "test_complex",
    "problem": "Complex reflection test",
    "generated": (
        "Wait, I need to reconsider this. On the other hand, maybe "
        "I should verify. Actually, let me double-check my reasoning. "
        "So the answer is 42."
    ),
    "reference_answer": "42",
}


# ============================================================================
# TESTS FOR EXISTING FUNCTIONS (should PASS)
# ============================================================================


class TestBuildDiagnosisPrompt:
    """Tests for build_diagnosis_prompt function."""

    def test_build_diagnosis_prompt_contains_schema(self) -> None:
        """Prompt should contain the expected JSON schema."""
        text = "Wait, let me think."
        prompt = build_diagnosis_prompt(text)

        # Check for JSON schema elements
        assert '"tokens"' in prompt
        assert '"text"' in prompt
        assert '"category"' in prompt
        assert '"context"' in prompt
        assert '"confidence"' in prompt

    def test_build_diagnosis_prompt_contains_taxonomy(self) -> None:
        """Prompt should contain reflection taxonomy examples."""
        text = "Test text"
        prompt = build_diagnosis_prompt(text)

        # Check for taxonomy categories
        assert "hesitation" in prompt
        assert "qualification" in prompt
        assert "verification" in prompt
        assert "redirection" in prompt

    def test_build_diagnosis_prompt_includes_input_text(self) -> None:
        """Prompt should include the input text to analyze."""
        text = "Wait, this is my test text with reflection."
        prompt = build_diagnosis_prompt(text)

        assert text in prompt

    def test_build_diagnosis_prompt_context_warning(self) -> None:
        """Prompt should warn about context-dependent judgment."""
        text = "Test"
        prompt = build_diagnosis_prompt(text)

        assert "Context matters" in prompt
        assert "Wait for the result" in prompt  # Example of non-reflection

    def test_build_diagnosis_prompt_empty_text(self) -> None:
        """Prompt should handle empty text."""
        text = ""
        prompt = build_diagnosis_prompt(text)

        # Should still return a valid prompt structure
        assert "JSON format" in prompt
        assert '"tokens"' in prompt


class TestEnsureOutputDir:
    """Tests for ensure_output_dir function."""

    def test_ensure_output_dir_creates_directory(self, tmp_path: Path) -> None:
        """Function should create directory if it doesn't exist."""
        new_dir = tmp_path / "new_output_dir"
        assert not new_dir.exists()

        result = ensure_output_dir(new_dir)

        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_output_dir_existing_directory(self, tmp_path: Path) -> None:
        """Function should return path for existing directory."""
        existing_dir = tmp_path / "existing_dir"
        existing_dir.mkdir()

        result = ensure_output_dir(existing_dir)

        assert result == existing_dir
        assert existing_dir.exists()

    def test_ensure_output_dir_nested_path(self, tmp_path: Path) -> None:
        """Function should create nested directories."""
        nested_path = tmp_path / "level1" / "level2" / "level3"

        result = ensure_output_dir(nested_path)

        assert nested_path.exists()
        assert result == nested_path

    def test_ensure_output_dir_string_path(self, tmp_path: Path) -> None:
        """Function should accept string path."""
        dir_path = str(tmp_path / "string_path")

        result = ensure_output_dir(dir_path)

        assert result.exists()
        assert isinstance(result, Path)


class TestReflectionTaxonomy:
    """Tests for the REFLECTION_TAXONOMY constant."""

    def test_taxonomy_is_dict(self) -> None:
        """Taxonomy should be a dictionary."""
        assert isinstance(REFLECTION_TAXONOMY, dict)

    def test_taxonomy_has_expected_categories(self) -> None:
        """Taxonomy should have expected reflection categories."""
        expected_categories = {
            "hesitation",
            "qualification",
            "verification",
            "redirection",
            "transition",
        }
        assert set(REFLECTION_TAXONOMY.keys()) == expected_categories

    def test_taxonomy_tokens_are_lists(self) -> None:
        """Each category should have a list of tokens."""
        for category, tokens in REFLECTION_TAXONOMY.items():
            assert isinstance(tokens, list), f"{category} tokens should be a list"
            assert len(tokens) > 0, f"{category} should have at least one token"


# ============================================================================
# TESTS FOR UNIMPLEMENTED FUNCTIONS (should FAIL or SKIP)
# These tests verify the expected interface for code that doesn't exist yet.
# ============================================================================


class TestReflectionJudge:
    """Tests for ReflectionJudge class.

    NOTE: ReflectionJudge is NOT YET IMPLEMENTED.
    These tests should FAIL until implementation is added.
    """

    def test_reflection_judge_init(self) -> None:
        """ReflectionJudge should initialize with model name."""
        # This test will fail because ReflectionJudge doesn't exist yet
        pytest.importorskip(
            "probing_reflection.reflection_diagnosis",
            reason="Waiting for ReflectionJudge implementation",
        )

        # After import, try to use the class - will fail if not implemented
        from probing_reflection.reflection_diagnosis import ReflectionJudge

        judge = ReflectionJudge("test-model")
        assert judge.model_name == "test-model"

    def test_reflection_judge_with_config(self) -> None:
        """ReflectionJudge should accept ReflectionDiagnosisConfig."""
        pytest.importorskip(
            "probing_reflection.reflection_diagnosis",
            reason="Waiting for ReflectionJudge implementation",
        )

        from probing_reflection.reflection_diagnosis import ReflectionJudge

        config = ReflectionDiagnosisConfig(
            model_name="test-model",
            batch_size=4,
        )
        judge = ReflectionJudge(config)
        assert judge.config == config


class TestDiagnoseSample:
    """Tests for diagnose_sample function."""

    @pytest.fixture
    def mock_judge(self) -> MockReflectionJudge:
        """Create a mock judge for testing."""
        return MockReflectionJudge()

    def test_diagnose_sample_with_reflection_tokens(self, mock_judge: MockReflectionJudge) -> None:
        """diagnose_sample should detect reflection tokens in sample."""
        result = diagnose_sample(mock_judge, SAMPLE_WITH_REFLECTION)

        assert "reflection_tokens" in result
        assert "reflection_count" in result
        assert isinstance(result["reflection_tokens"], list)
        assert result["reflection_count"] >= 1

    def test_diagnose_sample_without_reflection_tokens(
        self, mock_judge: MockReflectionJudge
    ) -> None:
        """diagnose_sample should return empty list for no reflection."""
        result = diagnose_sample(mock_judge, SAMPLE_WITHOUT_REFLECTION)

        assert result["reflection_tokens"] == []
        assert result["reflection_count"] == 0

    def test_diagnose_sample_empty_text(self, mock_judge: MockReflectionJudge) -> None:
        """diagnose_sample should handle empty text gracefully."""
        result = diagnose_sample(mock_judge, SAMPLE_EMPTY_TEXT)

        assert result["reflection_tokens"] == []
        assert result["reflection_count"] == 0

    def test_diagnose_sample_returns_sample_with_reflection_type(
        self, mock_judge: MockReflectionJudge
    ) -> None:
        """diagnose_sample should return SampleWithReflection compatible dict."""
        result = diagnose_sample(mock_judge, SAMPLE_WITH_REFLECTION)

        assert "problem_id" in result
        assert "problem" in result
        assert "generated" in result
        assert "reference_answer" in result
        assert "reflection_tokens" in result
        assert "reflection_count" in result
        assert "reflection_density" in result


class TestDiagnoseAll:
    """Tests for diagnose_all function."""

    @pytest.fixture
    def mock_judge(self) -> MockReflectionJudge:
        """Create a mock judge for testing."""
        return MockReflectionJudge()

    def test_diagnose_all_aggregates_statistics(
        self, tmp_path: Path, mock_judge: MockReflectionJudge
    ) -> None:
        """diagnose_all should aggregate statistics across all samples."""
        samples_path = tmp_path / "samples.jsonl"
        with open(samples_path, "w") as f:
            for sample in [
                SAMPLE_WITH_REFLECTION,
                SAMPLE_WITHOUT_REFLECTION,
                SAMPLE_WITH_MULTIPLE_REFLECTIONS,
            ]:
                f.write(json.dumps(sample) + "\n")

        output_dir = tmp_path / "output"
        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(output_dir),
        )

        samples, report = diagnose_all(config, judge=mock_judge)

        assert isinstance(report, dict)
        assert "total_samples" in report
        assert "total_tokens" in report
        assert "avg_tokens_per_sample" in report
        assert "category_distribution" in report
        assert isinstance(samples, list)
        assert len(samples) == 3

    def test_diagnose_all_returns_reflection_analysis_report(
        self, tmp_path: Path, mock_judge: MockReflectionJudge
    ) -> None:
        """diagnose_all should return ReflectionAnalysisReport compatible dict."""
        samples_path = tmp_path / "samples.jsonl"
        with open(samples_path, "w") as f:
            f.write(json.dumps(SAMPLE_WITH_REFLECTION) + "\n")

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
        )

        samples, report = diagnose_all(config, judge=mock_judge)

        required_fields = [
            "total_samples",
            "total_tokens",
            "avg_tokens_per_sample",
            "overall_density",
            "token_frequency",
            "category_distribution",
            "per_subject_stats",
            "per_level_stats",
            "processing_errors",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_diagnose_all_empty_input(self, tmp_path: Path) -> None:
        """diagnose_all should handle empty input file."""
        samples_path = tmp_path / "empty.jsonl"
        samples_path.touch()

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
        )

        samples, report = diagnose_all(config)

        assert report["total_samples"] == 0
        assert report["total_tokens"] == 0
        assert samples == []


class TestParseJsonResponse:
    """Tests for _parse_json_response helper function.

    NOTE: _parse_json_response is NOT YET IMPLEMENTED.
    These tests should FAIL until implementation is added.
    """

    def test_parse_json_response_valid(self) -> None:
        """_parse_json_response should parse valid JSON response."""
        pytest.importorskip(
            "probing_reflection.reflection_diagnosis",
            reason="Waiting for _parse_json_response implementation",
        )

        from probing_reflection.reflection_diagnosis import _parse_json_response

        response = (
            '{"tokens": [{"text": "wait", "category": "hesitation", '
            '"context": "Wait, let me think", "confidence": 0.9}]}'
        )
        result = _parse_json_response(response)

        assert "tokens" in result
        assert len(result["tokens"]) == 1
        assert result["tokens"][0]["text"] == "wait"

    def test_parse_json_response_empty_tokens(self) -> None:
        """_parse_json_response should handle empty tokens list."""
        pytest.importorskip(
            "probing_reflection.reflection_diagnosis",
            reason="Waiting for _parse_json_response implementation",
        )

        from probing_reflection.reflection_diagnosis import _parse_json_response

        response = '{"tokens": []}'
        result = _parse_json_response(response)

        assert result["tokens"] == []

    def test_parse_json_response_malformed_json(self) -> None:
        """_parse_json_response should handle malformed JSON."""
        pytest.importorskip(
            "probing_reflection.reflection_diagnosis",
            reason="Waiting for _parse_json_response implementation",
        )

        from probing_reflection.reflection_diagnosis import _parse_json_response

        # Malformed JSON - missing closing brace
        response = '{"tokens": [{"text": "wait"'
        result = _parse_json_response(response)

        # Should return empty tokens or error indicator
        assert result["tokens"] == [] or "error" in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestReflectionDiagnosisIntegration:
    """Integration tests for the full reflection diagnosis pipeline."""

    @pytest.fixture
    def mock_judge(self) -> MockReflectionJudge:
        """Create a mock judge for testing."""
        return MockReflectionJudge()

    def test_full_diagnosis_pipeline(self, tmp_path: Path, mock_judge: MockReflectionJudge) -> None:
        """Full pipeline should process samples and produce valid report."""
        samples_path = tmp_path / "samples.jsonl"
        with open(samples_path, "w") as f:
            for sample in [SAMPLE_WITH_REFLECTION, SAMPLE_COMPLEX_TOKENS]:
                f.write(json.dumps(sample) + "\n")

        output_dir = tmp_path / "diagnosis_output"
        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(output_dir),
        )

        samples, report = diagnose_all(config, judge=mock_judge)

        assert report["total_samples"] == 2
        assert report["total_tokens"] > 0
        assert len(report["category_distribution"]) > 0
        assert len(samples) == 2

    def test_diagnosis_with_subject_breakdown(
        self, tmp_path: Path, mock_judge: MockReflectionJudge
    ) -> None:
        """Diagnosis should provide per-subject statistics."""
        samples_data = [
            {**SAMPLE_WITH_REFLECTION, "subject": "Arithmetic"},
            {**SAMPLE_WITH_MULTIPLE_REFLECTIONS, "subject": "Algebra"},
        ]

        samples_path = tmp_path / "samples.jsonl"
        with open(samples_path, "w") as f:
            for sample in samples_data:
                f.write(json.dumps(sample) + "\n")

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
        )

        samples, report = diagnose_all(config, judge=mock_judge)

        assert "Arithmetic" in report["per_subject_stats"]
        assert "Algebra" in report["per_subject_stats"]
