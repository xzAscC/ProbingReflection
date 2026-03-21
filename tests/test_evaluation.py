"""Tests for evaluation utilities.

These tests verify the evaluation pipeline components including
answer extraction, LLM judging, and report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from probing_reflection.evaluation import (
    LLMJudge,
    evaluate,
    extract_boxed_answer,
    generate_report,
)
from probing_reflection.types import EvaluationConfig, EvaluationResult


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer function."""

    def test_extract_simple(self) -> None:
        """Simple boxed answer should be extracted correctly."""
        text = r"The answer is \boxed{42}"
        result = extract_boxed_answer(text)
        assert result == "42"

    def test_extract_nested(self) -> None:
        """Nested braces in boxed answer should be handled correctly."""
        text = r"Result: \boxed{\frac{1}{2}}"
        result = extract_boxed_answer(text)
        assert result == r"\frac{1}{2}"

    def test_extract_no_match(self) -> None:
        """Text without boxed answer should return None."""
        text = "No answer here"
        result = extract_boxed_answer(text)
        assert result is None

    def test_extract_multiple(self) -> None:
        """Should return first boxed answer when multiple present."""
        text = r"First \boxed{A}, second \boxed{B}"
        result = extract_boxed_answer(text)
        assert result == "A"

    def test_extract_whitespace_stripped(self) -> None:
        """Whitespace should be stripped from extracted answer."""
        text = r"The answer is \boxed{  42  }"
        result = extract_boxed_answer(text)
        assert result == "42"

    def test_extract_complex_nested(self) -> None:
        """Complex nested expressions should be extracted."""
        text = r"Answer: \boxed{\sqrt{x^2 + y^2}}"
        result = extract_boxed_answer(text)
        assert result == r"\sqrt{x^2 + y^2}"


class TestLLMJudgeBuildPrompt:
    """Tests for LLMJudge prompt building."""

    def test_build_prompt_format(self) -> None:
        """Prompt should contain verbosity bias warning."""
        judge = LLMJudge("test-model")
        prompt = judge.build_prompt("ref answer", "model answer")

        assert "Reference: ref answer" in prompt
        assert "Candidate: model answer" in prompt
        assert "Do NOT reward longer answers" in prompt
        assert "Conciseness is equally valuable" in prompt
        assert '"explanation"' in prompt
        assert '"equivalent"' in prompt
        assert '"confidence"' in prompt

    def test_build_prompt_json_format(self) -> None:
        """Prompt should specify JSON output format."""
        judge = LLMJudge("test-model")
        prompt = judge.build_prompt("A", "B")

        assert "JSON format" in prompt
        assert "true/false" in prompt
        assert "0.0-1.0" in prompt


class TestLLMJudgeParseJson:
    """Tests for LLMJudge JSON parsing."""

    def test_parse_json_response_valid(self) -> None:
        """Valid JSON response should be parsed correctly."""
        judge = LLMJudge("test-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        judge.model = mock_model
        judge.tokenizer = mock_tokenizer
        judge.device = MagicMock()

        prompt = judge.build_prompt("42", "42")
        json_response = '{"explanation": "Both are 42", "equivalent": true, "confidence": 0.95}'
        mock_tokenizer.decode.return_value = prompt + json_response
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_model.generate.return_value = [MagicMock()]

        result = judge._run_comparison("42", "42")

        assert result["explanation"] == "Both are 42"
        assert result["equivalent"] is True
        assert result["confidence"] == 0.95

    def test_parse_json_response_no_json(self) -> None:
        """Response without JSON should return parse error."""
        judge = LLMJudge("test-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        judge.model = mock_model
        judge.tokenizer = mock_tokenizer
        judge.device = MagicMock()

        mock_tokenizer.decode.return_value = "Some prompt textNo JSON here"
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_model.generate.return_value = [MagicMock()]

        result = judge._run_comparison("A", "B")

        assert "Parse error" in result["explanation"]
        assert result["equivalent"] is False
        assert result["confidence"] == 0.0

    def test_parse_json_response_invalid_json(self) -> None:
        """Response with invalid JSON should return parse error."""
        judge = LLMJudge("test-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        judge.model = mock_model
        judge.tokenizer = mock_tokenizer
        judge.device = MagicMock()

        mock_tokenizer.decode.return_value = "Prompt{invalid json}"
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_model.generate.return_value = [MagicMock()]

        result = judge._run_comparison("A", "B")

        assert "Parse error" in result["explanation"]
        assert result["equivalent"] is False


class TestLLMJudgePositionBias:
    """Tests for LLMJudge position bias mitigation."""

    def test_position_bias_mitigation(self) -> None:
        """Should detect and handle position bias."""
        judge = LLMJudge("test-model")

        with patch.object(judge, "_run_comparison") as mock_run:
            # Forward order returns equivalent=True
            # Reverse order returns equivalent=False (position bias detected)
            mock_run.side_effect = [
                {"explanation": "Forward comparison", "equivalent": True, "confidence": 0.9},
                {"explanation": "Reverse comparison", "equivalent": False, "confidence": 0.8},
            ]

            result = judge.judge_single("ref", "model")

            # Should return equivalent=False due to position bias
            assert result["equivalent"] is False
            assert "Position bias detected" in result["explanation"]
            assert "Forward comparison" in result["explanation"]
            assert "Reverse comparison" in result["explanation"]
            # Confidence should be minimum of both
            assert result["confidence"] == 0.8

    def test_position_bias_agreement(self) -> None:
        """Should return result when both orderings agree."""
        judge = LLMJudge("test-model")

        with patch.object(judge, "_run_comparison") as mock_run:
            mock_run.side_effect = [
                {"explanation": "Same answer", "equivalent": True, "confidence": 0.95},
                {"explanation": "Also same", "equivalent": True, "confidence": 0.90},
            ]

            result = judge.judge_single("42", "42")

            assert result["equivalent"] is True
            assert result["explanation"] == "Same answer"


class TestLLMJudgeConfidenceThreshold:
    """Tests for LLMJudge confidence threshold."""

    def test_confidence_threshold_applied(self) -> None:
        """Low confidence should set equivalent to False."""
        judge = LLMJudge("test-model", confidence_threshold=0.7)

        with patch.object(judge, "_run_comparison") as mock_run:
            # Both orderings agree but with low confidence
            mock_run.side_effect = [
                {"explanation": "Looks same", "equivalent": True, "confidence": 0.5},
                {"explanation": "Looks same", "equivalent": True, "confidence": 0.5},
            ]

            result = judge.judge_single("A", "B")

            # Equivalent should be False due to low confidence
            assert result["equivalent"] is False

    def test_confidence_threshold_passed(self) -> None:
        """High confidence should preserve equivalent=True."""
        judge = LLMJudge("test-model", confidence_threshold=0.7)

        with patch.object(judge, "_run_comparison") as mock_run:
            mock_run.side_effect = [
                {"explanation": "Exactly same", "equivalent": True, "confidence": 0.95},
                {"explanation": "Exactly same", "equivalent": True, "confidence": 0.95},
            ]

            result = judge.judge_single("42", "42")

            assert result["equivalent"] is True


class TestLLMJudgeBatch:
    """Tests for LLMJudge batch processing."""

    def test_judge_batch(self) -> None:
        """Batch judging should process all pairs."""
        judge = LLMJudge("test-model")

        with patch.object(judge, "judge_single") as mock_single:
            mock_single.side_effect = [
                {"explanation": "Match 1", "equivalent": True, "confidence": 0.9},
                {"explanation": "Match 2", "equivalent": False, "confidence": 0.8},
                {"explanation": "Match 3", "equivalent": True, "confidence": 0.95},
            ]

            pairs = [("ref1", "model1"), ("ref2", "model2"), ("ref3", "model3")]
            results = judge.judge_batch(pairs)

            assert len(results) == 3
            assert results[0]["equivalent"] is True
            assert results[1]["equivalent"] is False
            assert results[2]["equivalent"] is True


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_overall_accuracy(self) -> None:
        """Overall accuracy should be calculated correctly."""
        results: list[EvaluationResult] = [
            {
                "problem_id": "1",
                "extracted_answer": "42",
                "reference_answer": "42",
                "is_correct": True,
                "judge_explanation": "Match",
                "confidence": 0.9,
            },
            {
                "problem_id": "2",
                "extracted_answer": "24",
                "reference_answer": "42",
                "is_correct": False,
                "judge_explanation": "No match",
                "confidence": 0.8,
            },
            {
                "problem_id": "3",
                "extracted_answer": "42",
                "reference_answer": "42",
                "is_correct": True,
                "judge_explanation": "Match",
                "confidence": 0.95,
            },
        ]

        report = generate_report(results)

        assert report["total_samples"] == 3
        assert report["correct_count"] == 2
        assert report["overall_accuracy"] == pytest.approx(2 / 3)

    def test_grouping_by_subject(self) -> None:
        """Results should be grouped by subject with 'unknown' for None."""
        results: list[EvaluationResult] = [
            {
                "problem_id": "1",
                "extracted_answer": "A",
                "reference_answer": "A",
                "is_correct": True,
                "judge_explanation": "",
                "confidence": 0.9,
                "subject": "Algebra",
            },
            {
                "problem_id": "2",
                "extracted_answer": "B",
                "reference_answer": "B",
                "is_correct": True,
                "judge_explanation": "",
                "confidence": 0.9,
                "subject": "Algebra",
            },
            {
                "problem_id": "3",
                "extracted_answer": "C",
                "reference_answer": "D",
                "is_correct": False,
                "judge_explanation": "",
                "confidence": 0.9,
                "subject": "Geometry",
            },
            {
                "problem_id": "4",
                "extracted_answer": "E",
                "reference_answer": "E",
                "is_correct": True,
                "judge_explanation": "",
                "confidence": 0.9,
            },
            # No subject - should be "unknown"
        ]

        report = generate_report(results)

        assert "Algebra" in report["per_subject_accuracy"]
        assert "Geometry" in report["per_subject_accuracy"]
        assert "unknown" in report["per_subject_accuracy"]
        assert report["per_subject_accuracy"]["Algebra"] == pytest.approx(1.0)
        assert report["per_subject_accuracy"]["Geometry"] == pytest.approx(0.0)
        assert report["per_subject_accuracy"]["unknown"] == pytest.approx(1.0)

    def test_grouping_by_level(self) -> None:
        """Results should be grouped by level with 'unknown' for None."""
        results: list[EvaluationResult] = [
            {
                "problem_id": "1",
                "extracted_answer": "A",
                "reference_answer": "A",
                "is_correct": True,
                "judge_explanation": "",
                "confidence": 0.9,
                "level": 1,
            },
            {
                "problem_id": "2",
                "extracted_answer": "B",
                "reference_answer": "C",
                "is_correct": False,
                "judge_explanation": "",
                "confidence": 0.9,
                "level": 1,
            },
            {
                "problem_id": "3",
                "extracted_answer": "D",
                "reference_answer": "D",
                "is_correct": True,
                "judge_explanation": "",
                "confidence": 0.9,
                "level": 2,
            },
            {
                "problem_id": "4",
                "extracted_answer": "E",
                "reference_answer": "E",
                "is_correct": True,
                "judge_explanation": "",
                "confidence": 0.9,
            },
            # No level - should be "unknown"
        ]

        report = generate_report(results)

        assert "1" in report["per_level_accuracy"]
        assert "2" in report["per_level_accuracy"]
        assert "unknown" in report["per_level_accuracy"]
        assert report["per_level_accuracy"]["1"] == pytest.approx(0.5)
        assert report["per_level_accuracy"]["2"] == pytest.approx(1.0)
        assert report["per_level_accuracy"]["unknown"] == pytest.approx(1.0)

    def test_empty_results(self) -> None:
        """Empty results should produce zero accuracy report."""
        report = generate_report([])

        assert report["total_samples"] == 0
        assert report["correct_count"] == 0
        assert report["overall_accuracy"] == 0.0
        assert report["per_subject_accuracy"] == {}
        assert report["per_level_accuracy"] == {}


class TestEvaluate:
    """Tests for evaluate function."""

    def test_with_fixture(self, test_data_dir: Path) -> None:
        """Evaluate should work with fixture data and mocked judge."""
        fixture_path = str(test_data_dir / "eval_sample.jsonl")
        config = EvaluationConfig(judge_model_name="test-model", confidence_threshold=0.5)

        # Mock LLMJudge to avoid loading real model
        with patch("probing_reflection.evaluation.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge_class.return_value = mock_judge

            # Set up mock judge to return consistent results
            mock_judge.judge_batch.return_value = [
                {"explanation": "42 = 42", "equivalent": True, "confidence": 0.95},
                {"explanation": "24 != 42", "equivalent": False, "confidence": 0.9},
                {"explanation": "0.5 = 1/2", "equivalent": True, "confidence": 0.9},
                {"explanation": "Same", "equivalent": True, "confidence": 0.9},
                {"explanation": "Same", "equivalent": True, "confidence": 0.9},
                {"explanation": "4 = 4", "equivalent": True, "confidence": 0.95},
            ]

            report = evaluate(fixture_path, config)

            # Verify judge was called correctly
            mock_judge_class.assert_called_once()
            mock_judge.load_model.assert_called_once()
            mock_judge.judge_batch.assert_called_once()

            # Check report structure
            assert report["total_samples"] == 7
            assert "overall_accuracy" in report
            assert "results" in report

    def test_extraction_failure(self, tmp_path: Path) -> None:
        """Records without boxed answers should be marked as extraction failures."""
        # Create temp file with no boxed answer
        fixture_data = [
            {
                "problem_id": "no_box",
                "problem": "What is 2+2?",
                "generated": "The answer is 4.",
                "reference_answer": "4",
            },
        ]
        fixture_path = tmp_path / "test.jsonl"
        with open(fixture_path, "w") as f:
            for record in fixture_data:
                f.write(json.dumps(record) + "\n")

        config = EvaluationConfig(judge_model_name="test-model")

        with patch("probing_reflection.evaluation.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge_class.return_value = mock_judge

            report = evaluate(str(fixture_path), config)

            # Should have 1 result for the extraction failure
            assert report["total_samples"] == 1
            assert report["correct_count"] == 0

            # Check that the result shows extraction failure
            result = report["results"][0]
            assert result["extracted_answer"] is None
            assert "No boxed answer found" in result["judge_explanation"]
            assert result["is_correct"] is False

            # Judge should not have been called since no answers to judge
            mock_judge.judge_batch.assert_not_called()

    def test_mixed_extraction_success_failure(self, tmp_path: Path) -> None:
        """Should handle mix of successful and failed extractions."""
        fixture_data = [
            {
                "problem_id": "has_box",
                "problem": "What is 2+2?",
                "generated": r"The answer is \boxed{4}.",
                "reference_answer": "4",
            },
            {
                "problem_id": "no_box",
                "problem": "What is 3+3?",
                "generated": "The answer is 6.",
                "reference_answer": "6",
            },
        ]
        fixture_path = tmp_path / "test.jsonl"
        with open(fixture_path, "w") as f:
            for record in fixture_data:
                f.write(json.dumps(record) + "\n")

        config = EvaluationConfig(judge_model_name="test-model")

        with patch("probing_reflection.evaluation.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge_class.return_value = mock_judge

            mock_judge.judge_batch.return_value = [
                {"explanation": "4 = 4", "equivalent": True, "confidence": 0.95},
            ]

            report = evaluate(str(fixture_path), config)

            assert report["total_samples"] == 2
            # Only one correct (the one with boxed answer)
            assert report["correct_count"] == 1

            # Check results
            results_by_id = {r["problem_id"]: r for r in report["results"]}
            assert results_by_id["has_box"]["is_correct"] is True
            assert results_by_id["no_box"]["is_correct"] is False


class TestLLMJudgeLoadModel:
    """Tests for LLMJudge model loading."""

    def test_load_model_not_called_in_tests(self) -> None:
        """load_model should not be called in unit tests (requires GPU)."""
        # This test verifies the pattern: we mock load_model, never call it for real
        judge = LLMJudge("test-model")

        # Model should not be loaded initially
        assert judge.model is None
        assert judge.tokenizer is None
        assert judge.device is None

    def test_run_comparison_without_load_raises(self) -> None:
        """_run_comparison should raise if model not loaded."""
        judge = LLMJudge("test-model")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            judge._run_comparison("A", "B")
