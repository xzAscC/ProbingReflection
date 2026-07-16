"""Tests for RoscoeJudge class.

These tests verify the ROSCOE-based reasoning quality evaluation including
JSON parsing, score clamping, overall score calculation, filter logic,
and diagnosis categorization.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from probing_reflection.roscoe_metrics import RoscoeJudge

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def judge() -> RoscoeJudge:
    """Create a RoscoeJudge with default parameters."""
    return RoscoeJudge()


@pytest.fixture
def judge_custom() -> RoscoeJudge:
    """Create a RoscoeJudge with custom parameters."""
    return RoscoeJudge(model_name="custom-model", threshold=4.0)


# ============================================================================
# TESTS
# ============================================================================


class TestRoscoeJudgeInstantiation:
    """Tests for RoscoeJudge instantiation."""

    def test_instantiation_default_params(self) -> None:
        """Test RoscoeJudge() with defaults."""
        judge = RoscoeJudge()

        assert judge.model_name == "Qwen/Qwen3.5-27B"
        assert judge.threshold == 3.0

    def test_instantiation_custom_params(self) -> None:
        """Test with custom model_name and threshold."""
        judge = RoscoeJudge(model_name="custom-model-name", threshold=4.5)

        assert judge.model_name == "custom-model-name"
        assert judge.threshold == 4.5


class TestRoscoeJudgeParsing:
    """Tests for _parse_roscoe_response method."""

    def test_parse_response_valid_json(self, judge: RoscoeJudge) -> None:
        """Test parsing valid JSON response."""
        response = """{
            "faithfulness": 4.5,
            "coherence": 3.8,
            "informativeness": 4.2,
            "repetition": 5.0,
            "completeness": 3.5
        }"""

        result = judge._parse_roscoe_response(response)

        assert isinstance(result, dict)
        assert result["faithfulness"] == 4.5
        assert result["coherence"] == 3.8
        assert result["informativeness"] == 4.2
        assert result["repetition"] == 5.0
        assert result["completeness"] == 3.5

    def test_score_clamping_high(self, judge: RoscoeJudge) -> None:
        """Test scores > 5.0 are clamped to 5.0."""
        response = """{
            "faithfulness": 6.0,
            "coherence": 10.0,
            "informativeness": 7.5,
            "repetition": 8.0,
            "completeness": 5.5
        }"""

        result = judge._parse_roscoe_response(response)

        assert result["faithfulness"] == 5.0
        assert result["coherence"] == 5.0
        assert result["informativeness"] == 5.0
        assert result["repetition"] == 5.0
        assert result["completeness"] == 5.0

    def test_score_clamping_low(self, judge: RoscoeJudge) -> None:
        """Test scores < 1.0 are clamped to 1.0."""
        response = """{
            "faithfulness": 0.0,
            "coherence": -5.0,
            "informativeness": 0.5,
            "repetition": -1.0,
            "completeness": 0.8
        }"""

        result = judge._parse_roscoe_response(response)

        assert result["faithfulness"] == 1.0
        assert result["coherence"] == 1.0
        assert result["informativeness"] == 1.0
        assert result["repetition"] == 1.0
        assert result["completeness"] == 1.0

    def test_empty_response_handling(self, judge: RoscoeJudge) -> None:
        """Test empty/missing JSON returns defaults (1.0)."""
        response = ""

        result = judge._parse_roscoe_response(response)

        # All metrics should default to 1.0
        assert result["faithfulness"] == 1.0
        assert result["coherence"] == 1.0
        assert result["informativeness"] == 1.0
        assert result["repetition"] == 1.0
        assert result["completeness"] == 1.0

    def test_malformed_json_handling(self, judge: RoscoeJudge) -> None:
        """Test malformed JSON returns defaults."""
        response = '{"faithfulness": 4.0, "coherence": missing_quote, }'

        result = judge._parse_roscoe_response(response)

        # Should return defaults due to parsing failure
        assert result["faithfulness"] == 1.0
        assert result["coherence"] == 1.0


class TestRoscoeJudgeScoreCalculation:
    """Tests for overall score calculation and filter logic."""

    def test_overall_score_calculation(self, judge: RoscoeJudge) -> None:
        """Test mean of 5 metrics."""
        response = """{
            "faithfulness": 4.0,
            "coherence": 3.0,
            "informativeness": 5.0,
            "repetition": 2.0,
            "completeness": 4.0
        }"""

        result = judge._parse_roscoe_response(response)

        # (4.0 + 3.0 + 5.0 + 2.0 + 4.0) / 5 = 3.6
        assert result["overall_score"] == pytest.approx(3.6)

    def test_passed_filter_logic_above_threshold(self, judge: RoscoeJudge) -> None:
        """Test threshold comparison - above default threshold (3.0)."""
        response = """{
            "faithfulness": 4.0,
            "coherence": 4.0,
            "informativeness": 4.0,
            "repetition": 4.0,
            "completeness": 4.0
        }"""

        result = judge._parse_roscoe_response(response)

        # overall = 4.0, threshold = 3.0, so passed_filter = True
        assert result["passed_filter"] is True

    def test_passed_filter_logic_below_threshold(self, judge: RoscoeJudge) -> None:
        """Test threshold comparison - below default threshold (3.0)."""
        response = """{
            "faithfulness": 2.0,
            "coherence": 2.0,
            "informativeness": 2.0,
            "repetition": 2.0,
            "completeness": 2.0
        }"""

        result = judge._parse_roscoe_response(response)

        # overall = 2.0, threshold = 3.0, so passed_filter = False
        assert result["passed_filter"] is False

    def test_passed_filter_with_custom_threshold(self, judge_custom: RoscoeJudge) -> None:
        """Test threshold comparison with custom threshold (4.0)."""
        response = """{
            "faithfulness": 3.5,
            "coherence": 3.5,
            "informativeness": 3.5,
            "repetition": 3.5,
            "completeness": 3.5
        }"""

        result = judge_custom._parse_roscoe_response(response)

        # overall = 3.5, custom threshold = 4.0, so passed_filter = False
        assert result["passed_filter"] is False


class TestRoscoeJudgeDiagnosis:
    """Tests for diagnosis categorization."""

    def test_diagnosis_categories_high(self, judge: RoscoeJudge) -> None:
        """Test high category (>=4.0)."""
        response = """{
            "faithfulness": 4.5,
            "coherence": 5.0,
            "informativeness": 4.0,
            "repetition": 4.0,
            "completeness": 4.8
        }"""

        result = judge._parse_roscoe_response(response)
        diagnosis = result["diagnosis"]

        assert diagnosis["faithfulness"] == "high"
        assert diagnosis["coherence"] == "high"
        assert diagnosis["informativeness"] == "high"
        assert diagnosis["repetition"] == "high"
        assert diagnosis["completeness"] == "high"

    def test_diagnosis_categories_medium(self, judge: RoscoeJudge) -> None:
        """Test medium category (>=2.5, <4.0)."""
        response = """{
            "faithfulness": 3.0,
            "coherence": 2.5,
            "informativeness": 3.5,
            "repetition": 3.9,
            "completeness": 2.6
        }"""

        result = judge._parse_roscoe_response(response)
        diagnosis = result["diagnosis"]

        assert diagnosis["faithfulness"] == "medium"
        assert diagnosis["coherence"] == "medium"
        assert diagnosis["informativeness"] == "medium"
        assert diagnosis["repetition"] == "medium"
        assert diagnosis["completeness"] == "medium"

    def test_diagnosis_categories_low(self, judge: RoscoeJudge) -> None:
        """Test low category (<2.5)."""
        response = """{
            "faithfulness": 1.0,
            "coherence": 2.0,
            "informativeness": 1.5,
            "repetition": 2.4,
            "completeness": 1.2
        }"""

        result = judge._parse_roscoe_response(response)
        diagnosis = result["diagnosis"]

        assert diagnosis["faithfulness"] == "low"
        assert diagnosis["coherence"] == "low"
        assert diagnosis["informativeness"] == "low"
        assert diagnosis["repetition"] == "low"
        assert diagnosis["completeness"] == "low"

    def test_diagnosis_categories_mixed(self, judge: RoscoeJudge) -> None:
        """Test mixed categories across metrics."""
        response = """{
            "faithfulness": 4.5,
            "coherence": 3.0,
            "informativeness": 2.0,
            "repetition": 5.0,
            "completeness": 1.0
        }"""

        result = judge._parse_roscoe_response(response)
        diagnosis = result["diagnosis"]

        assert diagnosis["faithfulness"] == "high"
        assert diagnosis["coherence"] == "medium"
        assert diagnosis["informativeness"] == "low"
        assert diagnosis["repetition"] == "high"
        assert diagnosis["completeness"] == "low"


class TestRoscoeJudgeEvaluate:
    """Tests for the evaluate method with mocked inference."""

    def test_evaluate_returns_roscoe_evaluation(self, judge: RoscoeJudge) -> None:
        """Test evaluate returns proper RoscoeEvaluation structure."""
        mock_response = """{
            "faithfulness": 4.0,
            "coherence": 3.5,
            "informativeness": 4.5,
            "repetition": 3.0,
            "completeness": 4.0
        }"""

        with patch.object(judge, "_run_inference", return_value=mock_response):
            result = judge.evaluate("Test reasoning text")

        assert isinstance(result, dict)
        # Verify all required RoscoeEvaluation keys are present
        assert "faithfulness" in result
        assert "coherence" in result
        assert "informativeness" in result
        assert "repetition" in result
        assert "completeness" in result
        assert "overall_score" in result
        assert "passed_filter" in result
        assert "diagnosis" in result

    def test_evaluate_calls_run_inference(self, judge: RoscoeJudge) -> None:
        """Test evaluate calls _run_inference with correct prompt."""
        mock_response = (
            '{"faithfulness": 5.0, "coherence": 5.0, "informativeness": 5.0, '
            '"repetition": 5.0, "completeness": 5.0}'
        )

        with patch.object(judge, "_run_inference", return_value=mock_response) as mock_inference:
            judge.evaluate("Sample text for evaluation")

            mock_inference.assert_called_once()
            # Verify the prompt contains expected elements
            call_args = mock_inference.call_args
            assert call_args is not None
            prompt_arg = call_args[0][0]
            assert "Sample text for evaluation" in prompt_arg
