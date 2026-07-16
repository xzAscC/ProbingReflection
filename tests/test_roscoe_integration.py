"""Integration tests for ROSCOE feature in reflection diagnosis.

These tests verify the integration of RoscoeJudge with the reflection
diagnosis pipeline, including judge_type configuration support.

Mock judges are used to avoid actual LLM API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pytest

from probing_reflection.reflection_diagnosis import diagnose_all, diagnose_sample
from probing_reflection.types import (
    ReflectionDiagnosisConfig,
    ReflectionToken,
)


class JudgeProtocol(Protocol):
    """Protocol for judge objects used in diagnosis."""

    def judge(self, text: str) -> list[ReflectionToken]:
        """Analyze text and return reflection tokens."""
        ...


class MockReflectionJudge:
    """Mock ReflectionJudge that detects reflection tokens via keyword matching."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self.model_name: str = model_name
        self._reflection_keywords: set[str] = {
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


class MockRoscoeJudge:
    """Mock RoscoeJudge that simulates ROSCOE metric evaluation."""

    def __init__(self, model_name: str = "mock-model", threshold: float = 3.0) -> None:
        self.model_name: str = model_name
        self.threshold: float = threshold

    def judge(self, text: str) -> list[ReflectionToken]:
        """Return ROSCOE-derived reflection tokens based on text analysis."""
        word_count = len(text.split())
        has_reasoning_markers = any(
            marker in text.lower() for marker in ["because", "therefore", "since", "so", "thus"]
        )
        has_verification = any(
            marker in text.lower() for marker in ["check", "verify", "confirm", "correct"]
        )
        has_coherence = any(
            marker in text.lower() for marker in ["first", "then", "next", "finally", "step"]
        )

        faithfulness = 4.0 if has_reasoning_markers else 3.0
        coherence = 4.0 if has_coherence else 3.0
        informativeness = min(5.0, max(1.0, word_count / 20.0))
        repetition = 4.0 if word_count < 100 else 3.0
        completeness = 4.0 if has_verification else 3.0

        tokens: list[ReflectionToken] = [
            ReflectionToken(
                text="faithfulness",
                category="roscoe_metric",
                context=f"Score: {faithfulness:.1f}/5.0",
                confidence=faithfulness / 5.0,
            ),
            ReflectionToken(
                text="coherence",
                category="roscoe_metric",
                context=f"Score: {coherence:.1f}/5.0",
                confidence=coherence / 5.0,
            ),
            ReflectionToken(
                text="informativeness",
                category="roscoe_metric",
                context=f"Score: {informativeness:.1f}/5.0",
                confidence=informativeness / 5.0,
            ),
            ReflectionToken(
                text="repetition",
                category="roscoe_metric",
                context=f"Score: {repetition:.1f}/5.0",
                confidence=repetition / 5.0,
            ),
            ReflectionToken(
                text="completeness",
                category="roscoe_metric",
                context=f"Score: {completeness:.1f}/5.0",
                confidence=completeness / 5.0,
            ),
        ]

        return tokens


SAMPLE_WITH_REFLECTION = {
    "problem_id": "test_001",
    "problem": "What is 6 times 7?",
    "generated": "Wait, let me reconsider. Actually, the answer is 42.",
    "reference_answer": "42",
    "subject": "Arithmetic",
    "level": 1,
}

SAMPLE_WITH_REASONING = {
    "problem_id": "test_002",
    "problem": "Solve for x: 2x = 8",
    "generated": (
        "First, I need to isolate x. Therefore, I divide both sides by 2. "
        "So x = 4. Let me verify: 2 * 4 = 8. Yes, this is correct."
    ),
    "reference_answer": "4",
    "subject": "Algebra",
    "level": 2,
}

SAMPLE_SIMPLE = {
    "problem_id": "test_003",
    "problem": "What is 2 + 2?",
    "generated": "The answer is 4.",
    "reference_answer": "4",
    "subject": "Arithmetic",
    "level": 1,
}


class TestDiagnoseAllWithReflectionJudge:
    """Integration tests for diagnose_all with judge_type='reflection'."""

    @pytest.fixture
    def mock_reflection_judge(self) -> JudgeProtocol:
        """Create a mock ReflectionJudge for testing."""
        return MockReflectionJudge()

    def test_diagnose_all_with_reflection_judge_uses_reflection_tokens(
        self, tmp_path: Path, mock_reflection_judge: JudgeProtocol
    ) -> None:
        """Test that judge_type='reflection' uses ReflectionJudge behavior."""
        samples_path = tmp_path / "samples.jsonl"
        _ = samples_path.write_text(json.dumps(SAMPLE_WITH_REFLECTION) + "\n")

        output_dir = tmp_path / "output"
        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(output_dir),
            judge_type="reflection",
        )

        samples, report = diagnose_all(config, judge=mock_reflection_judge)

        assert len(samples) == 1
        assert report["total_samples"] == 1
        assert report["total_tokens"] >= 1
        assert "detected" in report["category_distribution"]

    def test_diagnose_all_with_reflection_judge_empty_input(self, tmp_path: Path) -> None:
        """Test that judge_type='reflection' handles empty input."""
        samples_path = tmp_path / "empty.jsonl"
        samples_path.touch()

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
            judge_type="reflection",
        )

        mock_judge = MockReflectionJudge()
        samples, report = diagnose_all(config, judge=mock_judge)

        assert report["total_samples"] == 0
        assert report["total_tokens"] == 0
        assert samples == []


class TestDiagnoseAllWithRoscoeJudge:
    """Integration tests for diagnose_all with judge_type='roscoe'."""

    @pytest.fixture
    def mock_roscoe_judge(self) -> JudgeProtocol:
        """Create a mock RoscoeJudge for testing."""
        return MockRoscoeJudge()

    def test_diagnose_all_with_roscoe_judge_uses_roscoe_metrics(
        self, tmp_path: Path, mock_roscoe_judge: JudgeProtocol
    ) -> None:
        """Test that judge_type='roscoe' uses RoscoeJudge behavior."""
        samples_path = tmp_path / "samples.jsonl"
        _ = samples_path.write_text(json.dumps(SAMPLE_WITH_REASONING) + "\n")

        output_dir = tmp_path / "output"
        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(output_dir),
            judge_type="roscoe",
        )

        samples, report = diagnose_all(config, judge=mock_roscoe_judge)

        assert len(samples) == 1
        assert report["total_samples"] == 1
        assert report["total_tokens"] == 5
        assert "roscoe_metric" in report["category_distribution"]

    def test_diagnose_all_with_roscoe_judge_multiple_samples(
        self, tmp_path: Path, mock_roscoe_judge: JudgeProtocol
    ) -> None:
        """Test that judge_type='roscoe' processes multiple samples correctly."""
        samples_path = tmp_path / "samples.jsonl"
        _ = samples_path.write_text(
            json.dumps(SAMPLE_WITH_REASONING) + "\n" + json.dumps(SAMPLE_SIMPLE) + "\n"
        )

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
            judge_type="roscoe",
        )

        samples, report = diagnose_all(config, judge=mock_roscoe_judge)

        assert len(samples) == 2
        assert report["total_samples"] == 2
        assert report["total_tokens"] == 10


class TestReflectionDiagnosisIntegration:
    """Integration tests for reflection_diagnosis module with RoscoeJudge."""

    @pytest.fixture
    def mock_roscoe_judge(self) -> JudgeProtocol:
        """Create a mock RoscoeJudge for testing."""
        return MockRoscoeJudge()

    def test_diagnose_sample_with_roscoe_judge(self, mock_roscoe_judge: JudgeProtocol) -> None:
        """Test diagnose_sample works correctly with RoscoeJudge."""
        result = diagnose_sample(mock_roscoe_judge, SAMPLE_WITH_REASONING)

        assert "reflection_tokens" in result
        assert "reflection_count" in result
        assert "reflection_density" in result
        assert result["reflection_count"] == 5

        for token in result["reflection_tokens"]:
            assert token["category"] == "roscoe_metric"

    def test_diagnose_all_with_roscoe_judge_full_pipeline(
        self, tmp_path: Path, mock_roscoe_judge: JudgeProtocol
    ) -> None:
        """Test full diagnosis pipeline with RoscoeJudge."""
        samples_data = [
            {**SAMPLE_WITH_REASONING, "subject": "Algebra", "level": 2},
            {**SAMPLE_SIMPLE, "subject": "Arithmetic", "level": 1},
        ]

        samples_path = tmp_path / "samples.jsonl"
        _ = samples_path.write_text("\n".join(json.dumps(sample) for sample in samples_data) + "\n")

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
            judge_type="roscoe",
        )

        _samples, report = diagnose_all(config, judge=mock_roscoe_judge)

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

        assert report["total_samples"] == 2
        assert report["processing_errors"] == 0
        assert "Algebra" in report["per_subject_stats"]
        assert "Arithmetic" in report["per_subject_stats"]
        assert "2" in report["per_level_stats"]
        assert "1" in report["per_level_stats"]

    def test_diagnose_all_comparison_reflection_vs_roscoe(self, tmp_path: Path) -> None:
        """Compare results between ReflectionJudge and RoscoeJudge."""
        samples_path = tmp_path / "samples.jsonl"
        _ = samples_path.write_text(json.dumps(SAMPLE_WITH_REFLECTION) + "\n")

        config = ReflectionDiagnosisConfig(
            input_path=str(samples_path),
            output_dir=str(tmp_path / "output"),
        )

        _, report_reflection = diagnose_all(config, judge=MockReflectionJudge("reflection-model"))
        _, report_roscoe = diagnose_all(config, judge=MockRoscoeJudge("roscoe-model"))

        assert report_reflection["total_samples"] == report_roscoe["total_samples"]
        assert report_reflection["total_tokens"] != report_roscoe["total_tokens"]
        assert report_reflection["category_distribution"] != report_roscoe["category_distribution"]
