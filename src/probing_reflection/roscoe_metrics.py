"""ROSCOE-based reasoning quality evaluation using LLM-as-Judge."""

from __future__ import annotations

from probing_reflection.judges import BaseLLMJudge
from probing_reflection.prompts import build_roscoe_prompt
from probing_reflection.types import ReflectionToken, RoscoeEvaluation


def _score(value: object) -> float:
    if isinstance(value, (str, int, float)):
        try:
            return float(value)
        except ValueError:
            return 1.0
    return 1.0


class RoscoeJudge(BaseLLMJudge):
    """LLM-based judge for ROSCOE reasoning quality metrics.

    Evaluates step-by-step reasoning quality using 5 core metrics
    on a 1-5 scale: faithfulness, coherence, informativeness,
    repetition, and completeness.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3.5-27B", threshold: float = 3.0) -> None:
        """Initialize the ROSCOE judge.

        Args:
            model_name: Name or path of the judge model.
            threshold: Threshold for passed_filter (default 3.0/5.0).
        """
        super().__init__(model_name)
        self.threshold = threshold

    def judge(self, text: str) -> list[ReflectionToken]:
        """Analyze text for reflection-like patterns using ROSCOE metrics.

        Maps ROSCOE quality evaluation to reflection token format for
        compatibility with the diagnosis pipeline.

        Args:
            text: The text to analyze.

        Returns:
            List of reflection tokens derived from ROSCOE evaluation.
        """
        evaluation = self.evaluate(text)
        return self._evaluation_to_tokens(evaluation, text)

    def _evaluation_to_tokens(
        self, evaluation: RoscoeEvaluation, text: str
    ) -> list[ReflectionToken]:
        """Convert ROSCOE evaluation to reflection tokens.

        Args:
            evaluation: The ROSCOE evaluation result.
            text: The original text being evaluated.

        Returns:
            List of reflection tokens based on evaluation metrics.
        """
        tokens: list[ReflectionToken] = []
        metric_scores: list[tuple[str, str, float]] = [
            ("faithfulness", evaluation["diagnosis"]["faithfulness"], evaluation["faithfulness"]),
            ("coherence", evaluation["diagnosis"]["coherence"], evaluation["coherence"]),
            (
                "informativeness",
                evaluation["diagnosis"]["informativeness"],
                evaluation["informativeness"],
            ),
            ("repetition", evaluation["diagnosis"]["repetition"], evaluation["repetition"]),
            ("completeness", evaluation["diagnosis"]["completeness"], evaluation["completeness"]),
        ]
        for metric_name, _diagnosis, score in metric_scores:
            tokens.append(
                ReflectionToken(
                    text=metric_name,
                    category="roscoe_metric",
                    context=f"Score: {score:.1f}/5.0",
                    confidence=score / 5.0,
                )
            )
        return tokens

    def evaluate(self, text: str, mode: str = "scoring") -> RoscoeEvaluation:
        """Evaluate reasoning quality using ROSCOE metrics.

        Args:
            text: The reasoning text to evaluate.
            mode: Evaluation mode - "scoring", "filtering", or "diagnosis".

        Returns:
            RoscoeEvaluation with all 5 metric scores and derived fields.
        """
        prompt = build_roscoe_prompt(text)
        response = self._run_inference(prompt, max_new_tokens=256)
        return self._parse_roscoe_response(response)

    def _parse_roscoe_response(self, response: str) -> RoscoeEvaluation:
        """Parse LLM response into RoscoeEvaluation.

        Args:
            response: The raw model output text.

        Returns:
            RoscoeEvaluation with all 8 fields populated.
        """
        result = self._parse_json_response(response)

        faithfulness = _score(result.get("faithfulness"))
        coherence = _score(result.get("coherence"))
        informativeness = _score(result.get("informativeness"))
        repetition = _score(result.get("repetition"))
        completeness = _score(result.get("completeness"))

        metrics = [
            max(1.0, min(5.0, m))
            for m in [faithfulness, coherence, informativeness, repetition, completeness]
        ]

        overall = sum(metrics) / 5.0

        def categorize(score: float) -> str:
            if score >= 4.0:
                return "high"
            elif score >= 2.5:
                return "medium"
            return "low"

        diagnosis = {
            "faithfulness": categorize(metrics[0]),
            "coherence": categorize(metrics[1]),
            "informativeness": categorize(metrics[2]),
            "repetition": categorize(metrics[3]),
            "completeness": categorize(metrics[4]),
        }

        return RoscoeEvaluation(
            faithfulness=metrics[0],
            coherence=metrics[1],
            informativeness=metrics[2],
            repetition=metrics[3],
            completeness=metrics[4],
            overall_score=overall,
            passed_filter=overall >= self.threshold,
            diagnosis=diagnosis,
        )
