"""LLM-based judge utilities for answer evaluation and reflection detection.

This module provides a base class for LLM judges and specific implementations
for answer comparison and reflection token detection. The base class handles
model loading, inference, and JSON response parsing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from probing_reflection.batch_utils import decode_generated_tokens
from probing_reflection.model_utils import GenerativeModel, get_device, load_model
from probing_reflection.prompts import build_comparison_prompt, build_diagnosis_prompt
from probing_reflection.types import JudgeVerdict, ReflectionToken


def _parse_json_response(response: str) -> dict[str, object]:
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return {}

        parsed: object = json.loads(response[start_idx:end_idx])
        if not isinstance(parsed, dict):
            return {}
        return {str(key): value for key, value in parsed.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (str, int, float)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


class BaseLLMJudge:
    """Base class for LLM-based judgment tasks.

    Provides common functionality for loading models, running inference,
    and parsing JSON responses.

    Attributes:
        model_name: Name or path of the judge model.
        model: The loaded language model (None until load_model is called).
        tokenizer: The loaded tokenizer (None until load_model is called).
        device: Device the model is running on (None until load_model is called).
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the judge with a model name.

        Args:
            model_name: Name or path of the judge model.
        """
        self.model_name = model_name
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.device: torch.device | None = None

    def load_model(self) -> None:
        """Load the model and tokenizer.

        Uses bfloat16 precision on CUDA and float32 on CPU.
        Sets up pad_token to eos_token if not present.
        """
        self.model, self.tokenizer = load_model(self.model_name)
        self.device = get_device()

    def _loaded_components(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model, self.tokenizer, self.device

    def _run_inference(self, prompt: str, max_new_tokens: int = 256) -> str:
        model, tokenizer, device = self._loaded_components()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = cast(GenerativeModel, model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        return decode_generated_tokens(tokenizer, outputs[0], input_ids.shape[-1])

    def _parse_json_response(self, response: str) -> dict[str, object]:
        """Parse JSON from a model response.

        Extracts and parses JSON from a response string, handling malformed
        and missing JSON gracefully.

        Args:
            response: The raw model output text.

        Returns:
            Parsed dict, or empty dict on failure.
        """
        return _parse_json_response(response)


class AnswerJudge(BaseLLMJudge):
    """LLM-based judge for comparing answer equivalence.

    Uses a language model to determine if two answers are semantically
    equivalent, with position bias mitigation through bidirectional comparison.

    Attributes:
        confidence_threshold: Minimum confidence to accept equivalent=True.
    """

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize the answer judge.

        Args:
            model_name: Name or path of the judge model.
            confidence_threshold: Minimum confidence to accept equivalent=True.
                Defaults to 0.7.
        """
        super().__init__(model_name)
        self.confidence_threshold = confidence_threshold

    def _run_comparison(self, answer_a: str, answer_b: str) -> JudgeVerdict:
        """Run a single comparison and parse the result.

        Args:
            answer_a: First answer in the comparison.
            answer_b: Second answer in the comparison.

        Returns:
            JudgeVerdict with explanation, equivalent, and confidence.
        """
        prompt = build_comparison_prompt(answer_a, answer_b)
        response = self._run_inference(prompt, max_new_tokens=256)

        result = self._parse_json_response(response)

        if not result:
            return {
                "explanation": "Parse error: No valid JSON found",
                "equivalent": False,
                "confidence": 0.0,
            }

        return {
            "explanation": str(result.get("explanation", "")),
            "equivalent": bool(result.get("equivalent", False)),
            "confidence": _to_float(result.get("confidence")),
        }

    def _apply_confidence_threshold(self, verdict: JudgeVerdict) -> JudgeVerdict:
        """Apply confidence threshold to a verdict.

        If confidence is below threshold, sets equivalent to False.

        Args:
            verdict: The original verdict.

        Returns:
            Verdict with equivalent potentially set to False.
        """
        if verdict["confidence"] < self.confidence_threshold:
            return {
                "explanation": verdict["explanation"],
                "equivalent": False,
                "confidence": verdict["confidence"],
            }
        return verdict

    def judge_single(self, ref_answer: str, model_answer: str) -> JudgeVerdict:
        """Judge a single answer pair with position bias mitigation.

        Runs comparison in both orderings and returns equivalent=False
        if results disagree (conservative approach).

        Args:
            ref_answer: The reference/gold answer.
            model_answer: The model's answer to evaluate.

        Returns:
            JudgeVerdict with the final decision.
        """
        verdict_ab = self._run_comparison(ref_answer, model_answer)
        verdict_ab = self._apply_confidence_threshold(verdict_ab)

        # Position bias mitigation: compare in reversed order
        verdict_ba = self._run_comparison(model_answer, ref_answer)
        verdict_ba = self._apply_confidence_threshold(verdict_ba)

        if verdict_ab["equivalent"] != verdict_ba["equivalent"]:
            return {
                "explanation": (
                    f"Position bias detected. Forward: {verdict_ab['explanation']}. "
                    f"Reverse: {verdict_ba['explanation']}"
                ),
                "equivalent": False,
                "confidence": min(verdict_ab["confidence"], verdict_ba["confidence"]),
            }

        return verdict_ab

    def judge_batch(self, pairs: list[tuple[str, str]]) -> list[JudgeVerdict]:
        """Judge multiple answer pairs.

        Args:
            pairs: List of (reference_answer, model_answer) tuples.

        Returns:
            List of JudgeVerdicts, one per pair.
        """
        return [self.judge_single(ref, model) for ref, model in pairs]


class ReflectionJudge(BaseLLMJudge):
    """LLM-based judge for identifying reflection tokens in text.

    Uses a language model to detect self-reflection tokens that indicate
    metacognitive moments like hesitation, verification, or self-correction.
    """

    def judge(self, text: str) -> list[ReflectionToken]:
        """Extract reflection tokens from text.

        Args:
            text: The text to analyze for reflection tokens.

        Returns:
            List of detected reflection tokens with their metadata.
        """
        prompt = build_diagnosis_prompt(text)
        response = self._run_inference(prompt, max_new_tokens=512)

        return self._parse_tokens_response(response)

    def _parse_tokens_response(self, response: str) -> list[ReflectionToken]:
        """Parse the JSON response containing reflection tokens.

        Args:
            response: The raw model output text.

        Returns:
            List of ReflectionToken dicts, or empty list if parsing fails.
        """
        result = self._parse_json_response(response)

        if "tokens" not in result:
            return []

        tokens: list[ReflectionToken] = []
        raw_tokens = result["tokens"]
        if not isinstance(raw_tokens, list):
            return []

        for token_data in raw_tokens:
            if not isinstance(token_data, Mapping):
                continue
            if all(k in token_data for k in ("text", "category", "context", "confidence")):
                tokens.append(
                    ReflectionToken(
                        text=str(token_data["text"]),
                        category=str(token_data["category"]),
                        context=str(token_data["context"]),
                        confidence=_to_float(token_data["confidence"]),
                    )
                )
        return tokens
