"""Evaluation utilities for answer extraction and comparison.

This module provides functions for extracting and comparing answers from
model outputs, particularly for LaTeX-formatted boxed answers.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from probing_reflection.types import (
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
    JudgeVerdict,
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the first \boxed{...} pattern from text.

    Handles nested braces by counting brace depth to find the matching
    closing brace.

    Args:
        text: The text to search for a boxed answer.

    Returns:
        The extracted answer with whitespace stripped, or None if no
        \boxed{...} pattern is found.

    Examples:
        >>> extract_boxed_answer("The answer is \\\\boxed{42}")
        '42'
        >>> extract_boxed_answer("Result: \\\\boxed{\\\\frac{1}{2}}")
        '\\\\frac{1}{2}'
        >>> extract_boxed_answer("No answer here") is None
        True
        >>> extract_boxed_answer("First \\\\boxed{A}, second \\\\boxed{B}")
        'A'
    """
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None

    start_pos = match.end()
    depth = 1
    pos = start_pos

    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth == 0:
        content = text[start_pos : pos - 1]
        return content.strip()

    return None


class LLMJudge:
    """LLM-based judge for comparing answer equivalence.

    Uses a language model to determine if two answers are semantically
    equivalent, with position bias mitigation through bidirectional comparison.

    Attributes:
        model_name: Name or path of the judge model.
        batch_size: Batch size for inference.
        confidence_threshold: Minimum confidence to accept equivalent=True.
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        device: Device the model is running on.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize the LLM judge.

        Args:
            model_name: Name or path of the judge model.
            batch_size: Batch size for inference. Defaults to 8.
            confidence_threshold: Minimum confidence to accept equivalent=True.
                If confidence is below threshold, equivalent is set to False.
                Defaults to 0.7.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.device: torch.device | None = None

    def load_model(self) -> None:
        """Load the model and tokenizer.

        Follows the inference.py pattern with bfloat16 precision on CUDA
        and float32 on CPU. Sets up pad_token to eos_token if not present.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        self.model.eval()  # type: ignore[no-untyped-call]
        self.model = self.model.to(self.device)  # type: ignore[arg-type]

    def build_prompt(self, answer_a: str, answer_b: str) -> str:
        """Create a comparison prompt for the judge model.

        The prompt is designed to mitigate verbosity bias by explicitly
        instructing not to reward longer answers.

        Args:
            answer_a: The first answer to compare.
            answer_b: The second answer to compare.

        Returns:
            Formatted prompt string for the judge model.
        """
        return f"""Compare these two mathematical answers. Are they semantically equivalent?

Reference: {answer_a}
Candidate: {answer_b}

CRITICAL: Do NOT reward longer answers. Conciseness is equally valuable.

First explain your reasoning step by step.
Then provide your verdict.

Respond in JSON format:
{{"explanation": "...", "equivalent": true/false, "confidence": 0.0-1.0}}"""

    def _run_comparison(self, answer_a: str, answer_b: str) -> JudgeVerdict:
        """Run a single comparison and parse the result.

        Args:
            answer_a: First answer in the comparison.
            answer_b: Second answer in the comparison.

        Returns:
            JudgeVerdict with explanation, equivalent, and confidence.
        """
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = self.build_prompt(answer_a, answer_b)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore[operator]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = str(generated_text)[len(prompt) :].strip()

        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)

                verdict: JudgeVerdict = {
                    "explanation": str(result.get("explanation", "")),
                    "equivalent": bool(result.get("equivalent", False)),
                    "confidence": float(result.get("confidence", 0.0)),
                }
                return verdict
            return {
                "explanation": "Parse error: No JSON found in response",
                "equivalent": False,
                "confidence": 0.0,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return {
                "explanation": "Parse error: Invalid JSON in response",
                "equivalent": False,
                "confidence": 0.0,
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

        Runs comparison in both orderings (ref vs model, model vs ref)
        and returns equivalent=False if results disagree (conservative approach).

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

        Note: This processes pairs sequentially with full position bias
        mitigation for each pair. True batching would require more complex
        handling of bidirectional comparisons.

        Args:
            pairs: List of (reference_answer, model_answer) tuples.

        Returns:
            List of JudgeVerdicts, one per pair.
        """
        results: list[JudgeVerdict] = []
        for ref_answer, model_answer in pairs:
            verdict = self.judge_single(ref_answer, model_answer)
            results.append(verdict)
        return results


def generate_report(results: list[EvaluationResult]) -> EvaluationReport:
    """Generate an aggregated evaluation report from individual results.

    Calculates overall accuracy and groups results by subject and level
    to provide per-category breakdowns.

    Args:
        results: List of EvaluationResult dictionaries from evaluating samples.

    Returns:
        EvaluationReport with overall statistics and per-category breakdowns.
    """
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    subject_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        subject = r.get("subject") or "unknown"
        subject_groups[subject].append(r)

    per_subject = {
        subj: sum(1 for r in group if r["is_correct"]) / len(group)
        for subj, group in subject_groups.items()
    }

    level_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        level = r.get("level")
        level_key = str(level) if level is not None else "unknown"
        level_groups[level_key].append(r)

    per_level = {
        level: sum(1 for r in group if r["is_correct"]) / len(group)
        for level, group in level_groups.items()
    }

    return EvaluationReport(
        overall_accuracy=correct / total if total > 0 else 0.0,
        total_samples=total,
        correct_count=correct,
        per_subject_accuracy=per_subject,
        per_level_accuracy=per_level,
        results=results,
    )


def evaluate(jsonl_path: str, config: EvaluationConfig) -> EvaluationReport:
    """Evaluate model outputs against reference answers.

    Loads a JSONL file containing model outputs, extracts boxed answers,
    and uses an LLM judge to compare against reference answers.

    Args:
        jsonl_path: Path to the JSONL file containing model outputs.
            Each line should be a JSON object with fields: problem_id, problem,
            generated, reference_answer, subject (optional), level (optional).
        config: Evaluation configuration with judge model settings.

    Returns:
        EvaluationReport with overall accuracy and per-category breakdowns.
    """
    records: list[dict[str, object]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    judge = LLMJudge(
        config.judge_model_name,
        config.batch_size,
        config.confidence_threshold,
    )
    judge.load_model()

    pairs_to_judge: list[tuple[dict[str, object], str]] = []
    results: list[EvaluationResult] = []

    for record in records:
        generated = str(record.get("generated", ""))
        extracted = extract_boxed_answer(generated)
        if extracted is None:
            results.append(
                EvaluationResult(
                    problem_id=str(record.get("problem_id", "")),
                    extracted_answer=None,
                    reference_answer=str(record.get("reference_answer", "")),
                    is_correct=False,
                    judge_explanation="No boxed answer found",
                    confidence=0.0,
                    subject=record.get("subject"),  # type: ignore[typeddict-item]
                    level=record.get("level"),  # type: ignore[typeddict-item]
                )
            )
        else:
            pairs_to_judge.append((record, extracted))

    if pairs_to_judge:
        pairs = [(p[1], str(p[0].get("reference_answer", ""))) for p in pairs_to_judge]
        verdicts = judge.judge_batch(pairs)

        for (record, extracted), verdict in zip(pairs_to_judge, verdicts, strict=True):
            results.append(
                EvaluationResult(
                    problem_id=str(record.get("problem_id", "")),
                    extracted_answer=extracted,
                    reference_answer=str(record.get("reference_answer", "")),
                    is_correct=verdict["equivalent"],
                    judge_explanation=verdict["explanation"],
                    confidence=verdict["confidence"],
                    subject=record.get("subject"),  # type: ignore[typeddict-item]
                    level=record.get("level"),  # type: ignore[typeddict-item]
                )
            )

    return generate_report(results)
