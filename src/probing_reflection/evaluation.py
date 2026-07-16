"""Evaluation utilities for answer extraction and comparison.

This module provides functions for extracting and comparing answers from
model outputs, particularly for LaTeX-formatted boxed answers.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from probing_reflection.judges import AnswerJudge
from probing_reflection.types import (
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the first \\boxed{...} pattern from text.

    Handles nested braces by counting brace depth to find the matching
    closing brace.

    Args:
        text: The text to search for a boxed answer.

    Returns:
        The extracted answer with whitespace stripped, or None if no
        \\boxed{...} pattern is found.
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

    judge = AnswerJudge(
        config.judge_model_name,
        config.confidence_threshold,
    )
    judge.load_model()

    pairs_to_judge: list[tuple[dict[str, object], str]] = []
    results: list[EvaluationResult] = []

    def optional_subject(record: dict[str, object]) -> str | None:
        value = record.get("subject")
        return str(value) if value is not None else None

    def optional_level(record: dict[str, object]) -> int | None:
        value = record.get("level")
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return int(str(value))

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
                    subject=optional_subject(record),
                    level=optional_level(record),
                )
            )
        else:
            pairs_to_judge.append((record, extracted))

    if pairs_to_judge:
        pairs = [
            (str(record.get("reference_answer", "")), extracted)
            for record, extracted in pairs_to_judge
        ]
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
                    subject=optional_subject(record),
                    level=optional_level(record),
                )
            )

    return generate_report(results)


def evaluate_gpqa(jsonl_path: str, config: EvaluationConfig | None = None) -> EvaluationReport:
    """Evaluate GPQA outputs using LLM judge (full text comparison).

    Unlike evaluate(), this does NOT use boxed extraction. It compares
    the full model output against the reference answer text.

    Args:
        jsonl_path: Path to JSONL file with model outputs.
        config: Evaluation configuration.

    Returns:
        EvaluationReport with accuracy metrics.
    """
    if config is None:
        config = EvaluationConfig()

    records: list[dict[str, object]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    judge = AnswerJudge(config.judge_model_name, config.confidence_threshold)
    judge.load_model()

    results: list[EvaluationResult] = []
    for record in records:
        generated = str(record.get("generated", ""))
        reference = str(record.get("reference_answer", ""))

        if not generated.strip():
            results.append(
                EvaluationResult(
                    problem_id=str(record.get("problem_id", "")),
                    extracted_answer=None,
                    reference_answer=reference,
                    is_correct=False,
                    judge_explanation="No output generated",
                    confidence=0.0,
                )
            )
        else:
            verdict = judge.judge_single(reference, generated)
            results.append(
                EvaluationResult(
                    problem_id=str(record.get("problem_id", "")),
                    extracted_answer=generated,
                    reference_answer=reference,
                    is_correct=verdict["equivalent"],
                    judge_explanation=verdict["explanation"],
                    confidence=verdict["confidence"],
                )
            )

    return generate_report(results)
