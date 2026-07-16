"""Reflection token diagnosis and extraction module.

This module provides functions for diagnosing self-reflection tokens
in model outputs using LLM-based judgment.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from probing_reflection.judges import ReflectionJudge
from probing_reflection.prompts import REFLECTION_TAXONOMY
from probing_reflection.roscoe_metrics import RoscoeJudge
from probing_reflection.types import (
    ReflectionAnalysisReport,
    ReflectionDiagnosisConfig,
    ReflectionToken,
    SampleWithReflection,
)


class ReflectionJudgeProtocol(Protocol):
    def judge(self, text: str) -> list[ReflectionToken]: ...


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_dir: Path to output directory.

    Returns:
        Path object for the directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def diagnose_sample(
    judge: ReflectionJudgeProtocol, sample: dict[str, object]
) -> SampleWithReflection:
    """Diagnose a single sample for reflection tokens.

    Analyzes the generated text from a sample to detect self-reflection
    tokens using the provided judge, and computes reflection metrics.

    Args:
        judge: A ReflectionJudge instance (must have load_model() called).
        sample: A dict containing at minimum 'generated' text and other
            sample fields like 'problem_id', 'problem', 'reference_answer'.

    Returns:
        A SampleWithReflection dict with all original fields plus:
            - reflection_tokens: List of detected ReflectionToken objects
            - reflection_count: Number of reflection tokens found
            - reflection_density: Reflection tokens per 100 words
    """
    generated = str(sample.get("generated", ""))

    subject_val = sample.get("subject")
    level_val = sample.get("level")
    subject: str | None = str(subject_val) if subject_val is not None else None
    level: int | None = (
        level_val
        if isinstance(level_val, int)
        else (int(str(level_val)) if level_val is not None else None)
    )

    if not generated.strip():
        return {
            "problem_id": str(sample.get("problem_id", "")),
            "problem": str(sample.get("problem", "")),
            "generated": "",
            "reference_answer": str(sample.get("reference_answer", "")),
            "subject": subject,
            "level": level,
            "reflection_tokens": [],
            "reflection_count": 0,
            "reflection_density": 0.0,
        }

    tokens = judge.judge(generated)
    token_count = len(tokens)

    words = generated.split()
    word_count = len(words)

    density = (token_count / word_count * 100) if word_count > 0 else 0.0

    result: SampleWithReflection = {
        "problem_id": str(sample.get("problem_id", "")),
        "problem": str(sample.get("problem", "")),
        "generated": generated,
        "reference_answer": str(sample.get("reference_answer", "")),
        "subject": subject,
        "level": level,
        "reflection_tokens": tokens,
        "reflection_count": token_count,
        "reflection_density": density,
    }
    return result


def diagnose_all(
    config: ReflectionDiagnosisConfig,
    judge: ReflectionJudgeProtocol | None = None,
) -> tuple[list[SampleWithReflection], ReflectionAnalysisReport]:
    """Diagnose reflection tokens across all samples in a JSONL file.

    Processes each sample through the reflection diagnosis pipeline and
    aggregates statistics about detected reflection tokens.

    Args:
        config: Configuration containing input_path, output_dir, and model settings.
        judge: Optional pre-configured ReflectionJudge or RoscoeJudge. If None,
            creates and loads a new judge based on config.judge_type.

    Returns:
        A tuple containing:
            - List of SampleWithReflection dicts with original fields plus analysis
            - ReflectionAnalysisReport with aggregated statistics
    """
    ensure_output_dir(config.output_dir)

    processed_samples: list[SampleWithReflection] = []
    total_samples = 0
    total_tokens = 0
    all_densities: list[float] = []
    token_frequency: dict[str, int] = defaultdict(int)
    category_distribution: dict[str, int] = defaultdict(int)
    per_subject_raw: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_density": 0.0}
    )
    per_level_raw: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_density": 0.0}
    )
    processing_errors = 0

    input_path = Path(config.input_path)
    with open(input_path) as f:
        lines = f.readlines()

    if judge is None and lines:
        if config.judge_type == "roscoe":
            judge = RoscoeJudge(config.model_name)
        else:
            judge = ReflectionJudge(config.model_name)
        judge.load_model()

    for line in tqdm(lines, desc="Diagnosing reflection tokens"):
        total_samples += 1
        try:
            if judge is None:
                raise RuntimeError("Model not loaded")
            sample = json.loads(line)
            result = diagnose_sample(judge, sample)
            processed_samples.append(result)

            total_tokens += result["reflection_count"]
            all_densities.append(result["reflection_density"])

            for token in result["reflection_tokens"]:
                token_frequency[token["text"]] += 1
                category_distribution[token["category"]] += 1

            subject = result.get("subject")
            if subject is not None:
                per_subject_raw[subject]["count"] += 1
                per_subject_raw[subject]["total_density"] += result["reflection_density"]

            level = result.get("level")
            if level is not None:
                level_key = str(level)
                per_level_raw[level_key]["count"] += 1
                per_level_raw[level_key]["total_density"] += result["reflection_density"]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
            processing_errors += 1

    avg_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0.0
    overall_density = sum(all_densities) / len(all_densities) if all_densities else 0.0

    per_subject_stats: dict[str, dict[str, float | int]] = {}
    for subject, stats in per_subject_raw.items():
        count = int(stats["count"])
        total_density = float(stats["total_density"])
        avg_density = total_density / count if count > 0 else 0.0
        per_subject_stats[subject] = {"count": count, "avg_density": avg_density}

    per_level_stats: dict[str, dict[str, float | int]] = {}
    for level_key, stats in per_level_raw.items():
        count = int(stats["count"])
        total_density = float(stats["total_density"])
        avg_density = total_density / count if count > 0 else 0.0
        per_level_stats[level_key] = {"count": count, "avg_density": avg_density}

    report = ReflectionAnalysisReport(
        total_samples=total_samples,
        total_tokens=total_tokens,
        avg_tokens_per_sample=avg_tokens_per_sample,
        overall_density=overall_density,
        token_frequency=dict(token_frequency),
        category_distribution=dict(category_distribution),
        per_subject_stats=per_subject_stats,
        per_level_stats=per_level_stats,
        processing_errors=processing_errors,
    )

    return processed_samples, report


def write_analysis_report(report: ReflectionAnalysisReport, output_path: Path | str) -> None:
    """Write a reflection analysis report to a JSON file.

    Args:
        report: The ReflectionAnalysisReport to write.
        output_path: Path to the output JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def write_analyzed_jsonl(
    samples: list[SampleWithReflection],
    output_path: Path | str,
) -> None:
    """Write analyzed samples to a JSONL file.

    Args:
        samples: List of samples with reflection analysis results.
        output_path: Path to the output JSONL file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


__all__ = [
    "REFLECTION_TAXONOMY",
    "ReflectionJudge",
    "RoscoeJudge",
    "diagnose_all",
    "diagnose_sample",
    "ensure_output_dir",
    "write_analysis_report",
    "write_analyzed_jsonl",
]
