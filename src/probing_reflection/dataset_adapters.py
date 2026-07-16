"""Unified dataset loading functions for evaluation datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict

from datasets.load import load_dataset


class DatasetSample(TypedDict):
    """Unified sample format for all datasets.

    Attributes:
        problem_id: Unique identifier for the problem.
        problem: The problem text/question.
        reference_answer: The ground truth answer.
        subject: Optional subject category.
        level: Optional difficulty level.
    """

    problem_id: str
    problem: str
    reference_answer: str
    subject: NotRequired[str | None]
    level: NotRequired[int | None]


def _record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Dataset rows must be mappings")
    return value


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_level(value: object) -> int | None:
    return int(str(value)) if value is not None else None


def load_math500() -> list[DatasetSample]:
    """Load MATH-500 dataset from HuggingFace.

    Returns:
        List of 500 DatasetSample dicts.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    samples: list[DatasetSample] = []
    for raw_item in dataset:
        item = _record(raw_item)
        sample: DatasetSample = {
            "problem_id": str(item["unique_id"]),
            "problem": str(item["problem"]),
            "reference_answer": str(item["answer"]),
            "subject": _optional_text(item.get("subject")),
            "level": _optional_level(item.get("level")),
        }
        samples.append(sample)

    return samples


def load_aime() -> list[DatasetSample]:
    """Load AIME 2024 dataset from HuggingFace.

    Returns:
        List of ~90 DatasetSample dicts.
    """
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")

    samples: list[DatasetSample] = []
    for raw_item in dataset:
        item = _record(raw_item)
        # Map available fields - check for id/problem_id variations
        problem_id = str(item.get("id", item.get("problem_id", "")))
        problem = str(item.get("problem", item.get("question", "")))

        sample: DatasetSample = {
            "problem_id": problem_id,
            "problem": problem,
            "reference_answer": str(item.get("answer", item.get("solution", ""))),
            "subject": "math",
            "level": None,
        }
        samples.append(sample)

    return samples


def load_gpqa_diamond() -> list[DatasetSample]:
    """Load GPQA Diamond dataset from HuggingFace.

    Returns:
        List of 198 DatasetSample dicts.
    """
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    samples: list[DatasetSample] = []
    for idx, raw_item in enumerate(dataset):
        item = _record(raw_item)
        # Format question with choices
        question = str(item.get("Question", item.get("question", "")))
        choice_1 = str(item.get("Choice 1", item.get("choice_1", "")))
        choice_2 = str(item.get("Choice 2", item.get("choice_2", "")))
        choice_3 = str(item.get("Choice 3", item.get("choice_3", "")))
        choice_4 = str(item.get("Choice 4", item.get("choice_4", "")))

        formatted_problem = (
            f"Question: {question}\n\nA) {choice_1}\nB) {choice_2}\nC) {choice_3}\nD) {choice_4}"
        )

        # Get the correct answer text
        correct_answer = str(item.get("Correct Answer", item.get("correct_answer", "")))

        sample: DatasetSample = {
            "problem_id": f"gpqa_diamond_{idx}",
            "problem": formatted_problem,
            "reference_answer": correct_answer,
            "subject": "science",
            "level": None,
        }
        samples.append(sample)

    return samples
