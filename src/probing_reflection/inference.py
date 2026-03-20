"""Inference module for running model inference on datasets.

This module provides utilities for running batch inference on
mathematical reasoning datasets with chain-of-thought prompting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM, AutoTokenizer

from probing_reflection.types import InferenceConfig


def prepare_batch(tokenizer: Any, problems: list[str]) -> dict[str, Any]:
    """Prepare a batch of problems for model inference.

    Sets up left padding for Qwen model compatibility.

    Args:
        tokenizer: The tokenizer to use.
        problems: List of problem strings to tokenize.

    Returns:
        Tokenized batch with input_ids and attention_mask.
    """
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = tokenizer(problems, padding=True, return_tensors=None)

    return cast(dict[str, Any], result)


def format_cot_prompt(problem: str) -> str:
    """Format a math problem with chain-of-thought prompting.

    Args:
        problem: The math problem text.

    Returns:
        Formatted prompt with CoT instructions.
    """
    return (
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
        f"\n\nProblem: {problem}\n\nSolution:"
    )


def load_model(config: InferenceConfig) -> tuple[Any, Any]:
    """Load model and tokenizer with bfloat16 precision.

    Args:
        config: Inference configuration containing model name.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
    )
    model.eval()  # type: ignore[no-untyped-call]

    return model, tokenizer


def run_inference(config: InferenceConfig) -> Path:
    """Run inference on dataset and save results to JSONL.

    Loads the model, processes the dataset in batches, generates responses,
    and saves results with all required fields.

    Args:
        config: Inference configuration with model, dataset, and output settings.

    Returns:
        Path to the output JSONL file.
    """
    model, tokenizer = load_model(config)
    dataset = load_dataset(config.dataset_name, split="test")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = config.batch_size
    results: list[dict[str, Any]] = []
    num_samples = len(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(0, num_samples, batch_size), desc="Running inference"):
        batch_end = min(i + batch_size, num_samples)
        batch_items = [dataset[j] for j in range(i, batch_end)]

        problems = [item["problem"] for item in batch_items]
        prompts = [format_cot_prompt(p) for p in problems]

        inputs = prepare_batch(tokenizer, prompts)
        input_ids = torch.tensor(inputs["input_ids"]).to(device)
        attention_mask = torch.tensor(inputs["attention_mask"]).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=1.05,
                    do_sample=False,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for item in batch_items:
                prompt = format_cot_prompt(item["problem"])
                single_input = tokenizer(prompt, return_tensors="pt")
                single_ids = single_input["input_ids"].to(device)
                single_mask = single_input["attention_mask"].to(device)

                with torch.no_grad():
                    output = model.generate(
                        input_ids=single_ids,
                        attention_mask=single_mask,
                        max_new_tokens=config.max_new_tokens,
                        repetition_penalty=1.05,
                        do_sample=False,
                    )

                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_text = str(generated_text)[len(prompt) :].strip()

                result_entry = {
                    "problem_id": item["unique_id"],
                    "problem": item["problem"],
                    "generated": generated_text,
                    "reference_answer": item["answer"],
                    "subject": item["subject"],
                    "level": item["level"],
                    "prompt": prompt,
                }
                results.append(result_entry)
            continue

        for j, (output, item) in enumerate(zip(outputs, batch_items)):  # noqa: B905
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            prompt = prompts[j]
            generated_text = str(generated_text)[len(prompt) :].strip()

            result_entry = {
                "problem_id": item["unique_id"],
                "problem": item["problem"],
                "generated": generated_text,
                "reference_answer": item["answer"],
                "subject": item["subject"],
                "level": item["level"],
                "prompt": prompt,
            }
            results.append(result_entry)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return output_path
