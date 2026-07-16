"""Inference module for running model inference on datasets.

This module provides utilities for running batch inference on
mathematical reasoning datasets with chain-of-thought prompting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from datasets.load import load_dataset
from tqdm import tqdm

from probing_reflection.batch_utils import decode_generated_tokens, prepare_batch
from probing_reflection.model_utils import GenerativeModel, load_model
from probing_reflection.prompts import format_cot_prompt
from probing_reflection.types import InferenceConfig


def run_inference(config: InferenceConfig) -> Path:
    """Run inference on dataset and save results to JSONL.

    Loads the model, processes the dataset in batches, generates responses,
    and saves results with all required fields.

    Args:
        config: Inference configuration with model, dataset, and output settings.

    Returns:
        Path to the output JSONL file.
    """
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.limit is not None and config.limit < 0:
        raise ValueError("limit must be non-negative")

    model, tokenizer = load_model(config.model_name)
    generative_model = cast(GenerativeModel, model)
    dataset = load_dataset(config.dataset_name, split="test")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = config.batch_size
    results: list[dict[str, str | int]] = []
    num_samples = len(dataset)
    if config.limit is not None:
        num_samples = min(num_samples, config.limit)

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
                outputs = generative_model.generate(
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
                    output = generative_model.generate(
                        input_ids=single_ids,
                        attention_mask=single_mask,
                        max_new_tokens=config.max_new_tokens,
                        repetition_penalty=1.05,
                        do_sample=False,
                    )

                generated_text = decode_generated_tokens(tokenizer, output[0], single_ids.shape[-1])

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
            prompt = prompts[j]
            generated_text = decode_generated_tokens(tokenizer, output, input_ids.shape[-1])

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
