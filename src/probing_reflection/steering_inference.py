"""Steering inference module for applying steering vectors during model inference.

This module provides utilities for loading steering vectors and models
with 4-bit quantization, and creating hooks to apply steering during inference.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from probing_reflection.types import SteeringInferenceConfig


def get_output_path(
    dataset: str, condition: str, base_dir: str = "outputs/steering_experiments"
) -> Path:
    """Get output path for a specific dataset and condition.

    Args:
        dataset: Dataset name (e.g., 'math500', 'aime', 'gpqa').
        condition: Steering condition (e.g., 'baseline', 'positive', 'negative').
        base_dir: Base directory for outputs.

    Returns:
        Path to the output JSONL file.
    """
    path = Path(base_dir) / dataset / condition / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def unload_model(model: PreTrainedModel) -> None:
    """Unload a model and free GPU memory.

    Args:
        model: The model to unload.
    """
    import gc

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_steering_vectors(path: Path | str) -> dict[int, Tensor]:
    """Load steering vectors from a .pt file.

    Args:
        path: Path to the .pt file saved by save_steering_vectors().

    Returns:
        Dict mapping layer index to steering vector tensor.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Steering vectors file not found: {file_path}")

    data = torch.load(file_path, map_location="cpu", weights_only=True)

    vectors: dict[int, Tensor] = {}
    for key, value in data.items():
        if key == "metadata":
            continue
        if key.startswith("layer_"):
            try:
                layer_idx = int(key.split("_")[1])
                vectors[layer_idx] = value
            except (IndexError, ValueError):
                continue

    if not vectors:
        raise ValueError(f"No valid layer keys found in {file_path}")

    return vectors


def load_model_4bit(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer with 4-bit quantization.

    Uses bitsandbytes NF4 quantization for memory efficiency.

    Args:
        model_name: Name or path of the model.

    Returns:
        Tuple of (model, tokenizer).
    """
    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()  # type: ignore[no-untyped-call]

    return model, tokenizer


def create_steering_hook(
    vector: Tensor, coefficient: float
) -> Callable[
    [object, tuple[object, ...], Tensor | tuple[Tensor, ...]], Tensor | tuple[Tensor, ...]
]:
    """Create a forward hook that adds steering vector to activations.

    Args:
        vector: The steering vector tensor to add.
        coefficient: Multiplier for the steering vector.

    Returns:
        A hook function compatible with register_forward_hook().
    """

    def hook(
        module: object, input: tuple[object, ...], output: Tensor | tuple[Tensor, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + coefficient * vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            output = output + coefficient * vector.to(output.device)
            return output

    return hook


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


def _prepare_batch(tokenizer: PreTrainedTokenizerBase, problems: list[str]) -> dict[str, Any]:
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


def _get_item_field(item: dict[str, Any], field_names: list[str], default: str = "") -> str:
    """Get a field from a dataset item, trying multiple possible field names.

    Args:
        item: Dataset item dictionary.
        field_names: List of possible field names to try.
        default: Default value if no field is found.

    Returns:
        The field value or default.
    """
    for name in field_names:
        if name in item:
            return str(item[name])
    return default


def run_steering_inference(config: SteeringInferenceConfig) -> Path:
    """Run steering inference on dataset and save results to JSONL.

    Loads steering vectors, applies them via forward hooks during generation,
    and saves results with all required fields.

    Args:
        config: SteeringInferenceConfig with all settings.

    Returns:
        Path to the output JSONL file.
    """
    vectors = load_steering_vectors(config.steering_vector_path)
    model, tokenizer = load_model_4bit(config.model_name)
    num_layers = len(model.model.layers)

    if config.layer_indices:
        for idx in config.layer_indices:
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {num_layers - 1}]")
        layer_indices = list(config.layer_indices)
    else:
        layer_indices = sorted(vectors.keys())

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.dataset_config:
        dataset = load_dataset(config.dataset_name, config.dataset_config, split="train")
    else:
        dataset = load_dataset(config.dataset_name, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = len(dataset)
    if config.limit is not None and config.limit < num_samples:
        num_samples = config.limit
    handles: list[Any] = []

    try:
        for layer_idx in layer_indices:
            if layer_idx in vectors:
                handle = model.model.layers[layer_idx].register_forward_hook(
                    create_steering_hook(vectors[layer_idx], config.coefficient)
                )
                handles.append(handle)

        with open(output_path, "w") as f:
            for i in tqdm(
                range(0, num_samples, config.batch_size), desc="Running steering inference"
            ):
                batch_end = min(i + config.batch_size, num_samples)
                batch_items = [dataset[j] for j in range(i, batch_end)]
                problems = [_get_item_field(item, ["problem", "question"]) for item in batch_items]
                prompts = [format_cot_prompt(p) for p in problems]

                inputs = _prepare_batch(tokenizer, prompts)
                input_ids = torch.tensor(inputs["input_ids"]).to(device)
                attention_mask = torch.tensor(inputs["attention_mask"]).to(device)

                try:
                    with torch.no_grad():
                        outputs = model.generate(  # type: ignore[operator]
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=config.max_new_tokens,
                            repetition_penalty=1.05,
                            do_sample=False,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    for item in batch_items:
                        problem = _get_item_field(item, ["problem", "question"])
                        prompt = format_cot_prompt(problem)
                        single_input = tokenizer(prompt, return_tensors="pt")
                        single_ids = single_input["input_ids"].to(device)
                        single_mask = single_input["attention_mask"].to(device)

                        with torch.no_grad():
                            output = model.generate(  # type: ignore[operator]
                                input_ids=single_ids,
                                attention_mask=single_mask,
                                max_new_tokens=config.max_new_tokens,
                                repetition_penalty=1.05,
                                do_sample=False,
                            )

                        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                        generated_text = str(generated_text)[len(prompt) :].strip()

                        result_entry = {
                            "problem_id": _get_item_field(item, ["unique_id", "id", "problem_id"]),
                            "problem": problem,
                            "generated": generated_text,
                            "reference_answer": _get_item_field(
                                item, ["answer", "solution", "reference"]
                            ),
                            "subject": _get_item_field(item, ["subject", "category"], ""),
                            "level": _get_item_field(item, ["level", "difficulty"], ""),
                            "prompt": prompt,
                        }
                        f.write(json.dumps(result_entry) + "\n")
                    continue

                for j, (output, item) in enumerate(zip(outputs, batch_items)):  # noqa: B905
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    prompt = prompts[j]
                    generated_text = str(generated_text)[len(prompt) :].strip()

                    problem = problems[j]
                    result_entry = {
                        "problem_id": _get_item_field(item, ["unique_id", "id", "problem_id"]),
                        "problem": problem,
                        "generated": generated_text,
                        "reference_answer": _get_item_field(
                            item, ["answer", "solution", "reference"]
                        ),
                        "subject": _get_item_field(item, ["subject", "category"], ""),
                        "level": _get_item_field(item, ["level", "difficulty"], ""),
                        "prompt": prompt,
                    }
                    f.write(json.dumps(result_entry) + "\n")

    finally:
        for handle in handles:
            handle.remove()

    return output_path
