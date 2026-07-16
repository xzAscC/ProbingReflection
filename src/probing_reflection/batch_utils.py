"""Batch processing utilities for model inference.

This module provides functions for preparing and processing batches
of inputs for language model inference, handling padding and tokenization
consistently across different modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from transformers import PreTrainedTokenizerBase


def prepare_batch(
    tokenizer: PreTrainedTokenizerBase, problems: list[str]
) -> dict[str, list[list[int]]]:
    """Prepare a batch of problems for model inference.

    Sets up left padding for autoregressive model compatibility
    and ensures pad_token is configured.

    Args:
        tokenizer: The tokenizer to use.
        problems: List of problem strings to tokenize.

    Returns:
        Tokenized batch with input_ids and attention_mask.
    """
    # Set padding side to left for autoregressive generation
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = tokenizer(problems, padding=True, return_tensors=None)

    return cast(dict[str, list[list[int]]], result)


def prepare_batch_with_tensors(
    tokenizer: PreTrainedTokenizerBase, problems: list[str], device: torch.device
) -> dict[str, torch.Tensor]:
    """Prepare a batch and convert to tensors on the specified device.

    Convenience function that combines prepare_batch with tensor conversion.

    Args:
        tokenizer: The tokenizer to use.
        problems: List of problem strings to tokenize.
        device: The device to move tensors to.

    Returns:
        Dict with input_ids and attention_mask as tensors on device.
    """
    batch = prepare_batch(tokenizer, problems)

    return {
        "input_ids": torch.tensor(batch["input_ids"]).to(device),
        "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
    }


def get_item_field(item: Mapping[str, object], field_names: list[str], default: str = "") -> str:
    """Get a field from a dataset item, trying multiple possible field names.

    Useful for handling datasets with varying field name conventions.

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
