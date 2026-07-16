"""Model loading and management utilities.

This module provides common functions for loading, managing, and unloading
language models with various configurations (standard, 4-bit quantization).

The utilities ensure consistent model loading across different modules
and provide memory management for GPU resources.
"""

from __future__ import annotations

import gc
from typing import Protocol, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class GenerativeModel(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class ModelLifecycle(Protocol):
    def eval(self) -> object: ...

    def to(self, device: torch.device) -> object: ...


class QuantizationConfigFactory(Protocol):
    def __call__(
        self,
        *,
        load_in_4bit: bool,
        bnb_4bit_quant_type: str,
        bnb_4bit_compute_dtype: torch.dtype,
        bnb_4bit_use_double_quant: bool,
    ) -> BitsAndBytesConfig: ...


def get_device() -> torch.device:
    """Get the appropriate device for model inference.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    """Get the appropriate dtype for the given device.

    Args:
        device: The target device. If None, uses get_device().

    Returns:
        torch.dtype: bfloat16 for CUDA, float32 for CPU.
    """
    if device is None:
        device = get_device()
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def setup_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Set up tokenizer with default settings.

    Ensures pad_token is set to eos_token if not present.

    Args:
        tokenizer: The tokenizer to configure.

    Returns:
        The configured tokenizer (modified in place).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer with specified precision.

    Args:
        model_name: Name or path of the model to load.
        device: Target device. If None, uses get_device().
        dtype: Data type for model weights. If None, uses get_dtype().

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    setup_tokenizer(tokenizer)

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype),
    )
    lifecycle = cast(ModelLifecycle, model)
    lifecycle.eval()
    lifecycle.to(device)

    return model, tokenizer


def load_model_4bit(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer with 4-bit quantization.

    Uses bitsandbytes NF4 quantization for memory efficiency.
    Best suited for large models that don't fit in memory.

    Args:
        model_name: Name or path of the model to load.

    Returns:
        Tuple of (model, tokenizer).
    """
    config_factory = cast(QuantizationConfigFactory, BitsAndBytesConfig)
    bnb_config = config_factory(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    setup_tokenizer(tokenizer)

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        ),
    )
    cast(ModelLifecycle, model).eval()

    return model, tokenizer


def unload_model(model: PreTrainedModel) -> None:
    """Unload a model and free GPU memory.

    Deletes the model and clears CUDA cache. Useful for
    freeing memory between different model operations.

    Args:
        model: The model to unload.
    """
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
