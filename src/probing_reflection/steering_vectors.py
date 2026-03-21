"""Steering vector extraction for modulating reflection behavior in LLMs.

This module provides functions to extract steering vectors from model
activations, which can be used to modulate self-reflection behavior
in language models.

The extraction pipeline involves:
1. Classifying samples into reflection (R) and non-reflection (N) sets
2. Finding token positions of reflection markers
3. Extracting activations at those positions
4. Computing difference-in-means vectors
5. Saving vectors for later injection
"""

from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from probing_reflection.reflection_diagnosis import REFLECTION_TAXONOMY
from probing_reflection.types import SampleWithReflection


def classify_samples(
    samples: list[SampleWithReflection],
    min_samples: int = 10,
) -> tuple[list[SampleWithReflection], list[SampleWithReflection]]:
    """Split samples into reflection (R) and non-reflection (N) sets.

    R set: samples with reflection_count > 0
    N set: samples with reflection_count == 0

    Args:
        samples: List of samples with reflection analysis.
        min_samples: Minimum required samples in each set.

    Returns:
        Tuple of (R_samples, N_samples).

    Raises:
        ValueError: If either R or N set has fewer than min_samples.
    """
    r_samples: list[SampleWithReflection] = []
    n_samples: list[SampleWithReflection] = []

    for sample in samples:
        if sample["reflection_count"] > 0:
            r_samples.append(sample)
        else:
            n_samples.append(sample)

    if len(r_samples) < min_samples:
        raise ValueError(
            f"R set has {len(r_samples)} samples, but min_samples={min_samples} is required"
        )
    if len(n_samples) < min_samples:
        raise ValueError(
            f"N set has {len(n_samples)} samples, but min_samples={min_samples} is required"
        )

    return (r_samples, n_samples)


def find_reflection_token_position(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    reflection_tokens: list[str] | None = None,
) -> int | None:
    """Find the position of the first reflection token in text.

    Args:
        tokenizer: Tokenizer to use for tokenization
        text: Text to search for reflection tokens
        reflection_tokens: List of tokens to search for (default: all from REFLECTION_TAXONOMY)

    Returns:
        Index of first reflection token in tokenized sequence, or None if not found
    """
    if reflection_tokens is None:
        reflection_tokens = [token for tokens in REFLECTION_TAXONOMY.values() for token in tokens]

    reflection_tokens_lower = [t.lower() for t in reflection_tokens]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    for idx, token in enumerate(tokens):
        token_lower = token.lower()
        for ref_token in reflection_tokens_lower:
            if ref_token in token_lower:
                return idx

    return None


def extract_activation_at_position(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    position: int,
    layer_indices: tuple[int, ...],
) -> dict[int, Tensor]:
    """Extract hidden state activations at a specific token position.

    Args:
        model: The language model to extract activations from
        tokenizer: Tokenizer for the model
        text: Text to process
        position: Token position to extract activation from
        layer_indices: Which layers to extract activations from

    Returns:
        Dict mapping layer index to activation tensor (on CPU)

    Raises:
        IndexError: If position is out of bounds
    """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple: (embedding, layer1, layer2, ..., final)
        # hidden_states[0] = embedding layer
        # hidden_states[i+1] = output of layer i
        seq_len = inputs["input_ids"].shape[1]
        if position >= seq_len:
            raise IndexError(f"Position {position} >= sequence length {seq_len}")

        result: dict[int, Tensor] = {}
        for layer in layer_indices:
            # +1 because hidden_states[0] is embedding
            hidden = outputs.hidden_states[layer + 1]
            # Extract at position, batch index 0, move to CPU
            result[layer] = hidden[0, position, :].cpu()

        return result


def compute_difference_in_means(
    r_activations: dict[int, list[Tensor]],
    n_activations: dict[int, list[Tensor]],
    layer_indices: tuple[int, ...],
) -> dict[int, Tensor]:
    """Compute steering vectors as difference in means.

    For each layer: v = mean(R_activations) - mean(N_activations)

    Args:
        r_activations: Dict mapping layer index to list of activation tensors
            (reflection samples)
        n_activations: Dict mapping layer index to list of activation tensors
            (non-reflection samples)
        layer_indices: Tuple of layer indices to compute vectors for

    Returns:
        Dict mapping layer index to steering vector tensor

    Raises:
        ValueError: If any layer is missing from either R or N activations
    """
    vectors: dict[int, Tensor] = {}

    for layer in layer_indices:
        if layer not in r_activations or not r_activations[layer]:
            raise ValueError(f"No R activations for layer {layer}")
        if layer not in n_activations or not n_activations[layer]:
            raise ValueError(f"No N activations for layer {layer}")

        # Stack and compute means
        r_stack = torch.stack(r_activations[layer])  # [N_R, hidden_dim]
        n_stack = torch.stack(n_activations[layer])  # [N_N, hidden_dim]

        r_mean = r_stack.mean(dim=0)  # [hidden_dim]
        n_mean = n_stack.mean(dim=0)  # [hidden_dim]

        vectors[layer] = (r_mean - n_mean).cpu()

    return vectors


def save_steering_vectors(
    vectors: dict[int, Tensor],
    metadata: dict[str, str | int | tuple[int, ...]],
    output_path: Path | str,
) -> None:
    """Save steering vectors to a .pt file with metadata.

    Args:
        vectors: Dict mapping layer index to steering vector tensor
        metadata: Metadata dict with model_name, layer_indices, r_count, n_count, etc.
        output_path: Path to save the .pt file

    Raises:
        ValueError: If any tensor is not on CPU
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for layer, tensor in vectors.items():
        if tensor.device.type != "cpu":
            raise ValueError(f"Tensor for layer {layer} is not on CPU")

    metadata_with_timestamp = dict(metadata)
    metadata_with_timestamp["timestamp"] = datetime.now().isoformat()

    save_dict: dict[str, Tensor | dict[str, str | int | tuple[int, ...]]] = {}
    for layer, tensor in vectors.items():
        save_dict[f"layer_{layer}"] = tensor
    save_dict["metadata"] = metadata_with_timestamp

    torch.save(save_dict, path)
