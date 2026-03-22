"""Linear probe module for detecting reflection tokens in model activations.

This module provides functions to train and evaluate linear probes that
classify activations as coming from reflection (R) or non-reflection (N)
samples. The probes use logistic regression on hidden state activations
extracted at reflection token positions.

The pipeline involves:
1. Collecting activations from R and N sets at token positions
2. Training LogisticRegression probes per layer
3. Evaluating probe accuracy on held-out test data
4. Visualizing activation separability with t-SNE/PCA
5. Saving probe weights for later analysis
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy import ndarray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.manifold import TSNE  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from probing_reflection.reflection_diagnosis import REFLECTION_TAXONOMY
from probing_reflection.steering_vectors import (
    extract_activation_at_position,
    find_reflection_token_position,
)
from probing_reflection.types import ProbeMetrics, SampleWithReflection

logger = logging.getLogger(__name__)


def collect_token_activations(
    samples: list[SampleWithReflection],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_indices: tuple[int, ...],
) -> tuple[dict[int, list[Tensor]], dict[int, list[Tensor]]]:
    """Extract activations from R set (reflection) and N set (non-reflection).

    Splits samples into:
    - R set: samples with reflection_count > 0 (extract at reflection token position)
    - N set: samples with reflection_count == 0 (extract at any taxonomy token position)

    For R samples: finds the first reflection token and extracts activation at that position.
    For N samples: finds the first taxonomy token position (any reflection-related word)
    and extracts activation there.

    Args:
        samples: List of samples with reflection analysis.
        model: The language model to extract activations from.
        tokenizer: Tokenizer for the model.
        layer_indices: Which layers to extract activations from.

    Returns:
        Tuple of (r_activations_by_layer, n_activations_by_layer).
        Each is a dict mapping layer index to list of activation tensors.
    """
    r_activations: dict[int, list[Tensor]] = {layer: [] for layer in layer_indices}
    n_activations: dict[int, list[Tensor]] = {layer: [] for layer in layer_indices}

    all_reflection_tokens = [token for tokens in REFLECTION_TAXONOMY.values() for token in tokens]

    for sample in tqdm(samples, desc="Collecting activations"):
        text = sample["generated"]

        if sample["reflection_count"] > 0:
            position = find_reflection_token_position(tokenizer, text)
            if position is None:
                logger.warning(f"Reflection token not found for R sample {sample['problem_id']}")
                continue

            try:
                activations = extract_activation_at_position(
                    model, tokenizer, text, position, layer_indices
                )
                for layer, tensor in activations.items():
                    r_activations[layer].append(tensor.cpu())
            except IndexError:
                logger.warning(f"Position out of bounds for R sample {sample['problem_id']}")
        else:
            position = find_reflection_token_position(tokenizer, text, all_reflection_tokens)
            if position is None:
                logger.warning(f"Taxonomy token not found for N sample {sample['problem_id']}")
                continue

            try:
                activations = extract_activation_at_position(
                    model, tokenizer, text, position, layer_indices
                )
                for layer, tensor in activations.items():
                    n_activations[layer].append(tensor.cpu())
            except IndexError:
                logger.warning(f"Position out of bounds for N sample {sample['problem_id']}")

    return r_activations, n_activations


def evaluate_probe(
    model: LogisticRegression,
    x_test: ndarray,  # noqa: N803
    y_test: ndarray,
    layer_index: int,
) -> ProbeMetrics:
    """Compute accuracy and return metrics.

    Evaluates the trained probe on test data and returns structured metrics.

    Args:
        model: Trained LogisticRegression model.
        x_test: Test features (activations).
        y_test: Test labels (1=R, 0=N).
        layer_index: Layer index this probe corresponds to.

    Returns:
        ProbeMetrics with layer_index, accuracy, and sample counts.
    """
    accuracy = model.score(x_test, y_test)

    return ProbeMetrics(
        layer_index=layer_index,
        accuracy=float(accuracy),
        train_samples=0,
        test_samples=len(y_test),
    )


def train_linear_probe(
    r_activations: dict[int, list[Tensor]],
    n_activations: dict[int, list[Tensor]],
    layer_indices: tuple[int, ...],
    test_size: float = 0.2,
) -> tuple[dict[int, LogisticRegression], list[ProbeMetrics]]:
    """Train LogisticRegression probes per layer.

    For each layer:
    1. Stack R and N activations into feature matrix X
    2. Create labels y (1=R, 0=N)
    3. Split into train/test with stratification
    4. Train LogisticRegression
    5. Evaluate on test set

    Args:
        r_activations: Dict mapping layer index to list of activation tensors (reflection).
        n_activations: Dict mapping layer index to list of activation tensors (non-reflection).
        layer_indices: Tuple of layer indices to train probes for.
        test_size: Fraction of data to use for testing (default 0.2 for 80/20 split).

    Returns:
        Tuple of (probes_by_layer, metrics_list).
        probes_by_layer maps layer index to trained LogisticRegression model.
        metrics_list contains ProbeMetrics for each layer.
    """
    probes: dict[int, LogisticRegression] = {}
    metrics: list[ProbeMetrics] = []

    for layer in layer_indices:
        r_list = r_activations.get(layer, [])
        n_list = n_activations.get(layer, [])

        if not r_list or not n_list:
            logger.warning(f"Skipping layer {layer}: missing activations")
            continue

        r_stack = np.stack([t.numpy() for t in r_list])
        n_stack = np.stack([t.numpy() for t in n_list])

        features = np.vstack([r_stack, n_stack])
        labels = np.concatenate([np.ones(len(r_list)), np.zeros(len(n_list))])

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42
        )

        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(x_train, y_train)

        probes[layer] = probe

        layer_metrics = ProbeMetrics(
            layer_index=layer,
            accuracy=float(probe.score(x_test, y_test)),
            train_samples=len(y_train),
            test_samples=len(y_test),
        )
        metrics.append(layer_metrics)

    return probes, metrics


def generate_tsne_plot(
    r_activations: dict[int, list[Tensor]],
    n_activations: dict[int, list[Tensor]],
    output_path: Path | str,
    layer_index: int = 0,
) -> None:
    """Generate t-SNE scatter plot.

    Combines R and N activations for a specific layer, applies t-SNE
    dimensionality reduction, and creates a scatter plot with different
    colors for each class.

    Args:
        r_activations: Dict mapping layer index to list of activation tensors (reflection).
        n_activations: Dict mapping layer index to list of activation tensors (non-reflection).
        output_path: Path to save the PNG file.
        layer_index: Which layer to visualize (default 0).
    """
    import matplotlib.pyplot as plt

    r_list = r_activations.get(layer_index, [])
    n_list = n_activations.get(layer_index, [])

    if not r_list or not n_list:
        logger.warning(f"Cannot generate t-SNE plot: no activations for layer {layer_index}")
        return

    r_stack = np.stack([t.numpy() for t in r_list])
    n_stack = np.stack([t.numpy() for t in n_list])

    features = np.vstack([r_stack, n_stack])
    labels = np.concatenate([np.ones(len(r_list)), np.zeros(len(n_list))])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embedded = tsne.fit_transform(features)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedded[labels == 1, 0],
        embedded[labels == 1, 1],
        c="blue",
        alpha=0.6,
        label="Reflection (R)",
        s=20,
    )
    plt.scatter(
        embedded[labels == 0, 0],
        embedded[labels == 0, 1],
        c="red",
        alpha=0.6,
        label="Non-reflection (N)",
        s=20,
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Activations (Layer {layer_index})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_pca_plot(
    r_activations: dict[int, list[Tensor]],
    n_activations: dict[int, list[Tensor]],
    output_path: Path | str,
    layer_index: int = 0,
) -> None:
    """Generate PCA scatter plot.

    Combines R and N activations for a specific layer, applies PCA
    dimensionality reduction, and creates a scatter plot with different
    colors for each class.

    Args:
        r_activations: Dict mapping layer index to list of activation tensors (reflection).
        n_activations: Dict mapping layer index to list of activation tensors (non-reflection).
        output_path: Path to save the PNG file.
        layer_index: Which layer to visualize (default 0).
    """
    import matplotlib.pyplot as plt

    r_list = r_activations.get(layer_index, [])
    n_list = n_activations.get(layer_index, [])

    if not r_list or not n_list:
        logger.warning(f"Cannot generate PCA plot: no activations for layer {layer_index}")
        return

    r_stack = np.stack([t.numpy() for t in r_list])
    n_stack = np.stack([t.numpy() for t in n_list])

    features = np.vstack([r_stack, n_stack])
    labels = np.concatenate([np.ones(len(r_list)), np.zeros(len(n_list))])

    pca = PCA(n_components=2, random_state=42)
    embedded = pca.fit_transform(features)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedded[labels == 1, 0],
        embedded[labels == 1, 1],
        c="blue",
        alpha=0.6,
        label="Reflection (R)",
        s=20,
    )
    plt.scatter(
        embedded[labels == 0, 0],
        embedded[labels == 0, 1],
        c="red",
        alpha=0.6,
        label="Non-reflection (N)",
        s=20,
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Visualization of Activations (Layer {layer_index})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_probe_weights(
    probes: dict[int, LogisticRegression],
    metrics: list[ProbeMetrics],
    output_path: Path | str,
    metadata: dict[str, str | int | tuple[int, ...]],
) -> None:
    """Save probe coefficients and metadata to .npz.

    Saves the trained probe weights (coefficients and intercepts) along
    with metrics and metadata to a numpy .npz file for later loading.

    Args:
        probes: Dict mapping layer index to trained LogisticRegression model.
        metrics: List of ProbeMetrics for each layer.
        output_path: Path to save the .npz file.
        metadata: Metadata dict with model_name, layer_indices, etc.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_arrays: dict[str, ndarray] = {}

    for layer, probe in probes.items():
        save_arrays[f"coef_layer_{layer}"] = probe.coef_
        save_arrays[f"intercept_layer_{layer}"] = probe.intercept_

    layer_indices_arr = np.array([m["layer_index"] for m in metrics], dtype=np.int32)
    accuracies = np.array([m["accuracy"] for m in metrics], dtype=np.float64)
    train_samples = np.array([m["train_samples"] for m in metrics], dtype=np.int32)
    test_samples = np.array([m["test_samples"] for m in metrics], dtype=np.int32)

    save_arrays["layer_indices"] = layer_indices_arr
    save_arrays["accuracies"] = accuracies
    save_arrays["train_samples"] = train_samples
    save_arrays["test_samples"] = test_samples

    for key, value in metadata.items():
        if isinstance(value, tuple):
            save_arrays[f"metadata_{key}"] = np.array(value, dtype=np.int32)
        elif isinstance(value, int):
            save_arrays[f"metadata_{key}"] = np.array([value], dtype=np.int32)
        elif isinstance(value, str):
            save_arrays[f"metadata_{key}"] = np.array([value], dtype=object)
        else:
            save_arrays[f"metadata_{key}"] = np.array([value], dtype=object)

    np.savez(path, **save_arrays)  # type: ignore[arg-type]
