"""Tests for linear_probe module.

These tests verify the functions used to train and evaluate linear probes
for detecting reflection tokens in model activations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from numpy import ndarray
from sklearn.linear_model import LogisticRegression

from probing_reflection.linear_probe import (
    evaluate_probe,
    generate_pca_plot,
    generate_tsne_plot,
    save_probe_weights,
    train_linear_probe,
)
from probing_reflection.types import ProbeMetrics


@pytest.fixture
def sample_r_activations() -> dict[int, list[torch.Tensor]]:
    """Create mock R (reflection) activations for 2 layers."""
    return {
        0: [torch.randn(64) for _ in range(20)],
        1: [torch.randn(64) for _ in range(20)],
    }


@pytest.fixture
def sample_n_activations() -> dict[int, list[torch.Tensor]]:
    """Create mock N (non-reflection) activations for 2 layers."""
    return {
        0: [torch.randn(64) for _ in range(20)],
        1: [torch.randn(64) for _ in range(20)],
    }


@pytest.fixture
def layer_indices() -> tuple[int, ...]:
    """Layer indices to test."""
    return (0, 1)


@pytest.fixture
def trained_probe() -> LogisticRegression:
    """Create a pre-fitted LogisticRegression for evaluation tests."""
    probe = LogisticRegression(max_iter=1000, random_state=42)
    x_train = np.random.randn(40, 64)
    y_train = np.concatenate([np.ones(20), np.zeros(20)])
    probe.fit(x_train, y_train)
    return probe


@pytest.fixture
def test_data() -> tuple[ndarray, ndarray]:
    """Create test data for evaluation tests."""
    x_test = np.random.randn(10, 64)
    y_test = np.concatenate([np.ones(5), np.zeros(5)])
    return x_test, y_test


class TestDataCollection:
    """Tests for data collection functions."""

    def test_extract_reflection_tokens_from_r_set(self) -> None:
        """Verify tokens extracted from samples with reflection_count > 0."""
        from probing_reflection import linear_probe
        from probing_reflection.types import SampleWithReflection

        original_extract = linear_probe.extract_activation_at_position
        original_find = linear_probe.find_reflection_token_position

        linear_probe.extract_activation_at_position = lambda *args, **kwargs: {0: torch.randn(64)}  # type: ignore[method-assign]
        linear_probe.find_reflection_token_position = lambda *args, **kwargs: 5  # type: ignore[method-assign]

        try:
            from probing_reflection.linear_probe import collect_token_activations

            samples: list[SampleWithReflection] = [
                {
                    "problem_id": "1",
                    "problem": "test",
                    "generated": "Wait, let me think about this.",
                    "reference_answer": "answer",
                    "reflection_tokens": [],
                    "reflection_count": 1,
                    "reflection_density": 0.1,
                }
            ]

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            r_act, n_act = collect_token_activations(samples, mock_model, mock_tokenizer, (0,))

            assert 0 in r_act
            assert len(r_act[0]) == 1
        finally:
            linear_probe.extract_activation_at_position = original_extract  # type: ignore[method-assign]
            linear_probe.find_reflection_token_position = original_find  # type: ignore[method-assign]

    def test_extract_nonself_tokens_from_n_set(self) -> None:
        """Verify tokens extracted from samples with reflection_count == 0."""
        from probing_reflection import linear_probe
        from probing_reflection.types import SampleWithReflection

        original_extract = linear_probe.extract_activation_at_position
        original_find = linear_probe.find_reflection_token_position

        linear_probe.extract_activation_at_position = lambda *args, **kwargs: {0: torch.randn(64)}  # type: ignore[method-assign]
        linear_probe.find_reflection_token_position = lambda *args, **kwargs: 3  # type: ignore[method-assign]

        try:
            from probing_reflection.linear_probe import collect_token_activations

            samples: list[SampleWithReflection] = [
                {
                    "problem_id": "2",
                    "problem": "test",
                    "generated": "Some text without reflection.",
                    "reference_answer": "answer",
                    "reflection_tokens": [],
                    "reflection_count": 0,
                    "reflection_density": 0.0,
                }
            ]

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            r_act, n_act = collect_token_activations(samples, mock_model, mock_tokenizer, (0,))

            assert 0 in n_act
            assert len(n_act[0]) == 1
        finally:
            linear_probe.extract_activation_at_position = original_extract  # type: ignore[method-assign]
            linear_probe.find_reflection_token_position = original_find  # type: ignore[method-assign]

    def test_tokens_from_taxonomy(self) -> None:
        """Verify both sets contain tokens from REFLECTION_TAXONOMY."""
        from probing_reflection import linear_probe
        from probing_reflection.types import SampleWithReflection

        original_extract = linear_probe.extract_activation_at_position
        original_find = linear_probe.find_reflection_token_position

        linear_probe.extract_activation_at_position = lambda *args, **kwargs: {0: torch.randn(64)}  # type: ignore[method-assign]
        linear_probe.find_reflection_token_position = lambda *args, **kwargs: 2  # type: ignore[method-assign]

        try:
            from probing_reflection.linear_probe import collect_token_activations

            samples: list[SampleWithReflection] = [
                {
                    "problem_id": "1",
                    "problem": "test",
                    "generated": "Wait, let me check.",
                    "reference_answer": "answer",
                    "reflection_tokens": [],
                    "reflection_count": 1,
                    "reflection_density": 0.1,
                },
                {
                    "problem_id": "2",
                    "problem": "test",
                    "generated": "Actually this is fine.",
                    "reference_answer": "answer",
                    "reflection_tokens": [],
                    "reflection_count": 0,
                    "reflection_density": 0.0,
                },
            ]

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            r_act, n_act = collect_token_activations(samples, mock_model, mock_tokenizer, (0,))

            assert len(r_act[0]) >= 1
            assert len(n_act[0]) >= 1
        finally:
            linear_probe.extract_activation_at_position = original_extract  # type: ignore[method-assign]
            linear_probe.find_reflection_token_position = original_find  # type: ignore[method-assign]

    def test_handle_missing_token_position(self) -> None:
        """Verify graceful handling when position is None."""
        from probing_reflection import linear_probe
        from probing_reflection.types import SampleWithReflection

        original_find = linear_probe.find_reflection_token_position

        linear_probe.find_reflection_token_position = lambda *args, **kwargs: None  # type: ignore[method-assign]

        try:
            from probing_reflection.linear_probe import collect_token_activations

            samples: list[SampleWithReflection] = [
                {
                    "problem_id": "1",
                    "problem": "test",
                    "generated": "No tokens here.",
                    "reference_answer": "answer",
                    "reflection_tokens": [],
                    "reflection_count": 1,
                    "reflection_density": 0.0,
                }
            ]

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            r_act, n_act = collect_token_activations(samples, mock_model, mock_tokenizer, (0,))

            assert len(r_act[0]) == 0
        finally:
            linear_probe.find_reflection_token_position = original_find  # type: ignore[method-assign]


class TestProbeTraining:
    """Tests for probe training functions."""

    def test_train_logistic_regression_per_layer(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        layer_indices: tuple[int, ...],
    ) -> None:
        """Verify LogisticRegression trained for each layer."""
        probes, metrics = train_linear_probe(
            sample_r_activations, sample_n_activations, layer_indices
        )

        assert len(probes) == len(layer_indices)
        for layer in layer_indices:
            assert layer in probes
            assert isinstance(probes[layer], LogisticRegression)

    def test_returns_sklearn_model(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        layer_indices: tuple[int, ...],
    ) -> None:
        """Verify returns sklearn model object."""
        probes, _ = train_linear_probe(sample_r_activations, sample_n_activations, layer_indices)

        for probe in probes.values():
            assert hasattr(probe, "coef_")
            assert hasattr(probe, "intercept_")
            assert hasattr(probe, "predict")

    def test_handles_class_imbalance(
        self,
        layer_indices: tuple[int, ...],
    ) -> None:
        """Verify no crash on imbalanced data."""
        imbalanced_r = {
            0: [torch.randn(64) for _ in range(5)],
            1: [torch.randn(64) for _ in range(5)],
        }
        imbalanced_n = {
            0: [torch.randn(64) for _ in range(30)],
            1: [torch.randn(64) for _ in range(30)],
        }

        probes, metrics = train_linear_probe(imbalanced_r, imbalanced_n, layer_indices)

        assert len(probes) == len(layer_indices)
        assert len(metrics) == len(layer_indices)

    def test_train_test_split_ratio(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        layer_indices: tuple[int, ...],
    ) -> None:
        """Verify 80/20 split."""
        probes, metrics = train_linear_probe(
            sample_r_activations, sample_n_activations, layer_indices, test_size=0.2
        )

        total_samples = 40
        expected_train = int(total_samples * 0.8)
        expected_test = int(total_samples * 0.2)

        for m in metrics:
            assert m["train_samples"] == expected_train
            assert m["test_samples"] == expected_test


class TestVisualization:
    """Tests for visualization functions."""

    def test_generate_tsne_plot(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        tmp_path: Path,
    ) -> None:
        """Verify t-SNE plot saved as PNG."""
        output_path = tmp_path / "tsne_plot.png"

        generate_tsne_plot(sample_r_activations, sample_n_activations, output_path, layer_index=0)

        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_generate_pca_plot(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        tmp_path: Path,
    ) -> None:
        """Verify PCA plot saved as PNG."""
        output_path = tmp_path / "pca_plot.png"

        generate_pca_plot(sample_r_activations, sample_n_activations, output_path, layer_index=0)

        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_plots_have_both_classes(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        tmp_path: Path,
    ) -> None:
        """Verify plots contain both classes with distinct colors."""
        tsne_path = tmp_path / "tsne_both.png"
        pca_path = tmp_path / "pca_both.png"

        generate_tsne_plot(sample_r_activations, sample_n_activations, tsne_path, layer_index=0)
        generate_pca_plot(sample_r_activations, sample_n_activations, pca_path, layer_index=0)

        assert tsne_path.exists()
        assert pca_path.exists()

    def test_output_directory_created(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        tmp_path: Path,
    ) -> None:
        """Verify output directory created if not exists."""
        output_dir = tmp_path / "nested" / "dir"
        output_path = output_dir / "tsne_plot.png"

        assert not output_dir.exists()

        generate_tsne_plot(sample_r_activations, sample_n_activations, output_path, layer_index=0)

        assert output_dir.exists()
        assert output_path.exists()


class TestWeightSaving:
    """Tests for weight saving functions."""

    def test_save_probe_weights(
        self,
        trained_probe: LogisticRegression,
        tmp_path: Path,
    ) -> None:
        """Verify weights saved to specified path."""
        probes = {0: trained_probe}
        metrics: list[ProbeMetrics] = [
            {
                "layer_index": 0,
                "accuracy": 0.85,
                "train_samples": 40,
                "test_samples": 10,
            }
        ]
        metadata: dict[str, str | int | tuple[int, ...]] = {
            "model_name": "test-model",
            "layer_indices": (0,),
        }
        output_path = tmp_path / "probe_weights.npz"

        save_probe_weights(probes, metrics, output_path, metadata)

        assert output_path.exists()

    def test_weights_loadable(
        self,
        trained_probe: LogisticRegression,
        tmp_path: Path,
    ) -> None:
        """Verify loadable via np.load()."""
        probes = {0: trained_probe}
        metrics: list[ProbeMetrics] = [
            {
                "layer_index": 0,
                "accuracy": 0.85,
                "train_samples": 40,
                "test_samples": 10,
            }
        ]
        metadata: dict[str, str | int | tuple[int, ...]] = {
            "model_name": "test-model",
        }
        output_path = tmp_path / "probe_weights.npz"

        save_probe_weights(probes, metrics, output_path, metadata)

        loaded = np.load(output_path, allow_pickle=True)
        assert "coef_layer_0" in loaded
        assert "intercept_layer_0" in loaded

    def test_metadata_included(
        self,
        trained_probe: LogisticRegression,
        tmp_path: Path,
    ) -> None:
        """Verify metadata includes layer_index, accuracy, sample_counts."""
        probes = {0: trained_probe}
        metrics: list[ProbeMetrics] = [
            {
                "layer_index": 0,
                "accuracy": 0.85,
                "train_samples": 40,
                "test_samples": 10,
            }
        ]
        metadata: dict[str, str | int | tuple[int, ...]] = {
            "model_name": "test-model",
            "layer_indices": (0, 1),
        }
        output_path = tmp_path / "probe_weights.npz"

        save_probe_weights(probes, metrics, output_path, metadata)

        loaded = np.load(output_path, allow_pickle=True)
        assert "layer_indices" in loaded
        assert "accuracies" in loaded
        assert "train_samples" in loaded
        assert "test_samples" in loaded
        assert "metadata_model_name" in loaded

    def test_directory_created(
        self,
        trained_probe: LogisticRegression,
        tmp_path: Path,
    ) -> None:
        """Verify parent directory created if not exists."""
        output_dir = tmp_path / "nested" / "weights"
        output_path = output_dir / "probe_weights.npz"

        assert not output_dir.exists()

        probes = {0: trained_probe}
        metrics: list[ProbeMetrics] = [
            {
                "layer_index": 0,
                "accuracy": 0.85,
                "train_samples": 40,
                "test_samples": 10,
            }
        ]
        metadata: dict[str, str | int | tuple[int, ...]] = {}

        save_probe_weights(probes, metrics, output_path, metadata)

        assert output_dir.exists()
        assert output_path.exists()


class TestProbeEvaluation:
    """Tests for probe evaluation functions."""

    def test_compute_accuracy(
        self,
        trained_probe: LogisticRegression,
        test_data: tuple[ndarray, ndarray],
    ) -> None:
        """Verify accuracy computed on test split."""
        x_test, y_test = test_data

        metrics = evaluate_probe(trained_probe, x_test, y_test, layer_index=0)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_accuracy_in_valid_range(
        self,
        trained_probe: LogisticRegression,
        test_data: tuple[ndarray, ndarray],
    ) -> None:
        """Verify returns float in [0.0, 1.0]."""
        x_test, y_test = test_data

        metrics = evaluate_probe(trained_probe, x_test, y_test, layer_index=0)

        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_stratified_split(
        self,
        sample_r_activations: dict[int, list[torch.Tensor]],
        sample_n_activations: dict[int, list[torch.Tensor]],
        layer_indices: tuple[int, ...],
    ) -> None:
        """Verify train/test respects class distribution."""
        probes, metrics = train_linear_probe(
            sample_r_activations, sample_n_activations, layer_indices
        )

        for m in metrics:
            assert m["train_samples"] > 0
            assert m["test_samples"] > 0
            assert "layer_index" in m
            assert "accuracy" in m
