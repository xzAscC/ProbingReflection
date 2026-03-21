"""Tests for steering vector extraction module.

These tests verify the functions used to extract and manipulate
steering vectors for modulating reflection behavior in LLMs.

This file follows TDD RED phase: all tests skip since the
steering_vectors module is not yet implemented.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

from probing_reflection.steering_vectors import (
    classify_samples,
    compute_difference_in_means,
    extract_activation_at_position,
    extract_batch_activations,
    extract_steering_vectors,
    find_reflection_token_position,
    save_steering_vectors,
)
from probing_reflection.types import ExtractVectorsConfig, SampleWithReflection


class TestClassify:
    """Tests for classify_samples function."""

    def test_classify_samples_basic(self, test_data_dir: Path) -> None:
        """classify_samples should split samples into R and N sets."""
        fixture_path = test_data_dir / "sample_reflection.jsonl"
        with open(fixture_path) as f:
            samples: list[SampleWithReflection] = [json.loads(line) for line in f]

        r_samples, n_samples = classify_samples(samples, min_samples=1)

        assert len(r_samples) == 10
        assert len(n_samples) == 5
        for sample in r_samples:
            assert sample["reflection_count"] > 0
        for sample in n_samples:
            assert sample["reflection_count"] == 0

    def test_classify_samples_min_samples_validation(self, test_data_dir: Path) -> None:
        """classify_samples should raise ValueError if min_samples not met."""
        fixture_path = test_data_dir / "sample_reflection.jsonl"
        with open(fixture_path) as f:
            samples: list[SampleWithReflection] = [json.loads(line) for line in f]

        with pytest.raises(ValueError, match="N set has 5 samples"):
            classify_samples(samples, min_samples=10)

    def test_classify_samples_empty_input(self, test_data_dir: Path) -> None:
        """classify_samples should handle empty input gracefully."""
        samples: list[SampleWithReflection] = []

        with pytest.raises(ValueError, match="R set has 0 samples"):
            classify_samples(samples, min_samples=1)

    def test_classify_samples_all_reflection(self, test_data_dir: Path) -> None:
        """classify_samples should handle case where all samples have reflection."""
        fixture_path = test_data_dir / "sample_reflection.jsonl"
        with open(fixture_path) as f:
            all_samples: list[SampleWithReflection] = [json.loads(line) for line in f]

        samples = [s for s in all_samples if s["reflection_count"] > 0]

        with pytest.raises(ValueError, match="N set has 0 samples"):
            classify_samples(samples, min_samples=1)

    def test_classify_samples_no_reflection(self, test_data_dir: Path) -> None:
        """classify_samples should handle case where no samples have reflection."""
        fixture_path = test_data_dir / "sample_reflection.jsonl"
        with open(fixture_path) as f:
            all_samples: list[SampleWithReflection] = [json.loads(line) for line in f]

        samples = [s for s in all_samples if s["reflection_count"] == 0]

        with pytest.raises(ValueError, match="R set has 0 samples"):
            classify_samples(samples, min_samples=1)


class TestFindPosition:
    """Tests for find_reflection_token_position function."""

    def test_find_reflection_token_basic(self) -> None:
        """find_reflection_token_position should return correct token index."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [101, 2094, 1037, 102]
        tokenizer.convert_ids_to_tokens.return_value = ["Hello", "Wait", "there", "."]

        result = find_reflection_token_position(tokenizer, "Hello Wait there.")

        assert result == 1

    def test_find_reflection_token_multiple_occurrences(self) -> None:
        """find_reflection_token_position should handle multiple reflection tokens."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [101, 2094, 1037, 2094, 102]
        tokenizer.convert_ids_to_tokens.return_value = ["But", "wait", "however", "now", "."]

        result = find_reflection_token_position(tokenizer, "But wait however now .")

        assert result == 0

    def test_find_reflection_token_not_found(self) -> None:
        """find_reflection_token_position should return None when token not in text."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [101, 2094, 102]
        tokenizer.convert_ids_to_tokens.return_value = ["Hello", "world", "."]

        result = find_reflection_token_position(tokenizer, "Hello world.")

        assert result is None

    def test_find_reflection_token_case_sensitivity(self) -> None:
        """find_reflection_token_position should handle case appropriately."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [101, 2094, 102]
        tokenizer.convert_ids_to_tokens.return_value = ["Hello", "WAIT", "there"]

        result = find_reflection_token_position(tokenizer, "Hello WAIT there")

        assert result == 1


class TestExtractSingle:
    """Tests for extract_activation_at_position function."""

    def test_extract_activation_basic(self) -> None:
        """extract_activation_at_position should return tensor of correct shape."""
        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3, 4, 5]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3, 4, 5]]).long()
        }

        hidden_dim = 768
        seq_len = 5
        mock_hidden = torch.randn(1, seq_len, hidden_dim)
        mock_output = MagicMock()
        mock_output.hidden_states = (None,) + (mock_hidden,) * 12
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        result = extract_activation_at_position(model, tokenizer, "test", 2, (0, 5))

        assert 0 in result
        assert 5 in result
        assert result[0].shape == (hidden_dim,)
        assert result[5].shape == (hidden_dim,)

    def test_extract_activation_invalid_position(self) -> None:
        """extract_activation_at_position should handle out-of-bounds position."""
        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3]]).long()
        }

        mock_output = MagicMock()
        mock_output.hidden_states = (None, torch.randn(1, 3, 768))
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        with pytest.raises(IndexError, match="Position 5 >= sequence length 3"):
            extract_activation_at_position(model, tokenizer, "test", 5, (0,))

    def test_extract_activation_layer_selection(self) -> None:
        """extract_activation_at_position should extract from specified layer."""
        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3]]).long()
        }

        hidden_dim = 768
        layer0_hidden = torch.ones(1, 3, hidden_dim)
        layer5_hidden = torch.full((1, 3, hidden_dim), 5.0)
        mock_output = MagicMock()
        mock_output.hidden_states = (
            None,
            layer0_hidden,
            torch.randn(1, 3, hidden_dim),
            torch.randn(1, 3, hidden_dim),
            torch.randn(1, 3, hidden_dim),
            torch.randn(1, 3, hidden_dim),
            layer5_hidden,
        )
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        result = extract_activation_at_position(model, tokenizer, "test", 1, (0, 5))

        assert torch.allclose(result[0], torch.ones(hidden_dim))
        assert torch.allclose(result[5], torch.full((hidden_dim,), 5.0))


class TestBatch:
    """Tests for extract_batch_activations function."""

    def test_batch_activations_basic(self) -> None:
        """extract_batch_activations should process multiple samples."""
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, this is a test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "This is simple.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]

        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.convert_ids_to_tokens.return_value = ["Wait", "this", "is", "test"]
        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3, 4]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3, 4]]).long()
        }

        hidden_dim = 768
        mock_hidden = torch.randn(1, 4, hidden_dim)
        mock_output = MagicMock()
        mock_output.hidden_states = (None,) + (mock_hidden,) * 12
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        r_acts, n_acts = extract_batch_activations(samples, model, tokenizer, (0, 5), batch_size=2)

        assert 0 in r_acts
        assert 5 in r_acts
        assert 0 in n_acts
        assert 5 in n_acts
        assert len(r_acts[0]) == 1
        assert len(n_acts[0]) == 1

    def test_batch_activations_shape(self) -> None:
        """extract_batch_activations should return stacked tensor of correct shape."""
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, test one.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "r2",
                "problem": "test",
                "generated": "Actually, test two.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple output.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]

        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.convert_ids_to_tokens.return_value = ["Wait", "test", "one", "."]
        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3, 4]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3, 4]]).long()
        }

        hidden_dim = 256
        mock_hidden = torch.randn(1, 4, hidden_dim)
        mock_output = MagicMock()
        mock_output.hidden_states = (None,) + (mock_hidden,) * 12
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        r_acts, n_acts = extract_batch_activations(samples, model, tokenizer, (3,), batch_size=2)

        assert 3 in r_acts
        assert len(r_acts[3]) == 2
        assert r_acts[3][0].shape == (hidden_dim,)
        assert r_acts[3][1].shape == (hidden_dim,)
        assert len(n_acts[3]) == 1

    def test_batch_activations_empty_batch(self) -> None:
        """extract_batch_activations should handle case where all samples are skipped."""
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "No reflection tokens here.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple output.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]

        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.encode.return_value = [1, 2, 3, 4]
        tokenizer.convert_ids_to_tokens.return_value = ["No", "tokens", "here", "."]
        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3, 4]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3, 4]]).long()
        }

        hidden_dim = 128
        mock_hidden = torch.randn(1, 4, hidden_dim)
        mock_output = MagicMock()
        mock_output.hidden_states = (None,) + (mock_hidden,) * 12
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        r_acts, n_acts = extract_batch_activations(samples, model, tokenizer, (0,), batch_size=1)

        assert 0 in r_acts
        assert 0 in n_acts
        assert len(r_acts[0]) == 0
        assert len(n_acts[0]) == 1

    def test_batch_activations_progress_callback(self) -> None:
        """extract_batch_activations should use tqdm progress bar."""
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple output.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]

        model = MagicMock()
        model.device = torch.device("cpu")
        tokenizer = MagicMock()

        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.convert_ids_to_tokens.return_value = ["Wait", "test", "."]
        tokenizer.return_value = {"input_ids": Tensor([[1, 2, 3]]).long()}
        tokenizer.__call__ = lambda text, return_tensors=None: {
            "input_ids": Tensor([[1, 2, 3]]).long()
        }

        hidden_dim = 64
        mock_hidden = torch.randn(1, 3, hidden_dim)
        mock_output = MagicMock()
        mock_output.hidden_states = (None,) + (mock_hidden,) * 12
        model.return_value = mock_output
        model.__call__ = lambda **kwargs: mock_output

        r_acts, n_acts = extract_batch_activations(samples, model, tokenizer, (0,), batch_size=1)

        assert len(r_acts[0]) == 1
        assert len(n_acts[0]) == 1


class TestDiffMeans:
    """Tests for compute_difference_in_means function."""

    def test_diff_means_basic(self) -> None:
        """compute_difference_in_means should compute R - N correctly."""
        r_activations = {
            0: [Tensor([1.0, 2.0]), Tensor([2.0, 3.0])],
            5: [Tensor([5.0, 6.0]), Tensor([7.0, 8.0])],
        }
        n_activations = {
            0: [Tensor([0.0, 1.0]), Tensor([1.0, 2.0])],
            5: [Tensor([3.0, 4.0]), Tensor([5.0, 6.0])],
        }

        result = compute_difference_in_means(r_activations, n_activations, (0, 5))

        assert 0 in result
        assert 5 in result
        assert torch.allclose(result[0], Tensor([1.0, 1.0]))
        assert torch.allclose(result[5], Tensor([2.0, 2.0]))

    def test_diff_means_normalization(self) -> None:
        """compute_difference_in_means should not normalize by default (raw difference)."""
        r_activations = {
            0: [Tensor([2.0, 0.0]), Tensor([4.0, 0.0])],
        }
        n_activations = {
            0: [Tensor([1.0, 0.0]), Tensor([1.0, 0.0])],
        }

        result = compute_difference_in_means(r_activations, n_activations, (0,))

        expected = Tensor([2.0, 0.0])
        assert torch.allclose(result[0], expected)

    def test_diff_means_mismatched_shapes(self) -> None:
        """compute_difference_in_means should validate layer coverage."""
        r_activations = {
            0: [Tensor([1.0, 2.0])],
        }
        n_activations = {
            0: [Tensor([1.0, 2.0])],
            5: [Tensor([3.0, 4.0])],
        }

        with pytest.raises(ValueError, match="No R activations for layer 5"):
            compute_difference_in_means(r_activations, n_activations, (0, 5))

    def test_diff_means_single_layer(self) -> None:
        """compute_difference_in_means should work with single layer tensor."""
        r_activations = {
            10: [Tensor([1.0, 2.0, 3.0]), Tensor([3.0, 4.0, 5.0])],
        }
        n_activations = {
            10: [Tensor([0.0, 0.0, 0.0]), Tensor([2.0, 2.0, 2.0])],
        }

        result = compute_difference_in_means(r_activations, n_activations, (10,))

        assert 10 in result
        expected = Tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result[10], expected)


class TestSave:
    """Tests for save_steering_vectors function."""

    def test_save_vectors_basic(self, tmp_path: Path) -> None:
        """save_steering_vectors should write files to disk."""
        vectors = {0: Tensor([1.0, 2.0]), 5: Tensor([3.0, 4.0])}
        metadata = {"model_name": "test-model", "r_count": 10, "n_count": 5}
        output_file = tmp_path / "test_vectors.pt"

        save_steering_vectors(vectors, metadata, output_file)

        assert output_file.exists()

        loaded = torch.load(output_file, weights_only=True)
        assert "layer_0" in loaded
        assert "layer_5" in loaded
        assert "metadata" in loaded
        assert torch.allclose(loaded["layer_0"], Tensor([1.0, 2.0]))
        assert torch.allclose(loaded["layer_5"], Tensor([3.0, 4.0]))

    def test_save_vectors_creates_directory(self, tmp_path: Path) -> None:
        """save_steering_vectors should create output directory if needed."""
        vectors = {0: Tensor([1.0, 2.0])}
        metadata = {"model_name": "test-model"}
        output_file = tmp_path / "subdir" / "nested" / "vectors.pt"

        save_steering_vectors(vectors, metadata, output_file)

        assert output_file.exists()
        assert output_file.parent.is_dir()

    def test_save_vectors_metadata(self, tmp_path: Path) -> None:
        """save_steering_vectors should include metadata in saved files."""
        vectors = {0: Tensor([1.0, 2.0])}
        metadata = {
            "model_name": "test-model",
            "layer_indices": (0, 5, 10),
            "r_count": 20,
            "n_count": 15,
        }
        output_file = tmp_path / "test_vectors.pt"

        save_steering_vectors(vectors, metadata, output_file)

        loaded = torch.load(output_file, weights_only=True)
        loaded_meta = loaded["metadata"]
        assert loaded_meta["model_name"] == "test-model"
        assert loaded_meta["layer_indices"] == (0, 5, 10)
        assert loaded_meta["r_count"] == 20
        assert loaded_meta["n_count"] == 15
        assert "timestamp" in loaded_meta

    def test_save_vectors_overwrite(self, tmp_path: Path) -> None:
        """save_steering_vectors should handle existing files appropriately."""
        vectors = {0: Tensor([1.0, 2.0])}
        metadata = {"model_name": "test-model"}
        output_file = tmp_path / "test_vectors.pt"

        save_steering_vectors(vectors, metadata, output_file)

        vectors2 = {0: Tensor([5.0, 6.0])}
        metadata2 = {"model_name": "updated-model"}
        save_steering_vectors(vectors2, metadata2, output_file)

        loaded = torch.load(output_file, weights_only=True)
        assert torch.allclose(loaded["layer_0"], Tensor([5.0, 6.0]))
        assert loaded["metadata"]["model_name"] == "updated-model"


class TestPipeline:
    """Tests for extract_steering_vectors end-to-end function."""

    def _create_mock_tokenizer(self, seq_len: int) -> MagicMock:
        """Helper to create a properly configured tokenizer mock."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(seq_len))
        mock_tokenizer.convert_ids_to_tokens.return_value = ["Wait"] + ["token"] * (seq_len - 1)
        mock_tokenizer.return_value = {"input_ids": Tensor([[1] * seq_len]).long()}
        return mock_tokenizer

    def _create_mock_model(self, hidden_dim: int, seq_len: int, num_layers: int) -> MagicMock:
        """Helper to create a properly configured model mock."""
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_hidden = torch.randn(1, seq_len, hidden_dim)
        mock_output = MagicMock()
        mock_output.hidden_states = (None,) + (mock_hidden,) * num_layers
        mock_model.return_value = mock_output
        mock_model.to.return_value = mock_model
        return mock_model

    def test_pipeline_basic(self, tmp_path: Path) -> None:
        """extract_steering_vectors should run full pipeline."""
        input_file = tmp_path / "test.jsonl"
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, this is a test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "This is simple.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        config = ExtractVectorsConfig(
            input_path=str(input_file),
            model_name="test-model",
            layer_indices=(0, 5),
            output_path=str(tmp_path / "output.pt"),
            min_samples=1,
            batch_size=1,
        )

        mock_model = self._create_mock_model(hidden_dim=64, seq_len=4, num_layers=12)
        mock_tokenizer = self._create_mock_tokenizer(seq_len=4)

        with (
            patch("probing_reflection.steering_vectors.AutoModelForCausalLM") as mock_model_cls,
            patch("probing_reflection.steering_vectors.AutoTokenizer") as mock_tokenizer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            result = extract_steering_vectors(config)

        assert "vectors" in result
        assert "metadata" in result
        assert 0 in result["vectors"]
        assert 5 in result["vectors"]

    def test_pipeline_returns_dict(self, tmp_path: Path) -> None:
        """extract_steering_vectors should return dict mapping layer to vector."""
        input_file = tmp_path / "test.jsonl"
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple output.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        config = ExtractVectorsConfig(
            input_path=str(input_file),
            model_name="test-model",
            layer_indices=(3,),
            output_path=str(tmp_path / "output.pt"),
            min_samples=1,
            batch_size=1,
        )

        hidden_dim = 128
        mock_model = self._create_mock_model(hidden_dim=hidden_dim, seq_len=3, num_layers=5)
        mock_tokenizer = self._create_mock_tokenizer(seq_len=3)

        with (
            patch("probing_reflection.steering_vectors.AutoModelForCausalLM") as mock_model_cls,
            patch("probing_reflection.steering_vectors.AutoTokenizer") as mock_tokenizer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            result = extract_steering_vectors(config)

        assert isinstance(result["vectors"], dict)
        assert 3 in result["vectors"]
        assert result["vectors"][3].shape == (hidden_dim,)

    def test_pipeline_multiple_layers(self, tmp_path: Path) -> None:
        """extract_steering_vectors should handle multiple layer indices."""
        input_file = tmp_path / "test.jsonl"
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        config = ExtractVectorsConfig(
            input_path=str(input_file),
            model_name="test-model",
            layer_indices=(0, 5, 10),
            output_path=str(tmp_path / "output.pt"),
            min_samples=1,
            batch_size=1,
        )

        mock_model = self._create_mock_model(hidden_dim=64, seq_len=3, num_layers=15)
        mock_tokenizer = self._create_mock_tokenizer(seq_len=3)

        with (
            patch("probing_reflection.steering_vectors.AutoModelForCausalLM") as mock_model_cls,
            patch("probing_reflection.steering_vectors.AutoTokenizer") as mock_tokenizer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            result = extract_steering_vectors(config)

        assert 0 in result["vectors"]
        assert 5 in result["vectors"]
        assert 10 in result["vectors"]
        assert result["metadata"]["layer_indices"] == (0, 5, 10)

    def test_pipeline_model_loading(self, tmp_path: Path) -> None:
        """extract_steering_vectors should load model from name or path."""
        input_file = tmp_path / "test.jsonl"
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        config = ExtractVectorsConfig(
            input_path=str(input_file),
            model_name="custom-model-path",
            layer_indices=(0,),
            output_path=str(tmp_path / "output.pt"),
            min_samples=1,
            batch_size=1,
        )

        mock_model = self._create_mock_model(hidden_dim=32, seq_len=3, num_layers=5)
        mock_tokenizer = self._create_mock_tokenizer(seq_len=3)

        with (
            patch("probing_reflection.steering_vectors.AutoModelForCausalLM") as mock_model_cls,
            patch("probing_reflection.steering_vectors.AutoTokenizer") as mock_tokenizer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            extract_steering_vectors(config)

            mock_model_cls.from_pretrained.assert_called_once()
            call_args = mock_model_cls.from_pretrained.call_args
            assert call_args[0][0] == "custom-model-path"
            assert "torch_dtype" in call_args[1]
            mock_tokenizer_cls.from_pretrained.assert_called_once_with("custom-model-path")

    def test_pipeline_logging(self, tmp_path: Path) -> None:
        """extract_steering_vectors should log progress during extraction."""
        input_file = tmp_path / "test.jsonl"
        samples: list[SampleWithReflection] = [
            {
                "problem_id": "r1",
                "problem": "test",
                "generated": "Wait, test.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 1,
                "reflection_density": 0.1,
            },
            {
                "problem_id": "n1",
                "problem": "test",
                "generated": "Simple.",
                "reference_answer": "answer",
                "reflection_tokens": [],
                "reflection_count": 0,
                "reflection_density": 0.0,
            },
        ]
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        config = ExtractVectorsConfig(
            input_path=str(input_file),
            model_name="test-model",
            layer_indices=(0,),
            output_path=str(tmp_path / "output.pt"),
            min_samples=1,
            batch_size=1,
        )

        mock_model = self._create_mock_model(hidden_dim=32, seq_len=3, num_layers=5)
        mock_tokenizer = self._create_mock_tokenizer(seq_len=3)

        with (
            patch("probing_reflection.steering_vectors.AutoModelForCausalLM") as mock_model_cls,
            patch("probing_reflection.steering_vectors.AutoTokenizer") as mock_tokenizer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            result = extract_steering_vectors(config)

        assert result["metadata"]["r_count"] == 1
        assert result["metadata"]["n_count"] == 1
        assert "model_name" in result["metadata"]
        assert "timestamp" in result["metadata"]
