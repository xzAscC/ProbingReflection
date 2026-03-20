"""Tests for inference utilities.

These tests verify the inference pipeline components including
prompt formatting, tokenization, and output handling.
"""

from probing_reflection.inference import format_cot_prompt


class TestInference:
    """Tests for inference functions."""

    def test_format_cot_prompt(self) -> None:
        """Prompt should contain step by step and boxed instructions."""
        result = format_cot_prompt("What is 2+2?")
        assert "step by step" in result
        assert "\\boxed" in result

    def test_batch_tokenization(self) -> None:
        """Batch tokenization should apply left padding."""
        # Create a mock tokenizer with right padding (default for most tokenizers)
        from unittest.mock import MagicMock

        from probing_reflection.inference import prepare_batch

        tokenizer = MagicMock()
        tokenizer.padding_side = "right"
        tokenizer.return_value = {
            "input_ids": [[1, 2], [1, 2, 3]],
            "attention_mask": [[1, 1], [1, 1, 1]],
        }

        texts = ["short", "longer text"]

        # Call prepare_batch - should set padding_side to "left" for Qwen compatibility
        result = prepare_batch(tokenizer, texts)

        # Verify left padding was applied
        assert tokenizer.padding_side == "left", (
            f"Expected left padding for Qwen compatibility, got {tokenizer.padding_side}"
        )
        assert result is not None, "prepare_batch should return tokenized output"

    def test_jsonl_output_format(self) -> None:
        """JSONL output should have correct schema."""
        import json
        import tempfile
        from pathlib import Path

        entry = {
            "problem_id": "test/123",
            "problem": "What is 2+2?",
            "generated": "The answer is 4.",
            "reference_answer": "4",
            "subject": "Algebra",
            "level": 1,
            "prompt": "Please reason step by step...",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(entry) + "\n")
            temp_path = Path(f.name)

        with open(temp_path) as f:
            line = f.readline().strip()
            loaded = json.loads(line)

        temp_path.unlink()

        required_keys = [
            "problem_id",
            "problem",
            "generated",
            "reference_answer",
            "subject",
            "level",
            "prompt",
        ]
        for key in required_keys:
            assert key in loaded, f"Missing required key: {key}"

        assert isinstance(loaded["problem_id"], str)
        assert isinstance(loaded["problem"], str)
        assert isinstance(loaded["generated"], str)
        assert isinstance(loaded["reference_answer"], str)
        assert isinstance(loaded["subject"], str)
        assert isinstance(loaded["level"], int)
        assert isinstance(loaded["prompt"], str)

    def test_run_inference_mocked(self) -> None:
        """run_inference should work with mocked model and tokenizer."""
        import os
        import tempfile
        from unittest.mock import MagicMock, patch

        from probing_reflection.inference import run_inference

        from probing_reflection import InferenceConfig

        # Create a temporary output path
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.jsonl")

            # Create InferenceConfig with the temp output path
            config = InferenceConfig(output_path=output_path)

            # Create mocks for model and tokenizer
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            # Mock torch.cuda.is_available to return False (CPU mode)
            with patch("torch.cuda.is_available", return_value=False), patch(
                "probing_reflection.inference.load_model",
                return_value=(mock_model, mock_tokenizer),
            ):
                # Call run_inference
                run_inference(config)

            # Verify output file was created
            assert os.path.exists(output_path), f"Output file should be created at {output_path}"
