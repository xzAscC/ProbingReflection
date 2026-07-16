# ProbingReflection

Official code for **[From Emergence to Control: Probing and Modulating
Self-Reflection in Language Models](https://arxiv.org/abs/2506.12217)**.

This repository studies where self-reflection appears in language-model
representations and how it can be measured and controlled. It includes
reproducible pipelines for inference, LLM-based evaluation, reflection-token
diagnosis, activation probing, steering-vector extraction, and steering
interventions.

## Features

- Run chain-of-thought inference on mathematical reasoning datasets.
- Evaluate generated answers with a bidirectional LLM-as-a-judge protocol.
- Detect and categorize self-reflection tokens in model outputs.
- Measure reasoning quality with ROSCOE-style faithfulness, coherence,
  informativeness, repetition, and completeness scores.
- Train layer-wise linear probes to test whether reflective and
  non-reflective token activations are linearly separable.
- Extract difference-in-means steering vectors from contrastive activations.
- Apply steering vectors during generation with configurable layers and
  coefficients.

## Installation

The project requires Python 3.12 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/xzAscC/ProbingReflection.git
cd ProbingReflection
uv sync
```

Model-backed experiments require access to the configured Hugging Face models
and sufficient CPU/GPU memory. Some datasets, including GPQA, may require
accepting their access conditions on Hugging Face.

## Command-Line Usage

List the available commands:

```bash
uv run probing-reflection --help
```

### Inference

```bash
uv run probing-reflection inference \
  --model Qwen/Qwen3.5-0.8B \
  --dataset HuggingFaceH4/MATH-500 \
  --batch-size 8 \
  --max-new-tokens 256 \
  --limit 100 \
  --output outputs/math500.jsonl
```

### Answer evaluation

```bash
uv run probing-reflection evaluate outputs/math500.jsonl \
  --model Qwen/Qwen3.5-27B \
  --confidence-threshold 0.7 \
  --output outputs/evaluation.json
```

### Reflection diagnosis

```bash
uv run probing-reflection reflection-diagnose \
  --input outputs/math500.jsonl \
  --model Qwen/Qwen3.5-27B \
  --output-dir outputs/reflection_diagnosis
```

### Steering-vector extraction

```bash
uv run probing-reflection extract-vectors \
  --input outputs/reflection_diagnosis/analyzed_samples.jsonl \
  --model Qwen/Qwen2.5-0.5B \
  --layers 8,12,16 \
  --min-samples 10 \
  --output outputs/steering_vectors.pt
```

Additional wrappers for evaluation, diagnosis, linear probing, vector
extraction, and steering inference are available under
`scripts/probing_reflection/`. Use `scripts/run_experiments.py` to coordinate
multi-dataset steering experiments and `scripts/generate_report.py` to create
summary reports.

## Project Layout

```text
src/probing_reflection/
├── inference.py              # Dataset inference pipeline
├── evaluation.py             # Answer evaluation and reports
├── reflection_diagnosis.py   # Reflection-token analysis
├── linear_probe.py           # Layer-wise linear probing
├── steering_vectors.py       # Steering-vector extraction
├── steering_inference.py     # Steered generation
├── judges.py                 # Shared LLM judge implementations
├── roscoe_metrics.py         # ROSCOE-style reasoning metrics
├── model_utils.py            # Model loading and lifecycle helpers
├── batch_utils.py            # Shared batching and decoding helpers
├── prompts.py                # Prompt templates and reflection taxonomy
└── types.py                  # Typed configurations and result schemas

tests/                        # Deterministic unit and integration tests
scripts/                      # Experiment and reporting entry points
docs/                         # Architecture and design documentation
```

Generated outputs, local datasets, figures, notebooks, and experiment state are
excluded from Git. The repository-level `.ignore` file keeps these local paths
searchable by OpenCode and ripgrep without uploading them to GitHub.

## Development

```bash
uv sync
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/
uv run pytest
```

The current test suite contains 159 deterministic tests and mocks external
model and dataset boundaries.

## Citation

If you use this code, please cite:

```bibtex
@article{zhu2025emergence,
  title={From emergence to control: Probing and modulating self-reflection in language models},
  author={Zhu, Xudong and Jiang, Jiachen and Khalili, Mohammad Mahdi and Zhu, Zhihui},
  journal={arXiv preprint arXiv:2506.12217},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
