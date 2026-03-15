# ProbingReflection

Source code for the paper **"From Emergence to Control: Probing and Modulating Self-Reflection in Language Models"**

## Overview

This project investigates self-reflection in Large Language Models (LLMs) through probing and steering techniques. We explore how reflection behaviors emerge and how they can be controlled through vector-based interventions.

## Key Concepts

- **Probing Vectors**: Techniques to detect and measure self-reflection patterns in model activations
- **Model Insertion**: Methods for injecting steering vectors to modulate reflection behavior
- **Reflection Analysis**: Frameworks for evaluating and understanding model self-reflection

## Installation

```bash
uv sync
```

## Development

```bash
# Lint check
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Run tests
uv run pytest
```

## Project Structure

```
.
├── src/probing_reflection/    # Source code
│   ├── __init__.py
│   └── py.typed
├── tests/                     # Test files
├── docs/                      # Documentation
│   └── design-docs/           # Design documents
├── AGENTS.md                  # AI agent instructions
├── ARCHITECTURE.md            # System architecture
└── pyproject.toml             # Project configuration
```

## Research Workflow

This project follows an AI-assisted research workflow. See AGENTS.md for detailed instructions on how AI agents should work in this repository.

## Citation

If you use this code, please cite:

```bibtex
@article{probing_reflection_2024,
  title={From Emergence to Control: Probing and Modulating Self-Reflection in Language Models},
  author={[Authors]},
  year={2024}
}
```

## License

[Add your license here]
