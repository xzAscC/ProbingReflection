# SPEC.md

Project-level specification for AI coding agents. This document defines WHAT to build, while AGENTS.md defines HOW to work.

## 1) Project Vision

ProbingReflection investigates self-reflection in Large Language Models through probing and steering techniques. This project explores how reflection behaviors emerge and how they can be controlled through vector-based interventions.

## 2) Tech Stack

- Python 3.12+
- Package manager: `uv`
- Linting/Formatting: `ruff`
- Type checking: `mypy`
- Testing: `pytest`

## 3) Project Structure

- `src/probing_reflection/` — Source code modules
- `tests/` — Test files mirroring source structure
- `docs/` — Documentation including:
  - `docs/design-docs/` — Design and experiment documentation
  - `.sisyphus/plans/` — Execution plans for tasks
  - `.sisyphus/roadmap.md` — Project roadmap
  - `docs/QUALITY_SCORE.md` — Quality tracking metrics

## 4) Three-Tier Boundaries

### ✅ Always Do

Actions the agent takes without asking:

- Run tests before commits
- Follow naming conventions (snake_case for functions/vars, PascalCase for classes)
- Update documentation when changing code behavior
- Add type hints to all function parameters and returns
- Write tests for new functionality
- Follow the commit criteria in AGENTS.md

### ⚠️ Ask First

Actions requiring human approval:

- Adding new dependencies
- Modifying `pyproject.toml` configuration
- Changing CI configuration (`.github/workflows/`)
- Creating new top-level directories
- Modifying existing public APIs
- Changing code style configuration

### 🚫 Never Do

Hard stops that require escalation:

- Commit secrets, API keys, or credentials
- Edit files in `.git/` or `__pycache__/`
- Remove tests without explicit approval
- Skip type hints on new code
- Push to `main` or `master` branch directly
- Modify `.python-version` without approval
- Delete documentation files
