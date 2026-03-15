"""Shared test fixtures for ProbingReflection tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_contrastive_pairs() -> list[dict[str, str]]:
    """Sample contrastive pairs for testing."""
    return [
        {
            "positive": "Let me think about this step by step.",
            "negative": "I'll just give you the answer.",
        },
        {
            "positive": "I should reconsider my approach here.",
            "negative": "This is definitely correct.",
        },
        {
            "positive": "Wait, I might be wrong about this.",
            "negative": "Trust me, I know what I'm doing.",
        },
    ]
