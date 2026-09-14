"""
tests/conftest.py

Shared pytest fixtures.
"""

import pytest
from transformers import ASTConfig


@pytest.fixture(scope="session")
def tiny_config():
    """2-layer randomly initialized AST backbone with the real input geometry (1024 x 128, hidden size 768)."""
    return ASTConfig(
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=1024,
        max_length=1024,
        num_mel_bins=128,
    )
