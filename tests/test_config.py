"""Tests for RetrievalConfig.from_name (fail-fast config-name parsing)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.config import ChunkingStrategy, RetrievalConfig, SearchStrategy


def test_from_name_parses_valid_config():
    cfg = RetrievalConfig.from_name("fixed_256__vector__k3")
    assert cfg.chunking == ChunkingStrategy.FIXED_256
    assert cfg.search == SearchStrategy.VECTOR
    assert cfg.top_k == 3


def test_from_name_parses_valid_raptor_config():
    cfg = RetrievalConfig.from_name("raptor_100__raptor_tree__k5")
    assert cfg.chunking == ChunkingStrategy.RAPTOR_100
    assert cfg.search == SearchStrategy.RAPTOR_TREE
    assert cfg.top_k == 5


def test_from_name_rejects_invalid_raptor_baseline_mix():
    """RAPTOR chunking paired with a non-RAPTOR search must raise, not
    silently construct a config that will fail deep inside the runner."""
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("raptor_100__vector__k3")


def test_from_name_rejects_baseline_chunking_with_raptor_search():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__raptor_tree__k3")


def test_from_name_rejects_malformed_name():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__vector")


def test_from_name_rejects_unknown_chunking():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("nonexistent_chunking__vector__k3")


def test_from_name_rejects_unknown_search():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__nonexistent_search__k3")


def test_from_name_rejects_bad_top_k():
    with pytest.raises(ValueError):
        RetrievalConfig.from_name("fixed_256__vector__three")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
