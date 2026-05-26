"""Tests for the RAPTOR disk caches and key derivation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.raptor.cache import (
    SummaryCache,
    hash_corpus,
    summary_key,
    tree_key,
)


def test_summary_key_changes_with_model():
    a = summary_key(["one"], model="m1", prompt_version="v1")
    b = summary_key(["one"], model="m2", prompt_version="v1")
    c = summary_key(["one"], model="m1", prompt_version="v2")
    assert len({a, b, c}) == 3


def test_summary_key_stable_for_same_inputs():
    a = summary_key(["one", "two"], model="m1", prompt_version="v1")
    b = summary_key(["one", "two"], model="m1", prompt_version="v1")
    assert a == b


def test_summary_cache_isolated_per_file(tmp_path):
    cache = SummaryCache(root=str(tmp_path))
    cache.put("k1", "first")
    cache.put("k2", "second")
    assert cache.get("k1") == "first"
    assert cache.get("k2") == "second"
    # Two keys -> two files
    files = list(tmp_path.iterdir())
    assert len(files) == 2


def test_hash_corpus_changes_on_content_edit(tmp_path):
    (tmp_path / "a.md").write_text("v1", encoding="utf-8")
    h1 = hash_corpus(str(tmp_path))
    (tmp_path / "a.md").write_text("v2", encoding="utf-8")
    h2 = hash_corpus(str(tmp_path))
    assert h1 != h2


def test_hash_corpus_changes_on_new_file(tmp_path):
    (tmp_path / "a.md").write_text("v1", encoding="utf-8")
    h1 = hash_corpus(str(tmp_path))
    (tmp_path / "b.md").write_text("v1", encoding="utf-8")
    h2 = hash_corpus(str(tmp_path))
    assert h1 != h2


def test_tree_key_changes_on_each_param():
    base = dict(
        corpus_hash="x",
        leaf_window=100,
        leaf_overlap=0,
        max_levels=4,
        umap_seed=42,
        gmm_seed=42,
        summarizer_model="m1",
        embedding_model="e1",
        prompt_version="v1",
        soft_assign_threshold=0.0,
    )
    base_key = tree_key(**base)
    for field in (
        "corpus_hash",
        "leaf_window",
        "leaf_overlap",
        "max_levels",
        "umap_seed",
        "gmm_seed",
        "summarizer_model",
        "embedding_model",
        "prompt_version",
    ):
        flipped = {**base}
        # Bump int or replace string
        if isinstance(flipped[field], int):
            flipped[field] = flipped[field] + 1
        else:
            flipped[field] = str(flipped[field]) + "_x"
        assert tree_key(**flipped) != base_key, f"{field} did not change the tree_key"
