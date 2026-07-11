"""Smoke tests for the RAPTOR subpackage.

Network-free: these tests build trees without calling an LLM by injecting a
stub summarizer. They cover the node/cache/clustering/index/retriever surface.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib

import numpy as np
import pytest

from src.config import SearchStrategy
from src.raptor.cache import SummaryCache, TreeCache, hash_corpus, summary_key, tree_key
from src.raptor.node import RaptorNode
from src.raptor.tree_index import RaptorIndex
from src.raptor.tree_retriever import QCondConfig, RaptorRetriever


def _has_umap() -> bool:
    try:
        importlib.import_module("umap")
        return True
    except Exception:
        return False


def test_node_to_chunk_joins_sources():
    n = RaptorNode(
        node_id="n1",
        text="summary text",
        embedding=[0.1, 0.2],
        level=1,
        token_count=3,
        source_files=["a.md", "b.md"],
    )
    chunk = n.to_chunk()
    assert "a.md" in chunk.source_file
    assert "b.md" in chunk.source_file
    assert ";" in chunk.source_file
    assert chunk.chunking_strategy == "raptor_L1"
    # Scorers use the structured list, not the joined display string.
    assert chunk.source_files == ["a.md", "b.md"]


def test_chunk_index_is_stable_across_processes():
    """chunk_index must be derived from a hash that is stable across
    process restarts, not Python's builtin `hash()` (randomized per-process
    for strings unless PYTHONHASHSEED is set before interpreter start)."""
    import hashlib

    node = RaptorNode(
        node_id="L2_C7", text="t", embedding=[0.1], level=2, token_count=1,
    )
    expected = int(hashlib.md5("L2_C7".encode("utf-8")).hexdigest()[:8], 16)
    assert node.to_chunk().chunk_index == expected

    # Same node_id -> same chunk_index regardless of other field values.
    other = RaptorNode(
        node_id="L2_C7", text="different text", embedding=[0.9], level=2,
        token_count=99,
    )
    assert other.to_chunk().chunk_index == expected


def test_hash_corpus_stable(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    h1 = hash_corpus(str(tmp_path))
    h2 = hash_corpus(str(tmp_path))
    assert h1 == h2
    # Touch content -> hash changes
    (tmp_path / "a.md").write_text("alpha v2", encoding="utf-8")
    assert hash_corpus(str(tmp_path)) != h1


def test_summary_cache_roundtrip(tmp_path):
    cache = SummaryCache(root=str(tmp_path))
    key = summary_key(["one", "two"], model="m1", prompt_version="v1")
    assert cache.get(key) is None
    cache.put(key, "summary text")
    assert cache.get(key) == "summary text"


def test_tree_key_changes_with_params():
    base = dict(
        corpus_hash="h",
        leaf_window=100,
        max_levels=4,
        umap_seed=42,
        gmm_seed=42,
        summarizer_model="m",
        embedding_model="e",
        prompt_version="v1",
        soft_assign_threshold=0.0,
    )
    a = tree_key(**base)
    b = tree_key(**{**base, "max_levels": 3})
    assert a != b


@pytest.mark.skipif(not _has_umap(), reason="umap-learn not installed")
def test_clustering_separates_obvious_groups():
    """Three well-separated Gaussian blobs -> at least three clusters."""
    from src.raptor.clustering import cluster_embeddings  # deferred import
    rng = np.random.default_rng(0)
    g1 = rng.normal(loc=(0, 0, 0), scale=0.1, size=(40, 3))
    g2 = rng.normal(loc=(10, 10, 10), scale=0.1, size=(40, 3))
    g3 = rng.normal(loc=(-10, 10, -10), scale=0.1, size=(40, 3))
    pts = np.vstack([g1, g2, g3])
    # Pad to embedding dim
    embs = np.hstack([pts, np.zeros((len(pts), 5))]).astype(np.float32)
    clusters = cluster_embeddings(embs, umap_seed=0, gmm_seed=0)
    assert len(clusters) >= 3, f"expected at least 3 clusters, got {len(clusters)}"


def test_raptor_index_collapsed_search():
    n_leaf = RaptorNode(
        node_id="l1",
        text="leaf",
        embedding=[1.0, 0.0, 0.0],
        level=0,
        token_count=1,
        source_files=["a.md"],
    )
    n_summary = RaptorNode(
        node_id="s1",
        text="summary",
        embedding=[0.0, 1.0, 0.0],
        level=1,
        token_count=1,
        source_files=["a.md"],
        children=["l1"],
    )
    index = RaptorIndex(
        nodes_by_id={n_leaf.node_id: n_leaf, n_summary.node_id: n_summary},
        root_ids=["s1"],
        tree_hash="t",
    )
    # Query aligned with leaf direction
    results = index.collapsed_search([1.0, 0.0, 0.0], top_k=2)
    assert results[0][0].node_id == "l1"
    # Query aligned with summary direction
    results = index.collapsed_search([0.0, 1.0, 0.0], top_k=2)
    assert results[0][0].node_id == "s1"


def _toy_tree() -> RaptorIndex:
    """Hand-built tiny tree: root -> two summaries, each over two leaves."""
    leaves = [
        RaptorNode(node_id=f"l{i}", text=f"leaf {i}", embedding=v, level=0,
                   token_count=2, source_files=[f"d{i}.md"])
        for i, v in enumerate([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ])
    ]
    s_left = RaptorNode(
        node_id="s_left", text="left half", embedding=[1.0, 1.0, 0.0, 0.0],
        level=1, token_count=2, source_files=["d0.md", "d1.md"],
        children=["l0", "l1"],
    )
    s_right = RaptorNode(
        node_id="s_right", text="right half", embedding=[0.0, 0.0, 1.0, 1.0],
        level=1, token_count=2, source_files=["d2.md", "d3.md"],
        children=["l2", "l3"],
    )
    root = RaptorNode(
        node_id="root", text="all", embedding=[1.0, 1.0, 1.0, 1.0],
        level=2, token_count=1,
        source_files=["d0.md", "d1.md", "d2.md", "d3.md"],
        children=["s_left", "s_right"],
    )
    nodes = {n.node_id: n for n in [*leaves, s_left, s_right, root]}
    return RaptorIndex(nodes_by_id=nodes, root_ids=["root"], tree_hash="toy")


def test_raptor_retriever_tree_mode_descends():
    index = _toy_tree()
    retriever = RaptorRetriever(index, embedding_model=_FakeEmbedder())
    # Force the embedder to return a vector aligned with the left half
    _FakeEmbedder.next_vec = [1.0, 0.0, 0.0, 0.0]
    result = retriever.retrieve("dummy", SearchStrategy.RAPTOR_TREE, top_k=3)
    assert len(result.chunks) <= 3
    ids = [c.source_file for c in result.chunks]
    # Must include at least one left-side leaf
    assert any("d0.md" in s or "d1.md" in s for s in ids)


def test_raptor_retriever_collapsed_mode_works():
    index = _toy_tree()
    retriever = RaptorRetriever(index, embedding_model=_FakeEmbedder())
    _FakeEmbedder.next_vec = [0.0, 0.0, 1.0, 0.0]
    result = retriever.retrieve("dummy", SearchStrategy.RAPTOR_COLLAPSED, top_k=2)
    assert result.chunks
    assert len(result.chunks) <= 2
    # Top hit should be on the right side
    assert "d2.md" in result.chunks[0].source_file or "d3.md" in result.chunks[0].source_file


def test_raptor_retriever_qcond_terminates_when_node_beats_children():
    """If a frontier node scores higher than its children, qcond should
    terminate at it rather than descend."""
    # Parent points at the query direction; children point orthogonally.
    leaves = [
        RaptorNode(node_id="l0", text="leaf0", embedding=[0.0, 1.0], level=0,
                   token_count=1, source_files=["a.md"]),
        RaptorNode(node_id="l1", text="leaf1", embedding=[0.0, 1.0], level=0,
                   token_count=1, source_files=["b.md"]),
    ]
    parent = RaptorNode(
        node_id="p", text="parent", embedding=[1.0, 0.0], level=1,
        token_count=1, source_files=["a.md", "b.md"],
        children=["l0", "l1"],
    )
    nodes = {n.node_id: n for n in [*leaves, parent]}
    index = RaptorIndex(nodes_by_id=nodes, root_ids=["p"], tree_hash="t")
    retriever = RaptorRetriever(
        index,
        embedding_model=_FakeEmbedder(),
        qcond_config=QCondConfig(tau_term=0.05),
    )
    _FakeEmbedder.next_vec = [1.0, 0.0]
    result = retriever.retrieve("dummy", SearchStrategy.RAPTOR_QCOND, top_k=3)
    assert len(result.chunks) <= 3
    top_ids = [c.source_file for c in result.chunks]
    # The parent should be the top hit; its chunk's source_file joins both children.
    assert top_ids, "qcond returned no chunks"
    assert ";" in top_ids[0], f"top hit should be the parent (joined sources); got {top_ids[0]}"


def test_raptor_retriever_qcond_handles_dangling_children():
    """A node whose children ids are all missing from the index (e.g. a
    dangling reference from a partial/stale build) must be treated as
    terminal instead of raising when scoring an empty child set."""
    node = RaptorNode(
        node_id="p", text="parent", embedding=[1.0, 0.0], level=1,
        token_count=1, source_files=["a.md"],
        children=["missing_child_1", "missing_child_2"],
    )
    index = RaptorIndex(nodes_by_id={"p": node}, root_ids=["p"], tree_hash="t")
    retriever = RaptorRetriever(index, embedding_model=_FakeEmbedder())
    _FakeEmbedder.next_vec = [1.0, 0.0]
    result = retriever.retrieve("dummy", SearchStrategy.RAPTOR_QCOND, top_k=3)
    assert len(result.chunks) <= 3
    assert result.chunks
    assert result.chunks[0].source_file == "a.md"


class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode."""

    next_vec: list[float] = [1.0, 0.0]

    def encode(self, text, show_progress_bar=False):
        if isinstance(text, str):
            return np.array(_FakeEmbedder.next_vec, dtype=np.float32)
        return np.array([_FakeEmbedder.next_vec for _ in text], dtype=np.float32)


class _StubSummarizer:
    """Deterministic stand-in for raptor.summarizer.Summarizer — no network."""

    class _StubLLMConfig:
        model = "stub-model"

    llm_config = _StubLLMConfig()

    def __init__(self):
        self.call_count = 0
        self.cache_hits = 0

    def summarize(self, passages: list[str]) -> str:
        self.call_count += 1
        return "summary: " + " ".join(passages)[:60]


@pytest.mark.skipif(not _has_umap(), reason="umap-learn not installed")
def test_raptor_tree_builder_deterministic_ids_across_builds(tmp_path, monkeypatch):
    """Two independent builds of the same tiny corpus (fresh caches, stub
    summarizer, stubbed clustering) must produce identical node ids and
    chunk_index values. Regression test for the former uuid4()-based
    summary-node ids, which made the same corpus build to different node
    identities on every run.
    """
    from src.raptor.tree_builder import RaptorBuildConfig, RaptorTreeBuilder
    from src.raptor.cache import SummaryCache, TreeCache
    import src.raptor.tree_builder as tree_builder_mod

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.md").write_text(
        "Alpha document about apples and orchards.", encoding="utf-8"
    )
    (corpus_dir / "b.md").write_text(
        "Beta document about rockets and engines.", encoding="utf-8"
    )

    # Force a deterministic 2-way split at every level regardless of the
    # real embedding values, so this test isolates node-id determinism from
    # BIC cluster-count selection (covered separately by
    # test_clustering_separates_obvious_groups).
    def _fixed_clusters(embeddings, **kwargs):
        n = len(embeddings)
        return [[i] for i in range(n)] if n == 2 else [list(range(n))]

    monkeypatch.setattr(tree_builder_mod, "cluster_embeddings", _fixed_clusters)

    def _build_once(tag: str):
        cfg = RaptorBuildConfig(leaf_window=100, max_levels=2)
        builder = RaptorTreeBuilder(
            build_config=cfg,
            tree_cache=TreeCache(root=str(tmp_path / f"trees_{tag}")),
            summary_cache=SummaryCache(root=str(tmp_path / f"summaries_{tag}")),
        )
        builder.summarizer = _StubSummarizer()
        return builder.build(str(corpus_dir))

    index1 = _build_once("run1")
    index2 = _build_once("run2")

    assert set(index1.nodes_by_id.keys()) == set(index2.nodes_by_id.keys())
    assert len(index1.nodes_by_id) > 2  # sanity: summary levels were actually built
    for node_id in index1.nodes_by_id:
        c1 = index1.nodes_by_id[node_id].to_chunk()
        c2 = index2.nodes_by_id[node_id].to_chunk()
        assert c1.chunk_index == c2.chunk_index


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
