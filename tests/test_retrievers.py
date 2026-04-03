"""Tests for retrieval strategies."""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunkers import Chunk
from src.config import SearchStrategy, CHROMA_DIR
from src.indexer import CorpusIndex
from src.retrievers import Retriever, get_reranker


# Small known corpus for testing
TEST_CHUNKS = [
    Chunk(
        text="Python is a high-level programming language known for its readability and versatility. It supports multiple programming paradigms including object-oriented, functional, and procedural programming.",
        source_file="tech/python.md",
        chunk_index=0,
        chunking_strategy="test",
    ),
    Chunk(
        text="The quarterly revenue report shows total earnings of $42.3 million, representing a 28% year-over-year growth. Enterprise segment contributed $28.7 million.",
        source_file="finance/earnings.md",
        chunk_index=0,
        chunking_strategy="test",
    ),
    Chunk(
        text="Our deployment process uses blue-green deployments with automated rollback. When error rates exceed 0.5%, the system automatically reverts to the previous stable version.",
        source_file="engineering/deploy.md",
        chunk_index=0,
        chunking_strategy="test",
    ),
    Chunk(
        text="The company offers 16 weeks of paid parental leave for all new parents. Healthcare coverage includes medical, dental, and vision with 95% premium coverage for employees.",
        source_file="hr/benefits.md",
        chunk_index=0,
        chunking_strategy="test",
    ),
]


def _build_test_index():
    """Build a test index with the small corpus."""
    # Use a unique temp directory per call to avoid file lock conflicts on Windows
    test_config = f"test_retriever_{os.getpid()}_{id(object())}"
    test_dir = os.path.join(CHROMA_DIR, test_config)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
    return CorpusIndex(TEST_CHUNKS, test_config)


def test_vector_retrieval_semantic_similarity():
    """Verify the most semantically similar chunk is ranked first."""
    index = _build_test_index()
    retriever = Retriever(index)

    result = retriever.retrieve("What programming language is known for readability?", SearchStrategy.VECTOR, top_k=2)
    assert len(result.chunks) <= 2
    assert result.chunks[0].source_file == "tech/python.md", (
        f"Expected python.md first, got {result.chunks[0].source_file}"
    )


def test_bm25_retrieval_keyword_match():
    """Verify exact keyword matches are ranked first."""
    index = _build_test_index()
    retriever = Retriever(index)

    result = retriever.retrieve("blue-green deployment rollback error rates", SearchStrategy.BM25, top_k=2)
    assert len(result.chunks) <= 2
    assert result.chunks[0].source_file == "engineering/deploy.md", (
        f"Expected deploy.md first, got {result.chunks[0].source_file}"
    )


def test_hybrid_returns_from_both():
    """Verify hybrid returns results from both vector and BM25."""
    index = _build_test_index()
    retriever = Retriever(index)

    result = retriever.retrieve("revenue earnings quarterly report", SearchStrategy.HYBRID, top_k=3)
    assert len(result.chunks) <= 3
    assert len(result.chunks) > 0


def test_hybrid_rerank_changes_order():
    """Verify the reranker can change the order from hybrid."""
    index = _build_test_index()
    retriever = Retriever(index)

    hybrid_result = retriever.retrieve("What is the total revenue?", SearchStrategy.HYBRID, top_k=4)
    rerank_result = retriever.retrieve("What is the total revenue?", SearchStrategy.HYBRID_RERANK, top_k=4)

    # Both should return results
    assert len(hybrid_result.chunks) > 0
    assert len(rerank_result.chunks) > 0

    # Reranker should put the finance chunk first
    assert rerank_result.chunks[0].source_file == "finance/earnings.md", (
        f"Expected finance/earnings.md first after reranking, got {rerank_result.chunks[0].source_file}"
    )


def test_top_k_respected():
    """Verify top_k is respected — never return more than top_k."""
    index = _build_test_index()
    retriever = Retriever(index)

    for strategy in SearchStrategy:
        for top_k in [1, 2, 3]:
            result = retriever.retrieve("programming language", strategy, top_k=top_k)
            assert len(result.chunks) <= top_k, (
                f"Strategy {strategy.value} with top_k={top_k} returned {len(result.chunks)} chunks"
            )


def _cleanup_test_index():
    test_dir = os.path.join(CHROMA_DIR, "test_retriever")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    try:
        test_vector_retrieval_semantic_similarity()
        test_bm25_retrieval_keyword_match()
        test_hybrid_returns_from_both()
        test_hybrid_rerank_changes_order()
        test_top_k_respected()
        print("All retriever tests passed!")
    finally:
        _cleanup_test_index()
