"""Tests for scoring methods."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunkers import Chunk
from src.scorers import RetrievalScorer, GoldSimilarityScorer, KeyFactScorer


def test_retrieval_scorer_precision():
    """Test context precision with known retrieved chunks and gold sources."""
    chunks = [
        Chunk(text="a", source_file="finance/q3_earnings.md", chunk_index=0, chunking_strategy="test"),
        Chunk(text="b", source_file="finance/q3_earnings.md", chunk_index=1, chunking_strategy="test"),
        Chunk(text="c", source_file="engineering/deploy.md", chunk_index=0, chunking_strategy="test"),
    ]
    scorer = RetrievalScorer()
    metrics = scorer.score(chunks, gold_source_ids=["finance/q3_earnings.md"])

    # 2 out of 3 chunks are from the gold source
    assert abs(metrics.context_precision - 2 / 3) < 0.01
    # 1 out of 1 gold source has at least one chunk
    assert metrics.context_recall == 1.0


def test_retrieval_scorer_recall():
    """Test context recall with multiple gold sources."""
    chunks = [
        Chunk(text="a", source_file="finance/q3_earnings.md", chunk_index=0, chunking_strategy="test"),
        Chunk(text="b", source_file="engineering/deploy.md", chunk_index=0, chunking_strategy="test"),
    ]
    scorer = RetrievalScorer()
    metrics = scorer.score(
        chunks,
        gold_source_ids=["finance/q3_earnings.md", "hr/policy.md", "engineering/deploy.md"],
    )

    # 2 out of 3 gold sources are covered
    assert abs(metrics.context_recall - 2 / 3) < 0.01


def test_retrieval_scorer_distractor_rate():
    """Test distractor rate calculation."""
    chunks = [
        Chunk(text="a", source_file="finance/q3_earnings.md", chunk_index=0, chunking_strategy="test"),
        Chunk(text="b", source_file="finance/q2_earnings.md", chunk_index=0, chunking_strategy="test"),
        Chunk(text="c", source_file="finance/q2_earnings.md", chunk_index=1, chunking_strategy="test"),
    ]
    scorer = RetrievalScorer()
    metrics = scorer.score(
        chunks,
        gold_source_ids=["finance/q3_earnings.md"],
        distractor_ids=["finance/q2_earnings.md"],
    )

    # 2 out of 3 chunks are distractors
    assert abs(metrics.distractor_rate - 2 / 3) < 0.01


def test_retrieval_scorer_empty():
    """Test with empty retrieved chunks."""
    scorer = RetrievalScorer()
    metrics = scorer.score([], gold_source_ids=["any.md"])
    assert metrics.context_precision == 0.0
    assert metrics.context_recall == 0.0


def test_gold_similarity_identical():
    """Test that identical strings score ~1.0."""
    scorer = GoldSimilarityScorer()
    text = "The revenue was $42.3 million with 28% growth."
    score = scorer.score(text, text)
    assert score > 0.95, f"Identical strings scored {score}"


def test_gold_similarity_paraphrase():
    """Test that paraphrases score >0.7."""
    scorer = GoldSimilarityScorer()
    gen = "Meridian earned $42.3M in Q3, growing 28% year-over-year."
    gold = "Q3 revenue was $42.3 million, representing 28% year-over-year growth."
    score = scorer.score(gen, gold)
    assert score > 0.7, f"Paraphrase scored {score}"


def test_gold_similarity_unrelated():
    """Test that unrelated strings score <0.3."""
    scorer = GoldSimilarityScorer()
    gen = "The cat sat on the mat and looked at the birds outside."
    gold = "Q3 revenue was $42.3 million, representing 28% year-over-year growth."
    score = scorer.score(gen, gold)
    assert score < 0.5, f"Unrelated strings scored {score}"


def test_key_fact_scorer_exact_match():
    """Test exact keyword matches."""
    scorer = KeyFactScorer()
    answer = "The revenue was $42.3 million with 28% growth."
    metrics = scorer.score(answer, ["42.3", "28%"])
    assert metrics.fact_recall == 1.0
    assert len(metrics.found_facts) == 2
    assert len(metrics.missing_facts) == 0


def test_key_fact_scorer_case_insensitive():
    """Test case-insensitive matching."""
    scorer = KeyFactScorer()
    answer = "Blue-Green deployment strategy with automated rollback."
    metrics = scorer.score(answer, ["blue-green", "ROLLBACK"])
    assert metrics.fact_recall == 1.0


def test_key_fact_scorer_numeric_variations():
    """Test numeric format variations."""
    scorer = KeyFactScorer()
    answer = "Revenue reached $42.3M with growth of 28 percent."
    metrics = scorer.score(answer, ["42.3", "28%"])
    assert metrics.fact_recall == 1.0, (
        f"Expected 1.0, got {metrics.fact_recall}. Missing: {metrics.missing_facts}"
    )


def test_key_fact_scorer_missing_facts():
    """Test that missing facts are correctly identified."""
    scorer = KeyFactScorer()
    answer = "The company grew significantly last quarter."
    metrics = scorer.score(answer, ["42.3", "28%", "blue-green"])
    assert metrics.fact_recall == 0.0
    assert len(metrics.missing_facts) == 3


def test_key_fact_scorer_partial():
    """Test partial fact recall."""
    scorer = KeyFactScorer()
    answer = "Revenue was $42.3 million but growth was modest."
    metrics = scorer.score(answer, ["42.3", "28%"])
    assert metrics.fact_recall == 0.5
    assert "42.3" in metrics.found_facts
    assert "28%" in metrics.missing_facts


if __name__ == "__main__":
    test_retrieval_scorer_precision()
    test_retrieval_scorer_recall()
    test_retrieval_scorer_distractor_rate()
    test_retrieval_scorer_empty()
    test_gold_similarity_identical()
    test_gold_similarity_paraphrase()
    test_gold_similarity_unrelated()
    test_key_fact_scorer_exact_match()
    test_key_fact_scorer_case_insensitive()
    test_key_fact_scorer_numeric_variations()
    test_key_fact_scorer_missing_facts()
    test_key_fact_scorer_partial()
    print("All scorer tests passed!")
