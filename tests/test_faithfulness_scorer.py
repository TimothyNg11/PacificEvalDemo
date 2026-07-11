"""Tests for the NLI faithfulness scorer.

Tests are network-free: they patch the CrossEncoder with a stub. This covers
the splitter, the per-sentence aggregation, and the edge cases.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.chunkers import Chunk
from src.faithfulness_scorer import FaithfulnessScorer


class _StubCrossEncoder:
    """Returns NLI logits where the entailment row is fixed per call."""

    def __init__(self, entail_score: float):
        self.entail_score = entail_score

    def predict(self, pairs):
        # 3-column output: [contradiction, entailment, neutral]
        n = len(pairs)
        return np.array(
            [[0.0, self.entail_score, 0.0] for _ in range(n)],
            dtype=np.float32,
        )


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, source_file="x.md", chunk_index=0, chunking_strategy="t")


def test_faithfulness_empty_answer_returns_perfect():
    scorer = FaithfulnessScorer(model=_StubCrossEncoder(0.5))
    m = scorer.score("", [_chunk("anything")])
    assert m.total_sentences == 0
    assert m.faithfulness == 1.0


def test_faithfulness_empty_context_returns_zero():
    scorer = FaithfulnessScorer(model=_StubCrossEncoder(0.5))
    m = scorer.score("This is a sentence.", [])
    assert m.faithfulness == 0.0
    assert m.total_sentences == 1


def test_faithfulness_high_entailment_passes_threshold():
    # entail_score=5.0 -> softmax over (0, 5, 0) -> p_entail very high
    scorer = FaithfulnessScorer(model=_StubCrossEncoder(5.0), threshold=0.5)
    m = scorer.score("One sentence. Two sentences.", [_chunk("ctx")])
    assert m.total_sentences == 2
    assert m.supported_sentences == 2
    assert m.faithfulness > 0.9


def test_faithfulness_low_entailment_fails_threshold():
    # entail_score=-5.0 -> p_entail very low
    scorer = FaithfulnessScorer(model=_StubCrossEncoder(-5.0), threshold=0.5)
    m = scorer.score("One sentence.", [_chunk("ctx")])
    assert m.supported_sentences == 0
    assert m.faithfulness < 0.1
