"""Local NLI-based faithfulness scoring.

For each (answer, retrieved context) pair, runs a cross-encoder NLI model to
estimate whether the answer is entailed by the context. Output is a 0-1
faithfulness score — higher means the answer is better supported by the
context (less hallucinated).

Fully local: uses `cross-encoder/nli-deberta-v3-base` via sentence-transformers.
No paid APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers required for FaithfulnessScorer"
    ) from e

from .chunkers import Chunk, split_sentences


FAITHFULNESS_MODEL = "cross-encoder/nli-deberta-v3-base"
# Cross-encoder NLI heads typically output [contradiction, entailment, neutral]
# but the index ordering depends on the model. For nli-deberta-v3-base the
# label order is ["contradiction", "entailment", "neutral"], so entailment is
# column 1. The per-sentence score used below is that entailment probability
# directly (after softmax), not a contradiction-adjusted margin.
ENTAILMENT_IDX = 1


@dataclass
class FaithfulnessMetrics:
    faithfulness: float            # mean per-sentence entailment score
    supported_sentences: int       # count of sentences with score >= threshold
    total_sentences: int
    threshold: float = 0.5


_reranker_instance = None


def _get_nli_model() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(FAITHFULNESS_MODEL)
    return _reranker_instance


class FaithfulnessScorer:
    """Per-answer faithfulness via cross-encoder NLI."""

    def __init__(self, model: CrossEncoder | None = None, threshold: float = 0.5):
        self.model = model or _get_nli_model()
        self.threshold = threshold

    def score(
        self, generated_answer: str, retrieved_chunks: list[Chunk]
    ) -> FaithfulnessMetrics:
        sentences = split_sentences(generated_answer)
        if not sentences:
            return FaithfulnessMetrics(
                faithfulness=1.0, supported_sentences=0, total_sentences=0,
                threshold=self.threshold,
            )
        context = "\n".join(c.text for c in retrieved_chunks)
        if not context.strip():
            return FaithfulnessMetrics(
                faithfulness=0.0, supported_sentences=0,
                total_sentences=len(sentences), threshold=self.threshold,
            )

        pairs = [(context, s) for s in sentences]
        logits = self.model.predict(pairs)  # shape (n_sentences, 3)
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 1:
            # Some models return a single logit — treat directly as entailment score.
            scores = 1.0 / (1.0 + np.exp(-logits))
        else:
            # Softmax over each row, take entailment probability
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=1, keepdims=True)
            scores = probs[:, ENTAILMENT_IDX]
        supported = int((scores >= self.threshold).sum())
        return FaithfulnessMetrics(
            faithfulness=float(scores.mean()),
            supported_sentences=supported,
            total_sentences=len(sentences),
            threshold=self.threshold,
        )
