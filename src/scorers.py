"""Scoring methods for retrieval and answer quality evaluation."""

import re
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunkers import Chunk
from .config import EMBEDDING_MODEL


@dataclass
class RetrievalMetrics:
    context_precision: float
    context_recall: float
    distractor_rate: float


@dataclass
class KeyFactMetrics:
    fact_recall: float
    found_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)


def _chunk_sources(chunk: Chunk) -> set[str]:
    """Return the set of source files a chunk covers.

    RAPTOR summary nodes set `chunk.source_files` to every constituent leaf
    source (see `RaptorNode.to_chunk`). Baseline chunkers leave
    `source_files` unset (None), so their chunk covers exactly one opaque
    source, `chunk.source_file` — it is never split or parsed, even if the
    filename itself happens to contain `;`.
    """
    if chunk.source_files:
        return set(chunk.source_files)
    return {chunk.source_file}


class RetrievalScorer:
    """Scores retrieval quality by comparing retrieved chunks to gold source files.

    Context precision and distractor rate use FRACTIONAL per-chunk credit:
    a chunk covering N sources, M of which are in the target set, scores
    M/N rather than 1.0. This matters for RAPTOR summary nodes, which can
    cover many source files (e.g. a root summary spanning 10 documents) —
    without fractional credit, a single such chunk would count as "fully
    relevant" merely because one of its many sources happens to be gold,
    inflating RAPTOR's precision relative to baseline chunkers (which always
    have exactly one source per chunk, so their fraction is naturally 0 or
    1). Context recall is unaffected: it is coverage over the union of
    retrieved sources, not a per-chunk average.
    """

    def score(
        self,
        retrieved_chunks: list[Chunk],
        gold_source_ids: list[str],
        distractor_ids: list[str] | None = None,
    ) -> RetrievalMetrics:
        if not retrieved_chunks:
            return RetrievalMetrics(
                context_precision=0.0,
                context_recall=0.0,
                distractor_rate=0.0,
            )

        total = len(retrieved_chunks)
        gold_set = set(gold_source_ids)

        # Context precision: mean per-chunk fractional credit |sources ∩ gold| / |sources|.
        precision_scores = [
            len(_chunk_sources(c) & gold_set) / len(_chunk_sources(c))
            for c in retrieved_chunks
        ]
        context_precision = sum(precision_scores) / total

        # Context recall: fraction of gold sources covered by some retrieved chunk.
        retrieved_sources: set[str] = set()
        for c in retrieved_chunks:
            retrieved_sources.update(_chunk_sources(c))
        covered = sum(1 for g in gold_source_ids if g in retrieved_sources)
        context_recall = covered / len(gold_source_ids) if gold_source_ids else 0.0

        # Distractor rate: same fractional treatment as precision, against distractors.
        distractor_rate = 0.0
        if distractor_ids:
            distractor_set = set(distractor_ids)
            distractor_scores = [
                len(_chunk_sources(c) & distractor_set) / len(_chunk_sources(c))
                for c in retrieved_chunks
            ]
            distractor_rate = sum(distractor_scores) / total

        return RetrievalMetrics(
            context_precision=context_precision,
            context_recall=context_recall,
            distractor_rate=distractor_rate,
        )


class GoldSimilarityScorer:
    """Scores answer quality by embedding similarity to gold answer."""

    def __init__(self, embedding_model=None):
        if embedding_model is not None:
            self.model = embedding_model
        else:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def score(self, generated_answer: str, gold_answer: str) -> float:
        embeddings = self.model.encode([generated_answer, gold_answer])
        a, b = embeddings[0], embeddings[1]
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        return max(0.0, min(1.0, cos_sim))


class KeyFactScorer:
    """Checks if specific key facts appear in the generated answer."""

    def score(self, answer: str, key_facts: list[str]) -> KeyFactMetrics:
        if not key_facts:
            return KeyFactMetrics(fact_recall=1.0, found_facts=[], missing_facts=[])

        answer_lower = answer.lower()
        found = []
        missing = []

        for fact in key_facts:
            if self._fact_matches(answer_lower, fact):
                found.append(fact)
            else:
                missing.append(fact)

        fact_recall = len(found) / len(key_facts)
        return KeyFactMetrics(
            fact_recall=fact_recall,
            found_facts=found,
            missing_facts=missing,
        )

    def _fact_matches(self, answer_lower: str, fact: str) -> bool:
        fact_lower = fact.lower()

        # Direct case-insensitive match
        if fact_lower in answer_lower:
            return True

        # Numeric matching: handle variations like "42.3M", "$42.3", "42.3 million"
        numeric_match = re.match(r'^(-?[\d,]+\.?\d*)(%?)$', fact.strip())
        if numeric_match:
            number = numeric_match.group(1)
            suffix = numeric_match.group(2)
            # Try various formats
            patterns = [
                re.escape(number),  # exact number
                r'\$' + re.escape(number),  # with dollar sign
                re.escape(number) + r'\s*[mMbBkK]',  # with magnitude suffix
                re.escape(number) + r'\s*million',
                re.escape(number) + r'\s*billion',
            ]
            if suffix == '%':
                patterns.extend([
                    re.escape(number) + r'\s*%',
                    re.escape(number) + r'\s*percent',
                ])

            for pattern in patterns:
                if re.search(pattern, answer_lower):
                    return True

        return False
