"""Retrieval strategies: vector, BM25, hybrid, hybrid+rerank."""

import time
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import CrossEncoder

from .chunkers import Chunk
from .config import SearchStrategy, RERANKER_MODEL
from .indexer import CorpusIndex


@dataclass
class RetrievalResult:
    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    strategy: str = ""


# Module-level reranker singleton
_reranker_instance = None


def get_reranker():
    """Load the reranker model once and reuse."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


class Retriever:
    """Retrieves chunks from a CorpusIndex using various strategies."""

    def __init__(self, index: CorpusIndex, reranker=None):
        self.index = index
        self.reranker = reranker or get_reranker()

    def retrieve(self, query: str, strategy: SearchStrategy, top_k: int) -> RetrievalResult:
        start = time.perf_counter()

        if strategy == SearchStrategy.VECTOR:
            results = self._retrieve_vector(query, top_k)
        elif strategy == SearchStrategy.BM25:
            results = self._retrieve_bm25(query, top_k)
        elif strategy == SearchStrategy.HYBRID:
            results = self._retrieve_hybrid(query, top_k)
        elif strategy == SearchStrategy.HYBRID_RERANK:
            results = self._retrieve_hybrid_rerank(query, top_k)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            retrieval_latency_ms=elapsed_ms,
            strategy=strategy.value,
        )

    def _retrieve_vector(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        query_embedding = self.index.embedding_model.encode(query).tolist()
        results = self.index.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Find the chunk by matching id
                chunk = self._find_chunk_by_id(doc_id)
                if chunk:
                    # ChromaDB returns distances for cosine; convert to similarity
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1.0 - distance
                    output.append((chunk, similarity))
        return output

    def _retrieve_bm25(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        tokenized_query = query.lower().split()
        scores = self.index.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        output = []
        for idx in top_indices:
            if scores[idx] > 0:
                output.append((self.index.chunks[idx], float(scores[idx])))
        return output

    def _retrieve_hybrid(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        # Get candidates from both strategies
        vector_results = self._retrieve_vector(query, top_k * 2)
        bm25_results = self._retrieve_bm25(query, top_k * 2)

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        k = 60  # Standard RRF constant

        for rank, (chunk, _score) in enumerate(vector_results):
            chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        for rank, (chunk, _score) in enumerate(bm25_results):
            chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        # Sort by RRF score and return top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]

    def _retrieve_hybrid_rerank(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        # Get more candidates from hybrid
        vector_results = self._retrieve_vector(query, top_k * 3)
        bm25_results = self._retrieve_bm25(query, top_k * 3)

        # RRF to get candidates
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        k = 60

        for rank, (chunk, _score) in enumerate(vector_results):
            chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        for rank, (chunk, _score) in enumerate(bm25_results):
            chunk_id = f"{chunk.source_file}_{chunk.chunk_index}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = chunk

        # Get top candidates for reranking
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k * 3]
        candidates = [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]

        # Rerank using cross-encoder
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranker score and return top_k
        scored = list(zip(candidates, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score)) for (chunk, _), score in scored[:top_k]]

    def _find_chunk_by_id(self, doc_id: str) -> Chunk | None:
        for chunk in self.index.chunks:
            if f"{chunk.source_file}_{chunk.chunk_index}" == doc_id:
                return chunk
        return None
