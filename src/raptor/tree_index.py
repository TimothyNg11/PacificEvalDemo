"""RaptorIndex: storage for a built RAPTOR tree.

Holds every node by id, root ids, and pre-computed numpy embedding matrix
for fast vector search across all nodes (used by the `raptor_collapsed`
retrieval strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .node import RaptorNode


@dataclass
class RaptorIndex:
    nodes_by_id: dict[str, RaptorNode]
    root_ids: list[str]
    tree_hash: str
    # Cached matrices for fast queries (populated post-construction)
    _all_ids: list[str] = field(default_factory=list)
    _all_embeddings: np.ndarray | None = None

    def __post_init__(self):
        if self._all_embeddings is None:
            self._build_collapsed_matrix()

    def _build_collapsed_matrix(self) -> None:
        ids = list(self.nodes_by_id.keys())
        if not ids:
            self._all_ids = []
            self._all_embeddings = np.zeros((0, 0), dtype=np.float32)
            return
        embs = np.array(
            [self.nodes_by_id[i].embedding for i in ids], dtype=np.float32
        )
        # L2-normalize so cosine == dot product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self._all_ids = ids
        self._all_embeddings = embs / norms

    def get(self, node_id: str) -> RaptorNode:
        return self.nodes_by_id[node_id]

    def children_of(self, node_id: str) -> list[RaptorNode]:
        node = self.nodes_by_id[node_id]
        return [self.nodes_by_id[cid] for cid in node.children if cid in self.nodes_by_id]

    def leaves(self) -> list[RaptorNode]:
        return [n for n in self.nodes_by_id.values() if n.is_leaf]

    def max_level(self) -> int:
        return max((n.level for n in self.nodes_by_id.values()), default=0)

    def collapsed_search(
        self, query_embedding: list[float] | np.ndarray, top_k: int
    ) -> list[tuple[RaptorNode, float]]:
        """Top-k cosine search across every node in the tree."""
        if self._all_embeddings is None or len(self._all_ids) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q = q / q_norm
        scores = self._all_embeddings @ q  # (n,)
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        # Sort the top-k by descending score
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            (self.nodes_by_id[self._all_ids[i]], float(scores[i]))
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self.nodes_by_id)
