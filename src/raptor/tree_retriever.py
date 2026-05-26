"""RaptorRetriever: three retrieval strategies over a built RAPTOR tree.

- raptor_tree: paper's top-down traversal (k children per level).
- raptor_collapsed: flatten all nodes and do a single vector top-k search.
- raptor_qcond: this repo's contribution — per-node descent decisions based
  on query-node score and the entropy of query-children scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from ..chunkers import Chunk
from ..config import EMBEDDING_MODEL, SearchStrategy
from ..retrievers import RetrievalResult
from .node import RaptorNode
from .tree_index import RaptorIndex


@dataclass
class QCondConfig:
    """Hyperparameters for query-conditional traversal."""
    # Terminate at node when node_score - max(child_score) > tau_term.
    tau_term: float = 0.05
    # Single-branch descend when entropy(softmax(child_scores)) < tau_focus.
    tau_focus: float = 0.6
    # When multi-branching, retain top-k children.
    k_branch: int = 2
    # Maximum number of descents per query.
    max_descents: int = 5
    # Temperature for softmax over child scores.
    softmax_temp: float = 0.1


class RaptorRetriever:
    """Retrieve from a RaptorIndex using one of three strategies."""

    def __init__(
        self,
        index: RaptorIndex,
        embedding_model: SentenceTransformer | None = None,
        qcond_config: QCondConfig | None = None,
        tree_k_per_level: int = 5,
    ):
        self.index = index
        self.embedder = embedding_model or SentenceTransformer(EMBEDDING_MODEL)
        self.qcond_config = qcond_config or QCondConfig()
        self.tree_k_per_level = tree_k_per_level

    def retrieve(
        self, query: str, strategy: SearchStrategy, top_k: int
    ) -> RetrievalResult:
        start = time.perf_counter()
        q_emb = self.embedder.encode(query).tolist()

        if strategy == SearchStrategy.RAPTOR_TREE:
            results = self._retrieve_tree(q_emb, top_k)
        elif strategy == SearchStrategy.RAPTOR_COLLAPSED:
            results = self._retrieve_collapsed(q_emb, top_k)
        elif strategy == SearchStrategy.RAPTOR_QCOND:
            results = self._retrieve_qcond(q_emb, top_k)
        else:
            raise ValueError(
                f"RaptorRetriever does not support strategy {strategy}"
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        chunks = [n.to_chunk() for n, _ in results]
        scores = [s for _, s in results]
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            retrieval_latency_ms=elapsed_ms,
            strategy=strategy.value,
        )

    # ---------- paper modes ----------

    def _retrieve_collapsed(
        self, q_emb: list[float], top_k: int
    ) -> list[tuple[RaptorNode, float]]:
        return self.index.collapsed_search(q_emb, top_k=top_k)

    def _retrieve_tree(
        self, q_emb: list[float], top_k: int
    ) -> list[tuple[RaptorNode, float]]:
        """Top-down traversal: keep top `tree_k_per_level` at each level,
        descend into their children, union all selected nodes across levels,
        then return top_k by score."""
        q = self._normalize(q_emb)
        # Start at roots
        frontier_ids = list(self.index.root_ids)
        selected: dict[str, float] = {}

        for _ in range(self.index.max_level() + 1):
            if not frontier_ids:
                break
            scored = self._score_nodes(q, frontier_ids)
            # Pick top tree_k_per_level from frontier
            scored.sort(key=lambda x: x[1], reverse=True)
            picked = scored[: self.tree_k_per_level]
            for node, score in picked:
                if node.node_id not in selected or score > selected[node.node_id]:
                    selected[node.node_id] = score
            # Descend into children of picked nodes
            next_frontier: list[str] = []
            for node, _ in picked:
                for cid in node.children:
                    if cid in self.index.nodes_by_id:
                        next_frontier.append(cid)
            # Dedup, preserve order
            seen: set[str] = set()
            frontier_ids = [c for c in next_frontier if not (c in seen or seen.add(c))]

        ranked = sorted(selected.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.index.nodes_by_id[nid], s) for nid, s in ranked]

    # ---------- contribution: query-conditional ----------

    def _retrieve_qcond(
        self, q_emb: list[float], top_k: int
    ) -> list[tuple[RaptorNode, float]]:
        """Per-node: terminate if node beats children; single-branch if one
        child dominates; multi-branch otherwise. Returns the nodes where the
        descent terminated, ranked by score."""
        q = self._normalize(q_emb)
        cfg = self.qcond_config
        terminal: dict[str, float] = {}
        frontier_ids = list(self.index.root_ids)
        descents = 0

        while frontier_ids and descents < cfg.max_descents:
            next_frontier: list[str] = []
            frontier_scored = self._score_nodes(q, frontier_ids)
            advanced = False
            for node, node_score in frontier_scored:
                if not node.children:
                    terminal[node.node_id] = max(
                        terminal.get(node.node_id, -np.inf), node_score
                    )
                    continue
                child_scored = self._score_nodes(q, node.children)
                child_scores = np.array(
                    [s for _, s in child_scored], dtype=np.float32
                )
                max_child = float(child_scores.max())

                # Terminate at this node
                if node_score - max_child > cfg.tau_term:
                    terminal[node.node_id] = max(
                        terminal.get(node.node_id, -np.inf), node_score
                    )
                    continue

                # Entropy over children
                logits = child_scores / max(cfg.softmax_temp, 1e-6)
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs /= probs.sum() + 1e-12
                ent = float(-np.sum(probs * np.log(probs + 1e-12)))
                max_ent = float(np.log(len(child_scores))) if len(child_scores) > 1 else 1.0
                norm_ent = ent / max(max_ent, 1e-6)

                if norm_ent < cfg.tau_focus:
                    # Single-branch descend into best child
                    best_idx = int(child_scores.argmax())
                    next_frontier.append(child_scored[best_idx][0].node_id)
                    advanced = True
                else:
                    # Multi-branch descend into top k_branch children
                    order = np.argsort(-child_scores)[: cfg.k_branch]
                    for idx in order:
                        next_frontier.append(child_scored[int(idx)][0].node_id)
                    advanced = True

            descents += 1
            # Dedup frontier
            seen: set[str] = set()
            frontier_ids = [c for c in next_frontier if not (c in seen or seen.add(c))]
            if not advanced:
                break

        # Any unterminated frontier nodes terminate now
        if frontier_ids:
            for node, score in self._score_nodes(q, frontier_ids):
                terminal[node.node_id] = max(
                    terminal.get(node.node_id, -np.inf), score
                )

        ranked = sorted(terminal.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.index.nodes_by_id[nid], s) for nid, s in ranked]

    # ---------- helpers ----------

    def _score_nodes(
        self, q_normalized: np.ndarray, node_ids: list[str]
    ) -> list[tuple[RaptorNode, float]]:
        nodes = [self.index.nodes_by_id[nid] for nid in node_ids if nid in self.index.nodes_by_id]
        if not nodes:
            return []
        embs = np.array([n.embedding for n in nodes], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        embs = embs / norms
        scores = embs @ q_normalized
        return list(zip(nodes, scores.tolist()))

    @staticmethod
    def _normalize(emb: list[float]) -> np.ndarray:
        v = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-10:
            return v
        return v / n
