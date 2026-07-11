"""RaptorRetriever: three retrieval strategies over a built RAPTOR tree.

- raptor_tree: the RAPTOR paper's top-down traversal mode (k children per
  level).
- raptor_collapsed: the paper's collapsed-tree mode — flatten all nodes and
  do a single vector top-k search. Table 2 in Sarthi et al. (2024) reports
  this as the paper's best-performing mode.
- raptor_qcond: this repo's original contribution — per-node descent
  decisions (terminate / single-branch / multi-branch) based on the
  query-node vs. best-child score gap and the entropy of query-children
  scores, rather than a fixed top-k-per-level traversal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from ..chunkers import Chunk, get_embedding_model
from ..config import SearchStrategy
from ..retrievers import RetrievalResult
from .node import RaptorNode
from .tree_index import RaptorIndex


def _dedup_preserve_order(ids: list[str]) -> list[str]:
    """Drop duplicates while keeping first-occurrence order."""
    return list(dict.fromkeys(ids))


@dataclass
class QCondConfig:
    """Hyperparameters for query-conditional traversal.

    `tau_level_scale`, `leaf_bias`, and `expand_terminal_leaves` are the
    three candidate fixes for qcond's early-termination-at-abstract-nodes
    problem (early v1 evaluation showed low context recall vs. collapsed
    search). All default to their v1-reproducing value (0.0 / 0.0 / False)
    so they're opt-in.
    """
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
    # Widens the terminate gap required at deeper (more abstract) levels:
    # effective threshold = tau_term + tau_level_scale * node.level. Leaves
    # are level 0, so this only affects internal nodes. 0.0 reproduces the
    # original level-independent tau_term.
    tau_level_scale: float = 0.0
    # Discounts node_score by leaf_bias * node.level in the terminate
    # DECISION only — the score stored for ranking is always the raw,
    # unbiased node_score. Makes it harder for an abstract node to
    # out-score its children just by sitting close to the query in
    # embedding space. 0.0 leaves the decision unbiased.
    leaf_bias: float = 0.0
    # When descent terminates at a non-leaf node, expand it to its
    # descendant leaves (each freshly scored against the query) instead of
    # returning the summary node itself. False reproduces v1 exactly.
    expand_terminal_leaves: bool = False


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
        self.embedder = embedding_model or get_embedding_model()
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
        """Flatten every node (leaf and summary) into one pool and do a
        single vector top-k search. The paper's "collapsed tree" retrieval
        mode — Table 2 in Sarthi et al. (2024) reports this as the
        best-performing mode against tree traversal."""
        return self.index.collapsed_search(q_emb, top_k=top_k)

    def _retrieve_tree(
        self, q_emb: list[float], top_k: int
    ) -> list[tuple[RaptorNode, float]]:
        """Top-down traversal: keep top `tree_k_per_level` at each level,
        descend into their children, union all selected nodes across levels,
        then return top_k by score. The paper's tree-traversal retrieval
        mode (Sarthi et al., 2024)."""
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
            frontier_ids = _dedup_preserve_order(next_frontier)

        ranked = sorted(selected.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.index.nodes_by_id[nid], s) for nid, s in ranked]

    # ---------- contribution: query-conditional ----------

    def _retrieve_qcond(
        self, q_emb: list[float], top_k: int
    ) -> list[tuple[RaptorNode, float]]:
        """Per-node: terminate if node beats children; single-branch if one
        child dominates; multi-branch otherwise. Returns the nodes where the
        descent terminated, ranked by score. This repo's original
        contribution (not from the paper): descent policy is conditioned
        per-node on the node-vs-best-child score gap (terminate) and on the
        entropy of query-children scores (single- vs multi-branch), rather
        than the paper's fixed top-k-per-level traversal.

        `QCondConfig.tau_level_scale`/`leaf_bias` make the terminate
        decision "adaptive scoping" — harder to stop at an abstract node
        the deeper it sits — and `expand_terminal_leaves` adds
        "leaf-precise retrieval within scope": once a scope (subtree) is
        chosen by termination, rank its individual leaves against the
        query instead of returning the summary node as one opaque unit.
        All three default to their v1-reproducing values.
        """
        q = self._normalize(q_emb)
        cfg = self.qcond_config
        pool: dict[str, float] = {}
        # Leaves already expanded into `pool`, shared across every
        # termination this call so a leaf reachable from two terminated
        # nodes (soft clustering gives leaves multiple parents) is only
        # scored and pooled once.
        visited_leaves: set[str] = set()
        frontier_ids = list(self.index.root_ids)
        descents = 0

        def terminate(node: RaptorNode, score: float) -> None:
            """Record a termination at `node`, scored at `score`.

            With `expand_terminal_leaves` off, or `node` already a leaf,
            pools `node` itself. Otherwise pools `node`'s descendant leaves
            instead, each freshly scored against the query.
            """
            if not cfg.expand_terminal_leaves or node.is_leaf:
                pool[node.node_id] = max(pool.get(node.node_id, -np.inf), score)
                return
            leaf_ids = self._collect_descendant_leaves(node, visited_leaves)
            if not leaf_ids:
                # Nothing resolvable below this node — fall back to it so
                # the scope isn't silently dropped from the results.
                pool[node.node_id] = max(pool.get(node.node_id, -np.inf), score)
                return
            for leaf, leaf_score in self._score_nodes(q, leaf_ids):
                pool[leaf.node_id] = max(pool.get(leaf.node_id, -np.inf), leaf_score)

        while frontier_ids and descents < cfg.max_descents:
            next_frontier: list[str] = []
            frontier_scored = self._score_nodes(q, frontier_ids)
            advanced = False
            for node, node_score in frontier_scored:
                resolved_children = [
                    cid for cid in node.children if cid in self.index.nodes_by_id
                ]
                if not resolved_children:
                    # No children, or every child id is unresolved (e.g. a
                    # dangling reference from a partial/stale build) — treat
                    # this node as terminal instead of scoring an empty set.
                    terminate(node, node_score)
                    continue
                child_scored = self._score_nodes(q, resolved_children)
                child_scores = np.array(
                    [s for _, s in child_scored], dtype=np.float32
                )
                max_child = float(child_scores.max())

                # Terminate at this node. tau_level_scale widens the
                # required gap at deeper (more abstract) levels; leaf_bias
                # additionally discounts node_score by level for this
                # comparison only — the score passed to terminate() below
                # is always the raw, unbiased node_score.
                tau_eff = cfg.tau_term + cfg.tau_level_scale * node.level
                biased_score = node_score - cfg.leaf_bias * node.level
                if biased_score - max_child > tau_eff:
                    terminate(node, node_score)
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
            frontier_ids = _dedup_preserve_order(next_frontier)
            if not advanced:
                break

        # Any unterminated frontier nodes terminate now
        if frontier_ids:
            for node, score in self._score_nodes(q, frontier_ids):
                terminate(node, score)

        ranked = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.index.nodes_by_id[nid], s) for nid, s in ranked]

    def _collect_descendant_leaves(
        self, node: RaptorNode, visited: set[str]
    ) -> list[str]:
        """Iterative DFS to `node`'s descendant leaves (level 0), skipping
        unresolved child ids and anything already in `visited` (marked as
        we go, so overlapping subtrees — from soft clustering — don't get
        walked or scored twice)."""
        if node.node_id in visited:
            return []
        leaves: list[str] = []
        stack = [node.node_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            n = self.index.nodes_by_id.get(nid)
            if n is None:
                continue
            if n.is_leaf:
                leaves.append(nid)
                continue
            for cid in n.children:
                if cid in self.index.nodes_by_id and cid not in visited:
                    stack.append(cid)
        return leaves

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
