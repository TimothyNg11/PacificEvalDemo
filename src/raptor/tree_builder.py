"""RaptorTreeBuilder: chunk a corpus into 100-token leaves, then recursively
cluster + summarize to build a tree of abstraction levels.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL, LLMConfig
from .cache import SummaryCache, TreeCache, hash_corpus, tree_key
from .clustering import cluster_embeddings
from .node import RaptorNode
from .prompts import PROMPT_VERSION
from .seed import set_global_seed
from .summarizer import Summarizer
from .tree_index import RaptorIndex


_tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class RaptorBuildConfig:
    leaf_window: int = 100
    leaf_overlap: int = 0
    max_levels: int = 4
    umap_seed: int = 42
    gmm_seed: int = 42
    soft_assign_threshold: float = 0.0  # placeholder; threshold is 1/k in clustering
    embedding_model: str = EMBEDDING_MODEL


class RaptorTreeBuilder:
    """Builds a RAPTOR tree from a corpus directory.

    Caching is two-tier: a full-tree pickle (one-shot cache hit) and a
    per-summary cache (granular reuse across param changes). On a cache miss
    for the tree, summaries are still served from the per-summary cache
    where possible.
    """

    def __init__(
        self,
        build_config: RaptorBuildConfig | None = None,
        llm_config: LLMConfig | None = None,
        tree_cache: TreeCache | None = None,
        summary_cache: SummaryCache | None = None,
    ):
        self.cfg = build_config or RaptorBuildConfig()
        self.llm_config = llm_config
        self.tree_cache = tree_cache if tree_cache is not None else TreeCache()
        self.summary_cache = summary_cache if summary_cache is not None else SummaryCache()
        self.embedder = SentenceTransformer(self.cfg.embedding_model)
        self.summarizer = Summarizer(llm_config=llm_config, cache=self.summary_cache)

    def build(self, corpus_dir: str) -> RaptorIndex:
        set_global_seed()

        corpus_hash = hash_corpus(corpus_dir)
        key = tree_key(
            corpus_hash=corpus_hash,
            leaf_window=self.cfg.leaf_window,
            leaf_overlap=self.cfg.leaf_overlap,
            max_levels=self.cfg.max_levels,
            umap_seed=self.cfg.umap_seed,
            gmm_seed=self.cfg.gmm_seed,
            summarizer_model=self.summarizer.llm_config.model,
            embedding_model=self.cfg.embedding_model,
            prompt_version=PROMPT_VERSION,
            soft_assign_threshold=self.cfg.soft_assign_threshold,
        )

        cached = self.tree_cache.get(key)
        if cached is not None:
            print(f"  [raptor] tree cache hit ({key[:12]}...)")
            return cached

        print(f"  [raptor] building tree ({key[:12]}...)")
        leaves = self._build_leaves(corpus_dir)
        print(f"  [raptor] {len(leaves)} leaves")

        nodes_by_id: dict[str, RaptorNode] = {n.node_id: n for n in leaves}
        current_level_nodes = leaves

        for level in range(1, self.cfg.max_levels + 1):
            if len(current_level_nodes) <= 1:
                break

            embeddings = np.array(
                [n.embedding for n in current_level_nodes], dtype=np.float32
            )
            clusters = cluster_embeddings(
                embeddings,
                umap_seed=self.cfg.umap_seed,
                gmm_seed=self.cfg.gmm_seed,
            )
            if len(clusters) <= 1 and len(clusters[0]) == len(current_level_nodes):
                # No further partitioning possible — single super-cluster
                break

            print(
                f"  [raptor] level {level}: clustering {len(current_level_nodes)} "
                f"nodes into {len(clusters)} clusters"
            )

            parents: list[RaptorNode] = []
            for cluster_idx, member_indices in enumerate(clusters):
                child_nodes = [current_level_nodes[i] for i in member_indices]
                summary_text = self.summarizer.summarize([c.text for c in child_nodes])
                if not summary_text:
                    continue
                summary_emb = self.embedder.encode(summary_text).tolist()
                source_files: list[str] = []
                seen: set[str] = set()
                for c in child_nodes:
                    for sf in c.source_files:
                        if sf not in seen:
                            seen.add(sf)
                            source_files.append(sf)
                parent = RaptorNode(
                    node_id=str(uuid.uuid4()),
                    text=summary_text,
                    embedding=summary_emb,
                    level=level,
                    token_count=len(_tokenizer.encode(summary_text)),
                    source_files=source_files,
                    children=[c.node_id for c in child_nodes],
                    cluster_id=cluster_idx,
                )
                parents.append(parent)
                nodes_by_id[parent.node_id] = parent

            if not parents:
                break
            current_level_nodes = parents

        root_ids = [n.node_id for n in current_level_nodes]
        index = RaptorIndex(
            nodes_by_id=nodes_by_id,
            root_ids=root_ids,
            tree_hash=key,
        )
        self.tree_cache.put(key, index)
        print(
            f"  [raptor] tree built: {len(index)} nodes, "
            f"depth={index.max_level()}, {len(root_ids)} roots; "
            f"summary calls={self.summarizer.call_count}, "
            f"summary cache hits={self.summarizer.cache_hits}"
        )
        return index

    def _build_leaves(self, corpus_dir: str) -> list[RaptorNode]:
        """Chunk corpus into ~`leaf_window`-token windows; embed."""
        leaves: list[RaptorNode] = []
        leaf_idx = 0
        for root, _dirs, files in os.walk(corpus_dir):
            for filename in sorted(files):
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, corpus_dir).replace("\\", "/")
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                tokens = _tokenizer.encode(text)
                start = 0
                while start < len(tokens):
                    end = min(start + self.cfg.leaf_window, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunk_text = _tokenizer.decode(chunk_tokens)
                    leaves.append(
                        RaptorNode(
                            node_id=f"leaf_{leaf_idx}",
                            text=chunk_text,
                            embedding=[],  # filled below
                            level=0,
                            token_count=len(chunk_tokens),
                            source_files=[rel_path],
                            children=[],
                            leaf_indices=[leaf_idx],
                            cluster_id=None,
                        )
                    )
                    leaf_idx += 1
                    if end == len(tokens):
                        break
                    if self.cfg.leaf_overlap > 0:
                        start = end - self.cfg.leaf_overlap
                    else:
                        start = end

        if leaves:
            texts = [n.text for n in leaves]
            embeddings = self.embedder.encode(texts, show_progress_bar=len(texts) > 200)
            for n, emb in zip(leaves, embeddings):
                n.embedding = emb.tolist()
        return leaves
