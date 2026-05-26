"""Disk-backed caches for expensive RAPTOR steps.

Two granularities:
- Per-summary cache: keyed by sha256(cluster_text + model + prompt_version).
  Survives parameter changes that don't alter cluster membership.
- Full-tree cache: keyed by sha256 of every parameter that influences the
  tree contents. Stored as a pickled RaptorIndex.
"""

import hashlib
import os
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tree_index import RaptorIndex


CACHE_ROOT = "data/raptor_cache"
SUMMARY_DIR = os.path.join(CACHE_ROOT, "summaries")
TREE_DIR = os.path.join(CACHE_ROOT, "trees")


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_corpus(corpus_dir: str) -> str:
    """Hash a corpus directory as sha256 over sorted (rel_path, sha256(content))."""
    parts = []
    for root, _dirs, files in os.walk(corpus_dir):
        for filename in sorted(files):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(root, filename)
            rel = os.path.relpath(filepath, corpus_dir).replace("\\", "/")
            with open(filepath, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()
            parts.append(f"{rel}:{content_hash}")
    parts.sort()
    return _sha256("|".join(parts))


def summary_key(cluster_texts: list[str], model: str, prompt_version: str) -> str:
    joined = "\n---\n".join(cluster_texts)
    return _sha256(f"{model}|{prompt_version}|{joined}")


class SummaryCache:
    """Per-summary cache (one file per summary)."""

    def __init__(self, root: str = SUMMARY_DIR):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.txt")

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def put(self, key: str, summary: str) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)


def tree_key(
    corpus_hash: str,
    leaf_window: int,
    leaf_overlap: int,
    max_levels: int,
    umap_seed: int,
    gmm_seed: int,
    summarizer_model: str,
    embedding_model: str,
    prompt_version: str,
    soft_assign_threshold: float,
) -> str:
    parts = [
        f"corpus={corpus_hash}",
        f"leaf_window={leaf_window}",
        f"leaf_overlap={leaf_overlap}",
        f"max_levels={max_levels}",
        f"umap_seed={umap_seed}",
        f"gmm_seed={gmm_seed}",
        f"summarizer_model={summarizer_model}",
        f"embedding_model={embedding_model}",
        f"prompt_version={prompt_version}",
        f"soft_assign_threshold={soft_assign_threshold:.4f}",
    ]
    return _sha256("|".join(parts))


class TreeCache:
    """Full-tree cache (one pickle per tree_hash)."""

    def __init__(self, root: str = TREE_DIR):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.pkl")

    def get(self, key: str) -> "RaptorIndex | None":
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def put(self, key: str, index: "RaptorIndex") -> None:
        path = self._path(key)
        with open(path, "wb") as f:
            pickle.dump(index, f)
