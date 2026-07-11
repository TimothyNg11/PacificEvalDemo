"""Configuration dataclasses and constants for the RAG benchmark."""

import os
from dataclasses import dataclass
from enum import Enum
from itertools import product


class ChunkingStrategy(Enum):
    FIXED_256 = "fixed_256"
    FIXED_512 = "fixed_512"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    RAPTOR_100 = "raptor_100"


class SearchStrategy(Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    RAPTOR_TREE = "raptor_tree"
    RAPTOR_COLLAPSED = "raptor_collapsed"
    RAPTOR_QCOND = "raptor_qcond"


# RAPTOR-only members
_RAPTOR_CHUNKINGS = {ChunkingStrategy.RAPTOR_100}
_RAPTOR_SEARCHES = {
    SearchStrategy.RAPTOR_TREE,
    SearchStrategy.RAPTOR_COLLAPSED,
    SearchStrategy.RAPTOR_QCOND,
}


@dataclass
class RetrievalConfig:
    chunking: ChunkingStrategy
    search: SearchStrategy
    top_k: int

    @property
    def name(self) -> str:
        return f"{self.chunking.value}__{self.search.value}__k{self.top_k}"

    @property
    def is_raptor(self) -> bool:
        return self.chunking in _RAPTOR_CHUNKINGS or self.search in _RAPTOR_SEARCHES

    @classmethod
    def from_name(cls, name: str) -> "RetrievalConfig":
        """Parse a config name like 'fixed_256__vector__k3' into a RetrievalConfig.

        Raises ValueError (never silently returns a bogus config) if the
        name is malformed, names an unknown chunking/search strategy, or
        pairs a chunking/search combination that `_config_is_valid` rejects
        (RAPTOR chunking requires a raptor_* search mode and vice versa).
        """
        parts = name.split("__")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid config name {name!r}: expected 'chunking__search__kN'"
            )
        chunking_str, search_str, k_str = parts

        try:
            chunking = ChunkingStrategy(chunking_str)
        except ValueError:
            raise ValueError(
                f"Invalid config name {name!r}: unknown chunking strategy "
                f"{chunking_str!r} (valid: {[c.value for c in ChunkingStrategy]})"
            )
        try:
            search = SearchStrategy(search_str)
        except ValueError:
            raise ValueError(
                f"Invalid config name {name!r}: unknown search strategy "
                f"{search_str!r} (valid: {[s.value for s in SearchStrategy]})"
            )
        if not k_str.startswith("k") or not k_str[1:].isdigit():
            raise ValueError(
                f"Invalid config name {name!r}: expected top_k like 'k3', got {k_str!r}"
            )
        top_k = int(k_str[1:])

        if not _config_is_valid(chunking, search):
            raise ValueError(
                f"Invalid config name {name!r}: chunking {chunking_str!r} and "
                f"search {search_str!r} cannot be combined - RAPTOR chunking "
                f"requires a raptor_* search mode and vice versa."
            )
        return cls(chunking=chunking, search=search, top_k=top_k)


def _config_is_valid(chunking: ChunkingStrategy, search: SearchStrategy) -> bool:
    """RAPTOR search strategies only pair with RAPTOR chunking, and vice versa."""
    chunk_is_raptor = chunking in _RAPTOR_CHUNKINGS
    search_is_raptor = search in _RAPTOR_SEARCHES
    return chunk_is_raptor == search_is_raptor


def generate_all_configs(include_raptor: bool = False) -> list[RetrievalConfig]:
    """Generate all valid config combinations.

    - Baseline (48 configs): 4 chunking × 4 search × 3 top_k.
    - With RAPTOR (57 configs): adds 1 chunking × 3 search × 3 top_k = 9 more.
    """
    configs = []
    for chunking, search, top_k in product(
        ChunkingStrategy, SearchStrategy, [3, 5, 10]
    ):
        if not _config_is_valid(chunking, search):
            continue
        if not include_raptor and (
            chunking in _RAPTOR_CHUNKINGS or search in _RAPTOR_SEARCHES
        ):
            continue
        configs.append(RetrievalConfig(chunking=chunking, search=search, top_k=top_k))
    return configs


# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CORPUS_DIR = "data/corpus"
EVAL_SET_PATH = "data/eval_set.yaml"
RESULTS_DIR = "results"
CHROMA_DIR = "data/chroma_indexes"

# LLM provider settings — set via --llm flag or environment variables
# Defaults to Ollama local
LLM_BASE_URL = os.environ.get("CONTEXTBENCH_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("CONTEXTBENCH_LLM_MODEL", "llama3.2:3b")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class LLMConfig:
    base_url: str = LLM_BASE_URL
    model: str = LLM_MODEL
    api_key: str = LLM_API_KEY

    @property
    def provider_name(self) -> str:
        if "openai.com" in self.base_url:
            return "openai"
        return "ollama"


# Preset configs for common providers
LLM_PRESETS = {
    "ollama": LLMConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.2:3b",
        api_key="not-needed",
    ),
    "openai": LLMConfig(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    ),
}
