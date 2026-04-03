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


class SearchStrategy(Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class RetrievalConfig:
    chunking: ChunkingStrategy
    search: SearchStrategy
    top_k: int

    @property
    def name(self) -> str:
        return f"{self.chunking.value}__{self.search.value}__k{self.top_k}"


def generate_all_configs() -> list[RetrievalConfig]:
    """Generate all 48 config combinations: 4 chunking × 4 search × 3 top_k."""
    configs = []
    for chunking, search, top_k in product(
        ChunkingStrategy, SearchStrategy, [3, 5, 10]
    ):
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
