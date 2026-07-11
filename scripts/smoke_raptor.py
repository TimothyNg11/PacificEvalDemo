"""End-to-end smoke test for RAPTOR on the existing 20-doc corpus.

Builds a RAPTOR tree (forcing shallow depth for speed), then runs each of the
three retrieval modes against three sanity questions. Asserts each mode
returns non-empty chunks and a non-trivial top score.

Requires a local Ollama running with the configured model pulled (default:
llama3.2:3b for speed). On a tiny corpus the tree build is ~30 seconds; the
result is cached so re-runs are near-instant.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    CORPUS_DIR,
    EMBEDDING_MODEL,
    LLM_PRESETS,
    SearchStrategy,
)
from src.raptor.tree_builder import RaptorBuildConfig, RaptorTreeBuilder
from src.raptor.tree_retriever import RaptorRetriever


SMOKE_QUERIES = [
    "What is the company's uptime SLA?",
    "How are quarterly earnings trending?",
    "What does the onboarding handbook say about the first week?",
]


def main():
    print("=" * 60)
    print("RAPTOR smoke test")
    print("=" * 60)

    build_cfg = RaptorBuildConfig(
        leaf_window=100,
        max_levels=2,  # shallow tree for smoke speed
    )
    # Use llama3.2:3b for fast summarization in the smoke test.
    llm_cfg = LLM_PRESETS["ollama"]
    print(f"Summarizer: {llm_cfg.model} @ {llm_cfg.base_url}")
    print(f"Embedder:   {EMBEDDING_MODEL}")
    print(f"Corpus:     {CORPUS_DIR}")

    builder = RaptorTreeBuilder(build_config=build_cfg, llm_config=llm_cfg)
    index = builder.build(CORPUS_DIR)

    print(f"\nTree built: {len(index)} nodes; depth={index.max_level()}; "
          f"{len(index.root_ids)} root(s).")

    retriever = RaptorRetriever(index)

    failures = 0
    for strategy in (
        SearchStrategy.RAPTOR_TREE,
        SearchStrategy.RAPTOR_COLLAPSED,
        SearchStrategy.RAPTOR_QCOND,
    ):
        print(f"\n--- {strategy.value} ---")
        for q in SMOKE_QUERIES:
            result = retriever.retrieve(q, strategy, top_k=3)
            if not result.chunks:
                print(f"  [FAIL] no chunks for: {q!r}")
                failures += 1
                continue
            top_score = result.scores[0] if result.scores else float("nan")
            sources = ", ".join(
                sorted({s for c in result.chunks for s in (c.source_files or [c.source_file])})
            )
            print(
                f"  {q!r}\n"
                f"    -> top_score={top_score:.3f}, latency={result.retrieval_latency_ms:.1f}ms\n"
                f"    sources: {sources}"
            )

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {failures} query/strategy combinations returned no results.")
        sys.exit(1)
    print("Smoke test passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
