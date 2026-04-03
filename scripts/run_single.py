"""CLI script for debugging a single config on a single question."""

import os
import sys

import click
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    RetrievalConfig, ChunkingStrategy, SearchStrategy,
    CORPUS_DIR, EVAL_SET_PATH, LLMConfig, LLM_PRESETS,
)
from src.indexer import IndexBuilder
from src.generator import AnswerGenerator
from src.retrievers import Retriever, get_reranker
from src.scorers import RetrievalScorer, GoldSimilarityScorer, KeyFactScorer


def parse_config_name(name: str) -> RetrievalConfig:
    parts = name.split("__")
    if len(parts) != 3:
        raise click.BadParameter(f"Invalid config name: {name}")
    chunking = ChunkingStrategy(parts[0])
    search = SearchStrategy(parts[1])
    top_k = int(parts[2].replace("k", ""))
    return RetrievalConfig(chunking=chunking, search=search, top_k=top_k)


@click.command()
@click.argument("config_name")
@click.argument("question_id")
@click.option(
    "--llm", default="auto", type=click.Choice(["auto", "ollama", "openai"]),
    help='LLM provider: "auto" (OpenAI if OPENAI_API_KEY is set, else Ollama), "ollama", or "openai"',
)
def main(config_name, question_id, llm):
    """Debug a single config on a single question.

    CONFIG_NAME: e.g., fixed_256__vector__k3
    QUESTION_ID: e.g., sf_001
    """
    # Parse config
    config = parse_config_name(config_name)
    print(f"Config: {config.name}")
    print(f"  Chunking: {config.chunking.value}")
    print(f"  Search: {config.search.value}")
    print(f"  Top-K: {config.top_k}")

    # Load eval set and find question
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = yaml.safe_load(f)

    question = None
    for q in eval_set:
        if q["id"] == question_id:
            question = q
            break

    if question is None:
        click.echo(f"Question '{question_id}' not found in eval set.", err=True)
        sys.exit(1)

    print(f"\nQuestion: {question['question']}")
    print(f"Category: {question['category']}")
    print(f"Difficulty: {question['difficulty']}")
    print(f"Gold sources: {question['gold_source_ids']}")

    # Build index
    print("\nBuilding index...")
    builder = IndexBuilder()
    indexes = builder.build_all_indexes(CORPUS_DIR)
    index = indexes[config.chunking.value]

    # Retrieve
    print("\nRetrieving...")
    reranker = get_reranker()
    retriever = Retriever(index, reranker=reranker)
    retrieval_result = retriever.retrieve(
        query=question["question"],
        strategy=config.search,
        top_k=config.top_k,
    )

    print(f"\nRetrieved {len(retrieval_result.chunks)} chunks in {retrieval_result.retrieval_latency_ms:.1f}ms:")
    for i, (chunk, score) in enumerate(zip(retrieval_result.chunks, retrieval_result.scores)):
        print(f"\n--- Chunk {i + 1} (score: {score:.4f}) ---")
        print(f"Source: {chunk.source_file} [chunk {chunk.chunk_index}]")
        print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))

    # Generate answer
    print("\n\nGenerating answer...")
    if llm == "auto":
        llm = "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"
    llm_config = LLM_PRESETS[llm]
    if llm == "openai" and not llm_config.api_key:
        click.echo("Error: OPENAI_API_KEY environment variable is not set.", err=True)
        sys.exit(1)
    print(f"Using LLM: {llm} ({llm_config.model} @ {llm_config.base_url})")
    generator = AnswerGenerator(llm_config=llm_config)
    gen_result = generator.generate(
        question=question["question"],
        context_chunks=retrieval_result.chunks,
    )

    print(f"\nGenerated answer ({gen_result.context_tokens} context tokens, "
          f"{gen_result.generation_latency_ms:.0f}ms):")
    print(gen_result.answer)

    # Score
    print("\n\n--- Scoring ---")

    retrieval_scorer = RetrievalScorer()
    retrieval_metrics = retrieval_scorer.score(
        retrieved_chunks=retrieval_result.chunks,
        gold_source_ids=question["gold_source_ids"],
        distractor_ids=question.get("distractors"),
    )
    print(f"Context Precision: {retrieval_metrics.context_precision:.3f}")
    print(f"Context Recall: {retrieval_metrics.context_recall:.3f}")
    print(f"Distractor Rate: {retrieval_metrics.distractor_rate:.3f}")

    gold_scorer = GoldSimilarityScorer()
    gold_sim = gold_scorer.score(gen_result.answer, question["gold_answer"])
    print(f"Gold Similarity: {gold_sim:.3f}")

    fact_scorer = KeyFactScorer()
    fact_metrics = fact_scorer.score(gen_result.answer, question.get("key_facts", []))
    print(f"Fact Recall: {fact_metrics.fact_recall:.3f}")
    if fact_metrics.found_facts:
        print(f"  Found: {fact_metrics.found_facts}")
    if fact_metrics.missing_facts:
        print(f"  Missing: {fact_metrics.missing_facts}")

    print(f"\n--- Gold Answer ---")
    print(question["gold_answer"])


if __name__ == "__main__":
    main()
