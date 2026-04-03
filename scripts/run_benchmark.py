"""CLI script to run the full benchmark."""

import os
import sys

import click

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RetrievalConfig, ChunkingStrategy, SearchStrategy, generate_all_configs, LLMConfig, LLM_PRESETS
from src.runner import BenchmarkRunner


def parse_config_name(name: str) -> RetrievalConfig:
    """Parse a config name like 'fixed_256__vector__k3' back into a RetrievalConfig."""
    parts = name.split("__")
    if len(parts) != 3:
        raise click.BadParameter(f"Invalid config name: {name}")

    chunking = ChunkingStrategy(parts[0])
    search = SearchStrategy(parts[1])
    top_k = int(parts[2].replace("k", ""))
    return RetrievalConfig(chunking=chunking, search=search, top_k=top_k)


@click.command()
@click.option(
    "--configs", default="all",
    help='Comma-separated config names, or "all" for full matrix',
)
@click.option(
    "--questions", default="all",
    help='Comma-separated question IDs, or "all"',
)
@click.option(
    "--llm", default="auto", type=click.Choice(["auto", "ollama", "openai"]),
    help='LLM provider: "auto" (OpenAI if OPENAI_API_KEY is set, else Ollama), "ollama", or "openai"',
)
def main(configs, questions, llm):
    """Run the full RAG benchmark."""
    # Parse config filter
    if configs == "all":
        config_list = generate_all_configs()
    else:
        config_list = [parse_config_name(c.strip()) for c in configs.split(",")]

    # Resolve LLM config
    if llm == "auto":
        llm = "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"
    llm_config = LLM_PRESETS[llm]
    if llm == "openai" and not llm_config.api_key:
        click.echo("Error: OPENAI_API_KEY environment variable is not set.", err=True)
        sys.exit(1)
    print(f"Using LLM: {llm} ({llm_config.model} @ {llm_config.base_url})")
    print(f"Running {len(config_list)} configurations...")

    # Initialize and run
    runner = BenchmarkRunner(llm_config=llm_config)

    # Filter questions if needed
    if questions != "all":
        question_ids = set(q.strip() for q in questions.split(","))
        runner.eval_set = [q for q in runner.eval_set if q["id"] in question_ids]
        print(f"Filtered to {len(runner.eval_set)} questions")

    results = runner.run(configs=config_list)

    # Print summary
    total = len(results)
    avg_similarity = sum(r.gold_similarity for r in results) / total if total else 0
    avg_fact_recall = sum(r.fact_recall for r in results) / total if total else 0

    print(f"\n{'=' * 60}")
    print(f"Benchmark Complete")
    print(f"{'=' * 60}")
    print(f"Total evaluations: {total}")
    print(f"Average gold similarity: {avg_similarity:.3f}")
    print(f"Average fact recall: {avg_fact_recall:.3f}")
    print(f"Results saved to results/raw/results.jsonl")


if __name__ == "__main__":
    main()
