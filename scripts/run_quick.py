"""Quick benchmark: tests one random question across diverse configs.

Designed for testing with Ollama (local LLM) where the full 1,200-eval
benchmark is too slow. Picks one random question, then generates diverse
retrieval configurations (varying chunking, search strategy, and top-k)
so you get a direct apples-to-apples comparison on every run.
"""

import os
import random
import sys

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    RetrievalConfig, ChunkingStrategy, SearchStrategy,
    LLMConfig, LLM_PRESETS, detect_llm_provider,
    _RAPTOR_CHUNKINGS, _RAPTOR_SEARCHES,
)
from src.runner import BenchmarkRunner

# Baseline-only by default — sampling across the full enum (now including
# RAPTOR) could produce invalid combos or require an expensive tree build
# that the quick script isn't meant to trigger.
ALL_CHUNKING = [c for c in ChunkingStrategy if c not in _RAPTOR_CHUNKINGS]
ALL_SEARCH = [s for s in SearchStrategy if s not in _RAPTOR_SEARCHES]
ALL_TOP_K = [3, 5, 10]
NUM_CONFIGS = 5


def sample_configs(n=NUM_CONFIGS, seed=None):
    """Sample n diverse retrieval configs by cycling through shuffled dimensions.

    Cycles through chunking strategies, search strategies, and top-k values
    so no single dimension dominates the sample.
    """
    rng = random.Random(seed)

    shuffled_chunking = ALL_CHUNKING[:]
    shuffled_search = ALL_SEARCH[:]
    shuffled_topk = ALL_TOP_K[:]

    rng.shuffle(shuffled_chunking)
    rng.shuffle(shuffled_search)
    rng.shuffle(shuffled_topk)

    configs = []
    for i in range(n):
        chunking = shuffled_chunking[i % len(shuffled_chunking)]
        search = shuffled_search[i % len(shuffled_search)]
        top_k = shuffled_topk[i % len(shuffled_topk)]
        configs.append(RetrievalConfig(chunking, search, top_k))

    return configs


@click.command()
@click.option(
    "--llm", default="auto", type=click.Choice(["auto", "ollama", "openai", "anthropic"]),
    help='LLM provider: "auto" (Anthropic if key is set, elif OpenAI if key is set, '
         'else Ollama), "ollama", "openai", or "anthropic"',
)
@click.option(
    "--num", "-n", default=NUM_CONFIGS, type=int,
    help=f"Number of random configurations to test (default: {NUM_CONFIGS}).",
)
@click.option(
    "--seed", "-s", default=None, type=int,
    help="Random seed for reproducible runs.",
)
def main(llm, num, seed):
    """Run a quick benchmark: one random question, many random configs.

    Picks a single question at random, then tests it against diverse
    retrieval configurations so you can directly compare how different
    chunking strategies, search strategies, and top-k values perform
    on the same question. The question changes each run (unless --seed
    is used).

    Results are saved to results/raw/results.jsonl and can be analyzed
    with generate_report.py.

    Examples:
        python scripts/run_quick.py              # 5 random configs, 1 random question
        python scripts/run_quick.py -n 10        # more configs
        python scripts/run_quick.py -n 5 -s 42   # reproducible with a seed
    """
    if llm == "auto":
        llm = detect_llm_provider()
    llm_config = LLM_PRESETS[llm]
    if llm in ("openai", "anthropic") and not llm_config.api_key:
        env_var = "OPENAI_API_KEY" if llm == "openai" else "ANTHROPIC_API_KEY"
        click.echo(f"Error: {env_var} environment variable is not set.", err=True)
        sys.exit(1)

    rng = random.Random(seed)
    runner = BenchmarkRunner(llm_config=llm_config)

    # Pick one random question
    question = rng.choice(runner.eval_set)
    runner.eval_set = [question]

    # Generate diverse configs
    configs = sample_configs(n=num, seed=seed)

    print(f"Using LLM: {llm} ({llm_config.model} @ {llm_config.base_url})")
    print(f"Running quick benchmark: {num} configs × 1 question = {num} evaluations")
    print(f"  Question:            {question['id']} — {question['question']}")
    print(f"  Chunking strategies: {sorted({c.chunking.value for c in configs})}")
    print(f"  Search strategies:   {sorted({c.search.value for c in configs})}")
    print(f"  Top-k values:        {sorted({c.top_k for c in configs})}")

    results = runner.run(configs=configs)

    total = len(results)
    avg_similarity = sum(r.gold_similarity for r in results) / total if total else 0
    avg_fact_recall = sum(r.fact_recall for r in results) / total if total else 0

    print(f"\n{'=' * 60}")
    print(f"Quick Benchmark Complete")
    print(f"{'=' * 60}")
    print(f"Total evaluations: {total}")
    print(f"Average gold similarity: {avg_similarity:.3f}")
    print(f"Average fact recall: {avg_fact_recall:.3f}")
    print(f"\nRun 'python scripts/generate_report.py' to generate charts and report.")


if __name__ == "__main__":
    main()
