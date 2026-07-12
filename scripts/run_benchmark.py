"""CLI script to run the full benchmark."""

import os
import sys

import click

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RetrievalConfig, generate_all_configs, LLMConfig, LLM_PRESETS, detect_llm_provider
from src.runner import BenchmarkRunner, results_path


def parse_config_name(name: str) -> RetrievalConfig:
    """Parse a config name like 'fixed_256__vector__k3' back into a RetrievalConfig."""
    try:
        return RetrievalConfig.from_name(name)
    except ValueError as e:
        raise click.BadParameter(str(e))


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
    "--llm", default="auto", type=click.Choice(["auto", "ollama", "openai", "anthropic"]),
    help='LLM provider: "auto" (Anthropic if ANTHROPIC_API_KEY is set, elif '
         'OpenAI if OPENAI_API_KEY is set, else Ollama), "ollama", "openai", or "anthropic"',
)
@click.option(
    "--include-raptor", is_flag=True, default=False,
    help="Include RAPTOR strategies (adds 9 configs: 1 chunking x 3 search x 3 top_k). "
         "Requires umap-learn; summarization uses whichever --llm provider is selected.",
)
@click.option(
    "--corpus-dir", default=None,
    help="Override the default corpus directory (data/corpus).",
)
@click.option(
    "--eval-set", "eval_set_path", default=None,
    help="Override the default eval set YAML (data/eval_set.yaml).",
)
@click.option(
    "--results-suffix", default="",
    help='Suffix appended to the results filename (e.g. "qasper" -> results_qasper.jsonl).',
)
@click.option(
    "--faithfulness", is_flag=True, default=False,
    help="Also compute NLI faithfulness score per answer (uses a local cross-encoder).",
)
@click.option(
    "--qasper-f1", is_flag=True, default=False,
    help="Also compute QASPER token-level F1/EM (only meaningful on QASPER eval sets).",
)
@click.option(
    "--qcond-config", "qcond_config_json", default=None,
    help='JSON overrides for QCondConfig, e.g. \'{"tau_focus": 0.85}\' — '
         "runs raptor_qcond with a dev-calibrated setting (scripts/sweep_qcond.py).",
)
def main(configs, questions, llm, include_raptor, corpus_dir, eval_set_path,
         results_suffix, faithfulness, qasper_f1, qcond_config_json):
    """Run the full RAG benchmark."""
    qcond_config = None
    if qcond_config_json:
        import json
        from src.raptor.tree_retriever import QCondConfig
        qcond_config = QCondConfig(**json.loads(qcond_config_json))
        print(f"Using qcond overrides: {qcond_config}")
    # Parse config filter
    if configs == "all":
        config_list = generate_all_configs(include_raptor=include_raptor)
    else:
        config_list = [parse_config_name(c.strip()) for c in configs.split(",")]
        # If the user passed RAPTOR configs by name, ensure the runner builds the index.
        if any(c.is_raptor for c in config_list):
            include_raptor = True

    # Resolve LLM config
    if llm == "auto":
        llm = detect_llm_provider()
    llm_config = LLM_PRESETS[llm]
    if llm in ("openai", "anthropic") and not llm_config.api_key:
        env_var = "OPENAI_API_KEY" if llm == "openai" else "ANTHROPIC_API_KEY"
        click.echo(f"Error: {env_var} environment variable is not set.", err=True)
        sys.exit(1)
    print(f"Using LLM: {llm} ({llm_config.model} @ {llm_config.base_url})")
    print(f"Running {len(config_list)} configurations (include_raptor={include_raptor})...")

    # Initialize and run
    from src.config import CORPUS_DIR, EVAL_SET_PATH
    runner_kwargs = dict(
        llm_config=llm_config,
        include_raptor=include_raptor,
        corpus_dir=corpus_dir or CORPUS_DIR,
        eval_set_path=eval_set_path or EVAL_SET_PATH,
        enable_faithfulness=faithfulness,
        enable_qasper_f1=qasper_f1,
        results_suffix=results_suffix,
        qcond_config=qcond_config,
    )
    runner = BenchmarkRunner(**runner_kwargs)

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
    print(f"Results saved to {results_path(results_suffix)}")


if __name__ == "__main__":
    main()
