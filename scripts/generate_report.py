"""CLI script to generate the analysis report from benchmark results."""

import os
import sys

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.runner import load_results
from src.analyzer import BenchmarkAnalyzer


@click.command()
@click.option(
    "--results-file", default="results/raw/results.jsonl",
    help="Path to the results JSONL file",
)
@click.option(
    "--output-dir", default="results",
    help="Directory to save report outputs",
)
def main(results_file, output_dir):
    """Generate analysis report from benchmark results."""
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} evaluation results")

    analyzer = BenchmarkAnalyzer(results)

    # Generate plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Generating quality vs. latency plot...")
    analyzer.plot_quality_vs_latency(os.path.join(plots_dir, "quality_vs_latency.png"))

    print("Generating strategy by category plot...")
    analyzer.plot_strategy_by_category(os.path.join(plots_dir, "strategy_by_category.png"))

    print("Generating chunking by category plot...")
    analyzer.plot_chunking_by_category(os.path.join(plots_dir, "chunking_by_category.png"))

    print("Generating top-k by category plot...")
    analyzer.plot_topk_by_category(os.path.join(plots_dir, "topk_by_category.png"))

    print("Generating failure modes plot...")
    analyzer.plot_failure_modes(os.path.join(plots_dir, "failure_modes.png"))

    print("Generating context tokens vs. quality plot...")
    analyzer.plot_context_tokens_vs_quality(
        os.path.join(plots_dir, "context_tokens_vs_quality.png")
    )

    # Generate summary CSV
    print("Generating summary table...")
    analyzer.generate_summary_table()

    # Generate markdown report
    report_path = os.path.join(output_dir, "report.md")
    print("Generating markdown report...")
    analyzer.generate_markdown_report(report_path)

    print(f"\nReport generated at {report_path}")


if __name__ == "__main__":
    main()
