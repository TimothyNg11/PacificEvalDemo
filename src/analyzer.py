"""Benchmark analysis: aggregation, visualization, and reporting."""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .runner import EvalResult
from .config import RESULTS_DIR


class BenchmarkAnalyzer:
    """Aggregates benchmark results and generates visualizations."""

    def __init__(self, results: list[EvalResult]):
        self.results = results
        self.result_dicts = [
            {
                "config_name": r.config_name,
                "question_id": r.question_id,
                "question_category": r.question_category,
                "question_difficulty": r.question_difficulty,
                "chunks_retrieved": r.chunks_retrieved,
                "retrieval_latency_ms": r.retrieval_latency_ms,
                "generation_latency_ms": r.generation_latency_ms,
                "total_latency_ms": r.total_latency_ms,
                "context_tokens": r.context_tokens,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
                "distractor_rate": r.distractor_rate,
                "gold_similarity": r.gold_similarity,
                "fact_recall": r.fact_recall,
                "generated_answer": r.generated_answer,
                "gold_answer": r.gold_answer,
                "qasper_f1": getattr(r, "qasper_f1", -1.0),
                "qasper_em": getattr(r, "qasper_em", -1.0),
                "faithfulness": getattr(r, "faithfulness", -1.0),
            }
            for r in results
        ]

    def _group_by_config(self) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in self.result_dicts:
            groups[r["config_name"]].append(r)
        return groups

    def _compute_config_averages(self, group: list[dict]) -> dict:
        keys = [
            "gold_similarity", "fact_recall", "context_precision",
            "context_recall", "distractor_rate", "retrieval_latency_ms",
            "generation_latency_ms", "total_latency_ms", "context_tokens",
        ]
        avgs = {}
        for key in keys:
            values = [r[key] for r in group]
            avgs[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
        # Optional metrics: only average over rows where they're populated (!= -1).
        for opt_key in ("qasper_f1", "qasper_em", "faithfulness"):
            values = [r.get(opt_key, -1.0) for r in group]
            values = [v for v in values if v is not None and v != -1.0]
            avgs[f"avg_{opt_key}"] = (sum(values) / len(values)) if values else float("nan")
        return avgs

    def generate_summary_table(self) -> dict:
        groups = self._group_by_config()
        rows = []
        for config_name, group in groups.items():
            avgs = self._compute_config_averages(group)
            row = {"config_name": config_name, **avgs}
            rows.append(row)

        rows.sort(key=lambda x: x["avg_gold_similarity"], reverse=True)

        # Save to CSV
        csv_path = os.path.join(RESULTS_DIR, "summary.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return {row["config_name"]: row for row in rows}

    def plot_quality_vs_latency(self, output_path: str):
        groups = self._group_by_config()

        search_colors = {
            "vector": "blue",
            "bm25": "orange",
            "hybrid": "green",
            "hybrid_rerank": "red",
        }
        chunk_markers = {
            "fixed_256": "o",
            "fixed_512": "s",
            "semantic": "^",
            "paragraph": "D",
        }

        fig, ax = plt.subplots(figsize=(12, 8))

        points = []
        for config_name, group in groups.items():
            avgs = self._compute_config_averages(group)
            parts = config_name.split("__")
            chunking = parts[0]
            search = parts[1]

            x = avgs["avg_total_latency_ms"]
            y = avgs["avg_gold_similarity"]
            points.append((x, y, config_name))

            ax.scatter(
                x, y,
                c=search_colors.get(search, "gray"),
                marker=chunk_markers.get(chunking, "o"),
                s=80, alpha=0.7, edgecolors="black", linewidth=0.5,
            )

        # Pareto frontier
        pareto = []
        for x, y, name in sorted(points, key=lambda p: p[0]):
            if not pareto or y > pareto[-1][1]:
                pareto.append((x, y, name))

        for x, y, name in pareto:
            ax.annotate(
                name, (x, y),
                fontsize=6, alpha=0.8,
                xytext=(5, 5), textcoords="offset points",
            )

        # Legend for search strategies
        for search, color in search_colors.items():
            ax.scatter([], [], c=color, label=f"Search: {search}", s=60)
        for chunking, marker in chunk_markers.items():
            ax.scatter([], [], c="gray", marker=marker, label=f"Chunk: {chunking}", s=60)

        ax.set_xlabel("Average Total Latency (ms)")
        ax.set_ylabel("Average Gold Similarity")
        ax.set_title("Answer Quality vs. Retrieval Latency")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_strategy_by_category(self, output_path: str):
        categories = [
            "single_doc_factual", "numerical_precision",
            "cross_doc_synthesis", "terminology_mismatch", "distractor_heavy",
        ]
        search_strategies = ["vector", "bm25", "hybrid", "hybrid_rerank"]

        # Compute avg gold_similarity per (category, search_strategy)
        cat_search_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in self.result_dicts:
            parts = r["config_name"].split("__")
            search = parts[1]
            cat_search_scores[(r["question_category"], search)].append(r["gold_similarity"])

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(categories))
        width = 0.2
        colors = ["blue", "orange", "green", "red"]

        for i, search in enumerate(search_strategies):
            values = []
            for cat in categories:
                scores = cat_search_scores.get((cat, search), [])
                values.append(sum(scores) / len(scores) if scores else 0.0)
            ax.bar(x + i * width, values, width, label=search, color=colors[i], alpha=0.8)

        ax.set_xlabel("Question Category")
        ax.set_ylabel("Avg Gold Similarity")
        ax.set_title("Search Strategy Performance by Question Category")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_chunking_by_category(self, output_path: str):
        categories = [
            "single_doc_factual", "numerical_precision",
            "cross_doc_synthesis", "terminology_mismatch", "distractor_heavy",
        ]
        chunking_strategies = ["fixed_256", "fixed_512", "semantic", "paragraph"]

        cat_chunk_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in self.result_dicts:
            chunking = r["config_name"].split("__")[0]
            cat_chunk_scores[(r["question_category"], chunking)].append(r["gold_similarity"])

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(categories))
        width = 0.2
        colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]

        for i, chunking in enumerate(chunking_strategies):
            values = []
            for cat in categories:
                scores = cat_chunk_scores.get((cat, chunking), [])
                values.append(sum(scores) / len(scores) if scores else 0.0)
            ax.bar(x + i * width, values, width, label=chunking, color=colors[i], alpha=0.8)

        ax.set_xlabel("Question Category")
        ax.set_ylabel("Avg Gold Similarity")
        ax.set_title("Chunking Strategy Performance by Question Category")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_topk_by_category(self, output_path: str):
        categories = [
            "single_doc_factual", "numerical_precision",
            "cross_doc_synthesis", "terminology_mismatch", "distractor_heavy",
        ]
        topk_values = [3, 5, 10]

        cat_topk_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
        for r in self.result_dicts:
            top_k = int(r["config_name"].split("__")[2].replace("k", ""))
            cat_topk_scores[(r["question_category"], top_k)].append(r["gold_similarity"])

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(categories))
        width = 0.25
        colors = ["#1abc9c", "#e74c3c", "#f39c12"]

        for i, top_k in enumerate(topk_values):
            values = []
            for cat in categories:
                scores = cat_topk_scores.get((cat, top_k), [])
                values.append(sum(scores) / len(scores) if scores else 0.0)
            ax.bar(x + i * width, values, width, label=f"k={top_k}", color=colors[i], alpha=0.8)

        ax.set_xlabel("Question Category")
        ax.set_ylabel("Avg Gold Similarity")
        ax.set_title("Top-K Performance by Question Category")
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_failure_modes(self, output_path: str):
        groups = self._group_by_config()

        # Classify each result
        config_failures: dict[str, dict[str, int]] = {}
        config_avgs: dict[str, float] = {}

        for config_name, group in groups.items():
            avgs = self._compute_config_averages(group)
            config_avgs[config_name] = avgs["avg_gold_similarity"]

            counts = {"correct": 0, "partial": 0, "wrong_source": 0, "insufficient": 0}
            for r in group:
                if r["gold_similarity"] >= 0.8 and r["fact_recall"] >= 0.8:
                    counts["correct"] += 1
                elif r["context_precision"] < 0.3:
                    counts["wrong_source"] += 1
                elif r["context_recall"] < 0.5:
                    counts["insufficient"] += 1
                elif r["gold_similarity"] >= 0.5:
                    counts["partial"] += 1
                else:
                    counts["wrong_source"] += 1  # fallback

            config_failures[config_name] = counts

        # Pick top 10 and bottom 5 by avg score
        sorted_configs = sorted(config_avgs.keys(), key=lambda k: config_avgs[k], reverse=True)
        selected = sorted_configs[:10] + sorted_configs[-5:]

        fig, ax = plt.subplots(figsize=(14, 8))
        modes = ["correct", "partial", "wrong_source", "insufficient"]
        mode_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

        y_pos = np.arange(len(selected))
        left = np.zeros(len(selected))

        for mode, color in zip(modes, mode_colors):
            values = []
            for cfg in selected:
                total = sum(config_failures[cfg].values())
                pct = config_failures[cfg][mode] / total * 100 if total else 0
                values.append(pct)
            ax.barh(y_pos, values, left=left, label=mode, color=color, alpha=0.85)
            left += np.array(values)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(selected, fontsize=7)
        ax.set_xlabel("Percentage of Questions")
        ax.set_title("Failure Mode Distribution by Configuration")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_context_tokens_vs_quality(self, output_path: str):
        groups = self._group_by_config()

        xs, ys = [], []
        for config_name, group in groups.items():
            avgs = self._compute_config_averages(group)
            xs.append(avgs["avg_context_tokens"])
            ys.append(avgs["avg_gold_similarity"])

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(xs, ys, alpha=0.6, edgecolors="black", linewidth=0.5, s=60)

        # Trend line (degree 2 polynomial)
        if len(xs) > 2:
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            coeffs = np.polyfit(xs_arr, ys_arr, 2)
            x_line = np.linspace(min(xs_arr), max(xs_arr), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, "r--", alpha=0.7, label="Trend (degree 2)")

        ax.set_xlabel("Average Context Tokens")
        ax.set_ylabel("Average Gold Similarity")
        ax.set_title("Context Size vs. Answer Quality")
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def generate_markdown_report(self, output_path: str):
        summary = self.generate_summary_table()
        configs_sorted = sorted(
            summary.values(),
            key=lambda x: x["avg_gold_similarity"],
            reverse=True,
        )

        best = configs_sorted[0] if configs_sorted else {}
        worst = configs_sorted[-1] if configs_sorted else {}

        # Compute strategy-level insights
        strategy_scores: dict[str, list[float]] = defaultdict(list)
        chunking_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self.result_dicts:
            parts = r["config_name"].split("__")
            search = parts[1]
            chunking = parts[0]
            strategy_scores[search].append(r["gold_similarity"])
            chunking_scores[chunking][r["question_category"]].append(r["gold_similarity"])

        strategy_avgs = {
            s: sum(v) / len(v) for s, v in strategy_scores.items() if v
        }
        best_strategy = max(strategy_avgs, key=strategy_avgs.get) if strategy_avgs else "N/A"

        # Reranking improvement
        hybrid_avg = strategy_avgs.get("hybrid", 0)
        rerank_avg = strategy_avgs.get("hybrid_rerank", 0)
        rerank_improvement = ((rerank_avg - hybrid_avg) / hybrid_avg * 100) if hybrid_avg else 0

        # Latency cost of reranking
        hybrid_latency = []
        rerank_latency = []
        for r in self.result_dicts:
            parts = r["config_name"].split("__")
            if parts[1] == "hybrid":
                hybrid_latency.append(r["retrieval_latency_ms"])
            elif parts[1] == "hybrid_rerank":
                rerank_latency.append(r["retrieval_latency_ms"])

        rerank_latency_cost = 0
        if hybrid_latency and rerank_latency:
            rerank_latency_cost = (
                sum(rerank_latency) / len(rerank_latency)
                - sum(hybrid_latency) / len(hybrid_latency)
            )

        # Find context token plateau
        groups = self._group_by_config()
        token_quality_pairs = []
        for config_name, group in groups.items():
            avgs = self._compute_config_averages(group)
            token_quality_pairs.append((avgs["avg_context_tokens"], avgs["avg_gold_similarity"]))
        token_quality_pairs.sort()

        plateau_tokens = "N/A"
        if len(token_quality_pairs) > 5:
            best_quality = max(q for _, q in token_quality_pairs)
            for tokens, quality in token_quality_pairs:
                if quality >= best_quality * 0.95:
                    plateau_tokens = f"{int(tokens)}"
                    break

        report = f"""# RAG Benchmark Results Report

## Summary

- **Best overall config**: `{best.get('config_name', 'N/A')}` (avg similarity: {best.get('avg_gold_similarity', 0):.3f})
- **Worst overall config**: `{worst.get('config_name', 'N/A')}` (avg similarity: {worst.get('avg_gold_similarity', 0):.3f})

## Key Findings

- Best search strategy overall: **{best_strategy}** (avg similarity: {strategy_avgs.get(best_strategy, 0):.3f})
- Reranking improves quality by **{rerank_improvement:.1f}%** but adds **{rerank_latency_cost:.0f}ms** latency
- Diminishing returns: quality plateaus after ~**{plateau_tokens}** context tokens

## Charts

### Answer Quality vs. Retrieval Latency
![Quality vs Latency](plots/quality_vs_latency.png)

### Search Strategy Performance by Question Category
![Strategy by Category](plots/strategy_by_category.png)

### Chunking Strategy Performance by Question Category
![Chunking by Category](plots/chunking_by_category.png)

### Top-K Performance by Question Category
![Top-K by Category](plots/topk_by_category.png)

### Failure Mode Distribution
![Failure Modes](plots/failure_modes.png)

### Context Size vs. Answer Quality
![Context Tokens vs Quality](plots/context_tokens_vs_quality.png)

## Summary Table

| Config | Avg Similarity | Avg Fact Recall | Avg Precision | Avg Recall | Avg Latency (ms) | Avg Context Tokens |
|--------|---------------|-----------------|---------------|------------|-------------------|--------------------|
"""

        for row in configs_sorted:
            report += (
                f"| {row['config_name']} "
                f"| {row['avg_gold_similarity']:.3f} "
                f"| {row['avg_fact_recall']:.3f} "
                f"| {row['avg_context_precision']:.3f} "
                f"| {row['avg_context_recall']:.3f} "
                f"| {row['avg_total_latency_ms']:.0f} "
                f"| {int(row['avg_context_tokens'])} |\n"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    # ---------- RAPTOR / Pareto / ablation plots ----------

    @staticmethod
    def _strategy_family(config_name: str) -> str:
        """Classify a config as baseline vs RAPTOR mode for plotting."""
        parts = config_name.split("__")
        search = parts[1] if len(parts) > 1 else ""
        if search.startswith("raptor_"):
            return search  # raptor_tree / raptor_collapsed / raptor_qcond
        return "baseline"

    def plot_pareto_frontier(self, output_path: str, metric: str = "gold_similarity"):
        """Token budget (context_tokens) vs quality, colored by strategy family.

        Highlights RAPTOR modes (tree / collapsed / qcond) against the
        baseline cloud so the reader can see whether RAPTOR sits on the
        Pareto frontier at any budget.
        """
        groups = self._group_by_config()

        family_colors = {
            "baseline": "lightgray",
            "raptor_tree": "#3498db",
            "raptor_collapsed": "#2ecc71",
            "raptor_qcond": "#e74c3c",
        }
        family_markers = {
            "baseline": "o",
            "raptor_tree": "^",
            "raptor_collapsed": "s",
            "raptor_qcond": "D",
        }

        fig, ax = plt.subplots(figsize=(12, 7))
        points: list[tuple[float, float, str, str]] = []
        for config_name, group in groups.items():
            avgs = self._compute_config_averages(group)
            x = avgs["avg_context_tokens"]
            y = avgs.get(f"avg_{metric}", avgs["avg_gold_similarity"])
            family = self._strategy_family(config_name)
            points.append((x, y, config_name, family))
            ax.scatter(
                x, y,
                c=family_colors.get(family, "gray"),
                marker=family_markers.get(family, "o"),
                s=110 if family != "baseline" else 50,
                alpha=0.85 if family != "baseline" else 0.45,
                edgecolors="black", linewidth=0.6,
                zorder=3 if family != "baseline" else 1,
            )

        # Pareto frontier: max y for each x (ascending x)
        sorted_pts = sorted(points, key=lambda p: p[0])
        frontier: list[tuple[float, float, str, str]] = []
        for x, y, name, family in sorted_pts:
            if not frontier or y > frontier[-1][1]:
                frontier.append((x, y, name, family))
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, "k--", alpha=0.4, label="Pareto frontier", linewidth=1.0)

        for family, color in family_colors.items():
            ax.scatter([], [], c=color, marker=family_markers[family],
                       label=family, s=80, edgecolors="black", linewidth=0.6)

        ax.set_xlabel("Average context tokens")
        ax.set_ylabel(f"Average {metric}")
        ax.set_title(f"Pareto frontier: token budget vs {metric}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_raptor_vs_baseline_by_category(self, output_path: str):
        """For each question category, plot avg gold_similarity for the best
        baseline config vs each RAPTOR mode. Highlights where RAPTOR helps."""
        # Find categories present
        categories = sorted({r["question_category"] for r in self.result_dicts})
        if not categories:
            return

        family_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self.result_dicts:
            family = self._strategy_family(r["config_name"])
            family_scores[family][r["question_category"]].append(r["gold_similarity"])

        families = [f for f in ("baseline", "raptor_tree", "raptor_collapsed", "raptor_qcond")
                    if f in family_scores]
        if len(families) <= 1:
            return  # no RAPTOR data; skip

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2.0), 6))
        x = np.arange(len(categories))
        width = 0.8 / len(families)
        colors = {
            "baseline": "#95a5a6",
            "raptor_tree": "#3498db",
            "raptor_collapsed": "#2ecc71",
            "raptor_qcond": "#e74c3c",
        }
        for i, fam in enumerate(families):
            values = []
            for cat in categories:
                scores = family_scores[fam].get(cat, [])
                values.append(sum(scores) / len(scores) if scores else 0.0)
            ax.bar(x + i * width, values, width, label=fam, color=colors.get(fam, "gray"), alpha=0.85)

        ax.set_xlabel("Question category")
        ax.set_ylabel("Avg gold similarity")
        ax.set_title("RAPTOR vs baseline by question category")
        ax.set_xticks(x + width * (len(families) - 1) / 2)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_qasper_f1_by_strategy(self, output_path: str):
        """Bar plot of avg QASPER F1 by search strategy. No-op if no F1 data."""
        rows = [r for r in self.result_dicts if r.get("qasper_f1", -1.0) != -1.0]
        if not rows:
            return
        per_search: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            parts = r["config_name"].split("__")
            search = parts[1] if len(parts) > 1 else "?"
            per_search[search].append(r["qasper_f1"])
        labels = sorted(per_search.keys())
        means = [sum(per_search[l]) / len(per_search[l]) for l in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#3498db" if l.startswith("raptor_") else "#95a5a6" for l in labels]
        ax.bar(labels, means, color=colors, alpha=0.85)
        ax.set_ylabel("Avg QASPER F1")
        ax.set_title("QASPER token-level F1 by search strategy")
        ax.set_ylim(0, max(means + [1.0]))
        for i, m in enumerate(means):
            ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_faithfulness_by_strategy(self, output_path: str):
        """Bar plot of avg NLI faithfulness by search strategy. No-op if missing."""
        rows = [r for r in self.result_dicts if r.get("faithfulness", -1.0) != -1.0]
        if not rows:
            return
        per_search: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            parts = r["config_name"].split("__")
            search = parts[1] if len(parts) > 1 else "?"
            per_search[search].append(r["faithfulness"])
        labels = sorted(per_search.keys())
        means = [sum(per_search[l]) / len(per_search[l]) for l in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#3498db" if l.startswith("raptor_") else "#95a5a6" for l in labels]
        ax.bar(labels, means, color=colors, alpha=0.85)
        ax.set_ylabel("Avg NLI faithfulness")
        ax.set_title("Faithfulness (NLI entailment) by search strategy")
        ax.set_ylim(0, 1.0)
        for i, m in enumerate(means):
            ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
