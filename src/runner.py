"""Benchmark runner: orchestrates retrieval, generation, and scoring."""

import json
import os
from dataclasses import dataclass, field, asdict

import yaml

from .config import (
    RetrievalConfig,
    SearchStrategy,
    LLMConfig,
    generate_all_configs,
    CORPUS_DIR,
    EVAL_SET_PATH,
    RESULTS_DIR,
)
from .indexer import IndexBuilder
from .generator import AnswerGenerator
from .retrievers import Retriever, get_reranker
from .scorers import RetrievalScorer, GoldSimilarityScorer, KeyFactScorer


@dataclass
class EvalResult:
    config_name: str
    question_id: str
    question_text: str
    question_category: str
    question_difficulty: str
    chunks_retrieved: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    context_tokens: int
    context_precision: float
    context_recall: float
    distractor_rate: float
    gold_similarity: float
    fact_recall: float
    missing_facts: list[str] = field(default_factory=list)
    generated_answer: str = ""
    gold_answer: str = ""


class BenchmarkRunner:
    """Orchestrates the full benchmark run."""

    def __init__(self, llm_config: LLMConfig | None = None):
        # Load eval set
        with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
            self.eval_set = yaml.safe_load(f)

        # Build indexes
        print("Building indexes...")
        self.index_builder = IndexBuilder()
        self.indexes = self.index_builder.build_all_indexes(CORPUS_DIR)

        # Initialize generator
        self.generator = AnswerGenerator(llm_config=llm_config)

        # Initialize scorers
        self.retrieval_scorer = RetrievalScorer()
        self.gold_similarity_scorer = GoldSimilarityScorer()
        self.key_fact_scorer = KeyFactScorer()

        # Load reranker once
        self.reranker = get_reranker()

    def run(self, configs: list[RetrievalConfig] | None = None) -> list[EvalResult]:
        if configs is None:
            configs = generate_all_configs()

        all_results = []
        total_configs = len(configs)

        for config_idx, config in enumerate(configs, 1):
            print(f"\n[{config_idx}/{total_configs}] Running config: {config.name}")

            # Get the index for this chunking strategy
            index = self.indexes[config.chunking.value]
            retriever = Retriever(index, reranker=self.reranker)

            for q_idx, question in enumerate(self.eval_set, 1):
                # Retrieve
                retrieval_result = retriever.retrieve(
                    query=question["question"],
                    strategy=config.search,
                    top_k=config.top_k,
                )

                # Generate
                gen_result = self.generator.generate(
                    question=question["question"],
                    context_chunks=retrieval_result.chunks,
                )

                # Score retrieval
                retrieval_metrics = self.retrieval_scorer.score(
                    retrieved_chunks=retrieval_result.chunks,
                    gold_source_ids=question["gold_source_ids"],
                    distractor_ids=question.get("distractors"),
                )

                # Score answer similarity
                gold_similarity = self.gold_similarity_scorer.score(
                    generated_answer=gen_result.answer,
                    gold_answer=question["gold_answer"],
                )

                # Score key facts
                key_fact_metrics = self.key_fact_scorer.score(
                    answer=gen_result.answer,
                    key_facts=question.get("key_facts", []),
                )

                total_latency = (
                    retrieval_result.retrieval_latency_ms
                    + gen_result.generation_latency_ms
                )

                result = EvalResult(
                    config_name=config.name,
                    question_id=question["id"],
                    question_text=question["question"],
                    question_category=question["category"],
                    question_difficulty=question["difficulty"],
                    chunks_retrieved=len(retrieval_result.chunks),
                    retrieval_latency_ms=retrieval_result.retrieval_latency_ms,
                    generation_latency_ms=gen_result.generation_latency_ms,
                    total_latency_ms=total_latency,
                    context_tokens=gen_result.context_tokens,
                    context_precision=retrieval_metrics.context_precision,
                    context_recall=retrieval_metrics.context_recall,
                    distractor_rate=retrieval_metrics.distractor_rate,
                    gold_similarity=gold_similarity,
                    fact_recall=key_fact_metrics.fact_recall,
                    missing_facts=key_fact_metrics.missing_facts,
                    generated_answer=gen_result.answer,
                    gold_answer=question["gold_answer"],
                )

                all_results.append(result)

                print(
                    f"  [{config.name}] Question {q_idx}/{len(self.eval_set)} "
                    f"({question['id']}) ... "
                    f"gold_similarity={gold_similarity:.2f} "
                    f"fact_recall={key_fact_metrics.fact_recall:.2f}"
                )

        # Save results
        output_path = os.path.join(RESULTS_DIR, "raw", "results.jsonl")
        save_results(all_results, output_path)
        print(f"\nResults saved to {output_path}")

        return all_results


def save_results(results: list[EvalResult], path: str):
    """Write each EvalResult as a JSON line to the file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")


def load_results(path: str) -> list[EvalResult]:
    """Read JSONL file back into list of EvalResult."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            results.append(EvalResult(**data))
    return results
