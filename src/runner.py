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
from .retrievers import get_reranker, make_retriever
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
    # Optional metrics; default to NaN-equivalent so missing values are visible.
    qasper_f1: float = -1.0
    qasper_em: float = -1.0
    faithfulness: float = -1.0
    faithfulness_supported: int = -1
    faithfulness_total: int = -1


def results_path(suffix: str, results_dir: str = RESULTS_DIR) -> str:
    """Compute the results.jsonl path for a given --results-suffix.

    Single-sourced so `BenchmarkRunner.run()` (which writes the file) and
    `run_benchmark.py`'s final summary print can never drift out of sync on
    the actual filename.
    """
    file_suffix = f"_{suffix}" if suffix else ""
    return os.path.join(results_dir, "raw", f"results{file_suffix}.jsonl")


class BenchmarkRunner:
    """Orchestrates the full benchmark run."""

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        corpus_dir: str = CORPUS_DIR,
        eval_set_path: str = EVAL_SET_PATH,
        include_raptor: bool = False,
        enable_faithfulness: bool = False,
        enable_qasper_f1: bool = False,
        results_suffix: str = "",
        qcond_config=None,
    ):
        self.include_raptor = include_raptor
        self.enable_faithfulness = enable_faithfulness
        self.enable_qasper_f1 = enable_qasper_f1
        self.results_suffix = results_suffix
        # Optional dev-calibrated QCondConfig for qcond configs (see
        # scripts/sweep_qcond.py); None = QCondConfig defaults.
        self.qcond_config = qcond_config

        # Load eval set
        with open(eval_set_path, "r", encoding="utf-8") as f:
            self.eval_set = yaml.safe_load(f)

        # Build indexes
        print("Building indexes...")
        self.index_builder = IndexBuilder()
        self.indexes = self.index_builder.build_all_indexes(
            corpus_dir,
            include_raptor=include_raptor,
            llm_config=llm_config,
        )

        # Initialize generator
        self.generator = AnswerGenerator(llm_config=llm_config)

        # Initialize scorers
        self.retrieval_scorer = RetrievalScorer()
        self.gold_similarity_scorer = GoldSimilarityScorer()
        self.key_fact_scorer = KeyFactScorer()

        # Optional scorers (lazy-imported so non-RAPTOR runs stay cheap)
        self.qasper_scorer = None
        if enable_qasper_f1:
            from .qasper_scorer import QasperF1Scorer
            self.qasper_scorer = QasperF1Scorer()
        self.faithfulness_scorer = None
        if enable_faithfulness:
            from .faithfulness_scorer import FaithfulnessScorer
            self.faithfulness_scorer = FaithfulnessScorer()

        # Load reranker once
        self.reranker = get_reranker()

    def run(self, configs: list[RetrievalConfig] | None = None) -> list[EvalResult]:
        if configs is None:
            configs = generate_all_configs(include_raptor=self.include_raptor)

        all_results = []
        total_configs = len(configs)

        # Checkpoint each result as it is produced so a crash or kill
        # partway through a long (LLM-billed) run doesn't lose everything.
        output_path = results_path(self.results_suffix)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        checkpoint = open(output_path, "w", encoding="utf-8")

        for config_idx, config in enumerate(configs, 1):
            print(f"\n[{config_idx}/{total_configs}] Running config: {config.name}")

            # Get the index for this chunking strategy
            index = self.indexes[config.chunking.value]
            retriever = make_retriever(
                index, reranker=self.reranker, qcond_config=self.qcond_config
            )

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

                # Optional QASPER F1/EM
                qf1 = -1.0
                qem = -1.0
                if self.qasper_scorer is not None:
                    gold_answers = question.get("gold_answers") or [question["gold_answer"]]
                    qm = self.qasper_scorer.score(gen_result.answer, gold_answers)
                    qf1, qem = qm.f1, qm.exact_match

                # Optional NLI faithfulness
                faith = -1.0
                faith_supp = -1
                faith_total = -1
                if self.faithfulness_scorer is not None:
                    fm = self.faithfulness_scorer.score(
                        gen_result.answer, retrieval_result.chunks
                    )
                    faith = fm.faithfulness
                    faith_supp = fm.supported_sentences
                    faith_total = fm.total_sentences

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
                    qasper_f1=qf1,
                    qasper_em=qem,
                    faithfulness=faith,
                    faithfulness_supported=faith_supp,
                    faithfulness_total=faith_total,
                )

                all_results.append(result)
                checkpoint.write(json.dumps(asdict(result)) + "\n")
                checkpoint.flush()

                print(
                    f"  [{config.name}] Question {q_idx}/{len(self.eval_set)} "
                    f"({question['id']}) ... "
                    f"gold_similarity={gold_similarity:.2f} "
                    f"fact_recall={key_fact_metrics.fact_recall:.2f}"
                )

        # Results were checkpointed row-by-row during the run.
        checkpoint.close()
        suffix = f"_{self.results_suffix}" if self.results_suffix else ""

        # Write a results manifest (reproducibility infra)
        try:
            from .manifest import write_manifest
            manifest_path = os.path.join(RESULTS_DIR, "raw", f"manifest{suffix}.json")
            write_manifest(
                results=all_results,
                output_path=manifest_path,
                config_metadata={
                    "include_raptor": self.include_raptor,
                    "enable_faithfulness": self.enable_faithfulness,
                    "enable_qasper_f1": self.enable_qasper_f1,
                },
            )
        except Exception as e:  # pragma: no cover
            print(f"  warning: failed to write manifest: {e}")

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
