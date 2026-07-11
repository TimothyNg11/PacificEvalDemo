# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG benchmark plus a faithful RAPTOR (Sarthi et al., ICLR 2024) reimplementation with one original contribution: query-conditional tree traversal (`raptor_qcond` in `src/raptor/tree_retriever.py`). The benchmark evaluates 48 flat-retrieval configurations (4 chunking × 4 search × 3 top-k) — 57 with `--include-raptor` — over a synthetic "Meridian Technologies" corpus, and optionally over a QASPER-derived corpus for comparison against the paper's published claims.

## Commands

```bash
pip install -r requirements.txt          # setup (Python 3.10+)

python -m pytest tests/                  # run all tests
python -m pytest tests/test_raptor.py -k test_name    # run a single test

python scripts/run_quick.py              # 5 random configs × 1 random question (-n N, -s SEED)
python scripts/run_benchmark.py --include-raptor      # full benchmark incl. 9 RAPTOR configs
python scripts/run_benchmark.py --configs "raptor_100__raptor_qcond__k5" --questions "sf_001"
python scripts/run_single.py fixed_512__hybrid__k5 sf_001   # debug one config/question step by step
python scripts/smoke_raptor.py           # RAPTOR end-to-end smoke (requires local Ollama)

python scripts/build_qasper_evalset.py   # one-time: QASPER corpus + eval set (30 papers / 60 questions)
python scripts/run_benchmark.py --include-raptor --corpus-dir data/corpus_qasper \
    --eval-set data/eval_set_qasper.yaml --qasper-f1 --faithfulness --results-suffix qasper
python scripts/check_paper_faithfulness.py results/raw/results_qasper.jsonl

python scripts/generate_report.py        # plots + CSV + report.md from results/raw/results.jsonl
python scripts/generate_report.py --results-file results/raw/results_qasper.jsonl
python scripts/clean_results.py          # remove results/, test indexes, __pycache__
```

Run everything from the repo root — scripts `sys.path.insert` relative to themselves and `src/config.py` uses repo-relative paths.

**LLM provider**: `--llm auto` (default) picks Anthropic if `ANTHROPIC_API_KEY` is set, else OpenAI if `OPENAI_API_KEY`, else local Ollama. Force with `--llm anthropic|openai|ollama`. `config.py` loads a repo-root `.env` (gitignored) at import, without overriding real env vars. The Anthropic path uses the official `anthropic` SDK (`claude-haiku-4-5`); Ollama/OpenAI share the `openai` SDK. All providers are reached through `generator.resolve_llm_client()`, which returns a uniform `chat(system, user, max_tokens)` wrapper.

## Architecture

Flat-retrieval pipeline, orchestrated by `runner.BenchmarkRunner`:

1. **`chunkers.py`** — splits corpus docs into `Chunk`s (fixed_256, fixed_512, semantic, paragraph, raptor_100); owns the shared embedding-model singleton `get_embedding_model()` and `split_sentences()`
2. **`indexer.py`** — one index per chunking strategy; flat strategies get a `CorpusIndex` (ChromaDB vector + BM25 over the same chunks), `raptor_100` gets a `RaptorIndex` built by `raptor/tree_builder.py`
3. **`retrievers.py`** — `make_retriever()` routes `CorpusIndex` → `Retriever` (vector/bm25/hybrid RRF/hybrid_rerank) and `RaptorIndex` → `RaptorRetriever` (reranker applies only to flat strategies)
4. **`generator.py`** — provider-agnostic answer generation (see LLM provider above)
5. **`scorers.py`** — deterministic scorers. `RetrievalScorer` gives *fractional* per-chunk precision credit (`|sources∩gold|/|sources|`) so multi-source RAPTOR summary nodes aren't counted as fully relevant — this keeps RAPTOR-vs-baseline comparisons unbiased. Optional: `qasper_scorer.py` (token F1/EM, max over `gold_answers` references), `faithfulness_scorer.py` (local NLI entailment)
6. **`runner.py`** — loops configs × questions, writes `EvalResult` rows to `results_path(suffix)` (`results/raw/results{_suffix}.jsonl`) plus a reproducibility manifest (`manifest.py`)
7. **`analyzer.py`** — plots (incl. `pareto_frontier`, `raptor_vs_baseline_by_category`, QASPER F1/faithfulness which no-op on baseline-only data), summary CSV, report.md

**RAPTOR** (`src/raptor/`): 100-token leaves → UMAP+GMM clustering with BIC-selected k and fixed 0.1 soft-assignment posterior threshold (paper-faithful; `soft_assign_threshold` in `RaptorBuildConfig`) → LLM cluster summaries (disk-cached by `(text, model, prompt_version)`) → recurse. Three retrieval modes: `raptor_tree` (paper's traversal), `raptor_collapsed` (paper's Table 2 winner), `raptor_qcond` (contribution: per-node terminate/single-branch/multi-branch from node score + child-score entropy, `QCondConfig`). Node ids and chunk indices are deterministic; trees are cached in `data/raptor_cache/` keyed on every build parameter.

Config names follow `{chunking}__{search}__k{top_k}`; parse/validate only via `RetrievalConfig.from_name()` (raises on invalid mixes like `raptor_100__vector`). RAPTOR chunks carry structured `source_files` provenance — never parse the `;`-joined `source_file` display string.

**Eval sets**: `data/eval_set.yaml` (25 questions, 5 categories) and generated `data/eval_set_qasper.yaml` (`gold_answers` holds all annotator references; F1 is max-over-references).

## Gotchas

- **Index/tree caching**: ChromaDB collections (`data/chroma_indexes/`) build only when empty; RAPTOR trees cache in `data/raptor_cache/`. Changing the corpus or chunkers without clearing these silently reuses stale indexes (tree cache keys include a corpus hash, Chroma's do not).
- QASPER loads via the HF Hub's `refs/convert/parquet` branch — `datasets>=4` cannot load the script-based `allenai/qasper` directly.
- Embedding/reranker/NLI models download from HuggingFace on first use; they're lazy module-level singletons.
- On Windows, importing `src.analyzer` standalone can segfault at interpreter teardown (torch/pyarrow/matplotlib DLL conflict); real `generate_report.py` runs are unaffected (exit 0, all outputs written).
- `results/`, `data/chroma_indexes/`, and `.env` are gitignored — never commit generated output or keys.

## Working Guidelines

Bias toward simplicity and surgical changes: no speculative features or configurability, touch only what the task requires, match existing style. State assumptions and surface tradeoffs before non-trivial changes. New behavior gets a test; keep `pytest tests/` green. Report failures honestly — this repo's README commits to reporting negative experimental results rather than cherry-picking.
