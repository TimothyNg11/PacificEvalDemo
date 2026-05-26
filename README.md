# RAG Benchmark

A benchmarking framework that evaluates how different retrieval strategies affect LLM answer quality. It indexes a corpus of synthetic company documents using multiple chunking strategies, retrieves context using multiple search strategies at varying top-k values, generates answers using an LLM, and scores the results against gold-standard answers.

## The Question This Answers

When building a RAG pipeline, which retrieval strategy should you use? The answer depends on the type of question being asked. This benchmark systematically evaluates **48 retrieval configurations** (4 chunking strategies × 4 search strategies × 3 top-k values) across 25 carefully designed questions to find out which strategies work best — and where they fail.

## Key Findings
Disclaimer: Results vary based on dataset; synthetic dataset was generated for purposes of project but findings may differ with real data.

- **More expensive strategies yield diminishing returns**: hybrid_rerank is the best search strategy (0.793 avg similarity) but only 1.3% better than plain hybrid at a cost of 785ms extra latency; similarly, k=10 uses 3.3x more tokens than k=3 for only a +0.047 similarity gain
- **Larger chunks and higher k inflate cost without proportional quality gains**: `fixed_512` averages 2,836 context tokens and 3,248ms latency — nearly 6x the tokens and 2x the latency of `paragraph` — for only a +0.048 similarity improvement
- **Answer quality has diminishing returns after ~1,200 context tokens** — beyond ~3,000 tokens there is no meaningful gain, suggesting retrieval precision matters far more than retrieval volume
- **Efficient configs rival the top performers**: `paragraph__vector__k10` (0.794 similarity, 868 tokens, 1,710ms) and `semantic__hybrid_rerank__k5` (0.794 similarity, 620 tokens, 2,893ms) approach the best score of 0.825 at a fraction of the cost
- **Wrong-source retrieval is the dominant failure mode** across all configurations — improving retrieval precision is the highest-leverage path to better answers

## Results

After running the benchmark, charts will be generated in `results/plots/`:

- **Quality vs. Latency** — Scatter plot showing the Pareto-optimal configurations
- **Strategy by Category** — Which search strategy is best for which question type
- **Chunking by Category** — Which chunking strategy is best for which question type
- **Top-K by Category** — How the number of retrieved chunks affects quality per question type
- **Failure Modes** — How different configs fail (wrong source, insufficient context, partial answers)
- **Context Tokens vs. Quality** — Diminishing returns from larger context windows

## How It Works

The benchmark evaluates retrieval along three axes:

1. **Chunking Strategy**: How documents are split into retrieval units
   - `fixed_256` — 256-token windows with 50-token overlap
   - `fixed_512` — 512-token windows with 100-token overlap
   - `semantic` — Split by sentence embedding similarity
   - `paragraph` — Split on paragraph boundaries

2. **Search Strategy**: How relevant chunks are found
   - `vector` — Cosine similarity on sentence-transformer embeddings
   - `bm25` — BM25 keyword search
   - `hybrid` — Reciprocal Rank Fusion of vector + BM25
   - `hybrid_rerank` — Hybrid + cross-encoder reranking

3. **Top-K**: How many chunks are retrieved (3, 5, or 10)

Questions are drawn from **5 categories** (25 total) designed to stress different retrieval weaknesses:

| Category | What it tests | Example |
|----------|--------------|---------|
| **Single-Doc Factual** | Answer lives in one document. Easy baseline — most strategies handle these. | "What deployment strategy does Meridian use?" |
| **Numerical Precision** | Answer requires retrieving a specific number. Tests whether chunking splits tables or figures away from their context. | "What was Q3 2024 total revenue and YoY growth?" |
| **Cross-Doc Synthesis** | Answer requires combining information from 2–4 different documents. Tests whether retrieval can surface all relevant pieces. | "How did the September outage affect engineering and the sales pipeline?" |
| **Terminology Mismatch** | Question uses different vocabulary than the source document (e.g., "turnover rate" vs. "attrition rate"). BM25 struggles here; vector search should succeed. | "What is Meridian's employee turnover rate?" |
| **Distractor-Heavy** | Multiple documents contain similar language (e.g., Q2 vs. Q3 earnings) but only one has the correct answer. Tests retrieval precision. | "What was enterprise revenue in Q3 2024, not Q2?" |

Scoring uses three methods (no paid API calls):
- **Gold Similarity**: Embedding cosine similarity between generated and gold answers
- **Key Fact Recall**: Deterministic check for specific facts in the answer
- **Retrieval Metrics**: Context precision, recall, and distractor rate

## RAPTOR (Sarthi et al., ICLR 2024) — Reimplementation + Contribution

The 4 baseline chunking strategies above all produce a flat list of chunks. **RAPTOR** builds a *recursive tree of LLM-generated summaries* over the corpus, then retrieves across multiple levels of abstraction. This benchmark contains a faithful reimplementation plus one original contribution.

### How RAPTOR works (this implementation, in [src/raptor/](src/raptor/))

1. **Leaves** — chunk every document into 100-token windows with no overlap ([chunk_raptor_100](src/chunkers.py)).
2. **Cluster** — UMAP (cosine, ~10 dims) reduce the leaf embeddings, then a two-stage Gaussian Mixture Model with BIC-selected cluster count produces soft-assigned clusters ([src/raptor/clustering.py](src/raptor/clustering.py)).
3. **Summarize** — each cluster is summarized by an LLM (local Ollama by default) into a single dense paragraph ([src/raptor/summarizer.py](src/raptor/summarizer.py)).
4. **Recurse** — the parent summaries become the new node set; repeat clustering + summarization until ≤1 cluster or `max_levels` is reached ([src/raptor/tree_builder.py](src/raptor/tree_builder.py)).
5. **Cache** — every summary is keyed on `(cluster_text, model, prompt_version)`; the full tree is keyed on every parameter that influences it. Rebuilds are free after the first run.

### Retrieval modes — two paper baselines plus one new

| Strategy | What it does | Notes |
|---|---|---|
| `raptor_tree` | Top-down descent: at each level, keep the top-k frontier nodes, descend into their children, union all picked nodes, rank by score. | Paper's "tree traversal" mode. |
| `raptor_collapsed` | Flatten every node (leaves + every summary at every level) into one pool, do a single top-k vector search. | Paper's winning mode in their Table 2. |
| `raptor_qcond` | **Original contribution.** Per-node decision policy: **terminate** when a node beats its children by more than `τ_term`; **single-branch** when child-score entropy is low (one child dominates); **multi-branch** otherwise. Returns the union of nodes where descent terminated. | No training. Hyperparameters in [`QCondConfig`](src/raptor/tree_retriever.py). |

The contribution targets the paper's weakest design choice — a fixed top-k descent at every level — which the paper itself flags as future work. `raptor_qcond` adapts its descent depth and branching factor per query, aiming to land on the **Pareto frontier** of `(retrieval_token_budget, answer_quality)`. If on QASPER it does not dominate the paper's two modes at matched budgets, that result will be reported honestly (no cherry-picking).

### Fully local, $0 spend

All RAPTOR LLM calls go through the same Ollama-backed `AnswerGenerator` used elsewhere in the repo. No paid APIs are required at any stage:

- **Summarization**: defaults to `qwen2.5:7b-instruct` (better summary quality than `llama3.2:3b`). Set via `OllamaConfig` or env vars on `LLM_PRESETS`.
- **Embeddings & rerankers**: sentence-transformers (already in `requirements.txt`).
- **QASPER F1 / Exact Match** ([src/qasper_scorer.py](src/qasper_scorer.py)): standard token-level F1 against gold answers — the metric the RAPTOR paper itself reports — no LLM judge needed.

### Running RAPTOR

```bash
# 1. Install dependencies (umap-learn + scikit-learn for RAPTOR, datasets for QASPER)
pip install -r requirements.txt

# 2. Pull the local summarizer (one-time, ~4.5GB)
ollama pull qwen2.5:7b-instruct
ollama pull llama3.2:3b   # smaller alternative used by the smoke test

# 3. Smoke test: builds a shallow tree on the existing 20-doc corpus,
#    runs all three RAPTOR retrieval modes against three sanity questions.
python scripts/smoke_raptor.py

# 4. Full benchmark on the existing Pacific corpus, baseline 48 configs + 9 RAPTOR = 57:
python scripts/run_benchmark.py --include-raptor

# 5. Just the RAPTOR configs:
python scripts/run_benchmark.py --configs \
  "raptor_100__raptor_tree__k5,raptor_100__raptor_collapsed__k5,raptor_100__raptor_qcond__k5"
```

The first tree build summarizes ~50–200 clusters and takes minutes; every subsequent run is served from cache.

### Running on QASPER (long-document corpus)

The hand-authored 20-doc corpus is short — RAPTOR shines on long documents. The repo includes a one-shot script that fetches the official QASPER dataset (AllenAI, NLP papers) and writes a corpus + eval set matching the existing schema.

```bash
# One-time: build a QASPER corpus + eval set (~30 papers, ~60 questions).
python scripts/build_qasper_evalset.py
# Outputs: data/corpus_qasper/*.md and data/eval_set_qasper.yaml

# Run the full RAPTOR + baseline benchmark on QASPER with extra scorers:
python scripts/run_benchmark.py \
    --include-raptor \
    --corpus-dir data/corpus_qasper \
    --eval-set data/eval_set_qasper.yaml \
    --qasper-f1 \
    --faithfulness \
    --results-suffix qasper

# Generate plots (including new Pareto frontier + RAPTOR-vs-baseline by category):
python scripts/generate_report.py --results-file results/raw/results_qasper.jsonl

# Sanity-check our implementation against the paper's directional claims:
python scripts/check_paper_faithfulness.py results/raw/results_qasper.jsonl
```

The paper-faithfulness check asserts: (1) `raptor_collapsed >= raptor_tree` on F1 (within tolerance, paper Table 2), and (2) the gap grows with question complexity (paper Table 3). It exits cleanly even when the assertion fails — the README is intended to report negative results honestly, not cherry-pick.

### Extra scorers and plots

- **`--qasper-f1`** — adds token-level F1 + EM against gold answers (the metric the RAPTOR paper reports on QASPER). Implemented in [src/qasper_scorer.py](src/qasper_scorer.py). Fully local, $0.
- **`--faithfulness`** — runs a local cross-encoder NLI model (`cross-encoder/nli-deberta-v3-base`) over each generated answer's sentences against the retrieved context. Outputs a 0–1 "answer supported by context" score per row. Implemented in [src/faithfulness_scorer.py](src/faithfulness_scorer.py).
- **New plots** ([src/analyzer.py](src/analyzer.py)): `pareto_frontier.png` (token budget vs quality, baseline vs RAPTOR modes), `raptor_vs_baseline_by_category.png` (where RAPTOR helps), `qasper_f1_by_strategy.png`, `faithfulness_by_strategy.png`. The new plots no-op when the relevant data isn't present, so they work on baseline-only runs too.

### Reproducibility

Every run writes [results/raw/manifest.json](results/raw/manifest.json) (or `manifest_{suffix}.json`) containing the git SHA, package versions, requirements.txt hash, machine info, and aggregate stats. Combined with the deterministic seeds in [src/raptor/seed.py](src/raptor/seed.py) and the corpus-content-hashed tree cache, results are reproducible across machines given the same lockfile.

CI runs the full test suite plus a config-count sanity check on every push to `main` ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

### Tests

```bash
python -m pytest tests/ -v
```

49 tests across 8 files cover: chunkers, baseline scorers (with the new `;`-separated source handling for RAPTOR), QASPER F1/EM, faithfulness scorer (with stubbed NLI model — no network), the cache + key derivation, the RAPTOR node adapter, both paper retrieval modes, and `raptor_qcond`'s "terminate at parent" behavior. One additional test on Gaussian-blob clustering is skipped when `umap-learn` isn't installed.

## Prerequisites

- **Python 3.10+**
- **An LLM provider** — either OpenAI (API key) or Ollama (local install)

## Setup

```bash
git clone https://github.com/TimothyNg11/PacificEvalDemo.git
cd PacificEvalDemo
pip install -r requirements.txt
```

### LLM Setup (pick one)

The scripts auto-detect which provider to use. If `OPENAI_API_KEY` is set, they use OpenAI. Otherwise they fall back to Ollama.

**Option A: OpenAI (fast, ~$0.50-2.00 for a full run)**

Set your API key before running any benchmark script:

```bash
# PowerShell
$env:OPENAI_API_KEY = "sk-proj-your-key-here"

# bash/zsh
export OPENAI_API_KEY="sk-proj-your-key-here"
```

**Option B: Ollama (free, but a lot slower)**

Ollama runs the LLM locally. The full benchmark takes 3-10 hours on CPU — use the quick benchmark for testing.

1. **Install Ollama** — download the desktop app from [ollama.com](https://ollama.com) and launch it. It runs in the background and serves the API automatically. No terminal commands needed.

   *Alternatively, on Linux/macOS you can install via CLI:*
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2:3b
   ollama serve  # keep running in a separate terminal
   ```

2. **Verify it's running** — the scripts connect to `http://localhost:11434`. If you see a connection error, make sure the Ollama app is open or `ollama serve` is running.

## Running the Benchmark

The first run will automatically build search indexes from the corpus (~30 seconds). Subsequent runs reuse the cached indexes.

### Quick Benchmark (5 evaluations, randomized)

Picks one random question, then tests it against 5 diverse retrieval configurations (varying chunking, search strategy, and top-k). Gives a direct apples-to-apples comparison on each run (~3-5 min with Ollama, ~30s with OpenAI).

```bash
python scripts/run_quick.py              # 5 random configs, 1 random question
python scripts/run_quick.py -n 10        # more configs
python scripts/run_quick.py -n 5 -s 42   # reproducible with a seed
python scripts/generate_report.py        # generate results
```

### Full Benchmark (1,200 evaluations)

All 48 configurations × 25 questions. Best with OpenAI (~20-40 min) or Ollama + GPU (~40-100 min).

```bash
python scripts/run_benchmark.py
python scripts/generate_report.py
```

### Debug a Single Configuration

Test one config on one question to inspect retrieval and generation step by step:

```bash
python scripts/run_single.py fixed_512__hybrid__k5 sf_001
```

### Run a Subset

```bash
# Specific configs
python scripts/run_benchmark.py --configs "fixed_256__vector__k3,semantic__hybrid_rerank__k5"

# Specific questions
python scripts/run_benchmark.py --questions "sf_001,np_001,cs_001"

# Force a specific LLM provider
python scripts/run_benchmark.py --llm openai
python scripts/run_benchmark.py --llm ollama
```

### Run Tests

```bash
python -m pytest tests/
```

### Clean Up

Remove all generated output (results, cached test indexes, Python bytecode):

```bash
python scripts/clean_results.py
```

## Next Steps

- **Reduce wrong-source retrieval** — Wrong-source is the dominant failure mode (~40–50% of results); adding metadata filtering, source-aware reranking, or query decomposition could improve precision
- **Query-dependent strategy routing** — Route terminology-mismatch and cross-doc questions to hybrid+rerank (where BM25 fails hardest) and simple factual/numerical questions to cheaper strategies like paragraph+vector
- **Optimize context budget** — Quality has diminishing returns around 1,200 tokens; capping context size or using smarter truncation could cut latency and cost without sacrificing answer quality
- **Benchmark across multiple LLMs** — Test whether a stronger generator model closes the gap between cheap retrieval configs and expensive ones, since ~50% of results are partial answers even with good retrieval

### RAPTOR follow-ups

All shipped in this repo — see the "Running on QASPER" and "Extra scorers and plots" sections above:
- ✅ QASPER long-document corpus builder ([scripts/build_qasper_evalset.py](scripts/build_qasper_evalset.py))
- ✅ Pareto frontier plot + RAPTOR-vs-baseline category breakdown ([src/analyzer.py](src/analyzer.py))
- ✅ QASPER F1/EM scorer ([src/qasper_scorer.py](src/qasper_scorer.py))
- ✅ Local NLI faithfulness scorer ([src/faithfulness_scorer.py](src/faithfulness_scorer.py))
- ✅ Paper-faithfulness assertion script ([scripts/check_paper_faithfulness.py](scripts/check_paper_faithfulness.py))
- ✅ Reproducibility manifest writer ([src/manifest.py](src/manifest.py))
- ✅ GitHub Actions CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))

Open work (would extend the contribution further):
- **Tree-depth and summarizer-size ablations** — sweep `max_levels ∈ {1,2,3,4}` and summarizer model (`llama3.2:3b` vs `qwen2.5:7b-instruct` vs `qwen2.5:14b-instruct`) to replicate the paper's Tables 2–3 directionally on QASPER.
- **Multi-Hop / Thematic / Aggregative question synthesis** — auto-generate cross-paper QASPER questions to stress where summary-aware retrieval should win biggest.
