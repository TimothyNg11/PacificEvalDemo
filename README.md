# RAG Benchmark

A benchmarking framework that evaluates how different retrieval strategies affect LLM answer quality. It indexes a corpus of synthetic company documents using multiple chunking strategies, retrieves context using multiple search strategies at varying top-k values, generates answers using an LLM, and scores the results against gold-standard answers.

## The Question This Answers

When building a RAG pipeline, which retrieval strategy should you use? The answer depends on the type of question being asked. This benchmark systematically evaluates **48 retrieval configurations** (4 chunking strategies × 4 search strategies × 3 top-k values) across 25 carefully designed questions to find out which strategies work best — and where they fail.

## Key Findings

All numbers below are from the committed runs in [docs/results/](docs/results/) (generator: `claude-haiku-4-5`; 1,425 synthetic evaluations across 57 configs, 900 QASPER evaluations across 15 configs — reproducibility manifests included). Results vary by dataset; the synthetic corpus is hand-authored, QASPER is real scientific papers.

**Baselines (synthetic corpus):**
- **More expensive strategies yield diminishing returns**: hybrid_rerank is the best search strategy (0.801 avg similarity) but only ~1.6% better than plain hybrid at ~1.3s extra latency per query
- **The best config** is `fixed_512__hybrid_rerank__k5` (0.826 similarity, 2,470 ctx tokens); cheaper configs like `fixed_256__hybrid__k5` reach 0.811 with half the tokens
- **Retrieval precision matters more than volume** — quality plateaus well before the largest context sizes

**RAPTOR (see the [Results section](#results-what-we-found) for the full story):**
- **The paper's headline claim reproduces**: `raptor_collapsed ≥ raptor_tree` on QASPER F1, and the gap moves the paper's way on complex questions (both faithfulness checks PASS)
- **RAPTOR is token-efficient on long documents**: on QASPER, `raptor_collapsed__k10` matches the best flat baselines' F1 at ~43% of their context budget (1,071 vs ~2,500 tokens)
- **RAPTOR summaries bridge vocabulary gaps**: on terminology-mismatch questions RAPTOR beats every flat baseline family (0.71–0.72 vs 0.678 similarity)
- **The original contribution (`raptor_qcond`) was a negative result, then repaired to parity**: v1 underperformed both paper modes on both corpora; a calibrated v2 (beam width was the binding constraint, not termination) matches collapsed search on the held-out QASPER set at a slightly lower token budget — full protocol and tables in [docs/results/qcond_v2/](docs/results/qcond_v2/comparison.md)

## Results

Committed results from the headline runs live in [docs/results/synthetic/](docs/results/synthetic/) and [docs/results/qasper/](docs/results/qasper/) (plots, summary tables, reports, and reproducibility manifests). Running the benchmark yourself regenerates the same charts under `results/plots/`:

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

The contribution targets the paper's weakest design choice — a fixed top-k descent at every level — which the paper itself flags as future work. `raptor_qcond` adapts its descent depth and branching factor per query, aiming to land on the **Pareto frontier** of `(retrieval_token_budget, answer_quality)`. We committed up front to reporting the result honestly if it failed to dominate the paper's modes — **and it did fail; see [Results](#results-what-we-found)**.

### Results: what we found

Both experiments were run end-to-end with `claude-haiku-4-5` as generator and summarizer; every plot, summary table, and reproducibility manifest is committed under [docs/results/](docs/results/) ([synthetic](docs/results/synthetic/report.md) · [QASPER](docs/results/qasper/report.md)).

**1. The paper's claims reproduce.** `scripts/check_paper_faithfulness.py` on the QASPER run passes both directional assertions: `raptor_collapsed ≥ raptor_tree` on token-level F1 (gap −0.002, within tolerance of the paper's Table 2 finding), and the collapsed-vs-tree gap moves in the paper's direction as question complexity rises (−0.006 on simple categories → +0.003 on complex).

**2. RAPTOR's value on long documents is token efficiency.** On QASPER (real NLP papers), `raptor_collapsed__k10` and `raptor_tree__k10` tie for second place on F1 (0.063) — statistically level with the best flat baseline (`fixed_512__hybrid__k5`, 0.065) — but collapsed does it at **1,071 context tokens vs ~2,500** for the top baselines. On the short synthetic corpus, flat retrieval wins outright (best RAPTOR config ranks 19/57): there is little abstraction hierarchy to exploit in 1–2 page documents.

**3. Where summaries help: vocabulary gaps.** On terminology-mismatch questions (query wording ≠ document wording), RAPTOR's paraphrased summaries beat every flat baseline family (tree 0.720 / collapsed 0.709 vs baselines 0.678 avg similarity). Where they hurt: numerical precision and single-doc factual lookups, where summarization dilutes exact figures.

**4. The contribution was a negative result — v1 — and was then repaired to parity (v2).** As first evaluated, `raptor_qcond` ranked last among RAPTOR modes on **both** corpora (synthetic: ranks 55–57/57; QASPER: bottom three of 15): its context recall (0.43–0.52 on QASPER) trailed collapsed (0.58–0.70). We then implemented the three hypothesized fixes as config knobs and calibrated them — see [docs/results/qcond_v2/](docs/results/qcond_v2/comparison.md). The diagnostic overturned the original hypothesis: early termination wasn't the binding constraint — **beam width was**. v1's `k_branch=2` descent explored ≤8 of ~150 leaves and returned only 3–8 candidates at k=10; the three termination mechanisms added no recall once the beam widened. The calibrated fix (`k_branch=5`, chosen on the synthetic set only, QASPER held out) transferred to the held-out set: qcond v2 improves every metric at every k (QASPER F1 +27% at k=10, recall 0.517→0.700 at equal tokens) and reaches **parity with collapsed on the 30-paper QASPER set** — statistically confirmed by a paired per-question analysis (95% CI on the F1 delta: [−0.004, +0.005]; [per-query analysis](docs/results/qcond_v2/per_query_analysis.md)). A pre-registered **scale-up to 250 papers (13,437 leaves)** then settled the question ([scaleup.md](docs/results/qcond_v2/scaleup.md)): the gap to collapsed reopens significantly (paired F1 CI [−0.0076, −0.0004]), and the tree only reaches **depth 2** — RAPTOR's ~40:1 clustering fanout means depth grows like log₄₀(N), so the "deep trees" that adaptive traversal targets barely exist at any practical scale. The contribution's final form: a calibrated negative result with a complete mechanism story, plus one advantage that survives — search-effort efficiency.

**Where qcond could pay off.** qcond examines only ~27% of the nodes collapsed search scores, while retaining 87% of its recall (and ~93% of its answer F1). With embedding dot-products that saving is worthless — a flat scan of 14K nodes is one free matrix multiply. But production RAG systems routinely use **cross-encoder rerankers** (this repo's `hybrid_rerank` included), and agentic systems increasingly score candidates with LLM calls — regimes where every node scored has real latency or dollar cost. There, a traversal that prunes 73% of the scoring work for a ~7% relative quality cost is a legitimate trade, with qcond acting as the candidate-scoping stage in front of an expensive scorer. The honest caveat: production pipelines usually shortlist via cheap ANN search before the cross-encoder, and qcond-as-scoper vs. ANN-shortlist-plus-rerank was not benchmarked here — that head-to-head is the natural follow-up (see Next Steps). The honest conclusion stands: per-query adaptive traversal does not *beat* collapsed search at matched budgets; it now matches it. The committed v1 rows in docs/results/{synthetic,qasper} used `k_branch=2`.

**Limitations.** Absolute F1 (~0.06) is far below the paper's reported numbers: our generator answers in verbose prose while QASPER golds are terse extractive spans, and token-overlap F1 punishes that mismatch — rankings are comparable, absolutes are not. Similarly, the strict per-sentence NLI faithfulness score is deflated for all configs; use it comparatively. The QASPER run covers all 9 RAPTOR configs plus the 6 strongest baselines from the synthetic run (15 of 57) to bound API cost. The generator is Haiku-class; a stronger generator may close gaps between retrieval configs.

### LLM providers and cost

Answer generation and RAPTOR summarization go through one provider-agnostic interface ([src/generator.py](src/generator.py)):

- **Anthropic** (`claude-haiku-4-5`, used for the committed results; the full experiment suite cost ≈ $10)
- **OpenAI** (`gpt-4o-mini`)
- **Ollama** (free, local; `llama3.2:3b` default)

Everything else — embeddings, reranking, QASPER F1/EM, NLI faithfulness, all retrieval scoring — runs locally at $0.

### Running RAPTOR

```bash
# 1. Install dependencies (umap-learn + scikit-learn for RAPTOR, datasets for QASPER)
pip install -r requirements.txt

# 2. Full benchmark on the synthetic corpus, baseline 48 configs + 9 RAPTOR = 57
#    (uses Anthropic/OpenAI if a key is set, else local Ollama):
python scripts/run_benchmark.py --include-raptor

# 3. Just the RAPTOR configs:
python scripts/run_benchmark.py --configs \
  "raptor_100__raptor_tree__k5,raptor_100__raptor_collapsed__k5,raptor_100__raptor_qcond__k5"

# Optional, Ollama-only smoke test (pull llama3.2:3b first): builds a shallow
# tree and runs all three RAPTOR retrieval modes against sanity questions.
python scripts/smoke_raptor.py
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

Every run writes `results/raw/manifest{_suffix}.json` containing the git SHA, package versions, requirements.txt hash, machine info, and aggregate stats — the manifests for the committed headline runs are in [docs/results/synthetic/manifest.json](docs/results/synthetic/manifest.json) and [docs/results/qasper/manifest.json](docs/results/qasper/manifest.json). Combined with the deterministic seeds in [src/raptor/seed.py](src/raptor/seed.py), deterministic node identity, and the corpus-content-hashed tree and index caches, results are reproducible across machines given the same lockfile.

CI runs the full test suite plus a config-count sanity check on every push to `main` ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

### Tests

```bash
python -m pytest tests/ -v
```

81 tests across 10 files cover: chunkers, retrieval scorers (including fractional multi-source credit for RAPTOR summary nodes), QASPER F1/EM, the faithfulness scorer (stubbed NLI model — no network), cache + key derivation, deterministic node identity, both paper retrieval modes, `raptor_qcond`'s terminate/branch behavior (including the dangling-child guard), tiny-input clustering, cross-corpus index isolation, the `.env` loader, provider auto-detection, and both LLM chat paths (stubbed — no network). Clustering tests are skipped when `umap-learn` isn't installed.

## Prerequisites

- **Python 3.10+**
- **An LLM provider** — Anthropic (API key), OpenAI (API key), or Ollama (local install)

## Setup

```bash
git clone https://github.com/TimothyNg11/PacificEvalDemo.git
cd PacificEvalDemo
pip install -r requirements.txt
```

### LLM Setup (pick one)

The scripts auto-detect the provider: `ANTHROPIC_API_KEY` → Anthropic, else `OPENAI_API_KEY` → OpenAI, else local Ollama. Keys can also live in a repo-root `.env` file (gitignored, loaded automatically, never overrides real env vars). Force a provider with `--llm anthropic|openai|ollama`.

**Option A: Anthropic (used for the committed results; full 57-config run ≈ $4-5 on claude-haiku-4-5)**

```bash
# .env file (recommended)
echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' > .env

# or PowerShell / bash
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Option B: OpenAI (~$0.50-2.00 for a full run on gpt-4o-mini)**

```bash
# PowerShell
$env:OPENAI_API_KEY = "sk-proj-your-key-here"

# bash/zsh
export OPENAI_API_KEY="sk-proj-your-key-here"
```

**Option C: Ollama (free, but a lot slower)**

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
python scripts/run_benchmark.py --llm anthropic
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

Open work (informed by the results above):
- ✅ **Fix `raptor_qcond`** — done; see [docs/results/qcond_v2/](docs/results/qcond_v2/comparison.md). All three candidate mechanisms were implemented and calibrated with a free retrieval-only sweep ([scripts/sweep_qcond.py](scripts/sweep_qcond.py)); the diagnostic showed beam width (`k_branch`), not termination, was the binding constraint. v2 reaches parity with collapsed on 30-paper QASPER.
- ✅ **Scale-up to deeper trees** — done; see [scaleup.md](docs/results/qcond_v2/scaleup.md). Answered negatively on both pre-registered hypotheses: at 250 papers the qcond-vs-collapsed gap reopens significantly, and RAPTOR trees plateau at depth 2 (~40:1 clustering fanout), so the deep-tree regime the termination mechanisms target is unreachable at practical corpus sizes. qcond's surviving advantage is search effort: ~27% of collapsed's node scorings for 87% of its recall.
- **qcond as a scoping stage for expensive scorers** — the untested follow-up suggested by the efficiency result: benchmark qcond-scoped cross-encoder scoring against the standard ANN-shortlist + cross-encoder rerank pipeline at matched quality, measuring scorer invocations and wall-clock. This is the head-to-head that would tell whether qcond's 73% scoring reduction survives contact with the production-standard alternative.
- **Tree-depth and summarizer-size ablations** — sweep `max_levels ∈ {1,2,3,4}` and summarizer model strength to replicate the paper's Tables 2–3 directionally on QASPER.
- **Short-answer generation prompt for QASPER** — QASPER golds are terse extractive spans; a concise-answer prompt would make absolute F1 comparable to the paper's reported numbers rather than only rankings.
- **Multi-Hop / Thematic / Aggregative question synthesis** — auto-generate cross-paper QASPER questions to stress where summary-aware retrieval should win biggest.
