# RAG Benchmark

A benchmarking framework that evaluates how different retrieval strategies affect LLM answer quality. It indexes a corpus of synthetic company documents using multiple chunking strategies, retrieves context using multiple search strategies at varying top-k values, generates answers using an LLM, and scores the results against gold-standard answers.

## The Question This Answers

When building a RAG pipeline, which retrieval strategy should you use? The answer depends on the type of question being asked. This benchmark systematically evaluates **48 retrieval configurations** (4 chunking strategies × 4 search strategies × 3 top-k values) across 25 carefully designed questions to find out which strategies work best — and where they fail.

## Key Findings

*Run the benchmark to populate this section with your own results. Example findings:*

- Hybrid search + reranking consistently outperforms pure vector search on multi-document questions, at an additional latency cost
- BM25 fails on terminology mismatch questions where the query uses different vocabulary than the source documents
- Answer quality plateaus after a certain context token threshold — retrieving more adds cost without improving answers
- Semantic chunking outperforms fixed-size chunking on questions requiring numerical precision

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
python scripts/generate_report.py
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

- **Query-dependent strategy selection** — Route simple factual questions to fast strategies (BM25, low top-k) and complex synthesis questions to thorough ones (hybrid+rerank, high top-k)
- **Benchmark across multiple LLMs** — Test if retrieval strategy matters more than model choice by comparing llama3.2:3b vs. larger models
- **Adaptive chunking** — Use document structure (headers, lists, tables) to create semantically meaningful chunks rather than fixed-size windows
