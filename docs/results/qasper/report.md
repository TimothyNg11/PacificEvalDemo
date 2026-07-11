# RAG Benchmark Results Report

## Summary

- **Best overall config**: `fixed_512__hybrid__k5` (avg similarity: 0.283)
- **Worst overall config**: `raptor_100__raptor_qcond__k5` (avg similarity: 0.219)

## Key Findings

- Best search strategy overall: **hybrid** (avg similarity: 0.277)
- Reranking improves quality by **-0.2%** but adds **1482ms** latency
- Diminishing returns: quality plateaus after ~**1257** context tokens

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
| fixed_512__hybrid__k5 | 0.283 | 0.750 | 0.447 | 0.700 | 3250 | 2467 |
| fixed_512__hybrid_rerank__k5 | 0.279 | 0.789 | 0.443 | 0.683 | 4956 | 2494 |
| fixed_512__hybrid_rerank__k3 | 0.274 | 0.738 | 0.506 | 0.617 | 3703 | 1476 |
| fixed_256__hybrid__k5 | 0.272 | 0.758 | 0.447 | 0.650 | 2682 | 1257 |
| raptor_100__raptor_tree__k10 | 0.270 | 0.771 | 0.298 | 0.750 | 2854 | 2195 |
| fixed_512__vector__k10 | 0.267 | 0.785 | 0.317 | 0.667 | 3365 | 4975 |
| raptor_100__raptor_collapsed__k10 | 0.265 | 0.771 | 0.397 | 0.700 | 2484 | 1070 |
| paragraph__vector__k10 | 0.251 | 0.750 | 0.375 | 0.717 | 2494 | 1113 |
| raptor_100__raptor_tree__k5 | 0.247 | 0.742 | 0.434 | 0.550 | 2185 | 559 |
| raptor_100__raptor_tree__k3 | 0.244 | 0.725 | 0.448 | 0.533 | 1944 | 334 |
| raptor_100__raptor_collapsed__k3 | 0.243 | 0.713 | 0.451 | 0.583 | 1997 | 318 |
| raptor_100__raptor_collapsed__k5 | 0.240 | 0.754 | 0.429 | 0.633 | 2179 | 530 |
| raptor_100__raptor_qcond__k10 | 0.226 | 0.721 | 0.126 | 0.517 | 2495 | 1048 |
| raptor_100__raptor_qcond__k3 | 0.222 | 0.696 | 0.312 | 0.433 | 1991 | 311 |
| raptor_100__raptor_qcond__k5 | 0.219 | 0.729 | 0.218 | 0.450 | 2057 | 523 |
