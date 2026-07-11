# RAG Benchmark Results Report

## Summary

- **Best overall config**: `fixed_512__hybrid_rerank__k5` (avg similarity: 0.826)
- **Worst overall config**: `raptor_100__raptor_qcond__k5` (avg similarity: 0.683)

## Key Findings

- Best search strategy overall: **hybrid_rerank** (avg similarity: 0.801)
- Reranking improves quality by **1.6%** but adds **1329ms** latency
- Diminishing returns: quality plateaus after ~**245** context tokens

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
| fixed_512__hybrid_rerank__k5 | 0.826 | 0.826 | 0.376 | 0.930 | 4351 | 2470 |
| fixed_512__hybrid__k5 | 0.823 | 0.738 | 0.352 | 0.873 | 2845 | 2352 |
| fixed_512__hybrid_rerank__k3 | 0.821 | 0.763 | 0.480 | 0.860 | 3469 | 1499 |
| fixed_512__vector__k10 | 0.820 | 0.751 | 0.236 | 0.947 | 3335 | 4652 |
| fixed_512__hybrid_rerank__k10 | 0.817 | 0.834 | 0.236 | 1.000 | 6617 | 4860 |
| fixed_256__hybrid__k10 | 0.816 | 0.786 | 0.280 | 0.947 | 2848 | 2414 |
| fixed_256__hybrid_rerank__k10 | 0.814 | 0.747 | 0.292 | 0.940 | 4903 | 2465 |
| fixed_256__vector__k10 | 0.812 | 0.759 | 0.296 | 0.940 | 3026 | 2431 |
| fixed_256__hybrid__k5 | 0.811 | 0.700 | 0.392 | 0.887 | 2464 | 1202 |
| fixed_512__bm25__k10 | 0.811 | 0.799 | 0.200 | 0.870 | 2956 | 4493 |
| fixed_512__hybrid__k10 | 0.809 | 0.814 | 0.228 | 0.947 | 3341 | 4596 |
| fixed_512__vector__k5 | 0.807 | 0.723 | 0.360 | 0.917 | 2946 | 2398 |
| fixed_256__bm25__k10 | 0.807 | 0.770 | 0.228 | 0.870 | 2856 | 2439 |
| fixed_256__hybrid_rerank__k3 | 0.806 | 0.622 | 0.453 | 0.807 | 2434 | 729 |
| semantic__hybrid_rerank__k10 | 0.805 | 0.591 | 0.300 | 0.900 | 5629 | 1369 |
| fixed_256__hybrid_rerank__k5 | 0.804 | 0.682 | 0.368 | 0.920 | 3177 | 1214 |
| paragraph__hybrid_rerank__k10 | 0.800 | 0.629 | 0.320 | 0.990 | 3520 | 860 |
| fixed_512__hybrid__k3 | 0.799 | 0.683 | 0.453 | 0.823 | 2424 | 1472 |
| raptor_100__raptor_collapsed__k10 | 0.799 | 0.727 | 0.354 | 0.940 | 2494 | 1428 |
| fixed_512__vector__k3 | 0.796 | 0.610 | 0.440 | 0.763 | 2856 | 1464 |
| fixed_512__bm25__k5 | 0.795 | 0.718 | 0.272 | 0.737 | 2711 | 2351 |
| paragraph__hybrid_rerank__k5 | 0.793 | 0.567 | 0.384 | 0.880 | 2594 | 410 |
| paragraph__hybrid__k10 | 0.789 | 0.681 | 0.324 | 0.960 | 2092 | 843 |
| semantic__hybrid_rerank__k5 | 0.789 | 0.503 | 0.376 | 0.840 | 3350 | 620 |
| fixed_256__hybrid__k3 | 0.786 | 0.663 | 0.467 | 0.747 | 2255 | 718 |
| paragraph__hybrid_rerank__k3 | 0.786 | 0.515 | 0.520 | 0.787 | 2072 | 245 |
| paragraph__vector__k10 | 0.785 | 0.613 | 0.312 | 0.990 | 2166 | 868 |
| semantic__hybrid__k10 | 0.783 | 0.699 | 0.332 | 0.980 | 2489 | 1318 |
| fixed_256__bm25__k3 | 0.782 | 0.553 | 0.373 | 0.687 | 2179 | 741 |
| semantic__vector__k10 | 0.781 | 0.613 | 0.276 | 0.930 | 2666 | 1232 |
| semantic__bm25__k10 | 0.776 | 0.589 | 0.224 | 0.870 | 2310 | 1413 |
| semantic__hybrid__k5 | 0.776 | 0.575 | 0.392 | 0.790 | 2209 | 683 |
| raptor_100__raptor_tree__k10 | 0.776 | 0.609 | 0.242 | 0.960 | 2865 | 2483 |
| fixed_256__vector__k5 | 0.776 | 0.602 | 0.360 | 0.757 | 2609 | 1194 |
| fixed_256__bm25__k5 | 0.770 | 0.595 | 0.336 | 0.777 | 2407 | 1225 |
| raptor_100__raptor_collapsed__k5 | 0.769 | 0.519 | 0.401 | 0.830 | 2000 | 748 |
| semantic__vector__k5 | 0.768 | 0.465 | 0.376 | 0.857 | 2015 | 571 |
| paragraph__hybrid__k5 | 0.765 | 0.561 | 0.416 | 0.850 | 1946 | 398 |
| paragraph__bm25__k10 | 0.765 | 0.509 | 0.240 | 0.840 | 1989 | 804 |
| paragraph__vector__k5 | 0.760 | 0.464 | 0.392 | 0.920 | 2071 | 405 |
| fixed_256__vector__k3 | 0.755 | 0.467 | 0.427 | 0.657 | 2285 | 704 |
| raptor_100__raptor_tree__k5 | 0.753 | 0.437 | 0.369 | 0.720 | 2255 | 748 |
| paragraph__hybrid__k3 | 0.753 | 0.473 | 0.467 | 0.737 | 1838 | 230 |
| semantic__hybrid_rerank__k3 | 0.751 | 0.447 | 0.440 | 0.767 | 2572 | 391 |
| semantic__hybrid__k3 | 0.750 | 0.439 | 0.493 | 0.717 | 1800 | 375 |
| semantic__bm25__k5 | 0.748 | 0.393 | 0.312 | 0.700 | 2120 | 662 |
| paragraph__bm25__k5 | 0.745 | 0.484 | 0.312 | 0.687 | 2535 | 388 |
| fixed_512__bm25__k3 | 0.741 | 0.443 | 0.320 | 0.537 | 2129 | 1413 |
| raptor_100__raptor_collapsed__k3 | 0.737 | 0.426 | 0.447 | 0.750 | 1901 | 452 |
| raptor_100__raptor_tree__k3 | 0.737 | 0.378 | 0.407 | 0.710 | 1752 | 461 |
| semantic__vector__k3 | 0.732 | 0.380 | 0.467 | 0.737 | 1913 | 369 |
| paragraph__vector__k3 | 0.731 | 0.443 | 0.453 | 0.767 | 2439 | 235 |
| semantic__bm25__k3 | 0.719 | 0.361 | 0.360 | 0.557 | 1783 | 407 |
| paragraph__bm25__k3 | 0.718 | 0.417 | 0.360 | 0.637 | 1540 | 223 |
| raptor_100__raptor_qcond__k10 | 0.706 | 0.348 | 0.168 | 0.607 | 2133 | 778 |
| raptor_100__raptor_qcond__k3 | 0.693 | 0.231 | 0.268 | 0.487 | 1901 | 353 |
| raptor_100__raptor_qcond__k5 | 0.683 | 0.303 | 0.207 | 0.607 | 1891 | 576 |
