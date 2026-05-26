"""UMAP dimensionality reduction + GMM soft clustering with BIC selection.

Faithful to the RAPTOR paper:
- Reduce embeddings via UMAP (cosine metric, ~10 dims).
- Two-stage GMM: a global GMM partitions the corpus, then within each global
  cluster a local GMM finds fine-grained sub-clusters. Cluster count at each
  stage is chosen by BIC over a small grid.
- Soft assignment: a node belongs to every cluster whose posterior probability
  is at least 1/n_clusters.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

try:
    import umap  # umap-learn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "umap-learn is required for RAPTOR clustering. "
        "Install with: pip install umap-learn"
    ) from e


def _reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    seed: int,
) -> np.ndarray:
    """UMAP reduction with cosine metric. Returns shape (n, n_components)."""
    n = len(embeddings)
    n_components = min(n_components, max(2, n - 2))
    n_neighbors = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _bic_best_gmm(
    points: np.ndarray, max_k: int, seed: int
) -> GaussianMixture:
    """Fit GMMs for k ∈ [1, max_k] and return the one minimizing BIC."""
    n = len(points)
    max_k = min(max_k, max(1, n - 1))
    best_gmm = None
    best_bic = np.inf
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                random_state=seed,
                covariance_type="full",
                reg_covar=1e-4,
                max_iter=200,
            )
            gmm.fit(points)
            bic = gmm.bic(points)
        except Exception:
            continue
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    if best_gmm is None:
        # Fall back to a single cluster
        best_gmm = GaussianMixture(n_components=1, random_state=seed).fit(points)
    return best_gmm


def _soft_assign(gmm: GaussianMixture, points: np.ndarray) -> list[list[int]]:
    """Soft-assign each point to every cluster where p >= 1/k.

    Returns: clusters[cluster_index] -> list of point indices.
    """
    probs = gmm.predict_proba(points)  # (n, k)
    k = probs.shape[1]
    threshold = 1.0 / k if k > 0 else 1.0
    clusters: list[list[int]] = [[] for _ in range(k)]
    for i, row in enumerate(probs):
        assigned = np.where(row >= threshold)[0]
        if len(assigned) == 0:
            assigned = [int(np.argmax(row))]
        for c in assigned:
            clusters[int(c)].append(i)
    # Drop empty clusters
    return [c for c in clusters if c]


def cluster_embeddings(
    embeddings: np.ndarray,
    umap_seed: int = 42,
    gmm_seed: int = 42,
    umap_components_global: int = 10,
    umap_components_local: int = 10,
    umap_neighbors: int = 15,
    max_global_k: int = 50,
    max_local_k: int = 10,
) -> list[list[int]]:
    """Two-stage UMAP + GMM clustering. Returns a list of clusters,
    each cluster being a list of indices into `embeddings`.

    Indices may appear in multiple clusters (soft assignment).
    """
    n = len(embeddings)
    if n <= 2:
        return [list(range(n))]

    # ---- Stage 1: global ----
    global_reduced = _reduce_umap(
        embeddings,
        n_components=umap_components_global,
        n_neighbors=umap_neighbors,
        seed=umap_seed,
    )
    global_max_k = min(max_global_k, max(1, n // 2))
    global_gmm = _bic_best_gmm(global_reduced, max_k=global_max_k, seed=gmm_seed)
    global_clusters = _soft_assign(global_gmm, global_reduced)

    # ---- Stage 2: local within each global cluster ----
    final_clusters: list[list[int]] = []
    for gc in global_clusters:
        if len(gc) <= 2:
            final_clusters.append(gc)
            continue
        local_embs = embeddings[gc]
        # For tiny clusters, skip local subdivision
        if len(gc) < 10:
            final_clusters.append(gc)
            continue
        local_reduced = _reduce_umap(
            local_embs,
            n_components=min(umap_components_local, max(2, len(gc) - 2)),
            n_neighbors=umap_neighbors,
            seed=umap_seed,
        )
        local_max_k = min(max_local_k, max(1, len(gc) // 2))
        local_gmm = _bic_best_gmm(local_reduced, max_k=local_max_k, seed=gmm_seed)
        local_subclusters = _soft_assign(local_gmm, local_reduced)
        # Map local indices back to global embedding indices
        for sub in local_subclusters:
            final_clusters.append([gc[i] for i in sub])

    # Filter out empties (defensive)
    return [c for c in final_clusters if c]
