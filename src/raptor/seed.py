"""Centralized seed control for reproducibility."""

import random

import numpy as np


DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed the RNGs that actually influence RAPTOR's tree build.

    UMAP and GMM take their own explicit seeds (`umap_seed`/`gmm_seed`), but
    this covers any incidental use of the global `random`/`numpy.random`
    state. Does NOT set `PYTHONHASHSEED`: that env var only affects hash
    randomization for interpreters started *after* it's set, so assigning it
    at runtime here has no effect on the current process — node identity
    must not depend on it (see `RaptorNode`/`RaptorTreeBuilder`).
    """
    random.seed(seed)
    np.random.seed(seed)
