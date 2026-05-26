"""Centralized seed control for reproducibility."""

import os
import random

import numpy as np


DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
