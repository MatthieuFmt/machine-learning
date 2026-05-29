"""Reproductibilité. À appeler en haut de chaque script run_*.py (cf. Règle 12)."""
from __future__ import annotations

import os
import random

import numpy as np

GLOBAL_SEED = 42


def set_global_seeds(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
