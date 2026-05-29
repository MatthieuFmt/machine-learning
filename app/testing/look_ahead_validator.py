"""Hooks anti-look-ahead à utiliser dans tous les modules de features."""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd


def assert_no_look_ahead(
    feature_fn: Callable,
    series: pd.Series | pd.DataFrame,
    n_samples: int = 100,
    seed: int = 42,
) -> None:
    """Vérifie que feature_fn(series[:n])[-1] == feature_fn(series)[n-1]
    pour 100 indices aléatoires."""
    rng = np.random.default_rng(seed)
    full = feature_fn(series)
    start = len(series) // 2
    if len(series) - 1 - start < n_samples:
        n_samples = max(1, len(series) - 1 - start)
    indices = rng.choice(range(start, len(series) - 1), n_samples, replace=False)

    for n in indices:
        truncated = feature_fn(series.iloc[: n + 1] if hasattr(series, "iloc") else series[: n + 1])
        full_val = _at(full, n)
        trunc_val = _at(truncated, n)
        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        if not np.isclose(full_val, trunc_val, rtol=1e-9, equal_nan=True):
            raise AssertionError(
                f"Look-ahead at idx {n}: full={full_val} vs truncated={trunc_val} "
                f"(feature_fn={feature_fn.__name__})"
            )


def _at(out, n: int):
    """Récupère la valeur scalaire à l'index positionnel n (Series ou DataFrame)."""
    if isinstance(out, pd.DataFrame):
        return out.iloc[n].sum()  # signature stable même multi-colonnes
    return out.iloc[n] if hasattr(out, "iloc") else out[n]


def look_ahead_safe(fn: Callable) -> Callable:
    """Décorateur de marquage. Toutes les fonctions de features doivent l'utiliser.
    Le test pytest `test_indicators_are_marked_safe` vérifie la présence du marqueur."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._look_ahead_safe = True  # type: ignore[attr-defined]
    return wrapper
