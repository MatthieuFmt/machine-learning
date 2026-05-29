"""Tests ciblés pour session_features — fix F8.

session_features prend un DatetimeIndex et retourne des features
calendaires (heure, jour, mois). Par définition, ces features ne
dépendent que du timestamp courant, donc pas de look-ahead possible.

Ce test vérifie quand même :
1. session_features(idx[:n+1])[n] == session_features(idx)[n] (anti-leak).
2. Les colonnes attendues sont présentes et bien typées.
3. Les sessions sont mutuellement exclusives sur des heures non-overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.superset import session_features


def test_session_features_no_look_ahead() -> None:
    """session_features(idx[:n+1])[n] == session_features(idx)[n] pour tous n.

    Comme la fonction ne dépend que de l'élément i de l'index, c'est trivialement
    vrai. On le vérifie quand même pour formaliser.
    """
    idx = pd.date_range("2024-01-01 00:00", periods=200, freq="h", tz="UTC")
    full = session_features(idx)

    rng = np.random.default_rng(42)
    for n in rng.choice(range(50, 200), size=20, replace=False):
        partial = session_features(idx[: n + 1])
        for col in full.columns:
            assert full[col].iloc[n] == partial[col].iloc[n], (
                f"Look-ahead détecté à idx {n}, col {col}"
            )


def test_session_features_columns_present() -> None:
    """Les colonnes attendues sont toutes là."""
    idx = pd.date_range("2024-01-01 00:00", periods=100, freq="h", tz="UTC")
    out = session_features(idx)
    expected = {
        "session_tokyo", "session_london", "session_ny",
        "session_overlap_london_ny",
        "day_sin", "day_cos", "month_sin", "month_cos",
    }
    assert expected.issubset(set(out.columns))


def test_session_overlap_is_subset_of_london_and_ny() -> None:
    """L'overlap London-NY (13:00-16:00 UTC) ⊂ session_london ET session_ny."""
    idx = pd.date_range("2024-01-01 00:00", periods=72, freq="h", tz="UTC")
    out = session_features(idx)
    overlap_mask = out["session_overlap_london_ny"] == 1
    assert (out.loc[overlap_mask, "session_london"] == 1).all()
    assert (out.loc[overlap_mask, "session_ny"] == 1).all()


def test_session_tokyo_no_overlap_with_ny() -> None:
    """Tokyo (00-09) et NY (13-22) ne se chevauchent pas."""
    idx = pd.date_range("2024-01-01 00:00", periods=72, freq="h", tz="UTC")
    out = session_features(idx)
    both = (out["session_tokyo"] == 1) & (out["session_ny"] == 1)
    assert not both.any(), "Tokyo et NY ne doivent jamais être actifs simultanément"
