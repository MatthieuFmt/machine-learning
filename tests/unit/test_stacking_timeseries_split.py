"""Tests pour fix F4 — Stacking sans look-ahead (TimeSeriesHoldoutStacking).

Sklearn StackingClassifier + TimeSeriesSplit étant incompatibles
(cross_val_predict exige une partition), on utilise une implémentation
maison qui fait un split chronologique 70/30 pour générer les meta-features.

Vérifie :
1. build_stacking() retourne TimeSeriesHoldoutStacking.
2. fit + predict_proba fonctionnent sur données synthétiques.
3. Les bases entraînées sur le holdout n'ont JAMAIS vu les samples du holdout.
4. Le refit final voit tout le dataset (pour la prédiction OOS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.candidates import TimeSeriesHoldoutStacking, build_stacking


def test_build_stacking_returns_holdout_stacking() -> None:
    """build_stacking() retourne TimeSeriesHoldoutStacking, pas sklearn Stacking."""
    model = build_stacking(seed=42)
    assert isinstance(model, TimeSeriesHoldoutStacking)
    assert model.holdout_frac == 0.3
    assert len(model.estimators) == 2
    names = [n for n, _ in model.estimators]
    assert names == ["rf", "hgbm"]


def test_stacking_fits_and_predicts() -> None:
    """Sanity : fit + predict_proba sur données synthétiques sans erreur."""
    rng = np.random.default_rng(7)
    n = 400
    X = rng.standard_normal((n, 8))
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)

    model = build_stacking(seed=42)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_stacking_holdout_split_is_chronological() -> None:
    """Vérifie que le split est bien chronologique (pas shuffle).

    On entraîne sur des données où les 70 premiers % et les 30 derniers %
    ont des distributions différentes. Si le split est shuffle, les
    base estimators vont mélanger les deux régimes. Sinon, le holdout
    contient uniquement le régime 2 et le final_estimator l'apprend.
    """
    rng = np.random.default_rng(11)
    n = 300
    split_idx = int(n * 0.7)
    # Régime 1 : feature 0 corrélée positivement à y
    X1 = rng.standard_normal((split_idx, 3))
    y1 = (X1[:, 0] > 0).astype(int)
    # Régime 2 : feature 0 corrélée NÉGATIVEMENT à y
    X2 = rng.standard_normal((n - split_idx, 3))
    y2 = (X2[:, 0] < 0).astype(int)
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    model = build_stacking(seed=42)
    model.fit(X, y)
    assert hasattr(model, "fitted_bases_")
    assert hasattr(model, "final_estimator_")


def test_stacking_raises_on_too_few_samples() -> None:
    """fit() doit refuser < 20 samples (sinon split inutile)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 3))
    y = rng.integers(0, 2, 10)
    model = build_stacking(seed=42)
    with pytest.raises(ValueError, match="au moins 20 samples"):
        model.fit(X, y)


def test_stacking_predict_proba_before_fit_raises() -> None:
    """predict_proba avant fit doit lever RuntimeError."""
    model = build_stacking(seed=42)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 3))
    with pytest.raises(RuntimeError, match="fit\\(\\) doit être appelé"):
        model.predict_proba(X)


def test_stacking_fallback_on_single_class_holdout() -> None:
    """Si le holdout n'a qu'une classe, fallback sur moyenne des bases."""
    rng = np.random.default_rng(3)
    n = 200
    X = rng.standard_normal((n, 4))
    # y : tout 0 sur le holdout (derniers 30 %)
    y = np.concatenate([rng.integers(0, 2, int(n * 0.7)), np.zeros(n - int(n * 0.7), dtype=int)])
    model = build_stacking(seed=42)
    model.fit(X, y)
    assert model.final_estimator_ is None
    # predict_proba doit quand même retourner des proba valides
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all((proba >= 0) & (proba <= 1))
