"""Tests pour app/models/meta_labeling_pipeline.py — fix F1.

Vérifie :
1. filter_signals_by_meta_proba conserve la direction du signal primaire.
2. Il ne génère JAMAIS de signal hors des bornes primary_signals != 0.
3. La rétention dépend bien du threshold.
4. assert_train_test_distribution_alignment détecte la rupture F1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.models.meta_labeling_pipeline import (
    assert_train_test_distribution_alignment,
    filter_signals_by_meta_proba,
)


def _build_synthetic_data(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Construit un DataFrame OHLC + features + signaux primaires."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 1.1000 + rng.standard_normal(n).cumsum() * 0.001
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.0005,
            "Low": close - 0.0005,
            "Close": close,
        },
        index=times,
    )
    features = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        },
        index=times,
    )
    # Signaux primaires : 1 toutes les 10 barres, -1 toutes les 17, sinon 0
    primary = pd.Series(0, index=times, dtype=int)
    primary.iloc[::10] = 1
    primary.iloc[::17] = -1
    return df, features, primary


def _train_dummy_model(features: pd.DataFrame, seed: int = 7) -> RandomForestClassifier:
    """RF trivial : random labels pour avoir un predict_proba calibré."""
    rng = np.random.default_rng(seed)
    y = (rng.random(len(features)) > 0.5).astype(int)
    model = RandomForestClassifier(
        n_estimators=20, max_depth=3, random_state=seed, n_jobs=-1,
    )
    model.fit(features.values, y)
    return model


def test_filter_preserves_direction() -> None:
    """Le filtre ne change JAMAIS la direction d'un signal primaire."""
    df, features, primary = _build_synthetic_data()
    model = _train_dummy_model(features)

    filtered = filter_signals_by_meta_proba(
        df=df, primary_signals=primary, features=features,
        model=model, threshold=0.0,  # threshold=0 → tout passe
    )

    # Sur les signaux retenus, la direction doit être identique à primary
    for ts in filtered.index[filtered != 0]:
        assert filtered.loc[ts] == primary.loc[ts], (
            f"Direction modifiée à {ts}: primary={primary.loc[ts]}, "
            f"filtered={filtered.loc[ts]}"
        )


def test_filter_only_at_primary_signals() -> None:
    """Aucun signal généré aux barres où primary_signals == 0."""
    df, features, primary = _build_synthetic_data()
    model = _train_dummy_model(features)

    filtered = filter_signals_by_meta_proba(
        df=df, primary_signals=primary, features=features,
        model=model, threshold=0.0,
    )

    zero_primary_idx = primary[primary == 0].index
    assert (filtered.loc[zero_primary_idx] == 0).all(), (
        "Des signaux ont été générés hors des barres primaires (rupture F1)"
    )


def test_filter_retention_depends_on_threshold() -> None:
    """Plus le threshold est haut, moins de signaux passent."""
    df, features, primary = _build_synthetic_data()
    model = _train_dummy_model(features)

    filtered_low = filter_signals_by_meta_proba(
        df=df, primary_signals=primary, features=features,
        model=model, threshold=0.0,
    )
    filtered_high = filter_signals_by_meta_proba(
        df=df, primary_signals=primary, features=features,
        model=model, threshold=0.95,
    )

    n_low = (filtered_low != 0).sum()
    n_high = (filtered_high != 0).sum()
    assert n_low >= n_high, (
        f"Threshold haut doit filtrer plus : low={n_low}, high={n_high}"
    )


def test_assert_distribution_alignment_passes_on_consistent_rate() -> None:
    """Le sanity check passe si test_rate ≈ train_rate."""
    # 50/an en train (600 trades sur 12 ans), 50/an en test (100 trades sur 2 ans)
    assert_train_test_distribution_alignment(
        primary_train_count=600,
        primary_test_count=100,
        n_train_years=12.0,
        n_test_years=2.0,
        tolerance=3.0,
    )


def test_assert_distribution_alignment_detects_rupture() -> None:
    """Le sanity check lève si test_rate >> train_rate (rupture F1)."""
    # 50/an en train, 240/an en test → ratio 4.8× > tolerance 3×
    with pytest.raises(AssertionError, match="Distribution train/test rompue"):
        assert_train_test_distribution_alignment(
            primary_train_count=600,
            primary_test_count=480,
            n_train_years=12.0,
            n_test_years=2.0,
            tolerance=3.0,
        )


def test_filter_raises_on_model_without_predict_proba() -> None:
    """Vérifie le type-check de l'argument model."""
    df, features, primary = _build_synthetic_data()

    class FakeModel:
        pass

    with pytest.raises(TypeError, match="predict_proba"):
        filter_signals_by_meta_proba(
            df=df, primary_signals=primary, features=features,
            model=FakeModel(), threshold=0.5,
        )


def test_filter_handles_empty_primary() -> None:
    """Aucun signal primaire → série de zéros."""
    df, features, _ = _build_synthetic_data()
    primary_empty = pd.Series(0, index=df.index, dtype=int)
    model = _train_dummy_model(features)

    filtered = filter_signals_by_meta_proba(
        df=df, primary_signals=primary_empty, features=features,
        model=model, threshold=0.5,
    )
    assert (filtered == 0).all()
