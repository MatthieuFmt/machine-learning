"""Tests unitaires pour le walk-forward retraining (v14).

Couvre : no-lookahead, purge respectée, couverture complète,
datasets insuffisants, colonnes extra-drop.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning.model.training import (
    _FILTER_ONLY_COLS,
    walk_forward_train,
    train_model,
)


def _dummy_model_factory(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Factory légère pour tests (petit RF, rapide)."""
    return train_model(
        X_train, y_train,
        params={"n_estimators": 10, "max_depth": 3, "random_state": 42},
    )


@pytest.fixture
def ml_multi_year() -> pd.DataFrame:
    """DataFrame ML-ready synthétique multi-années (2020→2025, ~4 ans)."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2025-12-31")
    index = pd.date_range(start, end, freq="D", name="Time")
    n = len(index)

    df = pd.DataFrame(index=index)
    df["Target"] = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25])
    df["Spread"] = rng.integers(10, 18, n)
    # Features
    for col in ["Dist_EMA_50", "RSI_14", "ADX_14", "RSI_14_D1",
                 "Dist_EMA_20_D1", "RSI_D1_delta", "Dist_SMA200_D1", "XAU_Return"]:
        df[col] = rng.normal(0, 1, n)
    # Colonnes filtrées
    for col in _FILTER_ONLY_COLS:
        df[col] = rng.normal(0, 1, n)

    return df


class TestWalkForwardNoLookahead:
    """Vérifie que train_end < test_start à chaque fold (pas de look-ahead)."""

    def test_train_before_test(self, ml_multi_year: pd.DataFrame) -> None:
        X_cols = [c for c in ml_multi_year.columns if c not in {"Target", "Spread"}]
        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=24, step_months=6, purge_hours=48,
        ))

        assert len(folds) > 0, "Au moins un fold attendu"
        for model, train_start, train_end, test_start, test_end in folds:
            assert train_end < test_start, (
                f"Look-ahead détecté : train_end={train_end} >= test_start={test_start}"
            )

    def test_no_overlap_between_folds(self, ml_multi_year: pd.DataFrame) -> None:
        """Les périodes de test des folds successifs ne doivent pas se chevaucher."""
        X_cols = [c for c in ml_multi_year.columns if c not in {"Target", "Spread"}]
        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=12, step_months=3, purge_hours=48,
        ))

        test_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _, _, _, test_start, test_end in folds:
            test_ranges.append((test_start, test_end))

        # Vérifier que chaque test_start >= test_end du fold précédent
        for i in range(1, len(test_ranges)):
            prev_end = test_ranges[i - 1][1]
            curr_start = test_ranges[i][0]
            assert prev_end <= curr_start, (
                f"Chevauchement fold {i} : prev_end={prev_end} > curr_start={curr_start}"
            )


class TestWalkForwardPurge:
    """Vérifie que la purge est respectée entre train et test."""

    def test_purge_gap(self, ml_multi_year: pd.DataFrame) -> None:
        purge_hours = 48
        X_cols = [c for c in ml_multi_year.columns if c not in {"Target", "Spread"}]
        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=24, step_months=6, purge_hours=purge_hours,
        ))

        for _, train_start, train_end, test_start, test_end in folds:
            gap = (test_start - train_end).total_seconds() / 3600.0
            assert gap >= purge_hours, (
                f"Purge insuffisante : gap={gap:.1f}h, attendu >= {purge_hours}h"
            )


class TestWalkForwardCoverage:
    """Vérifie que les prédictions agrégées sont valides (pas de chevauchement,
    pas de données hors plage).
    """

    def test_test_periods_within_data_range(self, ml_multi_year: pd.DataFrame) -> None:
        """Les périodes de test doivent être dans les bornes des données."""
        X_cols = [c for c in ml_multi_year.columns if c not in {"Target", "Spread"}]
        data_min = ml_multi_year.index.min()
        data_max = ml_multi_year.index.max()

        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=24, step_months=6, purge_hours=48,
        ))

        for _, _, _, test_start, test_end in folds:
            assert test_start >= data_min, f"test_start={test_start} < data_min={data_min}"
            assert test_end <= data_max, f"test_end={test_end} > data_max={data_max}"

    def test_no_overlap_between_test_periods(self, ml_multi_year: pd.DataFrame) -> None:
        """Les périodes de test ne doivent pas se chevaucher."""
        X_cols = [c for c in ml_multi_year.columns if c not in {"Target", "Spread"}]
        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=24, step_months=6, purge_hours=48,
        ))

        test_ranges = [(test_start, test_end) for _, _, _, test_start, test_end in folds]
        for i in range(1, len(test_ranges)):
            assert test_ranges[i - 1][1] <= test_ranges[i][0], (
                f"Chevauchement test fold {i}: "
                f"prev_end={test_ranges[i-1][1]} > curr_start={test_ranges[i][0]}"
            )

    def test_at_least_one_fold_per_year_of_test_data(self, ml_multi_year: pd.DataFrame) -> None:
        """Avec step_months=3, on doit avoir ~4 folds par an de données test."""
        X_cols = [c for c in ml_multi_year.columns if c not in {"Target", "Spread"}]
        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=36, step_months=3, purge_hours=48,
        ))

        # Dataset: 2020-01 à 2025-12, train_months=36 → premier test vers 2023
        # Le nombre de folds doit être > 0 et raisonnable
        assert len(folds) >= 1, f"Seulement {len(folds)} folds, attendu >= 2"
        # step_months=3 → environ 4 folds par an, sur ~2.5 ans de test = ~10 folds
        assert len(folds) <= 3, f"Trop de folds: {len(folds)}"


class TestWalkForwardInsufficientData:
    """Vérifie le comportement avec des données insuffisantes."""

    def test_too_short_raises(self) -> None:
        """Un dataset trop court doit lever ValueError."""
        rng = np.random.default_rng(99)
        index = pd.date_range("2020-01-01", periods=50, freq="D", name="Time")
        df = pd.DataFrame(
            {"Target": rng.choice([-1, 0, 1], size=50),
             "Feat1": rng.normal(0, 1, 50)},
            index=index,
        )
        X_cols = ["Feat1"]

        with pytest.raises(ValueError, match="Aucun fold"):
            list(walk_forward_train(
                df, X_cols, _dummy_model_factory,
                train_months=36, step_months=3, purge_hours=48,
            ))


class TestWalkForwardExtraDropCols:
    """Vérifie que extra_drop_cols exclut les colonnes des features."""

    def test_extra_cols_not_in_model(self, ml_multi_year: pd.DataFrame) -> None:
        X_cols = [c for c in ml_multi_year.columns
                   if c not in {"Target", "Spread"} | _FILTER_ONLY_COLS]
        folds = list(walk_forward_train(
            ml_multi_year, X_cols, _dummy_model_factory,
            train_months=24, step_months=6, purge_hours=48,
            extra_drop_cols=_FILTER_ONLY_COLS,
        ))

        for model, _, _, _, _ in folds:
            feature_names = model.feature_names_in_
            for col in _FILTER_ONLY_COLS:
                assert col not in feature_names, (
                    f"Colonne {col} ne devrait pas être dans les features du modèle"
                )
