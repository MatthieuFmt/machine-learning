"""Tests unitaires pour model/evaluation.py."""

import pandas as pd
import pytest

from learning_machine_learning.model.evaluation import (
    evaluate_model,
    feature_importance_impurity,
    feature_importance_permutation,
)
from learning_machine_learning.model.training import train_test_split_purge, train_model


class TestEvaluateModel:
    """evaluate_model — accuracy et classification_report."""

    def test_evaluate_returns_dict(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        result = evaluate_model(model, X_train, y_train)

        assert "accuracy" in result
        assert "report" in result
        assert isinstance(result["accuracy"], float)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert isinstance(result["report"], str)

    def test_perfect_model_on_train(self, ml_ready_synthetic: pd.DataFrame) -> None:
        """Sur l'entraînement, l'accuracy doit être bonne (> hasard)."""
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        # Overfit volontaire pour le test
        params = {"n_estimators": 100, "max_depth": 20, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        result = evaluate_model(model, X_train, y_train)
        # Sur l'entraînement, un RF profond doit avoir accuracy > 1/3 (hasard à 3 classes)
        assert result["accuracy"] > 0.4


class TestFeatureImportanceImpurity:
    """feature_importance_impurity — importances Gini."""

    def test_returns_dataframe(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        fi = feature_importance_impurity(model, X_cols)

        assert isinstance(fi, pd.DataFrame)
        assert "Indicateur" in fi.columns
        assert "Impurity_%" in fi.columns
        assert len(fi) == len(X_cols)
        assert fi["Impurity_%"].sum() == pytest.approx(100.0, abs=0.1)

    def test_sorted_descending(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        fi = feature_importance_impurity(model, X_cols)
        assert fi["Impurity_%"].is_monotonic_decreasing


class TestFeatureImportancePermutation:
    """feature_importance_permutation — permutation importance."""

    def test_returns_dataframe(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        fi = feature_importance_permutation(
            model, X_train, y_train, n_repeats=3, random_state=42, n_jobs=1,
        )

        assert isinstance(fi, pd.DataFrame)
        assert "Indicateur" in fi.columns
        assert "Permutation_mean" in fi.columns
        assert "Permutation_std" in fi.columns
        assert len(fi) == X_train.shape[1]

    def test_empty_x_returns_none(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        empty_X = X_train.iloc[:0]
        fi = feature_importance_permutation(
            model, empty_X, y_train.iloc[:0], n_repeats=3, random_state=42,
        )
        assert fi is None

    def test_sorted_descending(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        fi = feature_importance_permutation(
            model, X_train, y_train, n_repeats=3, random_state=42, n_jobs=1,
        )
        assert fi["Permutation_mean"].is_monotonic_decreasing
