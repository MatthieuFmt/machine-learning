"""Tests unitaires pour model/training.py."""

import pandas as pd
import pytest

from learning_machine_learning.model.training import train_test_split_purge, train_model


class TestTrainTestSplitPurge:
    """train_test_split_purge — split temporel avec purge anti-overlap."""

    def test_basic_split(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_cols, list)
        assert len(X_train) > 0
        assert len(y_train) == len(X_train)
        assert "Target" not in X_train.columns
        assert "Spread" not in X_train.columns
        assert y_train.name == "Target"

    def test_purge_excludes_recent(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, _, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        cutoff = pd.to_datetime("2023-01-01") - pd.Timedelta(hours=24)
        assert (X_train.index < cutoff).all()

    def test_no_train_before_data_raises(self, ml_ready_synthetic: pd.DataFrame) -> None:
        """Si cutoff est avant le début des données, ValueError."""
        with pytest.raises(ValueError):
            train_test_split_purge(
                ml_ready_synthetic, train_end_year=2021, purge_hours=24,
            )

    def test_no_target_in_features(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        assert "Target" not in X_cols
        assert "Spread" not in X_cols
        for col in X_cols:
            assert col in X_train.columns

    def test_empty_dataframe_raises(self) -> None:
        df = pd.DataFrame({"Target": [0, 1, -1], "Spread": [10, 11, 12]})
        # Pas de colonnes de features → X_train aura 0 colonnes, ce n'est pas interdit
        # Mais aucune donnée après 2021+purge car df n'a pas d'index datetime
        with pytest.raises(ValueError):
            train_test_split_purge(df, train_end_year=3000, purge_hours=24)

    def test_returns_correct_types(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_cols, list)
        assert all(isinstance(c, str) for c in X_cols)


class TestTrainModel:
    """train_model — entraînement du RandomForestClassifier."""

    def test_basic_train(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides avec ce cutoff")

        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": 42,
        }

        model = train_model(X_train, y_train, params)

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert set(model.classes_) == {-1.0, 0.0, 1.0}

    def test_model_reproduces_same_seed(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides avec ce cutoff")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model1 = train_model(X_train, y_train, params)
        model2 = train_model(X_train, y_train, params)

        # Mêmes prédictions
        preds1 = model1.predict(X_train)
        preds2 = model2.predict(X_train)
        assert (preds1 == preds2).all()

    def test_model_with_many_estimators(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, _ = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )

        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides avec ce cutoff")

        params = {"n_estimators": 50, "max_depth": 4, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)
        assert model.n_estimators == 50
