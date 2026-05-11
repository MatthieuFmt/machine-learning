"""Tests unitaires pour model/prediction.py."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.model.prediction import predict_oos
from learning_machine_learning.model.training import train_test_split_purge, train_model


class TestPredictOos:
    """predict_oos — prédictions out-of-sample par année."""

    def test_basic_prediction(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        result, class_map = predict_oos(
            model, ml_ready_synthetic, eval_year=2022, X_cols=X_cols,
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(class_map, dict)
        assert "Close_Reel_Direction" in result.columns
        assert "Prediction_Modele" in result.columns
        assert "Confiance_Baisse_%" in result.columns
        assert "Confiance_Neutre_%" in result.columns
        assert "Confiance_Hausse_%" in result.columns

    def test_no_data_for_year(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        # Année hors plage des données synthétiques (2022 uniquement)
        with pytest.raises(ValueError):
            predict_oos(model, ml_ready_synthetic, eval_year=2099, X_cols=X_cols)

    def test_class_map_reused(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        # Premier appel: class_map = None
        result1, class_map1 = predict_oos(
            model, ml_ready_synthetic, eval_year=2022, X_cols=X_cols,
        )

        # Deuxième appel: réutilisation du class_map
        result2, class_map2 = predict_oos(
            model, ml_ready_synthetic, eval_year=2022, X_cols=X_cols,
            class_map=class_map1,
        )

        assert class_map1 == class_map2
        assert len(result1) == len(result2)

    def test_prediction_values_make_sense(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        result, _ = predict_oos(model, ml_ready_synthetic, eval_year=2022, X_cols=X_cols)

        # Vérifier que les prédictions sont dans {-1, 0, 1}
        preds = result["Prediction_Modele"].values
        assert set(preds).issubset({-1.0, 0.0, 1.0})

        # Les confiances sont entre 0 et 100
        conf_cols = ["Confiance_Baisse_%", "Confiance_Neutre_%", "Confiance_Hausse_%"]
        for col in conf_cols:
            assert (result[col] >= 0).all()
            assert (result[col] <= 100).all()

        # La somme des 3 confiances ≈ 100
        sum_conf = result["Confiance_Baisse_%"] + result["Confiance_Neutre_%"] + result["Confiance_Hausse_%"]
        np.testing.assert_array_almost_equal(sum_conf.values, 100.0, decimal=0)

    def test_spread_included(self, ml_ready_synthetic: pd.DataFrame) -> None:
        X_train, y_train, X_cols = train_test_split_purge(
            ml_ready_synthetic, train_end_year=2022, purge_hours=24,
        )
        if len(X_train) == 0:
            pytest.skip("Données d'entraînement vides")

        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}
        model = train_model(X_train, y_train, params)

        result, _ = predict_oos(model, ml_ready_synthetic, eval_year=2022, X_cols=X_cols)
        assert "Spread" in result.columns
