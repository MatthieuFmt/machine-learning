"""Tests unitaires pour le régresseur GBM et predict_oos_regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from learning_machine_learning.model.prediction import predict_oos_regression
from learning_machine_learning.model.training import train_regressor


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_synthetic_regression_data(
    n: int = 400, seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Génère des données synthétiques quotidiennes de mi-2019 à mi-2020."""
    rng = np.random.default_rng(seed)
    n_features = 5
    X = rng.standard_normal((n, n_features))
    true_coef = rng.uniform(-2, 2, n_features)
    y = X @ true_coef + rng.normal(0, 0.5, n)

    dates = pd.date_range("2019-06-01", periods=n, freq="D")
    df = pd.DataFrame(
        X, columns=[f"feat_{i}" for i in range(n_features)], index=dates,
    )
    df["Target"] = y
    return df, pd.Series(y, index=dates, name="Target")


# ── Tests ────────────────────────────────────────────────────────────────

class TestTrainRegressor:
    """Tests de train_regressor."""

    def test_train_regressor_basic(self) -> None:
        """HistGradientBoostingRegressor s'entraîne sans erreur."""
        df, y = _make_synthetic_regression_data()
        X = df.drop(columns=["Target"])
        params = {
            "max_iter": 50, "max_depth": 3, "min_samples_leaf": 5,
            "learning_rate": 0.1, "loss": "absolute_error", "random_state": 42,
        }
        model = train_regressor(X, y, params)
        assert isinstance(model, HistGradientBoostingRegressor)
        score = model.score(X, y)
        assert score > 0.0

    def test_default_params_work(self) -> None:
        """Les paramètres GBM par défaut de ModelConfig fonctionnent."""
        df, y = _make_synthetic_regression_data()
        X = df.drop(columns=["Target"])
        params = {
            "max_iter": 200, "max_depth": 5, "min_samples_leaf": 50,
            "learning_rate": 0.05, "loss": "absolute_error", "random_state": 42,
        }
        model = train_regressor(X, y, params)
        assert isinstance(model, HistGradientBoostingRegressor)

    def test_few_samples_fallback(self) -> None:
        """Très peu d'échantillons — ne doit pas crasher."""
        df, y = _make_synthetic_regression_data(n=30)
        X = df.drop(columns=["Target"])
        params = {
            "max_iter": 20, "max_depth": 2, "min_samples_leaf": 2,
            "learning_rate": 0.1, "loss": "absolute_error", "random_state": 1,
        }
        model = train_regressor(X, y, params)
        assert isinstance(model, HistGradientBoostingRegressor)


class TestPredictOosRegression:
    """Tests de predict_oos_regression."""

    def test_predict_oos_regression_shape(self) -> None:
        """Sortie contient Predicted_Return, pas de Confiance_*_%."""
        df, _ = _make_synthetic_regression_data(n=400)
        X_cols = [c for c in df.columns if c.startswith("feat_")]

        train_mask = df.index.year <= 2019
        X_train = df.loc[train_mask, X_cols]
        y_train = df.loc[train_mask, "Target"]
        assert len(X_train) > 0, "Aucune donnée d'entraînement"
        params = {"max_iter": 30, "max_depth": 3, "random_state": 42}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = predict_oos_regression(model, df, eval_year=2020, X_cols=X_cols)

        assert "Predicted_Return" in preds.columns
        assert "Target" in preds.columns
        assert "Confiance_Hausse_%" not in preds.columns
        assert "Confiance_Baisse_%" not in preds.columns
        assert "Confiance_Neutre_%" not in preds.columns
        assert len(preds) > 0

    def test_predict_oos_regression_no_future_leak(self) -> None:
        """Les prédictions ne portent que sur l'année eval_year."""
        df, _ = _make_synthetic_regression_data(n=400)
        X_cols = [c for c in df.columns if c.startswith("feat_")]

        train_mask = df.index.year <= 2019
        X_train = df.loc[train_mask, X_cols]
        y_train = df.loc[train_mask, "Target"]
        params = {"max_iter": 30, "max_depth": 3, "random_state": 42}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = predict_oos_regression(model, df, eval_year=2020, X_cols=X_cols)
        assert all(preds.index.year == 2020)
        assert len(preds) > 0

    def test_predict_oos_regression_no_nan_predictions(self) -> None:
        """Aucune prédiction NaN dans Predicted_Return."""
        df, _ = _make_synthetic_regression_data(n=400)
        X_cols = [c for c in df.columns if c.startswith("feat_")]

        train_mask = df.index.year <= 2019
        X_train = df.loc[train_mask, X_cols]
        y_train = df.loc[train_mask, "Target"]
        params = {"max_iter": 30, "max_depth": 3, "random_state": 42}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = predict_oos_regression(model, df, eval_year=2020, X_cols=X_cols)
        assert not preds["Predicted_Return"].isna().any()
        assert preds["Predicted_Return"].dtype == np.float64

    def test_spearman_positive_on_trend(self) -> None:
        """Spearman > 0 : corrélation positive entre prédictions OOS et cible."""
        n = 200
        rng = np.random.default_rng(42)
        dates = pd.date_range("2019-01-01", periods=n, freq="D")

        # Cible = feature bruitée → GBM doit capter le signal
        signal = np.linspace(0, 5, n)
        y = signal + rng.normal(0, 0.3, n)
        X = np.column_stack([y + rng.normal(0, 0.1, n) for _ in range(3)])
        df = pd.DataFrame(X, columns=["f0", "f1", "f2"], index=dates)
        df["Target"] = y
        X_cols = ["f0", "f1", "f2"]

        train = df.iloc[:150]
        test = df.iloc[150:]

        params = {"max_iter": 200, "max_depth": 4, "random_state": 42}
        model = HistGradientBoostingRegressor(**params)
        model.fit(train[X_cols], train["Target"])

        preds = model.predict(test[X_cols])
        actual = test["Target"].values

        from scipy.stats import spearmanr
        rho, _ = spearmanr(preds, actual)
        assert rho > 0, f"Spearman={rho:.3f} doit être > 0"
