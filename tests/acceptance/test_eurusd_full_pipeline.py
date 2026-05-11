"""Tests d'acceptation du pipeline EUR/USD complet.

Valide que le pipeline s'exécute sans erreur du début à la fin,
que les sorties respectent le format attendu, et que les métriques
de base sont dans les plages plausibles.
"""

from __future__ import annotations

import pandas as pd
import pytest

from learning_machine_learning.config.registry import ConfigEntry, ConfigRegistry


@pytest.fixture(scope="session")
def eurusd_config_entry() -> ConfigEntry:
    """Entry de configuration EUR/USD avec chemins réels."""
    registry = ConfigRegistry()
    return registry.get("EURUSD")


class TestConfigIntegration:
    """Vérifie que la configuration EUR/USD est cohérente."""

    def test_registry_returns_entry(self, eurusd_config_entry: ConfigEntry) -> None:
        """Le registry retourne une entrée valide."""
        assert isinstance(eurusd_config_entry, ConfigEntry)
        assert eurusd_config_entry.instrument is not None
        assert eurusd_config_entry.model is not None
        assert eurusd_config_entry.backtest is not None
        assert eurusd_config_entry.paths is not None

    def test_instrument_eurusd(self, eurusd_config_entry: ConfigEntry) -> None:
        """L'instrument est bien EUR/USD."""
        instr = eurusd_config_entry.instrument
        assert instr.name == "EURUSD"
        assert instr.pip_size == 0.0001
        assert instr.pip_value_eur > 0
        assert any("XAU" in m for m in instr.macro_instruments)
        assert any("CHF" in m for m in instr.macro_instruments)

    def test_model_params_reasonable(self, eurusd_config_entry: ConfigEntry) -> None:
        """Les paramètres RF sont dans les plages attendues."""
        cfg = eurusd_config_entry.model
        assert 10 <= cfg.rf_params["n_estimators"] <= 500
        assert 3 <= cfg.rf_params["max_depth"] <= 20
        assert cfg.rf_params["random_state"] is not None

    def test_backtest_params_reasonable(self, eurusd_config_entry: ConfigEntry) -> None:
        """Les paramètres de backtest sont cohérents."""
        cfg = eurusd_config_entry.backtest
        assert cfg.tp_pips > 0
        assert cfg.sl_pips > 0
        assert cfg.window_hours > 0
        assert cfg.initial_capital > 0

    def test_paths_exist(self, eurusd_config_entry: ConfigEntry) -> None:
        """Les répertoires de données et résultats existent."""
        paths = eurusd_config_entry.paths
        assert paths.clean.exists()
        assert paths.results.exists()


class TestPipelineNoCrash:
    """Test que le pipeline EUR/USD s'exécute sans erreur fatale."""

    def test_pipeline_instantiation(self):
        """Le pipeline s'instancie sans erreur."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline
        pipeline = EurUsdPipeline()
        assert pipeline.instrument.name == "EURUSD"

    def test_load_data_returns_dict(self):
        """Le chargement des données retourne un dict."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline
        pipeline = EurUsdPipeline()
        data = pipeline.load_data()
        assert isinstance(data, dict)
        assert "h1" in data
        assert isinstance(data["h1"], pd.DataFrame)

    def test_build_features_returns_dataframe(self):
        """La construction des features retourne un DataFrame."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline
        pipeline = EurUsdPipeline()
        data = pipeline.load_data()
        ml = pipeline.build_features(data)
        assert isinstance(ml, pd.DataFrame)
        assert "Target" in ml.columns
        assert len(ml) > 0

    def test_train_model_returns_model_and_cols(self):
        """L'entraînement retourne un modèle et la liste des colonnes."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline
        from sklearn.ensemble import RandomForestClassifier

        pipeline = EurUsdPipeline()
        data = pipeline.load_data()
        ml = pipeline.build_features(data)
        model, X_cols = pipeline.train_model(ml)
        assert isinstance(model, RandomForestClassifier)
        assert isinstance(X_cols, list)
        assert len(X_cols) > 0

    def test_evaluate_model_returns_predictions(self):
        """L'évaluation retourne un dict de prédictions par année."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline

        pipeline = EurUsdPipeline()
        data = pipeline.load_data()
        ml = pipeline.build_features(data)
        model, X_cols = pipeline.train_model(ml)
        predictions = pipeline.evaluate_model(model, ml, X_cols)
        assert isinstance(predictions, dict)
        for year, preds_df in predictions.items():
            assert isinstance(preds_df, pd.DataFrame)
            assert "Prediction_Modele" in preds_df.columns

    def test_full_pipeline_run(self):
        """Le pipeline complet s'exécute sans erreur."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline

        pipeline = EurUsdPipeline()
        result = pipeline.run()

        assert "model" in result
        assert "predictions" in result
        assert "trades" in result
        assert "metrics" in result
        assert "X_cols" in result

        # Les trades doivent être un dict année → DataFrame
        for year, trades_df in result["trades"].items():
            assert isinstance(trades_df, pd.DataFrame)
            if len(trades_df) > 0:
                assert "Pips_Nets" in trades_df.columns
                assert "result" in trades_df.columns

    def test_full_pipeline_metrics_plausible(self):
        """Les métriques agrégées sont dans des plages plausibles."""
        from learning_machine_learning.pipelines.eurusd import EurUsdPipeline

        pipeline = EurUsdPipeline()
        result = pipeline.run()

        for year, metrics in result["metrics"].items():
            assert "win_rate" in metrics
            assert "sharpe" in metrics
            assert "dd" in metrics
            assert metrics["win_rate"] >= 0.0
            assert metrics["dd"] <= 0.0  # drawdown ≤ 0 par définition
