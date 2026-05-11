"""Pipeline de base abstrait pour tous les instruments.

Définit la séquence standard : clean → features → train → predict → backtest → report.
Chaque pipeline concret (EurUsdPipeline, BtcUsdPipeline) override les étapes spécifiques.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from learning_machine_learning.config.registry import ConfigRegistry


class BasePipeline(ABC):
    """Orchestrateur abstrait du pipeline ML/trading multi-actif.

    Chaque instrument concret hérite et fournit ses propres config/data/features.
    """

    def __init__(self, instrument_name: str) -> None:
        registry = ConfigRegistry()
        entry = registry.get(instrument_name)
        self.instrument = entry.instrument
        self.model_cfg = entry.model
        self.backtest_cfg = entry.backtest
        self.paths = entry.paths

    @abstractmethod
    def load_data(self) -> dict[str, Any]:
        """Charge les données brutes (H1, H4, D1, macro)."""

    @abstractmethod
    def build_features(self, data: dict[str, Any]) -> Any:
        """Construit le DataFrame ML-ready."""

    def train_model(self, ml_data: Any) -> Any:
        """Entraîne le modèle RandomForest."""
        from learning_machine_learning.model.training import (
            train_test_split_purge,
            train_model,
        )

        X_train, y_train, X_cols = train_test_split_purge(
            ml_data,
            train_end_year=self.model_cfg.train_end_year,
            purge_hours=self.model_cfg.purge_hours,
        )
        model = train_model(X_train, y_train, self.model_cfg.rf_params)
        return model, X_cols

    def evaluate_model(self, model: Any, ml_data: Any, X_cols: list[str]) -> dict:
        """Évalue le modèle sur val_year et test_year."""
        from learning_machine_learning.model.evaluation import (
            evaluate_model,
            feature_importance_impurity,
            feature_importance_permutation,
        )
        from learning_machine_learning.model.prediction import predict_oos

        results: dict[str, Any] = {}
        class_map = None

        for year in self.model_cfg.eval_years:
            preds_df, class_map = predict_oos(
                model, ml_data, eval_year=year, X_cols=X_cols, class_map=class_map,
            )
            results[year] = preds_df

        return results

    def run_backtest(
        self,
        predictions: Any,
        ml_data: Any,
        ohlcv_h1: Any,
    ) -> tuple[Any, dict]:
        """Exécute le backtest sur les prédictions."""
        from learning_machine_learning.backtest.simulator import simulate_trades
        from learning_machine_learning.backtest.sizing import weight_centered

        all_trades = {}
        all_metrics = {}

        for year, preds_df in predictions.items():
            # Joindre les prédictions avec OHLC H1 (simulate_trades a besoin de High/Low/Close)
            ohlc_cols = ["High", "Low", "Close"]
            ohlc_available = [c for c in ohlc_cols if c in ohlcv_h1.columns]
            if ohlc_available:
                year_ohlc = ohlcv_h1.loc[ohlcv_h1.index.year == year, ohlc_available]
                df_backtest = preds_df.join(year_ohlc, how="left")
            else:
                df_backtest = preds_df

            trades_df, n_signaux, n_filtres = simulate_trades(
                df=df_backtest,
                weight_func=weight_centered,
                tp_pips=self.backtest_cfg.tp_pips,
                sl_pips=self.backtest_cfg.sl_pips,
                window=self.backtest_cfg.window_hours,
                pip_size=self.instrument.pip_size,
            )
            all_trades[year] = trades_df

            from learning_machine_learning.backtest.metrics import compute_metrics

            year_data = ohlcv_h1[ohlcv_h1.index.year == year]
            if not year_data.empty:
                metrics = compute_metrics(
                    trades_df=trades_df,
                    annee=year,
                    df=year_data,
                    pip_value_eur=self.instrument.pip_value_eur,
                    initial_capital=self.backtest_cfg.initial_capital,
                    pip_size=self.instrument.pip_size,
                )
                all_metrics[year] = metrics

        return all_trades, all_metrics

    def run(self) -> dict[str, Any]:
        """Exécute le pipeline complet.

        Returns:
            Dict avec 'trades', 'metrics', 'predictions', 'model'.
        """
        data = self.load_data()
        ml_data = self.build_features(data)
        model, X_cols = self.train_model(ml_data)
        predictions = self.evaluate_model(model, ml_data, X_cols)
        trades, metrics = self.run_backtest(predictions, ml_data, data.get("h1"))

        return {
            "model": model,
            "predictions": predictions,
            "trades": trades,
            "metrics": metrics,
            "X_cols": X_cols,
        }
