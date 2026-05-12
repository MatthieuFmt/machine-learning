"""Pipeline de base abstrait pour tous les instruments.

Définit la séquence standard : clean → features → train → predict → backtest → report.
Chaque pipeline concret (EurUsdPipeline, BtcUsdPipeline) override les étapes spécifiques.

Support également le walk-forward retraining (v14).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from learning_machine_learning.config.registry import ConfigRegistry
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


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
            _FILTER_ONLY_COLS,
            train_test_split_purge,
            train_model,
        )

        X_train, y_train, X_cols = train_test_split_purge(
            ml_data,
            train_end_year=self.model_cfg.train_end_year,
            purge_hours=self.model_cfg.purge_hours,
            extra_drop_cols=_FILTER_ONLY_COLS,
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
        from learning_machine_learning.backtest.filters import (
            FilterPipeline,
            MomentumFilter,
            VolFilter,
            SessionFilter,
        )
        from learning_machine_learning.backtest.simulator import simulate_trades
        from learning_machine_learning.backtest.sizing import weight_centered

        # Construire le pipeline de filtres selon la config backtest
        filters: list = []
        cfg = self.backtest_cfg
        if cfg.use_momentum_filter:
            filters.append(
                MomentumFilter(threshold=cfg.momentum_filter_threshold)
            )
        if cfg.use_vol_filter:
            filters.append(
                VolFilter(
                    window=cfg.vol_filter_window,
                    multiplier=cfg.vol_filter_multiplier,
                )
            )
        if cfg.use_session_filter:
            filters.append(
                SessionFilter(
                    exclude_start=cfg.session_exclude_start,
                    exclude_end=cfg.session_exclude_end,
                )
            )
        filter_pipeline = FilterPipeline(filters) if filters else None

        all_trades = {}
        all_metrics = {}

        # Colonnes requises par les filtres de régime (MomentumFilter, VolFilter)
        FILTER_COLS: tuple[str, ...] = ("Dist_SMA200_D1", "ATR_Norm", "RSI_D1_delta")

        for year, preds_df in predictions.items():
            # Joindre les prédictions avec OHLC H1 (simulate_trades a besoin de High/Low/Close)
            ohlc_cols = ["High", "Low", "Close"]
            ohlc_available = [c for c in ohlc_cols if c in ohlcv_h1.columns]
            if ohlc_available:
                year_ohlc = ohlcv_h1.loc[ohlcv_h1.index.year == year, ohlc_available]
                df_backtest = preds_df.join(year_ohlc, how="left")
            else:
                df_backtest = preds_df

            # Injecter les colonnes requises par les filtres depuis ml_data
            filter_cols_present = [c for c in FILTER_COLS if c in ml_data.columns]
            if filter_cols_present:
                year_filter = ml_data.loc[ml_data.index.year == year, filter_cols_present]
                df_backtest = df_backtest.join(year_filter, how="left")

            trades_df, n_signaux, n_filtres = simulate_trades(
                df=df_backtest,
                weight_func=weight_centered,
                tp_pips=cfg.tp_pips,
                sl_pips=cfg.sl_pips,
                window=cfg.window_hours,
                pip_size=self.instrument.pip_size,
                seuil_confiance=cfg.confidence_threshold,
                commission_pips=cfg.commission_pips,
                slippage_pips=cfg.slippage_pips,
                filter_pipeline=filter_pipeline,
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

    def run_walk_forward(
        self,
        ml_data: Any,
        data: dict[str, Any],
        train_months: int = 36,
        step_months: int = 3,
    ) -> dict[str, Any]:
        """Exécute le pipeline en walk-forward retraining (v14).

        À chaque fold :
        1. Réentraîne le modèle sur [fold_start, fold_start + train_months).
        2. Prédit sur [train_end + purge, train_end + purge + step_months).
        3. Agrège toutes les prédictions OOS en une série unique.

        Args:
            ml_data: DataFrame ML-ready complet (index datetime).
            data: Dict des données brutes (doit contenir 'h1').
            train_months: Durée de la fenêtre d'entraînement en mois.
            step_months: Pas d'avancement entre les folds en mois.

        Returns:
            Dict avec :
            - 'predictions_agg': DataFrame des prédictions OOS agrégées.
            - 'trades_agg': DataFrame des trades simulés.
            - 'metrics_agg': Dict des métriques globales.
            - 'fold_count': Nombre de folds générés.
            - 'X_cols': Colonnes de features utilisées.
        """
        from learning_machine_learning.model.training import (
            _FILTER_ONLY_COLS,
            train_model,
            walk_forward_train,
        )
        from learning_machine_learning.model.prediction import predict_oos

        # 1. Déterminer X_cols (comme dans train_model)
        drop_cols = {"Target", "Spread"} | _FILTER_ONLY_COLS
        X_cols = [c for c in ml_data.columns if c not in drop_cols]

        # 2. Factory de modèle utilisant les hyperparamètres de la config
        rf_params = self.model_cfg.rf_params

        def model_factory(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
            return train_model(X_train, y_train, rf_params)

        # 3. Walk-forward : itérer sur les folds
        all_predictions: list[pd.DataFrame] = []
        fold_info: list[dict[str, Any]] = []

        folds = walk_forward_train(
            df=ml_data,
            X_cols=X_cols,
            model_factory=model_factory,
            train_months=train_months,
            step_months=step_months,
            purge_hours=self.model_cfg.purge_hours,
            extra_drop_cols=_FILTER_ONLY_COLS,
        )

        class_map = None
        for fold_idx, (model, train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
            # Prédire sur la période de test de ce fold
            test_mask = (ml_data.index >= test_start) & (ml_data.index < test_end)
            test_slice = ml_data.loc[test_mask]

            if test_slice.empty:
                continue

            # Construire un sous-DataFrame pour predict_oos
            X_test = test_slice[X_cols]
            preds_array = model.predict(X_test)
            probas = model.predict_proba(X_test)

            if class_map is None:
                class_map = {float(cls): int(idx) for idx, cls in enumerate(model.classes_)}

            def _get_col(class_key: float) -> "np.ndarray":
                import numpy as np
                if class_key in class_map:
                    return probas[:, class_map[class_key]]
                return np.zeros(len(probas), dtype=np.float64)

            fold_preds = pd.DataFrame(
                {
                    "Close_Reel_Direction": test_slice["Target"] if "Target" in test_slice.columns else np.nan,
                    "Prediction_Modele": preds_array,
                    "Confiance_Baisse_%": np.round(_get_col(-1.0) * 100, 2),
                    "Confiance_Neutre_%": np.round(_get_col(0.0) * 100, 2),
                    "Confiance_Hausse_%": np.round(_get_col(1.0) * 100, 2),
                },
                index=test_slice.index,
            )

            if "Spread" in test_slice.columns:
                fold_preds["Spread"] = test_slice["Spread"]

            all_predictions.append(fold_preds)
            fold_info.append({
                "fold": fold_idx,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "n_train": int(((ml_data.index >= train_start) & (ml_data.index < train_end)).sum()),
                "n_test": len(test_slice),
            })

        # 4. Agréger les prédictions
        if not all_predictions:
            raise ValueError("Aucune prédiction OOS générée par le walk-forward.")

        predictions_agg = pd.concat(all_predictions).sort_index()
        # Dédupliquer (les folds ne se chevauchent pas, mais par précaution)
        predictions_agg = predictions_agg[~predictions_agg.index.duplicated(keep="first")]

        # 5. Backtest sur les prédictions agrégées
        # Structurer comme evaluate_model pour run_backtest (dict année -> DataFrame)
        predictions_by_year: dict[int, pd.DataFrame] = {}
        for year in predictions_agg.index.year.unique():
            predictions_by_year[int(year)] = predictions_agg[predictions_agg.index.year == year]

        trades_agg, metrics_agg = self.run_backtest(
            predictions_by_year, ml_data, data.get("h1"),
        )

        n_folds = len(all_predictions)
        logger.info(
            "Walk-forward terminé : %d folds, %d prédictions agrégées, %d années couvertes.",
            n_folds, len(predictions_agg), len(predictions_by_year),
        )

        return {
            "predictions_agg": predictions_agg,
            "trades_agg": trades_agg,
            "metrics_agg": metrics_agg,
            "fold_count": n_folds,
            "fold_info": fold_info,
            "X_cols": X_cols,
        }

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
