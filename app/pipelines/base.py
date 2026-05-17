"""Pipeline de base abstrait pour tous les instruments.

Définit la séquence standard : clean → features → train → predict → backtest → report.
Chaque pipeline concret (EurUsdPipeline, BtcUsdPipeline) override les étapes spécifiques.

Support également le walk-forward retraining (v14).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.config.backtest import BacktestConfig

logger = get_logger(__name__)


class BasePipeline(ABC):
    """Orchestrateur abstrait du pipeline ML/trading multi-actif.

    Chaque instrument concret hérite et fournit ses propres config/data/features.
    """

    def __init__(
        self,
        instrument_name: str,
        backtest_cfg: BacktestConfig | None = None,
    ) -> None:
        raise NotImplementedError(
            "v1 pipeline — non porté : ConfigRegistry n'existe plus dans app/. "
            "Utiliser un pipeline v4 (ex: app/pipelines/us30.py)."
        )

    @abstractmethod
    def load_data(self) -> dict[str, Any]:
        """Charge les données brutes (H1, H4, D1, macro)."""

    @abstractmethod
    def build_features(self, data: dict[str, Any]) -> Any:
        """Construit le DataFrame ML-ready."""

    def train_model(self, ml_data: Any) -> Any:
        """Entraîne le modèle (RandomForest ou GBM Regressor selon target_mode)."""
        raise NotImplementedError(
            "v1 pipeline — non porté : app/model/training.py inexistant. "
            "Utiliser app/models/build.py pour l'entraînement v4."
        )

    def evaluate_model(self, model: Any, ml_data: Any, X_cols: list[str]) -> dict:
        """Évalue le modèle sur val_year et test_year (classifieur ou régression)."""
        raise NotImplementedError(
            "v1 pipeline — non porté : app/model/prediction.py inexistant. "
            "Utiliser app/models/ pour les prédictions v4."
        )

    def run_backtest(
        self,
        predictions: Any,
        ml_data: Any,
        ohlcv_h1: Any,
    ) -> tuple[Any, dict]:
        """Exécute le backtest sur les prédictions."""
        from app.backtest.filters import (
            FilterPipeline,
            MomentumFilter,
            VolFilter,
            SessionFilter,
            CalendarFilter,
        )
        from app.backtest.simulator import (
            simulate_trades,
            simulate_trades_continuous,
        )
        # Fallback: weight_centered n'existe pas dans app.backtest.sizing
        try:
            from app.backtest.sizing import weight_centered  # type: ignore[no-redef]
        except ImportError:
            weight_centered = lambda x: np.ones_like(x)  # type: ignore[no-redef]

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
        if cfg.use_calendar_filter:
            filters.append(
                CalendarFilter(
                    exclude_window_minutes=cfg.calendar_exclude_window_minutes,
                    impact_threshold=cfg.calendar_impact_threshold,
                )
            )
        filter_pipeline = FilterPipeline(filters) if filters else None

        all_trades = {}
        all_metrics = {}

        # Colonnes requises par les filtres de régime (MomentumFilter, VolFilter)
        FILTER_COLS: tuple[str, ...] = (
            "Dist_SMA200_D1", "ATR_Norm", "RSI_D1_delta",
            "near_high_impact_event",
        )

        # Router la fonction de simulation selon target_mode
        if self.instrument.target_mode == "forward_return":
            simulate_func = simulate_trades_continuous
            simulate_kwargs: dict[str, Any] = {
                "signal_threshold": cfg.continuous_signal_threshold,
            }
        else:
            simulate_func = simulate_trades
            simulate_kwargs = {"seuil_confiance": cfg.confidence_threshold}

        for year, preds_df in predictions.items():
            # Joindre les prédictions avec OHLC H1 (simulate_trades a besoin de High/Low/Close)
            ohlc_cols = ["High", "Low", "Close"]
            ohlc_available = [c for c in ohlc_cols if c in ohlcv_h1.columns]
            if ohlc_available:
                year_ohlc = ohlcv_h1.loc[ohlcv_h1.index.year == year, ohlc_available]
                df_backtest = preds_df.join(year_ohlc, how="left")
            else:
                df_backtest = preds_df

            # Fallback Spread si absent (BTC n'a pas de spread dans les CSV bruts)
            if "Spread" not in df_backtest.columns:
                df_backtest["Spread"] = 0.0

            # Injecter les colonnes requises par les filtres depuis ml_data
            filter_cols_present = [c for c in FILTER_COLS if c in ml_data.columns]
            if filter_cols_present:
                year_filter = ml_data.loc[ml_data.index.year == year, filter_cols_present]
                df_backtest = df_backtest.join(year_filter, how="left")

            trades_df, _n_signaux, _n_filtres = simulate_func(
                df=df_backtest,
                weight_func=weight_centered,
                tp_pips=cfg.tp_pips * self.instrument.tp_sl_scale_factor,
                sl_pips=cfg.sl_pips * self.instrument.tp_sl_scale_factor,
                window=cfg.window_hours,
                pip_size=self.instrument.pip_size,
                commission_pips=cfg.commission_pips,
                slippage_pips=cfg.slippage_pips,
                filter_pipeline=filter_pipeline,
                **simulate_kwargs,
            )

            all_trades[year] = trades_df

            from app.backtest.metrics import compute_metrics

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
        raise NotImplementedError(
            "v1 pipeline — non porté : app/model/training.py inexistant. "
            "Utiliser app/models/ pour l'entraînement v4."
        )

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
