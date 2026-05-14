"""Pipeline US30 D1 — Trend-Following avec features mono-TF minimales.

Hypothèse v2-01 :
- Cible : directional_clean (binaire symétrique, horizon 5j, bruit 0.5×ATR)
- Features : 6-8 mono-D1, pas de H4, pas de macro, pas de calendrier
- Modèle : RF 200 arbres, depth=4, leaf=20, class_weight=balanced_subsample
- Split : train ≤ 2022, val 2023, test 2024-2025
- Backtest : TP=200, SL=100, commission=3, slippage=5, vol_filter only
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning_v2.config.backtest import BacktestConfig
from learning_machine_learning_v2.config.instruments import Us30Config
from learning_machine_learning_v2.core.logging import get_logger
from learning_machine_learning_v2.targets.labels import compute_directional_clean

logger = get_logger(__name__)

# Hyperparamètres RF figés ex ante — NE PAS MODIFIER après premier run
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

# Split temporel figé ex ante — NE PAS MODIFIER après premier run
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI vectorisé — Wilder's smoothing."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()

    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain[i] = (1 - alpha) * avg_gain[i - 1] + alpha * gain[i]
        avg_loss[i] = (1 - alpha) * avg_loss[i - 1] + alpha * loss[i]

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0.0)
    rsi_arr = 100.0 - (100.0 / (1.0 + rs))
    return rsi_arr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX vectorisé — indicateur de force de tendance."""
    n = len(close)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0.0
    down_move[0] = 0.0

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_arr = np.full(n, np.nan)
    atr_arr[period - 1] = tr[:period].mean()
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    plus_di[period - 1] = plus_dm[:period].mean()
    minus_di[period - 1] = minus_dm[:period].mean()

    alpha = 1.0 / period
    for i in range(period, n):
        atr_arr[i] = (1 - alpha) * atr_arr[i - 1] + alpha * tr[i]
        plus_di[i] = (1 - alpha) * plus_di[i - 1] + alpha * plus_dm[i]
        minus_di[i] = (1 - alpha) * minus_di[i - 1] + alpha * minus_dm[i]

    plus_di_norm = 100.0 * plus_di / atr_arr
    minus_di_norm = 100.0 * minus_di / atr_arr
    dx = 100.0 * np.abs(plus_di_norm - minus_di_norm) / (plus_di_norm + minus_di_norm + 1e-10)

    adx_arr = np.full(n, np.nan)
    adx_arr[2 * period - 2] = dx[period - 1 : 2 * period - 1].mean()
    for i in range(2 * period - 1, n):
        adx_arr[i] = (1 - alpha) * adx_arr[i - 1] + alpha * dx[i]

    return adx_arr


class Us30Pipeline:
    """Pipeline US30 D1 — v2 Hypothesis 01.

    Mono-TF D1, features minimales, RF fortement régularisé,
    protocole anti-snooping strict (split figé, un seul regard).

    N'hérite PAS de BasePipeline pour éviter la dépendance à
    ConfigRegistry v1 qui ne connaît pas US30.
    """

    def __init__(self) -> None:
        self.instrument = Us30Config()
        self.backtest_cfg = BacktestConfig(
            tp_pips=200.0,
            sl_pips=100.0,
            window_hours=120,
            commission_pips=3.0,
            slippage_pips=5.0,
            confidence_threshold=0.55,
            use_momentum_filter=False,
            use_vol_filter=True,
            use_session_filter=False,
            use_calendar_filter=False,
        )

    def load_data(self) -> dict[str, Any]:
        """Charge US30 D1 et H4 depuis cleaned-data/.

        H4 est chargé mais pas utilisé dans build_features (gardé pour v2-02).
        """
        from pathlib import Path

        cleaned_dir = Path("cleaned-data")
        data: dict[str, Any] = {}

        for tf in ("D1", "H4"):
            path = cleaned_dir / f"USA30IDXUSD_{tf}_cleaned.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} introuvable. Exécuter d'abord : "
                    f"python scripts/inspect_us30_csv.py"
                )
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info(
                "Charge %s : %d barres, %s -> %s",
                path.name,
                len(df),
                df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d"),
            )
            data[f"us30_{tf.lower()}"] = df

        return data

    def build_features(
        self, data: dict[str, Any], train_end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Construit 6-8 features mono-D1 + Target directional_clean.

        Features :
        - RSI_14, ADX_14, Dist_SMA50, Dist_SMA200, ATR_Norm, Log_Return_5d
        - Volume_Ratio (si Volume/TickVol présent)
        """
        d1 = data["us30_d1"].sort_index()
        close = d1["Close"].values.astype(np.float64)
        high = d1["High"].values.astype(np.float64)
        low = d1["Low"].values.astype(np.float64)
        n = len(close)

        ml = pd.DataFrame(index=d1.index)

        # ── Conserver OHLC dans ml pour le backtest ──
        for col in ("Open", "High", "Low", "Close"):
            if col in d1.columns:
                ml[col] = d1[col]

        # ── Target : directional_clean ──
        ml["Target"] = compute_directional_clean(
            d1, horizon_hours=120, noise_atr=0.5, atr_period=14,
        )

        # ── Feature 1 : RSI_14 ──
        ml["RSI_14"] = _rsi(close, 14)

        # ── Feature 2 : ADX_14 ──
        ml["ADX_14"] = _adx(high, low, close, 14)

        # ── Feature 3 : Dist_SMA50 (%) ──
        sma50 = pd.Series(close).rolling(50, min_periods=1).mean().values
        ml["Dist_SMA50"] = (close - sma50) / sma50 * 100.0

        # ── Feature 4 : Dist_SMA200 (%) ──
        sma200 = pd.Series(close).rolling(200, min_periods=1).mean().values
        ml["Dist_SMA200"] = (close - sma200) / sma200 * 100.0

        # ── Feature 5 : ATR_Norm ──
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0], tr3[0] = 0.0, 0.0
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_raw = np.full(n, np.nan)
        atr_raw[13] = tr[:14].mean()
        alpha = 1.0 / 14
        for i in range(14, n):
            atr_raw[i] = (1 - alpha) * atr_raw[i - 1] + alpha * tr[i]
        ml["ATR_Norm"] = atr_raw / close * 100.0

        # ── Feature 6 : Log_Return_5d ──
        log_ret = np.full(n, np.nan)
        log_ret[5:] = np.log(close[5:] / close[:-5])
        ml["Log_Return_5d"] = log_ret

        # ── Feature 7 (optionnelle) : Volume_Ratio ──
        vol_col = None
        for candidate in ("Volume", "TickVol"):
            if candidate in d1.columns:
                vol_col = candidate
                break

        if vol_col:
            vol = d1[vol_col].values.astype(np.float64)
            vol_sma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values
            ml["Volume_Ratio"] = np.where(vol_sma20 > 0, vol / vol_sma20, 1.0)
            logger.info("Feature Volume_Ratio construite (colonne=%s)", vol_col)
        else:
            logger.warning("Pas de Volume/TickVol — Volume_Ratio indisponible")

        # ── Nettoyage ──
        n_before = len(ml)
        ml = ml.dropna()
        n_after = len(ml)
        logger.info(
            "Features : %d colonnes, %d -> %d lignes (%.1f%% loss)",
            len(ml.columns),
            n_before,
            n_after,
            (n_before - n_after) / max(n_before, 1) * 100,
        )

        if "Target" in ml.columns:
            target_dist = ml["Target"].value_counts(normalize=True).sort_index()
            logger.info(
                "Distribution Target : -1=%.1f%%, 0=%.1f%%, 1=%.1f%%",
                target_dist.get(-1, 0) * 100,
                target_dist.get(0, 0) * 100,
                target_dist.get(1, 0) * 100,
            )

        return ml

    def train_model(
        self, ml_data: pd.DataFrame
    ) -> tuple[RandomForestClassifier, list[str]]:
        """Entraîne RF sur train ≤ 2022, ignore val 2023."""
        drop_cols = {"Target", "Spread"}
        X_cols = [c for c in ml_data.columns if c not in drop_cols]
        y_col = "Target"

        train_mask = ml_data.index <= TRAIN_END
        val_mask = (ml_data.index >= VAL_START) & (ml_data.index <= VAL_END)
        test_mask = ml_data.index >= TEST_START

        X_train = ml_data.loc[train_mask, X_cols]
        y_train = ml_data.loc[train_mask, y_col]

        valid_train = y_train.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        if X_train.empty:
            raise ValueError(
                "X_train vide — vérifier les données train ≤ 2022-12-31"
            )

        logger.info(
            "Split : train=%d (<=%s), val=%d (%s->%s), test=%d (>=%s)",
            len(X_train),
            TRAIN_END.strftime("%Y-%m-%d"),
            val_mask.sum(),
            VAL_START.strftime("%Y-%m-%d"),
            VAL_END.strftime("%Y-%m-%d"),
            test_mask.sum(),
            TEST_START.strftime("%Y-%m-%d"),
        )

        train_dist = y_train.value_counts(normalize=True).sort_index()
        logger.info(
            "Train distribution : -1=%.1f%%, 0=%.1f%%, 1=%.1f%%",
            train_dist.get(-1, 0) * 100,
            train_dist.get(0, 0) * 100,
            train_dist.get(1, 0) * 100,
        )

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        importances = sorted(
            zip(X_cols, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("Feature importance (top 5) :")
        for feat, imp in importances[:5]:
            logger.info("  %-20s %.4f", feat, imp)

        return model, X_cols

    def predict(
        self,
        model: RandomForestClassifier,
        ml_data: pd.DataFrame,
        X_cols: list[str],
    ) -> dict[int, pd.DataFrame]:
        """Prédit sur val (2023) et test (2024-2025)."""
        results: dict[int, pd.DataFrame] = {}

        for year in (2023, 2024, 2025):
            year_mask = ml_data.index.year == year
            year_data = ml_data.loc[year_mask]
            if year_data.empty:
                logger.warning("Aucune donnée pour l'année %d", year)
                continue

            X = year_data[X_cols]
            preds = model.predict(X)
            probas = model.predict_proba(X)

            class_map = {
                float(cls): int(idx) for idx, cls in enumerate(model.classes_)
            }

            def _get_col(class_key: float) -> np.ndarray:
                if class_key in class_map:
                    return probas[:, class_map[class_key]]
                return np.zeros(len(probas), dtype=np.float64)

            pred_df = pd.DataFrame(
                {
                    "Prediction_Modele": preds,
                    "Confiance_Baisse_%": np.round(_get_col(-1.0) * 100, 2),
                    "Confiance_Neutre_%": np.round(_get_col(0.0) * 100, 2),
                    "Confiance_Hausse_%": np.round(_get_col(1.0) * 100, 2),
                },
                index=year_data.index,
            )

            for col in ("Open", "High", "Low", "Close"):
                if col in year_data.columns:
                    pred_df[col] = year_data[col]
                else:
                    # Fallback depuis le D1 brut
                    pass

            pred_df["Spread"] = 0.0
            results[year] = pred_df

            logger.info("Prédictions %d : %d barres", year, len(pred_df))

        return results

    def run_backtest(
        self,
        predictions: dict[int, pd.DataFrame],
        ml_data: pd.DataFrame,
    ) -> tuple[dict[int, pd.DataFrame], dict[int, dict]]:
        """Backtest simplifié — VolFilter uniquement, pas de join H1."""
        from learning_machine_learning_v2.backtest.filters import (
            FilterPipeline,
            VolFilter,
        )
        from learning_machine_learning_v2.backtest.metrics import compute_metrics
        from learning_machine_learning_v2.backtest.simulator import simulate_trades
        from learning_machine_learning_v2.backtest.sizing import weight_centered

        cfg = self.backtest_cfg

        filters = [
            VolFilter(
                window=cfg.vol_filter_window,
                multiplier=cfg.vol_filter_multiplier,
            )
        ]
        filter_pipeline = FilterPipeline(filters)

        all_trades: dict[int, pd.DataFrame] = {}
        all_metrics: dict[int, dict] = {}

        for year, preds_df in predictions.items():
            # Injecter ATR_Norm pour VolFilter
            if "ATR_Norm" in ml_data.columns:
                year_atr = ml_data.loc[
                    ml_data.index.year == year, ["ATR_Norm"]
                ]
                preds_df = preds_df.join(year_atr, how="left")

            trades_df, n_signaux, n_filtres = simulate_trades(
                df=preds_df,
                weight_func=weight_centered,
                tp_pips=cfg.tp_pips * self.instrument.tp_sl_scale_factor,
                sl_pips=cfg.sl_pips * self.instrument.tp_sl_scale_factor,
                window=cfg.window_hours,
                pip_size=self.instrument.pip_size,
                seuil_confiance=cfg.confidence_threshold,
                commission_pips=cfg.commission_pips,
                slippage_pips=cfg.slippage_pips,
                filter_pipeline=filter_pipeline,
            )

            all_trades[year] = trades_df
            logger.info(
                "Backtest %d : %d signaux, %d trades, filtres=%s",
                year,
                n_signaux,
                len(trades_df),
                n_filtres,
            )

            metrics = compute_metrics(
                trades_df=trades_df,
                annee=year,
                df=preds_df,
                pip_value_eur=self.instrument.pip_value_eur,
                initial_capital=cfg.initial_capital,
                pip_size=self.instrument.pip_size,
            )
            all_metrics[year] = metrics
            logger.info(
                "Métriques %d : Sharpe=%.3f, WR=%.1f%%, Trades=%d, PnL=%.1f pips",
                year,
                metrics["sharpe"],
                metrics["win_rate"],
                metrics["trades"],
                metrics["profit_net"],
            )

        return all_trades, all_metrics

    def save_report(
        self,
        metrics: dict[int, dict],
        trades: dict[int, pd.DataFrame],
        path: str,
    ) -> None:
        """Sauvegarde le rapport JSON avec Sharpe OOS agrégé."""
        import json
        from pathlib import Path

        from learning_machine_learning_v2.backtest.metrics import sharpe_ratio

        report: dict[str, Any] = {
            "hypothesis": "v2_01",
            "instrument": self.instrument.name,
            "primary_tf": self.instrument.primary_tf,
            "target_mode": "directional_clean",
            "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
            "backtest_config": {
                "tp_pips": self.backtest_cfg.tp_pips,
                "sl_pips": self.backtest_cfg.sl_pips,
                "window_hours": self.backtest_cfg.window_hours,
                "commission_pips": self.backtest_cfg.commission_pips,
                "slippage_pips": self.backtest_cfg.slippage_pips,
                "confidence_threshold": self.backtest_cfg.confidence_threshold,
            },
            "split": {
                "train_end": str(TRAIN_END.date()),
                "val": f"{VAL_START.date()} -> {VAL_END.date()}",
                "test_start": str(TEST_START.date()),
            },
            "metrics": {},
        }

        for year, m in metrics.items():
            report["metrics"][str(year)] = {
                k: v
                for k, v in m.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }

        # Sharpe agrégé test 2024-2025
        test_years = [y for y in trades if y >= 2024]
        if test_years:
            all_test = pd.concat([trades[y] for y in test_years])
            daily_pips = all_test["Pips_Nets"].resample("D").sum().dropna()
            daily_returns = (
                daily_pips
                * self.instrument.pip_value_eur
                / self.backtest_cfg.initial_capital
            )
            report["sharpe"] = float(sharpe_ratio(daily_returns.values))
        else:
            report["sharpe"] = 0.0

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Rapport sauvegardé : %s", path)

    def run(self) -> dict[str, Any]:
        """Exécute le pipeline complet.

        Returns:
            Dict avec 'model', 'predictions', 'trades', 'metrics', 'X_cols'.
        """
        data = self.load_data()
        ml = self.build_features(data)
        model, X_cols = self.train_model(ml)
        predictions = self.predict(model, ml, X_cols)
        trades, metrics = self.run_backtest(predictions, ml)
        return {
            "model": model,
            "predictions": predictions,
            "trades": trades,
            "metrics": metrics,
            "X_cols": X_cols,
        }
