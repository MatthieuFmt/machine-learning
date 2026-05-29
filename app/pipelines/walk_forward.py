"""Walk-forward avec méta-labeling RF — pivot v4 B1.

Splits temporels en segments de retrain_months, expanding window.
Le méta-modèle est ré-entraîné à chaque date de retrain sur toutes
les données ≤ retrain_date (moins embargo 2 jours).

Calcule le sizing au risque 2 % pour chaque trade afin que
compute_metrics (mode A1) et validate_edge disposent de position_size_lots
et pnl (€).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.deterministic import run_deterministic_backtest
from app.backtest.metrics import sharpe_daily_from_trades
from app.backtest.sizing import compute_position_size, expected_pnl_eur
from app.config.instruments import AssetConfig
from app.core.logging import get_logger
from app.models.meta_labeling import MetaLabelingConfig, MetaLabelingRF

logger = get_logger(__name__)


@dataclass
class WalkForwardSegment:
    """Métriques d'un segment walk-forward."""

    start: pd.Timestamp
    end: pd.Timestamp
    n_train: int
    n_oos_trades: int
    sharpe_oos: float
    wr_oos: float
    meta_disabled: bool = False
    threshold: float = 0.50


def _trades_to_dataframe(
    trades: list[dict],
    cfg: AssetConfig | None = None,
    capital_eur: float = 10_000.0,
    risk_pct: float = 0.02,
) -> pd.DataFrame:
    """Convertit la liste de trades du backtest déterministe en DataFrame.

    En mode sizing (cfg non-None), calcule position_size_lots et pnl (€)
    pour que compute_metrics (mode A1) et validate_edge fonctionnent.

    Args:
        trades: Liste de dicts avec pips_net, result, entry_time, exit_time, signal.
        cfg: AssetConfig pour sizing. Si None, pas de colonne sizing.
        capital_eur: Capital de référence pour sizing.
        risk_pct: Risque par trade (défaut 2 %).

    Returns:
        DataFrame indexé par entry_time avec colonnes Pips_Nets, Pips_Bruts, result,
        et optionnellement position_size_lots, pnl.
    """
    if not trades:
        cols = ["Pips_Nets", "Pips_Bruts", "result"]
        if cfg is not None:
            cols.extend(["position_size_lots", "pnl"])
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.set_index("entry_time")
    df["Pips_Nets"] = df["pips_net"].astype(float)
    df["Pips_Bruts"] = df["pips_net"].astype(float)
    df["result"] = df["result"].astype(str)

    if cfg is not None:
        # Sizing au risque 2% sur SL fixe
        entry_prices = df["entry_price"].astype(float).values
        signals_signed = df["signal"].astype(int).values
        sl_prices = np.where(
            signals_signed == 1,
            entry_prices - cfg.sl_points * cfg.pip_size,
            entry_prices + cfg.sl_points * cfg.pip_size,
        )
        lots = np.array([
            compute_position_size(ep, sl, capital_eur, risk_pct, cfg)
            for ep, sl in zip(entry_prices, sl_prices, strict=True)
        ], dtype=float)
        df["position_size_lots"] = lots
        df["pnl"] = expected_pnl_eur(
            df["Pips_Nets"].values, lots, cfg
        )
    return df


def walk_forward_meta(
    df: pd.DataFrame,
    strat,
    cfg: AssetConfig,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    target_builder: Callable[[pd.DataFrame, pd.Series], pd.Series],
    retrain_months: int = 6,
    test_start: str = "2024-01-01",
    capital_eur: float = 10_000.0,
    meta_config: MetaLabelingConfig | None = None,
) -> tuple[pd.DataFrame, list[WalkForwardSegment]]:
    """Walk-forward avec méta-labeling RF sur la période [test_start, fin].

    Pour chaque date de retrain (1er janvier et 1er juillet) :
    1. Backtest baseline sur train ≤ retrain_dt − 2j (embargo).
    2. Extrait les features aux barres d'entrée des trades train.
    3. Entraîne MetaLabelingRF.
    4. Calibre le seuil sur train (max Sharpe, rétention ≥ 20%).
    5. Backtest baseline sur OOS, filtre avec méta-modèle.
    6. Agrège les métriques.

    Args:
        df: DataFrame OHLCV indexé par datetime, trié chronologiquement.
        strat: Stratégie déterministe (e.g. DonchianBreakout).
        cfg: AssetConfig avec coûts calibrés.
        feature_builder: Fonction df → DataFrame de features.
        target_builder: Fonction (df, pnl_brut) → pd.Series binaire (1/0).
        retrain_months: Fréquence de réentraînement en mois.
        test_start: Date de début du test OOS (str "YYYY-MM-DD").
        capital_eur: Capital initial en euros.
        meta_config: Configuration du méta-modèle (défaut: MetaLabelingConfig()).

    Returns:
        (all_trades_oos_df, segments) où all_trades_oos_df est un DataFrame
        de tous les trades OOS filtrés et segments la liste des métriques
        par segment walk-forward.
    """
    if meta_config is None:
        meta_config = MetaLabelingConfig()

    df = df.sort_index()
    test_start_ts = pd.Timestamp(test_start)

    # Normaliser les timezones : df.index peut être tz-aware (UTC)
    df_end = df.index[-1]
    if test_start_ts.tz is None and df_end.tz is not None:
        test_start_ts = test_start_ts.tz_localize(df_end.tz)
    elif test_start_ts.tz is not None and df_end.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)

    # Générer les dates de retrain : 1er janvier, 1er juillet depuis test_start
    retrain_dates = pd.date_range(
        start=test_start_ts, end=df_end, freq="6MS", inclusive="both"
    )
    if len(retrain_dates) == 0:
        retrain_dates = pd.DatetimeIndex([test_start_ts])

    # S'assurer que la première date de retrain est bien test_start
    if retrain_dates[0] > test_start_ts:
        # Insérer test_start comme première date
        retrain_dates = pd.DatetimeIndex([test_start_ts] + list(retrain_dates))

    # Coûts pour run_deterministic_backtest
    # cost_total = 2 * (commission_pips + slippage_pips)
    # On veut cost_total = spread_pips + slippage_pips
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    all_oos_trades: list[pd.DataFrame] = []
    segments: list[WalkForwardSegment] = []

    for i, retrain_dt in enumerate(retrain_dates):
        # Segment OOS : de retrain_dt à la prochaine date de retrain (ou fin)
        if i + 1 < len(retrain_dates):
            segment_end = retrain_dates[i + 1] - pd.Timedelta(days=1)
        else:
            segment_end = df.index[-1]

        # Train : toutes les barres ≤ retrain_dt − 2 jours (embargo)
        train_end = retrain_dt - pd.Timedelta(days=2)
        df_train = df.loc[:train_end]
        df_oos = df.loc[retrain_dt:segment_end]

        if df_train.empty or df_oos.empty:
            logger.warning(
                "Segment %s: train (%d barres) ou OOS (%d barres) vide, skip.",
                retrain_dt.date(), len(df_train), len(df_oos),
            )
            continue

        # ── Étape 1: Backtest baseline sur train ────────────────────────
        signals_train = strat.generate_signals(df_train)
        bt_train = run_deterministic_backtest(
            df=df_train,
            signals=signals_train,
            tp_pips=cfg.tp_points,
            sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost,
            pip_size=cfg.pip_size,
        )
        trades_train: list[dict] = bt_train.get("trades", [])

        if len(trades_train) < 10:
            logger.warning(
                "Segment %s: seulement %d trades train, skip méta.",
                retrain_dt.date(), len(trades_train),
            )
            # Fallback: baseline pure sur OOS
            signals_oos = strat.generate_signals(df_oos)
            bt_oos = run_deterministic_backtest(
                df=df_oos, signals=signals_oos,
                tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
            )
            trades_oos = _trades_to_dataframe(bt_oos.get("trades", []), cfg=cfg, capital_eur=capital_eur)
            seg = WalkForwardSegment(
                start=retrain_dt,
                end=segment_end,
                n_train=len(trades_train),
                n_oos_trades=len(trades_oos),
                sharpe_oos=bt_oos.get("sharpe", 0.0),
                wr_oos=bt_oos.get("wr", 0.0),
                meta_disabled=True,
                threshold=0.0,
            )
            segments.append(seg)
            if not trades_oos.empty:
                all_oos_trades.append(trades_oos)
            continue

        # ── Étape 2: Features aux barres d'entrée des trades train ──────
        # df_train contient déjà tout l'historique ≤ train_end, donc les
        # rolling indicators ont assez de warmup.
        features_all_train = feature_builder(df_train)
        entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_train])

        # Aligner les features avec les entry_times
        common_idx = features_all_train.index.intersection(entry_times_train)
        if len(common_idx) < 5:
            logger.warning(
                "Segment %s: seulement %d features alignées, skip méta.",
                retrain_dt.date(), len(common_idx),
            )
            # Fallback baseline
            signals_oos = strat.generate_signals(df_oos)
            bt_oos = run_deterministic_backtest(
                df=df_oos, signals=signals_oos,
                tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
            )
            trades_oos = _trades_to_dataframe(bt_oos.get("trades", []), cfg=cfg, capital_eur=capital_eur)
            seg = WalkForwardSegment(
                start=retrain_dt, end=segment_end,
                n_train=len(trades_train),
                n_oos_trades=len(trades_oos),
                sharpe_oos=bt_oos.get("sharpe", 0.0),
                wr_oos=bt_oos.get("wr", 0.0),
                meta_disabled=True, threshold=0.0,
            )
            segments.append(seg)
            if not trades_oos.empty:
                all_oos_trades.append(trades_oos)
            continue

        x_train = features_all_train.loc[common_idx]

        # ── Étape 3: Build target (meta-labels) ─────────────────────────
        # Aligner les trades avec les entry times dans common_idx
        trades_train_df = _trades_to_dataframe(trades_train, cfg=cfg, capital_eur=capital_eur)
        pnl_aligned = trades_train_df.loc[
            trades_train_df.index.intersection(common_idx), "Pips_Nets"
        ]
        y_train = target_builder(df_train, pnl_aligned)

        # ── Étape 4: Train MetaLabelingRF ───────────────────────────────
        meta = MetaLabelingRF(config=meta_config)
        meta.fit(x_train, y_train)

        # ── Étape 5: Calibrate threshold ────────────────────────────────
        if not meta.disabled:
            # Construire un mapping entry_time → trade_index pour le backtest_fn
            entry_to_trade: dict[pd.Timestamp, dict] = {}
            for t in trades_train:
                et = pd.Timestamp(t["entry_time"])
                entry_to_trade[et] = t

            def _sharpe_for_threshold(mask: pd.Series, _entry_map: dict = entry_to_trade) -> float:  # noqa: B023
                """Calcule le Sharpe sur les trades train filtrés par le masque."""
                _trades_map = _entry_map
                accepted_indices = set(mask[mask].index)
                filtered_trades = [
                    t for et, t in _trades_map.items()
                    if et in accepted_indices
                ]
                if len(filtered_trades) < 5:
                    return -np.inf
                return sharpe_daily_from_trades(filtered_trades)

            meta.calibrate_threshold(x_train, _sharpe_for_threshold)

        # ── Étape 6: Appliquer sur OOS ──────────────────────────────────
        signals_oos = strat.generate_signals(df_oos)

        if meta.disabled:
            # Baseline pure
            filtered_signals = signals_oos
        else:
            # Générer les features pour tout l'OOS avec historique complet
            # pour que les rolling indicators (ex: SMA 200) aient assez de warmup.
            df_oos_with_history = df.loc[:segment_end]
            features_all_oos = feature_builder(df_oos_with_history)
            # Ne garder que les barres OOS (retrain_dt → segment_end)
            features_all_oos = features_all_oos.loc[retrain_dt:segment_end]

            # Identifier les barres avec signal ≠ 0
            signal_bars = signals_oos[signals_oos != 0]
            if len(signal_bars) == 0:
                bt_oos = {"sharpe": 0.0, "wr": 0.0, "total_trades": 0, "trades": []}
                seg = WalkForwardSegment(
                    start=retrain_dt, end=segment_end,
                    n_train=len(trades_train),
                    n_oos_trades=0,
                    sharpe_oos=0.0, wr_oos=0.0,
                    meta_disabled=meta.disabled,
                    threshold=meta.threshold,
                )
                segments.append(seg)
                continue

            # Aligner features avec les barres de signal
            common_oos_idx = features_all_oos.index.intersection(signal_bars.index)
            if len(common_oos_idx) == 0:
                bt_oos = {"sharpe": 0.0, "wr": 0.0, "total_trades": 0, "trades": []}
                seg = WalkForwardSegment(
                    start=retrain_dt, end=segment_end,
                    n_train=len(trades_train),
                    n_oos_trades=0,
                    sharpe_oos=0.0, wr_oos=0.0,
                    meta_disabled=meta.disabled,
                    threshold=meta.threshold,
                )
                segments.append(seg)
                continue

            x_oos_signal = features_all_oos.loc[common_oos_idx]

            # Prédire le masque
            keep_mask = meta.predict(x_oos_signal)
            keep_indices = set(common_oos_idx[keep_mask])

            # Filtrer les signaux OOS
            filtered_signals = signals_oos.copy()
            for idx in signal_bars.index:
                if idx not in keep_indices:
                    filtered_signals.loc[idx] = 0

        # Backtest OOS avec signaux filtrés
        bt_oos = run_deterministic_backtest(
            df=df_oos,
            signals=filtered_signals,
            tp_pips=cfg.tp_points,
            sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost,
            pip_size=cfg.pip_size,
        )

        trades_oos = _trades_to_dataframe(bt_oos.get("trades", []), cfg=cfg, capital_eur=capital_eur)
        seg = WalkForwardSegment(
            start=retrain_dt,
            end=segment_end,
            n_train=len(trades_train),
            n_oos_trades=int(bt_oos.get("total_trades", 0)),
            sharpe_oos=float(bt_oos.get("sharpe", 0.0)),
            wr_oos=float(bt_oos.get("wr", 0.0)),
            meta_disabled=meta.disabled,
            threshold=meta.threshold if not meta.disabled else 0.0,
        )
        segments.append(seg)

        if not trades_oos.empty:
            all_oos_trades.append(trades_oos)

    # ── Agrégation ──────────────────────────────────────────────────────
    if all_oos_trades:
        all_trades_df = pd.concat(all_oos_trades)
        all_trades_df = all_trades_df.sort_index()
    else:
        all_trades_df = pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"])

    return all_trades_df, segments
