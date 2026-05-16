"""Walk-forward avec fenêtre train rolling 3 ans (pas expansive).

Pivot v4 B3 — H_new2. Contrairement à walk_forward.py (expanding window),
ce module utilise une fenêtre glissante de 3 ans pour l'entraînement,
recalibrée tous les 6 mois. Testé sur la période 2024+.

Règles :
- Une seule position à la fois (backtest stateful).
- Train rolling : 3 dernières années avant chaque date de retrain.
- Embargo 2 jours entre train_end et oos_start.
- Méta-labeling RF recalibré sur chaque segment train.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.deterministic import run_deterministic_backtest
from app.backtest.metrics import compute_metrics, sharpe_daily_from_trades
from app.backtest.sizing import compute_position_size, expected_pnl_eur
from app.config.instruments import AssetConfig
from app.core.logging import get_logger
from app.models.meta_labeling import MetaLabelingConfig, MetaLabelingRF

logger = get_logger(__name__)


@dataclass
class RollingSegment:
    """Métriques d'un segment walk-forward rolling."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    n_train: int
    n_oos: int
    sharpe_oos: float
    meta_disabled: bool = False
    threshold: float = 0.50


def _trades_list_to_dataframe(
    trades: list[dict],
    cfg: AssetConfig | None = None,
    capital_eur: float = 10_000.0,
    risk_pct: float = 0.02,
) -> pd.DataFrame:
    """Convertit la liste de trades du backtest déterministe en DataFrame.

    Inspiré de walk_forward._trades_to_dataframe avec sizing au risque 2 %.
    """
    if not trades:
        cols = ["Pips_Nets", "Pips_Bruts", "result", "entry_price", "signal"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.set_index("entry_time")
    df["Pips_Nets"] = df["pips_net"].astype(float)
    df["Pips_Bruts"] = df["pips_net"].astype(float)
    df["result"] = df["result"].astype(str)

    if cfg is not None:
        entry_prices = df["entry_price"].astype(float).values
        signals_signed = df["signal"].astype(int).values
        sl_prices = np.where(
            signals_signed == 1,
            entry_prices - cfg.sl_points * cfg.pip_size,
            entry_prices + cfg.sl_points * cfg.pip_size,
        )
        lots = np.array(
            [
                compute_position_size(ep, sl, capital_eur, risk_pct, cfg)
                for ep, sl in zip(entry_prices, sl_prices, strict=True)
            ],
            dtype=float,
        )
        df["position_size_lots"] = lots
        df["pnl"] = expected_pnl_eur(df["Pips_Nets"].values, lots, cfg)
    return df


def walk_forward_rolling(
    df: pd.DataFrame,
    strat,
    cfg: AssetConfig,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    target_builder: Callable[[pd.DataFrame, pd.Series], pd.Series],
    train_window_years: int = 3,
    retrain_months: int = 6,
    test_start: str = "2024-01-01",
    capital_eur: float = 10_000.0,
    embargo_days: int = 2,
    meta_config: MetaLabelingConfig | None = None,
) -> tuple[pd.DataFrame, list[RollingSegment]]:
    """Walk-forward rolling : fenêtre train glissante de train_window_years.

    Pour chaque date de retrain (tous les retrain_months depuis test_start) :
    1. Sélectionne les train_window_years précédant retrain_dt (rolling, pas expansive).
    2. Applique embargo_days entre train_end et retrain_dt.
    3. Backtest baseline sur train, entraîne méta-labeling RF, calibre seuil.
    4. Backtest baseline sur OOS, filtre avec méta-modèle.
    5. Agrège les trades OOS filtrés et les métriques par segment.

    Args:
        df: DataFrame OHLCV indexé par datetime, trié chronologiquement.
        strat: Stratégie déterministe avec méthode generate_signals(df).
        cfg: AssetConfig avec coûts calibrés.
        feature_builder: Fonction df → DataFrame de features.
        target_builder: Fonction (df, pnl_brut) → pd.Series binaire (1/0).
        train_window_years: Durée de la fenêtre train glissante en années.
        retrain_months: Fréquence de réentraînement en mois.
        test_start: Date de début du test OOS (str "YYYY-MM-DD").
        capital_eur: Capital initial en euros.
        embargo_days: Jours d'embargo entre train_end et oos_start.
        meta_config: Configuration du méta-modèle (défaut: MetaLabelingConfig()).

    Returns:
        (all_trades_oos_df, segments) — trades OOS agrégés et liste des
        métriques par segment.
    """
    if meta_config is None:
        meta_config = MetaLabelingConfig()

    df = df.sort_index()
    test_start_ts = pd.Timestamp(test_start)

    # Normaliser les timezones
    df_end = df.index[-1]
    if test_start_ts.tz is None and df_end.tz is not None:
        test_start_ts = test_start_ts.tz_localize(df_end.tz)
    elif test_start_ts.tz is not None and df_end.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)

    # Générer les dates de retrain tous les retrain_months
    freq_str = f"{retrain_months}MS"
    retrain_dates = pd.date_range(
        start=test_start_ts, end=df_end, freq=freq_str, inclusive="both"
    )
    if len(retrain_dates) == 0:
        retrain_dates = pd.DatetimeIndex([test_start_ts])
    if retrain_dates[0] > test_start_ts:
        retrain_dates = pd.DatetimeIndex([test_start_ts, *list(retrain_dates)])

    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    all_oos_trades: list[pd.DataFrame] = []
    segments: list[RollingSegment] = []

    for i, retrain_dt in enumerate(retrain_dates):
        # Segment OOS : de retrain_dt à la prochaine date de retrain (ou fin)
        if i + 1 < len(retrain_dates):
            oos_end = retrain_dates[i + 1] - pd.Timedelta(days=1)
        else:
            oos_end = df.index[-1]

        # Train rolling : train_window_years avant retrain_dt, moins embargo
        train_start = retrain_dt - pd.DateOffset(years=train_window_years)
        train_end = retrain_dt - pd.Timedelta(days=embargo_days)

        df_train = df.loc[train_start:train_end]
        df_oos = df.loc[retrain_dt:oos_end]

        if len(df_train) < 250 or df_oos.empty:
            logger.warning(
                "Segment %s: train (%d barres) ou OOS (%d barres) insuffisant, skip.",
                retrain_dt.date(),
                len(df_train),
                len(df_oos),
            )
            continue

        # ── Étape 1: Backtest baseline sur train ──────────────────────────
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

        if len(trades_train) < 20:
            logger.warning(
                "Segment %s: seulement %d trades train, skip méta.",
                retrain_dt.date(),
                len(trades_train),
            )
            continue

        # ── Étape 2: Features aux barres d'entrée des trades train ────────
        features_all_train = feature_builder(df_train)
        entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_train])
        common_idx = features_all_train.index.intersection(entry_times_train)

        if len(common_idx) < 5:
            logger.warning(
                "Segment %s: seulement %d features alignées, skip méta.",
                retrain_dt.date(),
                len(common_idx),
            )
            continue

        x_train = features_all_train.loc[common_idx]

        # ── Étape 3: Build target (meta-labels) ───────────────────────────
        trades_train_df = _trades_list_to_dataframe(trades_train, cfg=cfg, capital_eur=capital_eur)
        pnl_aligned = trades_train_df.loc[
            trades_train_df.index.intersection(common_idx), "Pips_Nets"
        ]
        y_train = target_builder(df_train, pnl_aligned)

        # ── Étape 4: Train MetaLabelingRF ──────────────────────────────────
        meta = MetaLabelingRF(config=meta_config)
        meta.fit(x_train, y_train)

        # ── Étape 5: Calibrate threshold ───────────────────────────────────
        if not meta.disabled:
            entry_to_trade: dict[pd.Timestamp, dict] = {}
            for t in trades_train:
                et = pd.Timestamp(t["entry_time"])
                entry_to_trade[et] = t

            def _sharpe_for_threshold(mask: pd.Series, _emap: dict = entry_to_trade) -> float:  # noqa: B023
                accepted_indices = set(mask[mask].index)
                filtered_trades = [
                    t for et, t in _emap.items() if et in accepted_indices
                ]
                if len(filtered_trades) < 5:
                    return -np.inf
                return sharpe_daily_from_trades(filtered_trades)

            meta.calibrate_threshold(x_train, _sharpe_for_threshold)

        # ── Étape 6: Appliquer sur OOS ─────────────────────────────────────
        signals_oos = strat.generate_signals(df_oos)

        if meta.disabled:
            bt_oos = run_deterministic_backtest(
                df=df_oos, signals=signals_oos,
                tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
            )
            trades_oos = _trades_list_to_dataframe(
                bt_oos.get("trades", []), cfg=cfg, capital_eur=capital_eur,
            )
        else:
            df_oos_with_history = df.loc[:oos_end]
            features_all_oos = feature_builder(df_oos_with_history)
            features_all_oos = features_all_oos.loc[retrain_dt:oos_end]

            signal_bars = signals_oos[signals_oos != 0]
            if len(signal_bars) == 0:
                seg = RollingSegment(
                    train_start=train_start, train_end=train_end,
                    oos_start=retrain_dt, oos_end=oos_end,
                    n_train=len(trades_train), n_oos=0, sharpe_oos=0.0,
                    meta_disabled=meta.disabled, threshold=meta.threshold,
                )
                segments.append(seg)
                continue

            common_oos_idx = features_all_oos.index.intersection(signal_bars.index)
            if len(common_oos_idx) == 0:
                seg = RollingSegment(
                    train_start=train_start, train_end=train_end,
                    oos_start=retrain_dt, oos_end=oos_end,
                    n_train=len(trades_train), n_oos=0, sharpe_oos=0.0,
                    meta_disabled=meta.disabled, threshold=meta.threshold,
                )
                segments.append(seg)
                continue

            x_oos_signal = features_all_oos.loc[common_oos_idx]
            keep_mask = meta.predict(x_oos_signal)
            keep_indices = set(common_oos_idx[keep_mask])

            filtered_signals = signals_oos.copy()
            for idx in signal_bars.index:
                if idx not in keep_indices:
                    filtered_signals.loc[idx] = 0

            bt_oos = run_deterministic_backtest(
                df=df_oos, signals=filtered_signals,
                tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
            )
            trades_oos = _trades_list_to_dataframe(
                bt_oos.get("trades", []), cfg=cfg, capital_eur=capital_eur,
            )

        m = compute_metrics(trades_oos, asset_cfg=cfg, capital_eur=capital_eur)
        seg = RollingSegment(
            train_start=train_start,
            train_end=train_end,
            oos_start=retrain_dt,
            oos_end=oos_end,
            n_train=len(trades_train),
            n_oos=int(m["trades"]),
            sharpe_oos=float(m["sharpe"]),
            meta_disabled=meta.disabled,
            threshold=meta.threshold if not meta.disabled else 0.0,
        )
        segments.append(seg)

        if not trades_oos.empty:
            all_oos_trades.append(trades_oos)

    if all_oos_trades:
        all_trades_df = pd.concat(all_oos_trades).sort_index()
    else:
        all_trades_df = pd.DataFrame(
            columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"]
        )

    return all_trades_df, segments
