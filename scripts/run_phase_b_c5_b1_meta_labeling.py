"""Pivot v4 Phase B C5 â€” B1 : MÃ©ta-labeling RF sur GBPUSD H4 (rf, Sharpe outer +3.45).

âš ï¸ Consomme 1 n_trial. Lecture OOS test set â‰¥ 2024 = unique.

Flow :
1. Charge GBPUSD H4, cutoff train â‰¤ 2022-12-31, test â‰¥ 2024-01-01.
2. Construit le superset de features, sÃ©lectionne le top 15 C5 pour GBPUSD H4.
3. Walk-forward expanding window, retrain 6M depuis 2024-01-01 :
   a. EntraÃ®ne le modÃ¨le primaire rf (hyperparams C5) sur train â‰¤ retrain_date
   b. GÃ©nÃ¨re les signaux primaires â†’ backtest dÃ©terministe â†’ trades train
   c. Extrait les features aux barres d'entrÃ©e des trades train
   d. EntraÃ®ne MetaLabelingRF (2áµ‰ modÃ¨le) pour filtrer les faux signaux
   e. Applique le filtre mÃ©ta sur le segment OOS
   f. Backtest OOS avec et sans mÃ©ta-labeling â†’ comparaison Sharpe
4. AgrÃ¨ge tous les trades OOS, calcule mÃ©triques, validate_edge, read_oos.
5. Sauvegarde dans predictions/phase_b_c5_b1_gbpusd_h4.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.backtest.deterministic import run_deterministic_backtest  # noqa: E402
from app.backtest.metrics import compute_metrics, sharpe_daily_from_trades  # noqa: E402
from app.config.features_selected import FEATURES_SELECTED  # noqa: E402
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.superset import build_superset  # noqa: E402
from app.models.meta_labeling import MetaLabelingConfig, MetaLabelingRF  # noqa: E402
from app.testing.snooping_guard import read_oos  # noqa: E402

logger = get_logger(__name__)

# â”€â”€ Constantes du couple â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ASSET = "GBPUSD"
TF = "H4"
COUPLE_KEY = (ASSET, TF)
TRAIN_CUTOFF = pd.Timestamp("2022-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
RETRAIN_MONTHS = 6
CAPITAL_EUR = 10_000.0
RISK_PCT = 0.02
EMBARGO_DAYS = 2


def _build_features_for_split(df_split: pd.DataFrame) -> pd.DataFrame:
    """Construit le superset et sÃ©lectionne le top 15 C5 pour GBPUSD H4."""
    superset = build_superset(df_split, asset=ASSET)
    selected = list(FEATURES_SELECTED[COUPLE_KEY])
    available = [c for c in selected if c in superset.columns]
    missing = set(selected) - set(available)
    if missing:
        logger.warning("Features C5 manquantes dans le superset : %s", sorted(missing))
    return superset[available].dropna()


def _build_target_winner(_df: pd.DataFrame, pnl_brut: pd.Series) -> pd.Series:
    """Cible binaire : 1 si trade gagnant (pnl brut > 0)."""
    return (pnl_brut > 0).astype(int)


def _generate_bootstrap_signals(df: pd.DataFrame) -> pd.Series:
    """GÃ©nÃ¨re les signaux bootstrap basÃ©s sur les terciles du trend score.

    Top 20% trend â†’ LONG, bottom 20% â†’ SHORT. UtilisÃ© comme gÃ©nÃ©rateur
    primaire pour le flow mÃ©ta-labeling B1 (Ã©quivalent du Donchian dans
    les autres scripts c5).

    Args:
        df: DataFrame OHLC.

    Returns:
        SÃ©rie 1/-1/0 mÃªme index que df.
    """
    features = _build_features_for_split(df)
    if features.empty:
        return pd.Series(0, index=df.index, dtype=int)

    trend_cols = [
        c for c in ["slope_sma_20", "slope_sma_50", "dist_sma_200"]
        if c in features.columns
    ]
    if not trend_cols:
        return pd.Series(0, index=df.index, dtype=int)

    trend_score = features[trend_cols].mean(axis=1)
    signals = pd.Series(0, index=df.index, dtype=int)
    q80 = trend_score.quantile(0.80)
    q20 = trend_score.quantile(0.20)
    signals.loc[features.index[trend_score > q80]] = 1
    signals.loc[features.index[trend_score < q20]] = -1
    return signals


def _train_primary_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """EntraÃ®ne le modÃ¨le primaire rf avec les hyperparams C5 pour GBPUSD H4."""
    hp = HYPERPARAMS_TUNED[COUPLE_KEY]
    params = hp["params"]
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        min_samples_leaf=params.get("min_samples_leaf", 10),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train.values, y_train.values)
    return model


def _primary_signals(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    primary_signals: pd.Series | None = None,
) -> pd.Series:
    """Fix F1 : primary RF filtre les signaux bootstrap (jamais directionnel).

    La direction provient EXCLUSIVEMENT du signal bootstrap (top/bottom 20%
    du trend score). Le RF primaire filtre par P(winner) > threshold.
    La MetaLabelingRF en aval ajoute un second niveau de filtrage.

    Args:
        df: DataFrame OHLC.
        model: RF primaire entraÃ®nÃ© sur (features Ã  entrÃ©e bootstrap, y=winner).
        primary_signals: Signaux bootstrap sur df. Si None, ils sont gÃ©nÃ©rÃ©s
            ici via _generate_bootstrap_signals(df).
    """
    from app.models.meta_labeling_pipeline import filter_signals_by_meta_proba

    hp = HYPERPARAMS_TUNED[COUPLE_KEY]
    threshold = hp["threshold"]

    if primary_signals is None:
        primary_signals = _generate_bootstrap_signals(df)

    features = _build_features_for_split(df)
    if features.empty:
        return pd.Series(0, index=df.index, dtype=int)

    return filter_signals_by_meta_proba(
        df=df,
        primary_signals=primary_signals,
        features=features,
        model=model,
        threshold=threshold,
    )


def _trades_to_dataframe(
    trades: list[dict],
    cfg: Any,
    capital_eur: float = CAPITAL_EUR,
    risk_pct: float = RISK_PCT,
) -> pd.DataFrame:
    """Convertit la liste de trades du backtest en DataFrame avec sizing."""
    if not trades:
        return pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"])

    from app.backtest.sizing import compute_position_size, expected_pnl_eur

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.set_index("entry_time")
    df["Pips_Nets"] = df["pips_net"].astype(float)
    df["Pips_Bruts"] = df["pips_net"].astype(float)
    df["result"] = df["result"].astype(str)

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
    df["pnl"] = expected_pnl_eur(df["Pips_Nets"].values, lots, cfg)
    return df


def main() -> int:
    set_global_seeds()

    # â”€â”€ 1. Chargement GBPUSD H4 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"Chargement {ASSET} {TF}...")
    df = load_asset(ASSET, TF)
    cfg = ASSET_CONFIGS[ASSET]
    print(f"  {len(df)} barres, {df.index.min().date()} â†’ {df.index.max().date()}")

    # â”€â”€ 2. PrÃ©paration des dates de retrain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0
    retrain_dates = pd.date_range(start=TEST_START, end=df.index[-1], freq="6MS", inclusive="both")
    if len(retrain_dates) == 0:
        retrain_dates = pd.DatetimeIndex([TEST_START])

    all_trades_baseline: list[pd.DataFrame] = []
    all_trades_meta: list[pd.DataFrame] = []
    segments: list[dict[str, Any]] = []

    print(f"\nWalk-forward mÃ©ta-labeling (retrain {RETRAIN_MONTHS}M, test â‰¥ {TEST_START.date()})...")
    print(f"  {len(retrain_dates)} segments de retrain")

    for i, retrain_dt in enumerate(retrain_dates):
        if i + 1 < len(retrain_dates):
            segment_end = retrain_dates[i + 1] - pd.Timedelta(days=1)
        else:
            segment_end = df.index[-1]

        train_end = retrain_dt - pd.Timedelta(days=EMBARGO_DAYS)
        df_train = df.loc[:train_end]
        df_oos = df.loc[retrain_dt:segment_end]

        if df_train.empty or df_oos.empty:
            logger.warning("Segment %s: train (%d) ou OOS (%d) vide, skip.", retrain_dt.date(), len(df_train), len(df_oos))
            continue

        print(f"\nâ”€â”€ Segment {retrain_dt.date()} â†’ {segment_end.date()} â”€â”€")
        print(f"   Train: {df_train.index.min().date()} â†’ {df_train.index.max().date()} ({len(df_train)} barres)")
        print(f"   OOS:   {df_oos.index.min().date()} â†’ {df_oos.index.max().date()} ({len(df_oos)} barres)")

        # â”€â”€ 3a. EntraÃ®ner le modÃ¨le primaire sur train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            features_train = _build_features_for_split(df_train)
            if features_train.empty:
                logger.warning("Aucune feature train, skip segment.")
                continue

            # GÃ©nÃ©rer des signaux "baseline" pour crÃ©er les targets d'entraÃ®nement
            # On utilise un modÃ¨le prÃ©-entraÃ®nÃ© bootstrap pour gÃ©nÃ©rer des signaux initiaux
            # StratÃ©gie bootstrap : signaux basÃ©s sur les features de tendance
            trend_cols = [c for c in ["slope_sma_20", "slope_sma_50", "dist_sma_200"] if c in features_train.columns]
            if trend_cols:
                trend_score = features_train[trend_cols].mean(axis=1)
                bootstrap_signals = pd.Series(0, index=features_train.index, dtype=int)
                # Top 20% bullish â†’ LONG, bottom 20% bearish â†’ SHORT
                q80 = trend_score.quantile(0.80)
                q20 = trend_score.quantile(0.20)
                bootstrap_signals[trend_score > q80] = 1
                bootstrap_signals[trend_score < q20] = -1
            else:
                bootstrap_signals = pd.Series(0, index=features_train.index, dtype=int)

            # Backtest bootstrap pour gÃ©nÃ©rer des trades et labels
            bt_bootstrap = run_deterministic_backtest(
                df=df_train,
                signals=bootstrap_signals,
                tp_pips=cfg.tp_points,
                sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
                asset_config=cfg,
            )
            trades_bootstrap: list[dict] = bt_bootstrap.get("trades", [])

            if len(trades_bootstrap) < 10:
                logger.warning("Bootstrap: seulement %d trades train, skip mÃ©ta.", len(trades_bootstrap))
                # Fallback: utiliser les signaux bootstrap directement sur OOS
                signals_oos = pd.Series(0, index=df_oos.index, dtype=int)
                features_oos = _build_features_for_split(df_oos)
                common_oos = df_oos.index.intersection(features_oos.index)
                if len(common_oos) > 0 and trend_cols:
                    trend_oos = features_oos.loc[common_oos, trend_cols].mean(axis=1)
                    signals_oos.loc[common_oos[trend_oos > q80]] = 1
                    signals_oos.loc[common_oos[trend_oos < q20]] = -1
                bt_oos = run_deterministic_backtest(
                    df=df_oos, signals=signals_oos,
                    tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                    window_hours=cfg.window_hours,
                    commission_pips=cfg.commission_pips,
                    slippage_pips=half_cost,
                    pip_size=cfg.pip_size,
                    asset_config=cfg,
                )
                trades_oos_df = _trades_to_dataframe(bt_oos.get("trades", []), cfg=cfg)
                segments.append({
                    "start": str(retrain_dt.date()), "end": str(segment_end.date()),
                    "n_train_trades": len(trades_bootstrap),
                    "n_oos_trades_baseline": int(bt_oos.get("total_trades", 0)),
                    "n_oos_trades_meta": int(bt_oos.get("total_trades", 0)),
                    "sharpe_baseline": float(bt_oos.get("sharpe", 0.0)),
                    "sharpe_meta": float(bt_oos.get("sharpe", 0.0)),
                    "meta_disabled": True, "threshold": 0.0,
                })
                if not trades_oos_df.empty:
                    all_trades_baseline.append(trades_oos_df)
                    all_trades_meta.append(trades_oos_df)
                continue

            # EntraÃ®ner le primaire rf sur les signaux bootstrap
            entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_bootstrap])
            common_train_idx = features_train.index.intersection(entry_times_train)
            if len(common_train_idx) < 5:
                logger.warning("Seulement %d features alignÃ©es avec trades train.", len(common_train_idx))
                continue

            X_primary = features_train.loc[common_train_idx]
            trades_df_bootstrap = _trades_to_dataframe(trades_bootstrap, cfg=cfg)
            pnl_aligned = trades_df_bootstrap.loc[
                trades_df_bootstrap.index.intersection(common_train_idx), "Pips_Nets"
            ]
            y_primary = _build_target_winner(df_train, pnl_aligned)

            if y_primary.nunique() < 2:
                logger.warning("Une seule classe dans y_primary, skip mÃ©ta.")
                continue

            primary_model = _train_primary_model(X_primary, y_primary)
            print(f"   ModÃ¨le primaire rf entraÃ®nÃ© sur {len(X_primary)} trades train")

            # â”€â”€ 3b. GÃ©nÃ©rer signaux primaires sur train â†’ mÃ©ta-labels â”€â”€â”€
            signals_train_primary = _primary_signals(df_train, primary_model)
            bt_train_primary = run_deterministic_backtest(
                df=df_train,
                signals=signals_train_primary,
                tp_pips=cfg.tp_points,
                sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
                asset_config=cfg,
            )
            trades_train_primary: list[dict] = bt_train_primary.get("trades", [])

            if len(trades_train_primary) < 10:
                logger.warning("Primaire: seulement %d trades train.", len(trades_train_primary))
                # Fallback: baseline sans mÃ©ta sur OOS
                signals_oos = _primary_signals(df_oos, primary_model)
                bt_oos = run_deterministic_backtest(
                    df=df_oos, signals=signals_oos,
                    tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                    window_hours=cfg.window_hours,
                    commission_pips=cfg.commission_pips,
                    slippage_pips=half_cost,
                    pip_size=cfg.pip_size,
                    asset_config=cfg,
                )
                trades_oos_df = _trades_to_dataframe(bt_oos.get("trades", []), cfg=cfg)
                segments.append({
                    "start": str(retrain_dt.date()), "end": str(segment_end.date()),
                    "n_train_trades": len(trades_train_primary),
                    "n_oos_trades_baseline": int(bt_oos.get("total_trades", 0)),
                    "n_oos_trades_meta": int(bt_oos.get("total_trades", 0)),
                    "sharpe_baseline": float(bt_oos.get("sharpe", 0.0)),
                    "sharpe_meta": float(bt_oos.get("sharpe", 0.0)),
                    "meta_disabled": True, "threshold": 0.0,
                })
                if not trades_oos_df.empty:
                    all_trades_baseline.append(trades_oos_df)
                    all_trades_meta.append(trades_oos_df)
                continue

            # â”€â”€ 3c. Extraire features aux barres d'entrÃ©e â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            entry_times_primary = pd.to_datetime([t["entry_time"] for t in trades_train_primary])
            common_meta_idx = features_train.index.intersection(entry_times_primary)
            if len(common_meta_idx) < 5:
                logger.warning("Seulement %d features pour mÃ©ta.", len(common_meta_idx))
                continue

            x_meta_train = features_train.loc[common_meta_idx]
            trades_primary_df = _trades_to_dataframe(trades_train_primary, cfg=cfg)
            pnl_meta_aligned = trades_primary_df.loc[
                trades_primary_df.index.intersection(common_meta_idx), "Pips_Nets"
            ]
            y_meta = _build_target_winner(df_train, pnl_meta_aligned)

            # â”€â”€ 3d. EntraÃ®ner MetaLabelingRF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            meta_config = MetaLabelingConfig(
                n_estimators=HYPERPARAMS_TUNED[COUPLE_KEY]["params"].get("n_estimators", 100),
                max_depth=HYPERPARAMS_TUNED[COUPLE_KEY]["params"].get("max_depth", 10),
                min_samples_leaf=HYPERPARAMS_TUNED[COUPLE_KEY]["params"].get("min_samples_leaf", 10),
                threshold_candidates=(0.45, 0.50, 0.55, 0.60),
                min_trade_retention=0.20,
            )
            meta = MetaLabelingRF(config=meta_config)
            meta.fit(x_meta_train, y_meta)

            # Calibrer le seuil sur train
            if not meta.disabled:
                entry_to_trade: dict[pd.Timestamp, dict] = {}
                for t in trades_train_primary:
                    et = pd.Timestamp(t["entry_time"])
                    entry_to_trade[et] = t

                def _sharpe_for_threshold(mask: pd.Series, _entry_map: dict = entry_to_trade) -> float:
                    accepted_indices = set(mask[mask].index)
                    filtered_trades = [
                        t for et, t in _entry_map.items()
                        if et in accepted_indices
                    ]
                    if len(filtered_trades) < 5:
                        return -np.inf
                    return sharpe_daily_from_trades(filtered_trades)

                meta.calibrate_threshold(x_meta_train, _sharpe_for_threshold)
                print(f"   MÃ©ta-labeling: seuil={meta.threshold:.2f}, disabled={meta.disabled}")

            # â”€â”€ 3e. Appliquer sur OOS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Baseline: signaux primaires sans filtre
            signals_oos_primary = _primary_signals(df_oos, primary_model)
            bt_oos_baseline = run_deterministic_backtest(
                df=df_oos,
                signals=signals_oos_primary,
                tp_pips=cfg.tp_points,
                sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost,
                pip_size=cfg.pip_size,
                asset_config=cfg,
            )
            trades_oos_baseline = _trades_to_dataframe(bt_oos_baseline.get("trades", []), cfg=cfg)

            # Avec mÃ©ta-labeling
            if meta.disabled:
                trades_oos_meta = trades_oos_baseline.copy() if not trades_oos_baseline.empty else trades_oos_baseline
            else:
                # Features OOS avec historique complet pour rolling indicators
                df_oos_with_history = df.loc[:segment_end]
                features_oos_full = _build_features_for_split(df_oos_with_history)
                features_oos = features_oos_full.loc[features_oos_full.index.isin(df_oos.index)]

                signal_bars = signals_oos_primary[signals_oos_primary != 0]
                if len(signal_bars) == 0 or features_oos.empty:
                    trades_oos_meta = trades_oos_baseline.copy() if not trades_oos_baseline.empty else trades_oos_baseline
                else:
                    common_oos_idx = features_oos.index.intersection(signal_bars.index)
                    if len(common_oos_idx) == 0:
                        trades_oos_meta = trades_oos_baseline.copy() if not trades_oos_baseline.empty else trades_oos_baseline
                    else:
                        x_oos_signal = features_oos.loc[common_oos_idx]
                        keep_mask = meta.predict(x_oos_signal)
                        keep_indices = set(common_oos_idx[keep_mask])

                        filtered_signals = signals_oos_primary.copy()
                        for idx in signal_bars.index:
                            if idx not in keep_indices:
                                filtered_signals.loc[idx] = 0

                        bt_oos_meta = run_deterministic_backtest(
                            df=df_oos,
                            signals=filtered_signals,
                            tp_pips=cfg.tp_points,
                            sl_pips=cfg.sl_points,
                            window_hours=cfg.window_hours,
                            commission_pips=cfg.commission_pips,
                            slippage_pips=half_cost,
                            pip_size=cfg.pip_size,
                            asset_config=cfg,
                        )
                        trades_oos_meta = _trades_to_dataframe(bt_oos_meta.get("trades", []), cfg=cfg)

            # â”€â”€ 3f. MÃ©triques segment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            sharpe_base = float(bt_oos_baseline.get("sharpe", 0.0))
            n_base = int(bt_oos_baseline.get("total_trades", 0))
            n_meta = len(trades_oos_meta)
            sharpe_meta = sharpe_base  # fallback
            if n_meta > 0 and "pnl" in trades_oos_meta.columns:
                pnl_series = trades_oos_meta["pnl"]
                if len(pnl_series) >= 2:
                    returns = pnl_series.pct_change().dropna() if (pnl_series != 0).all() else pnl_series.diff().dropna()
                    if len(returns) >= 2 and returns.std() > 0:
                        sharpe_meta = float(returns.mean() / returns.std() * np.sqrt(252))

            segments.append({
                "start": str(retrain_dt.date()),
                "end": str(segment_end.date()),
                "n_train_trades": len(trades_train_primary),
                "n_oos_trades_baseline": n_base,
                "n_oos_trades_meta": n_meta,
                "sharpe_baseline": sharpe_base,
                "sharpe_meta": sharpe_meta,
                "meta_disabled": meta.disabled,
                "threshold": meta.threshold if not meta.disabled else 0.0,
            })
            print(f"   Baseline: {n_base} trades, Sharpe={sharpe_base:.3f}")
            print(f"   MÃ©ta:     {n_meta} trades, Sharpe={sharpe_meta:.3f}")

            if not trades_oos_baseline.empty:
                all_trades_baseline.append(trades_oos_baseline)
            if not trades_oos_meta.empty:
                all_trades_meta.append(trades_oos_meta)

        except Exception as exc:
            logger.error("Erreur segment %s: %s", retrain_dt.date(), exc)
            continue

    # â”€â”€ 4. AgrÃ©gation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if all_trades_meta:
        all_meta_df = pd.concat(all_trades_meta).sort_index()
    else:
        all_meta_df = pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"])

    if all_trades_baseline:
        all_baseline_df = pd.concat(all_trades_baseline).sort_index()
    else:
        all_baseline_df = pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"])

    # â”€â”€ 5. MÃ©triques globales â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics_meta: dict[str, Any] = {}
    metrics_baseline: dict[str, Any] = {}

    if not all_meta_df.empty:
        metrics_meta = compute_metrics(all_meta_df, asset_cfg=cfg, capital_eur=CAPITAL_EUR, df=df)
    else:
        metrics_meta = {"sharpe": 0.0, "trades": 0, "win_rate": 0.0, "max_dd_pct": 0.0}

    if not all_baseline_df.empty:
        metrics_baseline = compute_metrics(all_baseline_df, asset_cfg=cfg, capital_eur=CAPITAL_EUR, df=df)
    else:
        metrics_baseline = {"sharpe": 0.0, "trades": 0, "win_rate": 0.0, "max_dd_pct": 0.0}

    # â”€â”€ 6. validate_edge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    equity_meta: pd.Series
    if not all_meta_df.empty and "pnl" in all_meta_df.columns:
        equity_meta = all_meta_df["pnl"].cumsum() + CAPITAL_EUR
    else:
        equity_meta = pd.Series([CAPITAL_EUR], index=[df.index[0]])

    n_trials_cumul = 26  # hÃ©ritÃ©s Phase A + C1-C5 (Ã  ajuster par utilisateur aprÃ¨s JOURNAL.md)
    report = validate_edge(
        equity=equity_meta,
        trades=all_meta_df if not all_meta_df.empty else pd.DataFrame(columns=["pnl"]),
        n_trials=n_trials_cumul,
    )

    # â”€â”€ 7. read_oos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    read_oos(
        prompt="pivot_v4_phase_b_c5_b1",
        hypothesis="B1_C5_GBPUSD_H4_meta_labeling",
        sharpe=float(metrics_meta.get("sharpe", 0.0)),
        n_trades=int(metrics_meta.get("trades", 0)),
    )

    # â”€â”€ 8. Sauvegarde â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out: dict[str, Any] = {
        "hypothesis": "B1_C5_GBPUSD_H4_meta_labeling",
        "phase": "Phase B C5 â€” B1",
        "asset": ASSET,
        "tf": TF,
        "model_primary": "rf",
        "hyperparams_primary": HYPERPARAMS_TUNED[COUPLE_KEY]["params"],
        "threshold_primary": HYPERPARAMS_TUNED[COUPLE_KEY]["threshold"],
        "features": list(FEATURES_SELECTED[COUPLE_KEY]),
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "test_start": str(TEST_START.date()),
        "retrain_months": RETRAIN_MONTHS,
        "capital_eur": CAPITAL_EUR,
        "risk_per_trade": RISK_PCT,
        "n_trials_cumul": n_trials_cumul,
        "config": {
            "spread_pips": cfg.spread_pips,
            "slippage_pips": cfg.slippage_pips,
            "tp_points": cfg.tp_points,
            "sl_points": cfg.sl_points,
            "window_hours": cfg.window_hours,
            "pip_size": cfg.pip_size,
            "pip_value_eur": cfg.pip_value_eur,
        },
        "metrics_baseline": {k: v for k, v in metrics_baseline.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "metrics_meta": {k: v for k, v in metrics_meta.items() if isinstance(v, (int, float, str, bool, type(None)))},
        "sharpe_improvement": float(metrics_meta.get("sharpe", 0.0)) - float(metrics_baseline.get("sharpe", 0.0)),
        "segments": segments,
        "validate_edge": {
            "go": report.go,
            "reasons": report.reasons,
            "metrics": report.metrics,
        },
    }

    out_path = Path("predictions/phase_b_c5_b1_gbpusd_h4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"Phase B C5 â€” B1 GBPUSD H4 terminÃ©.")
    print(f"  Sharpe baseline : {metrics_baseline.get('sharpe', 0):.3f}")
    print(f"  Sharpe mÃ©ta     : {metrics_meta.get('sharpe', 0):.3f}")
    print(f"  AmÃ©lioration    : {out['sharpe_improvement']:+.3f}")
    print(f"  Trades baseline : {metrics_baseline.get('trades', 0)}")
    print(f"  Trades mÃ©ta     : {metrics_meta.get('trades', 0)}")
    print(f"  Verdict         : {'GO âœ…' if report.go else 'NO-GO âŒ'}")
    print(f"  Raisons         : {report.reasons}")
    print(f"\nRÃ©sultats sauvegardÃ©s : {out_path}")
    return 0 if report.go else 1


if __name__ == "__main__":
    sys.exit(main())
