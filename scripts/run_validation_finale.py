"""Prompt 18 â€” Validation finale GO/NO-GO production.

Replay complet du portfolio des 6 stratÃ©gies GO sur test 2024-2025.
Produit le rapport predictions/validation_finale.json et docs/v3_final_report.md.

RÃ¨gles :
- n_trials = 29 (28 hÃ©ritÃ©s JOURNAL.md + 1 prompt 18)
- 0 modification de config
- Benchmarks B&H equal-weight + Monte Carlo random obligatoires
- verify_no_snooping avant verdict
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.backtest.deterministic import run_deterministic_backtest  # noqa: E402
from app.backtest.metrics import (  # noqa: E402
    compute_metrics,
    sharpe_ratio,
)
from app.backtest.sizing import compute_position_size, expected_pnl_eur  # noqa: E402
from app.config.features_selected import FEATURES_SELECTED  # noqa: E402
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.superset import build_superset  # noqa: E402
from app.models.candidates import build_stacking  # noqa: E402
from app.strategies.donchian import DonchianBreakout  # noqa: E402
from app.testing.snooping_guard import (  # noqa: E402
    n_trials_from_history,
    n_unique_hypotheses,
    read_oos,
)

logger = get_logger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. Configuration
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

TRAIN_CUTOFF = pd.Timestamp("2022-12-31", tz="UTC")
# Fix F14 : 2023 est la validation set (constitution Â§3). UtilisÃ© pour
# calibrer le seuil de probabilitÃ© sans consommer de n_trials sur le test.
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
CAPITAL_EUR = 10_000.0
RISK_PCT = 0.02
DONCHIAN_N = 20
DONCHIAN_M = 20

# Si True, le seuil de probabilitÃ© est calibrÃ© sur 2023 (val set) plutÃ´t
# que d'utiliser HYPERPARAMS_TUNED[(asset, tf)]["threshold"]. Fix F14.
CALIBRATE_THRESHOLD_ON_VAL = False
# Fix F5 : n_trials calculÃ© dynamiquement depuis read_history plutÃ´t
# qu'une constante hardcodÃ©e. La valeur historique 29 est utilisÃ©e comme
# plancher si l'historique n'a pas Ã©tÃ© correctement maintenu.
N_TRIALS_FLOOR = 29

# 6 stratÃ©gies GO validÃ©es
STRATEGIES_DONCHIAN_ML: list[dict[str, Any]] = [
    {"asset": "GBPUSD", "tf": "D1", "model": "rf", "json_ref": "predictions/phase_b_c5_extra_gbpusd_d1.json"},
    {"asset": "EURUSD", "tf": "D1", "model": "stacking", "json_ref": "predictions/phase_b_c5_extra_eurusd_d1.json"},
    {"asset": "USDCHF", "tf": "D1", "model": "stacking", "json_ref": "predictions/phase_b_c5_extra_usdchf_d1.json"},
    {"asset": "ETHUSD", "tf": "D1", "model": "hgbm", "json_ref": "predictions/phase_b_c5_b3_ethusd_d1.json"},
]

STRATEGIES_WALKFORWARD: list[dict[str, Any]] = [
    {"asset": "GBPUSD", "tf": "H4", "model": "rf", "json_ref": "predictions/phase_b_c5_b1_gbpusd_h4.json", "type": "meta_labeling_wf"},
    {"asset": "EURUSD", "tf": "H4", "model": "rf", "json_ref": "predictions/h_new3_eurusd_h4.json", "type": "meanrev_wf"},
]

ASSETS_FOR_BENCHMARK = sorted(set(
    s["asset"] for s in STRATEGIES_DONCHIAN_ML + STRATEGIES_WALKFORWARD
))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. Helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _build_features(df: pd.DataFrame, asset: str, tf: str) -> pd.DataFrame:
    """Construit le superset et sÃ©lectionne le top 15 pour le couple."""
    superset = build_superset(df, asset=asset)
    key = (asset, tf)
    selected = list(FEATURES_SELECTED.get(key, []))
    available = [c for c in selected if c in superset.columns]
    missing = set(selected) - set(available)
    if missing:
        logger.warning("Features manquantes %s: %s", key, sorted(missing))
    return superset[available].dropna()


def _generate_donchian_signals(df: pd.DataFrame) -> pd.Series:
    """Signaux Donchian (N=20, M=20)."""
    strategy = DonchianBreakout(params={"N": DONCHIAN_N, "M": DONCHIAN_M})
    return strategy.generate_signals(df)


def _target_winner(pnl_net: pd.Series) -> pd.Series:
    return (pnl_net > 0).astype(int)


def _trades_to_equity(
    trades: list[dict],
    cfg: Any,
    capital_eur: float = CAPITAL_EUR,
    risk_pct: float = RISK_PCT,
) -> tuple[pd.Series, pd.DataFrame]:
    """Convertit trades â†’ equity curve + DataFrame avec sizing.

    Returns:
        (equity: pd.Series indexÃ©e par entry_time, trades_df: pd.DataFrame)
    """
    if not trades:
        empty_idx = pd.DatetimeIndex([], tz="UTC")
        return pd.Series(dtype=float), pd.DataFrame(
            columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"],
        )

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.set_index("entry_time").sort_index()
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

    equity = capital_eur + df["pnl"].cumsum()
    return equity, df


def _train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    asset: str,
    tf: str,
) -> Any:
    """EntraÃ®ne le modÃ¨le selon le type (rf/hgbm/stacking) avec hyperparams C5."""
    key = (asset, tf)
    hp = HYPERPARAMS_TUNED.get(key, {})
    params = hp.get("params", {})

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            min_samples_leaf=params.get("min_samples_leaf", 10),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "hgbm":
        model = HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            max_leaf_nodes=params.get("max_leaf_nodes", 15),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            class_weight="balanced",
            random_state=42,
            early_stopping=False,
        )
    elif model_type == "stacking":
        model = build_stacking(seed=42)
    else:
        raise ValueError(f"ModÃ¨le inconnu: {model_type}")

    model.fit(X.values, y.values)
    return model


def _generate_model_signals(
    df: pd.DataFrame,
    model: Any,
    asset: str,
    tf: str,
    primary_signals: pd.Series | None = None,
) -> pd.Series:
    """MÃ©ta-labeling fidÃ¨le (fix F1) : filtre des signaux primaires par P(winner).

    Args:
        df: DataFrame OHLC du segment Ã©valuÃ©.
        model: ModÃ¨le entraÃ®nÃ© sur (features Ã  l'entrÃ©e, y=winner).
        asset, tf: Identifiant du couple (pour features_selected / threshold).
        primary_signals: Signaux primaires Donchian (1/-1/0). REQUIS â€” la
            distribution test doit correspondre Ã  la distribution train.

    Returns:
        SÃ©rie identique Ã  primary_signals avec les signaux dont la probabilitÃ©
        de winner est sous le threshold remis Ã  0. La DIRECTION provient
        toujours du signal primaire, jamais d'un trend_sign synthÃ©tique.
    """
    if primary_signals is None:
        raise ValueError(
            "primary_signals est requis (fix F1). Ne pas appeler "
            "_generate_model_signals sans signaux primaires Donchian."
        )

    key = (asset, tf)
    hp = HYPERPARAMS_TUNED.get(key, {})
    threshold = hp.get("threshold", 0.5)

    features = _build_features(df, asset, tf)
    if features.empty:
        return pd.Series(0, index=df.index, dtype=int)

    signals = pd.Series(0, index=df.index, dtype=int)
    primary_mask = primary_signals.reindex(df.index, fill_value=0) != 0
    candidate_idx = df.index[primary_mask].intersection(features.index)
    if len(candidate_idx) == 0:
        return signals

    X_candidates = features.loc[candidate_idx]
    proba = model.predict_proba(X_candidates.values)[:, 1]
    keep_mask = proba > threshold

    kept_idx = candidate_idx[keep_mask]
    signals.loc[kept_idx] = primary_signals.loc[kept_idx].astype(int)
    return signals


def _calibrate_threshold_on_val(
    df_full: pd.DataFrame,
    model: Any,
    asset: str,
    tf: str,
    cfg: Any,
    half_cost: float,
    threshold_candidates: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
) -> tuple[float, dict[float, float]]:
    """Fix F14 : calibre le seuil de probabilitÃ© sur 2023 (val set).

    Pour chaque seuil candidat :
    1. GÃ©nÃ¨re les signaux Donchian sur le val set (2023).
    2. Filtre par P(winner) > threshold.
    3. Backtest, calcule Sharpe linÃ©aire (fix F2 via sharpe_daily_from_trades).
    4. Retient le seuil maximisant le Sharpe sur 2023.

    Le test set â‰¥ 2024 n'est JAMAIS consultÃ© ici â†’ pas de consommation de n_trials.

    Args:
        df_full: DataFrame OHLC complet (assez d'historique pour les features).
        model: ModÃ¨le dÃ©jÃ  entraÃ®nÃ© sur train â‰¤ 2022.
        asset, tf: Identifiant du couple.
        cfg: AssetConfig pour costs/TP/SL.
        half_cost: spread+slippage divisÃ© par 2.
        threshold_candidates: Seuils Ã  tester.

    Returns:
        (best_threshold, dict {threshold: sharpe_val}). Si val vide ou aucun
        signal, retourne (0.50, {}).
    """
    df_val = df_full.loc[VAL_START:VAL_END]
    if df_val.empty:
        logger.warning("Val 2023 vide pour %s %s â€” fallback threshold=0.50", asset, tf)
        return 0.50, {}

    donchian_val = _generate_donchian_signals(df_val)
    if (donchian_val != 0).sum() < 5:
        logger.warning(
            "Trop peu de signaux Donchian sur val 2023 (%d) â€” fallback 0.50",
            int((donchian_val != 0).sum()),
        )
        return 0.50, {}

    val_scores: dict[float, float] = {}
    for t in threshold_candidates:
        # Override temporaire via kwargs : on appelle filter_signals_by_meta_proba directement
        from app.models.meta_labeling_pipeline import filter_signals_by_meta_proba
        features_val = _build_features(df_full, asset, tf)
        if features_val.empty:
            continue
        signals_val = filter_signals_by_meta_proba(
            df=df_val,
            primary_signals=donchian_val,
            features=features_val.loc[features_val.index.isin(df_val.index)],
            model=model,
            threshold=t,
        )
        if (signals_val != 0).sum() < 3:
            val_scores[t] = float("-inf")
            continue

        bt = run_deterministic_backtest(
            df=df_val, signals=signals_val,
            tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
        )
        trades = bt.get("trades", [])
        if len(trades) < 3:
            val_scores[t] = float("-inf")
            continue
        val_scores[t] = float(sharpe_ratio(
            pd.Series([t["pips_net"] for t in trades]) / cfg.sl_points,
            annual_factor=252.0,
        ))

    finite_scores = {k: v for k, v in val_scores.items() if np.isfinite(v)}
    if not finite_scores:
        logger.warning("Aucun seuil viable sur val 2023 â€” fallback 0.50")
        return 0.50, val_scores
    best_t = max(finite_scores, key=finite_scores.get)
    logger.info(
        "Seuil calibrÃ© sur val 2023 pour %s %s : %.2f (Sharpe val=%.3f)",
        asset, tf, best_t, finite_scores[best_t],
    )
    return best_t, val_scores


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. Backtest Donchian + ML (simple train/test)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def backtest_donchian_ml(
    asset: str, tf: str, model_type: str,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Rejoue le pipeline Donchian+ML. Retourne (equity, trades_df, metrics)."""
    print(f"\n{'='*60}")
    print(f"[Donchian+ML] {asset} {tf} ({model_type})")
    print(f"{'='*60}")

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]
    key = (asset, tf)
    hp = HYPERPARAMS_TUNED.get(key, {})

    print(f"  {len(df)} barres, {df.index.min().date()} â†’ {df.index.max().date()}")

    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]

    if df_train.empty or df_test.empty:
        logger.error("Split vide pour %s %s", asset, tf)
        return pd.Series(dtype=float), pd.DataFrame(), {}

    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # Train: gÃ©nÃ©ration target Donchian
    donchian_signals_train = _generate_donchian_signals(df_train)
    n_signals_train = int((donchian_signals_train != 0).sum())
    print(f"  Signaux Donchian train: {n_signals_train}")

    bt_train = run_deterministic_backtest(
        df=df_train, signals=donchian_signals_train,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    trades_train: list[dict] = bt_train.get("trades", [])
    print(f"  Trades Donchian train: {len(trades_train)}")

    if len(trades_train) < 20:
        logger.warning("Seulement %d trades train, insuffisant.", len(trades_train))
        return pd.Series(dtype=float), pd.DataFrame(), {}

    # Features + labels train
    features_train = _build_features(df_train, asset, tf)
    if features_train.empty:
        return pd.Series(dtype=float), pd.DataFrame(), {}

    entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_train])
    common_train_idx = features_train.index.intersection(entry_times_train)
    if len(common_train_idx) < 10:
        logger.warning("Alignement features/trades insuffisant.")
        return pd.Series(dtype=float), pd.DataFrame(), {}

    X_train = features_train.loc[common_train_idx]
    _, trades_df_train = _trades_to_equity(trades_train, cfg=cfg)
    pnl_aligned = trades_df_train.loc[
        trades_df_train.index.intersection(common_train_idx), "Pips_Nets"
    ]
    y_train = _target_winner(pnl_aligned)

    if y_train.nunique() < 2:
        logger.warning("Classe unique dans y_train.")
        return pd.Series(dtype=float), pd.DataFrame(), {}

    # EntraÃ®nement
    model = _train_model(X_train, y_train, model_type, asset, tf)
    acc_train = float((model.predict(X_train.values) == y_train.values).mean())
    print(f"  Accuracy train: {acc_train:.3f}")

    # Fix F14 : calibration optionnelle du seuil sur 2023 (val set).
    if CALIBRATE_THRESHOLD_ON_VAL:
        calibrated_threshold, val_scores = _calibrate_threshold_on_val(
            df_full=df, model=model, asset=asset, tf=tf,
            cfg=cfg, half_cost=half_cost,
        )
        print(f"  Seuil calibrÃ© sur val 2023 : {calibrated_threshold:.2f}")
        print(f"    Scores Sharpe val par seuil : {val_scores}")
        # Override le seuil HYPERPARAMS_TUNED pour ce couple, en mÃ©moire
        HYPERPARAMS_TUNED.setdefault(key, {})["threshold"] = calibrated_threshold

    # Test: signaux Donchian â†’ filtrage par mÃ©ta-labeling (fix F1)
    df_test_with_history = df.loc[:df_test.index[-1]]
    features_test = _build_features(df_test_with_history, asset, tf)
    features_test = features_test.loc[features_test.index.isin(df_test.index)]

    donchian_signals_test = _generate_donchian_signals(df_test)
    n_primary_test = int((donchian_signals_test != 0).sum())
    print(f"  Signaux Donchian test (primaires): {n_primary_test}")

    signals_test = _generate_model_signals(
        df_test, model, asset, tf,
        primary_signals=donchian_signals_test,
    )
    n_test_signals = int((signals_test != 0).sum())
    print(f"  Signaux test (aprÃ¨s meta-filter): {n_test_signals}")

    if n_test_signals == 0:
        logger.warning("0 signal sur test pour %s %s", asset, tf)
        return pd.Series(dtype=float), pd.DataFrame(), {}

    bt_test = run_deterministic_backtest(
        df=df_test, signals=signals_test,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    trades_test: list[dict] = bt_test.get("trades", [])
    equity, trades_df = _trades_to_equity(trades_test, cfg=cfg)

    metrics = compute_metrics(trades_df, asset_cfg=cfg, capital_eur=CAPITAL_EUR, df=df_test) if not trades_df.empty else {}
    print(f"  Trades test: {len(trades_test)}, Sharpe: {metrics.get('sharpe', 0):.2f}")

    info = {
        "asset": asset, "tf": tf, "model": model_type,
        "n_train_trades": len(trades_train),
        "n_test_trades": len(trades_test),
        "accuracy_train": acc_train,
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "wr": float(metrics.get("win_rate", 0.0)),
        "max_dd_pct": float(metrics.get("max_dd_pct", 0.0)),
    }
    return equity, trades_df, info


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. Backtest walk-forward mÃ©ta-labeling (GBPUSD H4)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def backtest_wf_meta_labeling(
    asset: str, tf: str, model_type: str,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Walk-forward expanding window avec mÃ©ta-labeling RF.

    Logique simplifiÃ©e reproduisant run_phase_b_c5_b1_meta_labeling.py.
    """
    print(f"\n{'='*60}")
    print(f"[WF Meta-Labeling] {asset} {tf} ({model_type})")
    print(f"{'='*60}")

    from app.models.meta_labeling import MetaLabelingConfig, MetaLabelingRF  # noqa: E402

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]
    key = (asset, tf)
    hp = HYPERPARAMS_TUNED.get(key, {})
    threshold = hp.get("threshold", 0.5)

    print(f"  {len(df)} barres, {df.index.min().date()} â†’ {df.index.max().date()}")

    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0
    retrain_dates = pd.date_range(
        start=TEST_START, end=df.index[-1], freq="6MS", inclusive="both",
    )
    if len(retrain_dates) == 0:
        retrain_dates = pd.DatetimeIndex([TEST_START])

    all_equities: list[pd.Series] = []
    all_trades_dfs: list[pd.DataFrame] = []
    n_train_total = 0

    print(f"  {len(retrain_dates)} segments de retrain")

    for i, retrain_dt in enumerate(retrain_dates):
        if i + 1 < len(retrain_dates):
            segment_end = retrain_dates[i + 1] - pd.Timedelta(days=1)
        else:
            segment_end = df.index[-1]

        embargo_days = 2
        train_end = retrain_dt - pd.Timedelta(days=embargo_days)
        df_train = df.loc[:train_end]
        df_oos = df.loc[retrain_dt:segment_end]

        if df_train.empty or df_oos.empty:
            continue

        # Construction features + target
        features_train = _build_features(df_train, asset, tf)
        if features_train.empty:
            continue

        # Signaux bootstrap pour gÃ©nÃ©rer des labels d'entraÃ®nement
        trend_cols = [c for c in ["slope_sma_20", "slope_sma_50", "dist_sma_200"]
                      if c in features_train.columns]
        if not trend_cols:
            continue

        trend_score = features_train[trend_cols].mean(axis=1)
        bootstrap_signals = pd.Series(0, index=features_train.index, dtype=int)
        q80 = trend_score.quantile(0.80)
        q20 = trend_score.quantile(0.20)
        bootstrap_signals[trend_score > q80] = 1
        bootstrap_signals[trend_score < q20] = -1

        bt_bootstrap = run_deterministic_backtest(
            df=df_train, signals=bootstrap_signals,
            tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
        )
        trades_bootstrap: list[dict] = bt_bootstrap.get("trades", [])
        if len(trades_bootstrap) < 10:
            continue

        entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_bootstrap])
        common_idx = features_train.index.intersection(entry_times_train)
        if len(common_idx) < 5:
            continue

        X_primary = features_train.loc[common_idx]
        _, trades_df_bs = _trades_to_equity(trades_bootstrap, cfg=cfg)
        pnl_aligned = trades_df_bs.loc[
            trades_df_bs.index.intersection(common_idx), "Pips_Nets"
        ]
        y_primary = _target_winner(pnl_aligned)
        if y_primary.nunique() < 2:
            continue

        primary_model = _train_model(X_primary, y_primary, model_type, asset, tf)
        n_train_total += len(X_primary)

        # Signaux primaires sur OOS : Donchian (fix F1) puis filtre meta-label
        donchian_oos = _generate_donchian_signals(df_oos)
        signals_primary = _generate_model_signals(
            df_oos, primary_model, asset, tf,
            primary_signals=donchian_oos,
        )
        n_primary = int((signals_primary != 0).sum())

        # MÃ©ta-labeling sur signaux primaires
        meta_labeler = MetaLabelingRF(
            config=MetaLabelingConfig(
                n_estimators=100, max_depth=4, min_samples_leaf=10,
            ),
        )
        # EntraÃ®ner le mÃ©ta-labeler sur les trades train
        meta_labeler.fit(X_primary, y_primary)

        # Backtest des signaux primaires sur OOS
        bt_oos_primary = run_deterministic_backtest(
            df=df_oos, signals=signals_primary,
            tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
        )
        trades_primary_oos: list[dict] = bt_oos_primary.get("trades", [])

        # Filtrer les trades OOS avec le mÃ©ta-modÃ¨le
        if trades_primary_oos and not meta_labeler.disabled:
            entry_times_oos = pd.to_datetime(
                [t["entry_time"] for t in trades_primary_oos]
            )
            features_oos = _build_features(df_oos, asset, tf)
            common_oos = features_oos.index.intersection(entry_times_oos)
            if len(common_oos) > 0:
                X_oos = features_oos.loc[common_oos]
                meta_mask = meta_labeler.predict(X_oos)  # ndarray[bool]
                kept_set = set(common_oos[meta_mask])
                trades_oos = [
                    t for t in trades_primary_oos
                    if pd.Timestamp(t["entry_time"]) in kept_set
                ]
            else:
                trades_oos = trades_primary_oos
        else:
            trades_oos = trades_primary_oos

        if trades_oos:
            equity_seg, trades_df_seg = _trades_to_equity(trades_oos, cfg=cfg)
            all_equities.append(equity_seg)
            all_trades_dfs.append(trades_df_seg)

    if not all_equities:
        logger.warning("Aucun trade pour %s %s WF", asset, tf)
        return pd.Series(dtype=float), pd.DataFrame(), {}

    # ConcatÃ©ner les equity curves (chaque segment repart du capital prÃ©cÃ©dent)
    all_trades = pd.concat(all_trades_dfs).sort_index()
    equity = CAPITAL_EUR + all_trades["pnl"].cumsum()

    metrics = compute_metrics(all_trades, asset_cfg=cfg, capital_eur=CAPITAL_EUR, df=df.loc[TEST_START:])
    print(f"  Trades totaux: {len(all_trades)}, Sharpe: {metrics.get('sharpe', 0):.2f}")

    info = {
        "asset": asset, "tf": tf, "model": model_type,
        "n_train_total": n_train_total,
        "n_test_trades": len(all_trades),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "wr": float(metrics.get("win_rate", 0.0)),
        "max_dd_pct": float(metrics.get("max_dd_pct", 0.0)),
    }
    return equity, all_trades, info


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. Backtest walk-forward mean-reversion + meta (EURUSD H4)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def backtest_wf_meanrev(
    asset: str, tf: str, model_type: str,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Walk-forward mean-reversion RSI+BB + mÃ©ta-labeling RF (EURUSD H4)."""
    print(f"\n{'='*60}")
    print(f"[WF Mean-Rev] {asset} {tf} ({model_type})")
    print(f"{'='*60}")

    from app.features.indicators import adx, atr, rsi  # noqa: E402
    from app.pipelines.walk_forward import walk_forward_meta  # noqa: E402
    from app.strategies.mean_reversion import MeanReversionRSIBB  # noqa: E402

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]

    print(f"  {len(df)} barres, {df.index.min().date()} â†’ {df.index.max().date()}")

    def build_features_h4(df_wf: pd.DataFrame) -> pd.DataFrame:
        close = df_wf["Close"]
        high = df_wf["High"]
        low = df_wf["Low"]
        atr14 = atr(high, low, close, 14)
        adx14 = adx(high, low, close, 14)["adx_line"]
        sma50 = close.rolling(50).mean()
        out = pd.DataFrame({
            "RSI_14": rsi(close, 14),
            "ADX_14": adx14,
            "Dist_SMA_50": (close - sma50) / atr14,
            "ATR_Norm_14": atr14 / close,
            "Log_Return_5": np.log(close / close.shift(5)),
            "BB_Width": (close.rolling(20).std() * 4) / close,
            "Hour_UTC": pd.Series(df_wf.index.hour, index=df_wf.index, dtype=float),
            "Is_London_NY_Overlap": pd.Series(
                ((df_wf.index.hour >= 13) & (df_wf.index.hour < 17)).astype(int),
                index=df_wf.index, dtype=float,
            ),
        }, index=df_wf.index)
        return out.dropna()

    strat = MeanReversionRSIBB(
        rsi_period=14, rsi_long=30, rsi_short=70,
        bb_period=20, bb_mult=2.0,
    )

    all_trades_oos, _segments = walk_forward_meta(
        df=df, strat=strat, cfg=cfg,
        feature_builder=build_features_h4,
        target_builder=lambda _df, pnl: (pnl > 0).astype(int),
        retrain_months=6, test_start="2024-01-01",
        capital_eur=CAPITAL_EUR,
    )

    if all_trades_oos.empty:
        logger.warning("0 trade walk-forward pour %s %s", asset, tf)
        return pd.Series(dtype=float), pd.DataFrame(), {}

    equity = (
        CAPITAL_EUR
        + (
            all_trades_oos["Pips_Nets"]
            * all_trades_oos["position_size_lots"]
            * cfg.pip_value_eur
        ).cumsum()
    )

    metrics = compute_metrics(all_trades_oos, asset_cfg=cfg, capital_eur=CAPITAL_EUR, df=df.loc[TEST_START:])
    print(f"  Trades WF: {len(all_trades_oos)}, Sharpe: {metrics.get('sharpe', 0):.2f}")

    info = {
        "asset": asset, "tf": tf, "model": model_type,
        "n_test_trades": len(all_trades_oos),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "wr": float(metrics.get("win_rate", 0.0)),
        "max_dd_pct": float(metrics.get("max_dd_pct", 0.0)),
    }
    return equity, all_trades_oos, info


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. Benchmarks
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def buy_and_hold_benchmark(
    assets: list[str],
    start: str = "2024-01-01",
    end: str | None = None,
) -> pd.Series:
    """Equity B&H equal weight des actifs retenus, frais inclus Ã  l'achat."""
    returns = pd.DataFrame()
    for a in assets:
        try:
            df_bh = load_asset(a, "D1")
            if end:
                df_bh = df_bh.loc[start:end]
            else:
                df_bh = df_bh.loc[start:]
            if df_bh.empty:
                continue
            ret = df_bh["Close"].pct_change().dropna()
            # Frais d'entrÃ©e : spread/2
            cfg_bh = ASSET_CONFIGS[a]
            half_cost_pct = (cfg_bh.spread_pips * cfg_bh.pip_size) / (2 * df_bh["Close"].iloc[0])
            ret.iloc[0] -= half_cost_pct
            returns[a] = ret
        except Exception as exc:
            logger.warning("B&H %s: %s", a, exc)

    if returns.empty:
        return pd.Series(dtype=float)

    portfolio_ret = returns.mean(axis=1)
    return CAPITAL_EUR * (1.0 + portfolio_ret).cumprod()


def monte_carlo_random_benchmark(
    sleeve_specs: list[dict[str, Any]],
    sleeve_trade_rates: dict[str, float],
    n_iter: int = 500,
    start: str = "2024-01-01",
) -> np.ndarray:
    """Fix F6 : Monte Carlo reprÃ©sentatif du portfolio rÃ©el.

    Pour chaque itÃ©ration :
      1. Pour chaque sleeve (asset, tf) du portfolio, gÃ©nÃ¨re des signaux
         alÃ©atoires Ã  la mÃªme FRÃ‰QUENCE observÃ©e que la stratÃ©gie rÃ©elle.
      2. Backteste chacun â†’ equity en â‚¬.
      3. Combine en portfolio equal-weight (mÃªme schÃ©ma que build_portfolio).
      4. Calcule le Sharpe daily linÃ©aire (fix F2) sur le portfolio random.

    Args:
        sleeve_specs: liste de dicts {"asset": ..., "tf": ...} (une par sleeve).
        sleeve_trade_rates: dict {"ASSET_TF": trades_per_bar}. Calibre la
            frÃ©quence de signal pour chaque sleeve. Par dÃ©faut 0.05 si absent.
        n_iter: Nombre d'itÃ©rations Monte Carlo.
        start: Date de dÃ©but OOS.

    Returns:
        Array des Sharpe portfolio random (longueur n_iter).
    """
    # PrÃ©-charger les df par sleeve
    sleeve_data: dict[str, tuple[pd.DataFrame, Any, float]] = {}
    for spec in sleeve_specs:
        asset, tf = spec["asset"], spec["tf"]
        cfg = ASSET_CONFIGS[asset]
        try:
            df = load_asset(asset, tf).loc[start:]
            if df.empty:
                continue
            sleeve_data[f"{asset}_{tf}"] = (df, cfg, (cfg.spread_pips + cfg.slippage_pips) / 2.0)
        except Exception as exc:
            logger.warning("MC: skip %s %s: %s", asset, tf, exc)

    if not sleeve_data:
        logger.warning("Aucune donnÃ©e sleeve dispo pour Monte Carlo.")
        return np.array([])

    sharpes = np.zeros(n_iter, dtype=float)

    for i in range(n_iter):
        rng = np.random.default_rng(seed=i)

        # Backtest random sur chaque sleeve, collecter daily PnL linÃ©aires
        sleeve_daily_pnls: list[pd.Series] = []

        for name, (df, cfg, half_cost) in sleeve_data.items():
            n_bars = len(df)
            signal_freq = sleeve_trade_rates.get(name, 0.05)
            # Signaux alÃ©atoires (long/short 50/50, bernoulli signal_freq)
            direction = rng.choice([1, -1], size=n_bars)
            entry_mask = rng.random(n_bars) < signal_freq
            signals = pd.Series(
                np.where(entry_mask, direction, 0),
                index=df.index, dtype=int,
            )

            bt = run_deterministic_backtest(
                df=df, signals=signals,
                tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
                window_hours=cfg.window_hours,
                commission_pips=cfg.commission_pips,
                slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
            )
            trades_mc: list[dict] = bt.get("trades", [])
            if not trades_mc:
                continue

            _, trades_df = _trades_to_equity(trades_mc, cfg=cfg)
            if trades_df.empty:
                continue
            # PnL daily en â‚¬ (linÃ©aire, capital fixe â€” cohÃ©rent avec fix F2)
            pnl_daily = trades_df["pnl"].resample("D").sum().fillna(0.0)
            sleeve_daily_pnls.append(pnl_daily)

        if not sleeve_daily_pnls:
            sharpes[i] = 0.0
            continue

        # Portfolio equal-weight des PnL daily â†’ Sharpe linÃ©aire annualisÃ©
        all_idx = sleeve_daily_pnls[0].index
        for series in sleeve_daily_pnls[1:]:
            all_idx = all_idx.union(series.index)
        pnl_matrix = pd.DataFrame(index=all_idx)
        for k, series in enumerate(sleeve_daily_pnls):
            pnl_matrix[f"s{k}"] = series.reindex(all_idx, fill_value=0.0)
        # equal-weight = moyenne
        portfolio_daily_pnl = pnl_matrix.mean(axis=1)
        daily_ret = portfolio_daily_pnl / CAPITAL_EUR  # retours linÃ©aires
        sharpes[i] = float(sharpe_ratio(daily_ret)) if len(daily_ret) > 1 else 0.0

    return sharpes


def _estimate_sleeve_trade_rate(
    asset: str,
    tf: str,
    n_trades: int,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> float:
    """Estime trades_per_bar observÃ© pour calibrer la frÃ©quence MC.

    Args:
        asset, tf: identifiant du couple.
        n_trades: nb de trades produits par la stratÃ©gie rÃ©elle.
        start, end: bornes OOS.

    Returns:
        Fraction de barres avec un signal. ClampÃ©e Ã  [0.001, 0.5].
    """
    try:
        df = load_asset(asset, tf)
        if end is not None:
            df = df.loc[start:end]
        else:
            df = df.loc[start:]
        n_bars = max(len(df), 1)
        rate = n_trades / n_bars
        return float(np.clip(rate, 0.001, 0.5))
    except Exception:
        return 0.05


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7. Portfolio construction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_portfolio(
    sleeve_equities: dict[str, pd.Series],
) -> tuple[pd.Series, pd.Series]:
    """Combine les equity curves en portfolio equal-risk.

    Returns:
        (portfolio_equity: pd.Series, daily_returns: pd.Series)
    """
    # Aligner toutes les equity curves sur une grille commune
    union_dates = sorted(set().union(*[eq.index for eq in sleeve_equities.values()]))
    all_dates = pd.DatetimeIndex(union_dates)
    if all_dates.tz is None:
        all_dates = all_dates.tz_localize("UTC")
    else:
        all_dates = all_dates.tz_convert("UTC")

    # Construire les returns quotidiens pour chaque sleeve
    daily_returns = pd.DataFrame(index=all_dates)
    for name, equity in sleeve_equities.items():
        # Resample equity to daily
        equity_daily = equity.resample("D").last().ffill()
        ret = equity_daily.pct_change().fillna(0.0)
        daily_returns[name] = ret

    daily_returns = daily_returns.fillna(0.0)

    # Equal-risk weights: 1/n allocation
    n = len(sleeve_equities)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    weight = 1.0 / n
    portfolio_ret = daily_returns.mean(axis=1)  # equal weight
    portfolio_equity = CAPITAL_EUR * (1.0 + portfolio_ret).cumprod()

    return portfolio_equity, portfolio_ret


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 8. Main
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main() -> int:
    set_global_seeds()

    # Fix F5 : n_trials dÃ©rivÃ© du snooping_guard (n_reads â‰¥ floor 29).
    # On expose aussi n_unique_hypotheses pour comparaison mÃ©thodologique.
    n_trials_cumul = n_trials_from_history(min_floor=N_TRIALS_FLOOR)
    n_uniq = n_unique_hypotheses()

    print("=" * 70)
    print("PROMPT 18 â€” VALIDATION FINALE GO/NO-GO")
    print(f"n_trials (n_reads, plancher {N_TRIALS_FLOOR}) = {n_trials_cumul}")
    print(f"n_unique_hypotheses (alternative) = {n_uniq}")
    print("=" * 70)

    # â”€â”€ 8a. Backtest de chaque stratÃ©gie â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sleeve_equities: dict[str, pd.Series] = {}
    all_trades_combined: list[pd.DataFrame] = []
    strategy_infos: list[dict[str, Any]] = []

    # Donchian + ML (4 stratÃ©gies)
    for s in STRATEGIES_DONCHIAN_ML:
        try:
            equity, trades_df, info = backtest_donchian_ml(
                s["asset"], s["tf"], s["model"],
            )
            if not equity.empty and not trades_df.empty:
                name = f"{s['asset']}_{s['tf']}"
                sleeve_equities[name] = equity
                all_trades_combined.append(trades_df)
                strategy_infos.append(info)
        except Exception as exc:
            logger.error("Ã‰chec %s %s: %s", s["asset"], s["tf"], exc, exc_info=True)

    # Walk-forward mÃ©ta-labeling (GBPUSD H4)
    for s in STRATEGIES_WALKFORWARD:
        try:
            if s["type"] == "meta_labeling_wf":
                equity, trades_df, info = backtest_wf_meta_labeling(
                    s["asset"], s["tf"], s["model"],
                )
            else:
                equity, trades_df, info = backtest_wf_meanrev(
                    s["asset"], s["tf"], s["model"],
                )
            if not equity.empty and not trades_df.empty:
                name = f"{s['asset']}_{s['tf']}"
                sleeve_equities[name] = equity
                all_trades_combined.append(trades_df)
                strategy_infos.append(info)
        except Exception as exc:
            logger.error("Ã‰chec %s %s WF: %s", s["asset"], s["tf"], exc, exc_info=True)

    print(f"\n{'='*60}")
    print(f"Sleeves backtestÃ©s avec succÃ¨s : {len(sleeve_equities)}/6")
    for name in sleeve_equities:
        print(f"  âœ… {name}")
    print(f"{'='*60}")

    if len(sleeve_equities) == 0:
        logger.error("Aucun sleeve backtestÃ© avec succÃ¨s.")
        return 1

    # â”€â”€ 8b. Portfolio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    portfolio_equity, portfolio_returns = build_portfolio(sleeve_equities)
    portfolio_daily_returns = portfolio_equity.pct_change().dropna()

    # Trades combinÃ©s
    all_trades_df = pd.concat(all_trades_combined).sort_index() if all_trades_combined else pd.DataFrame()

    # â”€â”€ 8c. validate_edge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    read_oos(
        prompt="18",
        hypothesis="validation_finale_portfolio",
        sharpe=float(sharpe_ratio(portfolio_daily_returns)) if len(portfolio_daily_returns) > 1 else 0.0,
        n_trades=len(all_trades_df),
    )

    report = validate_edge(
        equity=portfolio_equity,
        trades=all_trades_df if not all_trades_df.empty else pd.DataFrame({"pnl": []}),
        n_trials=n_trials_cumul,
    )

    # â”€â”€ 8d. Benchmarks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nâ”€â”€ Benchmarks â”€â”€")

    # B&H equal weight
    bh_equity = buy_and_hold_benchmark(ASSETS_FOR_BENCHMARK)
    if len(bh_equity) > 1:
        bh_returns = bh_equity.pct_change().dropna()
        sr_bh = float(sharpe_ratio(bh_returns)) if len(bh_returns) > 1 else 0.0
    else:
        sr_bh = 0.0
    print(f"  B&H equal-weight Sharpe: {sr_bh:.3f}")

    # Monte Carlo random multi-asset (fix F6)
    # Calibrer la frÃ©quence par sleeve depuis les trades observÃ©s
    test_end = pd.Timestamp.now(tz="UTC")
    sleeve_trade_rates: dict[str, float] = {}
    for info in strategy_infos:
        name = f"{info['asset']}_{info['tf']}"
        n_trades = int(info.get("n_test_trades", 0))
        sleeve_trade_rates[name] = _estimate_sleeve_trade_rate(
            info["asset"], info["tf"], n_trades, TEST_START, test_end,
        )

    mc_sleeve_specs = [
        {"asset": info["asset"], "tf": info["tf"]} for info in strategy_infos
    ]
    mc_sharpes = monte_carlo_random_benchmark(
        sleeve_specs=mc_sleeve_specs,
        sleeve_trade_rates=sleeve_trade_rates,
        n_iter=500,
    )
    if len(mc_sharpes) > 0:
        p95_random = float(np.percentile(mc_sharpes, 95))
        p50_random = float(np.percentile(mc_sharpes, 50))
    else:
        p95_random = 0.0
        p50_random = 0.0
    print(f"  Monte Carlo P50 Sharpe: {p50_random:.3f}, P95: {p95_random:.3f}")
    print(f"  MC sleeve trade rates: {sleeve_trade_rates}")

    sr_portfolio = float(report.metrics.get("sharpe", 0.0))
    bench_bh_ok = sr_portfolio >= sr_bh + 0.3
    bench_mc_ok = sr_portfolio > p95_random
    benches_ok = bench_bh_ok and bench_mc_ok

    print(f"  Portfolio Sharpe: {sr_portfolio:.3f}")
    print(f"  Beat B&H+0.3: {'âœ…' if bench_bh_ok else 'âŒ'} (B&H={sr_bh:.3f}, Portfolio={sr_portfolio:.3f})")
    print(f"  Beat P95 random: {'âœ…' if bench_mc_ok else 'âŒ'} (P95={p95_random:.3f}, Portfolio={sr_portfolio:.3f})")

    # â”€â”€ 8e. Verdict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reasons = list(report.reasons)
    if not bench_bh_ok:
        reasons.append(f"Portfolio Sharpe {sr_portfolio:.2f} < B&H+0.3 ({sr_bh + 0.3:.2f})")
    if not bench_mc_ok:
        reasons.append(f"Portfolio Sharpe {sr_portfolio:.2f} <= P95 random ({p95_random:.2f})")

    go = report.go and benches_ok

    print(f"\n{'='*60}")
    print(f"VERDICT FINAL : {'âœ… GO' if go else 'âŒ NO-GO'}")
    if reasons:
        for r in reasons:
            print(f"  âš ï¸  {r}")
    print(f"{'='*60}")

    # â”€â”€ 8f. Rapport JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    output = {
        "config": {
            "sleeves": [
                {"asset": s["asset"], "tf": s["tf"], "model": s["model"]}
                for s in STRATEGIES_DONCHIAN_ML + STRATEGIES_WALKFORWARD
            ],
            "portfolio_weighting": "equal_risk",
            "vol_targeting": False,
            "capital_eur": CAPITAL_EUR,
            "risk_per_trade": RISK_PCT,
        },
        "metrics": {
            "sharpe_portfolio": sr_portfolio,
            "dsr": float(report.metrics.get("dsr", float("nan"))),
            "p_value": float(report.metrics.get("p_value", float("nan"))),
            "max_dd": float(report.metrics.get("max_dd", 0.0)),
            "wr": float(report.metrics.get("wr", 0.0)),
            "trades_per_year": float(report.metrics.get("trades_per_year", 0.0)),
            "n_trades": int(report.metrics.get("n_trades", 0)),
        },
        "benchmarks": {
            "bh_sharpe": sr_bh,
            "bh_ok": bench_bh_ok,
            "mc_p95_sharpe": p95_random,
            "mc_p50_sharpe": p50_random,
            "mc_ok": bench_mc_ok,
        },
        "strategy_details": strategy_infos,
        "go": go,
        "reasons": reasons,
        "n_trials": n_trials_cumul,
        "n_unique_hypotheses": n_uniq,
        "n_trials_floor": N_TRIALS_FLOOR,
        "date": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    out_json = Path("predictions/validation_finale.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(output, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRapport JSON sauvegardÃ© : {out_json}")

    # â”€â”€ 8g. Rapport humain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _write_human_report(output, sr_portfolio, sr_bh, p95_random, go)

    # â”€â”€ 8h. Snooping check (ne pas lock sans confirmation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nâš ï¸  Si verdict GO :")
    print("   1. python scripts/verify_no_snooping.py")
    print("   2. from app.testing.snooping_guard import lock; lock(prompt='18')")
    print("   âš ï¸  Le lock est IRRÃ‰VERSIBLE â€” confirmation utilisateur requise.")

    return 0 if go else 1


def _write_human_report(
    output: dict,
    sr_portfolio: float,
    sr_bh: float,
    p95_random: float,
    go: bool,
) -> None:
    """GÃ©nÃ¨re le rapport Markdown docs/v3_final_report.md."""
    metrics = output["metrics"]
    reasons = output["reasons"]

    lines = [
        "# Rapport de validation finale â€” Prompt 18",
        "",
        f"**Date** : {output['date']}",
        f"**n_trials cumulÃ©** : {output['n_trials']}",
        f"**Verdict** : {'âœ… GO â€” PRODUCTION' if go else 'âŒ NO-GO â€” RETOUR EN RECHERCHE'}",
        "",
        "## StratÃ©gies / Sleeves retenus",
        "",
        "| Actif | TF | ModÃ¨le | Sharpe | WR | Trades | Max DD |",
        "|---|---|---|---|---|---|---|",
    ]

    for s in output["strategy_details"]:
        # Fix F11 : `wr` est dÃ©jÃ  en % (ex: 65.83 = 65.83%), mÃªme chose pour
        # `max_dd_pct`. Utiliser {:.1f}% au lieu de {:.1%}.
        lines.append(
            f"| {s['asset']} | {s['tf']} | {s['model']} | "
            f"{s.get('sharpe', 0):.2f} | {s.get('wr', 0):.1f}% | "
            f"{s.get('n_test_trades', 0)} | {s.get('max_dd_pct', 0):.1f}% |"
        )

    lines += [
        "",
        "## CritÃ¨res de la constitution",
        "",
        "| CritÃ¨re | Cible | ObservÃ© | Verdict |",
        "|---|---|---|---|",
        f"| Sharpe | â‰¥ 1.0 | {sr_portfolio:.2f} | {'âœ…' if sr_portfolio >= 1.0 else 'âŒ'} |",
        f"| DSR | > 0, p < 0.05 | {metrics['dsr']:.2f} (p={metrics['p_value']:.3f}) | {'âœ…' if metrics['dsr'] > 0 and metrics['p_value'] < 0.05 else 'âŒ'} |",
        f"| Max DD | < 15% | {metrics['max_dd']:.1%} | {'âœ…' if abs(metrics['max_dd']) < 0.15 else 'âŒ'} |",
        f"| WR | > 30% | {metrics['wr']:.1%} | {'âœ…' if metrics['wr'] > 0.30 else 'âŒ'} |",
        f"| Trades/an | â‰¥ 30 | {metrics['trades_per_year']:.1f} | {'âœ…' if metrics['trades_per_year'] >= 30 else 'âŒ'} |",
        "",
        "## Benchmarks",
        "",
        "| Benchmark | Cible | ObservÃ© | Verdict |",
        "|---|---|---|---|",
        f"| Beat B&H+0.3 | Sharpe > {sr_bh + 0.3:.2f} | {sr_portfolio:.2f} | {'âœ…' if output['benchmarks']['bh_ok'] else 'âŒ'} |",
        f"| Beat P95 random | Sharpe > {p95_random:.2f} | {sr_portfolio:.2f} | {'âœ…' if output['benchmarks']['mc_ok'] else 'âŒ'} |",
        "",
        "## Verdict",
        "",
    ]

    if go:
        lines += [
            "### âœ… GO â€” Passage en Phase 4 (production)",
            "",
            "Tous les critÃ¨res sont satisfaits. Le portfolio peut Ãªtre dÃ©ployÃ©.",
            "",
            "**Prochaine Ã©tape** : Prompt 19 â€” `19_h18_walk_forward_continu.md`",
        ]
    else:
        lines += [
            "### âŒ NO-GO â€” ItÃ©ration requise",
            "",
            "**Raisons** :",
        ]
        for r in reasons:
            lines.append(f"- {r}")
        lines += [
            "",
            "**Actions recommandÃ©es** :",
        ]
        for r in reasons:
            if "Sharpe" in r and "B&H" not in r and "random" not in r:
                lines.append("- Prompt 14 (vol targeting) ou prompt 11 (mÃ©ta-labeling)")
            elif "DD" in r:
                lines.append("- Prompt 15 (vol targeting) ou prompt 14 (corrÃ©lation)")
            elif "WR" in r:
                lines.append("- Prompt 10 (rÃ©gime) ou prompt 11 (mÃ©ta-labeling)")
            elif "Trades" in r:
                lines.append("- Prompt 16 (TF) ou prompt 08 (ajouter stratÃ©gies)")
            elif "DSR" in r:
                lines.append("- Revoir n_trials, rÃ©duire hypothÃ¨ses testÃ©es")
            elif "B&H" in r:
                lines.append("- Le portfolio n'apporte pas d'edge vs buy-and-hold")
            elif "random" in r:
                lines.append("- Le portfolio ne bat pas des signaux alÃ©atoires")

    md_path = Path("docs/v3_final_report.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport humain sauvegardÃ© : {md_path}")


if __name__ == "__main__":
    sys.exit(main())
