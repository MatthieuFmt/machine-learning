"""Prompt 18 — Validation finale GO/NO-GO production.

Replay complet du portfolio des 6 stratégies GO sur test 2024-2025.
Produit le rapport predictions/validation_finale.json et docs/v3_final_report.md.

Règles :
- n_trials = 29 (28 hérités JOURNAL.md + 1 prompt 18)
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
from sklearn.calibration import CalibratedClassifierCV
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
from app.testing.snooping_guard import read_oos  # noqa: E402

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_CUTOFF = pd.Timestamp("2022-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
CAPITAL_EUR = 10_000.0
RISK_PCT = 0.02
DONCHIAN_N = 20
DONCHIAN_M = 20
N_TRIALS_CUMUL = 29  # 28 (JOURNAL.md C5) + 1 (prompt 18)

# 6 stratégies GO validées
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_features(df: pd.DataFrame, asset: str, tf: str) -> pd.DataFrame:
    """Construit le superset et sélectionne le top 15 pour le couple."""
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
    """Convertit trades → equity curve + DataFrame avec sizing.

    Returns:
        (equity: pd.Series indexée par entry_time, trades_df: pd.DataFrame)
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
    """Entraîne le modèle selon le type (rf/hgbm/stacking) avec hyperparams C5."""
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
        raise ValueError(f"Modèle inconnu: {model_type}")

    model.fit(X.values, y.values)
    return model


def _generate_model_signals(
    df: pd.DataFrame,
    model: Any,
    asset: str,
    tf: str,
) -> pd.Series:
    """Génère signaux directionnels (1=LONG, -1=SHORT) depuis le modèle."""
    key = (asset, tf)
    hp = HYPERPARAMS_TUNED.get(key, {})
    threshold = hp.get("threshold", 0.5)

    features = _build_features(df, asset, tf)
    if features.empty:
        return pd.Series(0, index=df.index, dtype=int)

    common_idx = df.index.intersection(features.index)
    if len(common_idx) == 0:
        return pd.Series(0, index=df.index, dtype=int)

    features_aligned = features.loc[common_idx]
    proba = model.predict_proba(features_aligned.values)[:, 1]

    signals = pd.Series(0, index=df.index, dtype=int)

    trend_cols = [c for c in ["slope_sma_20", "slope_sma_50", "dist_sma_200"] if c in features_aligned.columns]
    if trend_cols:
        trend_sign = features_aligned[trend_cols].mean(axis=1).apply(np.sign)
    else:
        trend_sign = pd.Series(1, index=features_aligned.index)

    long_mask = (proba > threshold) & (trend_sign > 0)
    short_mask = (proba > threshold) & (trend_sign < 0)

    signals.loc[common_idx[long_mask]] = 1
    signals.loc[common_idx[short_mask]] = -1
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Backtest Donchian + ML (simple train/test)
# ═══════════════════════════════════════════════════════════════════════════════

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

    print(f"  {len(df)} barres, {df.index.min().date()} → {df.index.max().date()}")

    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]

    if df_train.empty or df_test.empty:
        logger.error("Split vide pour %s %s", asset, tf)
        return pd.Series(dtype=float), pd.DataFrame(), {}

    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # Train: génération target Donchian
    donchian_signals_train = _generate_donchian_signals(df_train)
    n_signals_train = int((donchian_signals_train != 0).sum())
    print(f"  Signaux Donchian train: {n_signals_train}")

    bt_train = run_deterministic_backtest(
        df=df_train, signals=donchian_signals_train,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size,
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

    # Entraînement
    model = _train_model(X_train, y_train, model_type, asset, tf)
    acc_train = float((model.predict(X_train.values) == y_train.values).mean())
    print(f"  Accuracy train: {acc_train:.3f}")

    # Test: signaux + backtest
    df_test_with_history = df.loc[:df_test.index[-1]]
    features_test = _build_features(df_test_with_history, asset, tf)
    features_test = features_test.loc[features_test.index.isin(df_test.index)]

    signals_test = _generate_model_signals(df_test, model, asset, tf)
    n_test_signals = int((signals_test != 0).sum())
    print(f"  Signaux test: {n_test_signals}")

    if n_test_signals == 0:
        logger.warning("0 signal sur test pour %s %s", asset, tf)
        return pd.Series(dtype=float), pd.DataFrame(), {}

    bt_test = run_deterministic_backtest(
        df=df_test, signals=signals_test,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size,
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


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Backtest walk-forward méta-labeling (GBPUSD H4)
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_wf_meta_labeling(
    asset: str, tf: str, model_type: str,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Walk-forward expanding window avec méta-labeling RF.

    Logique simplifiée reproduisant run_phase_b_c5_b1_meta_labeling.py.
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

    print(f"  {len(df)} barres, {df.index.min().date()} → {df.index.max().date()}")

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

        # Signaux bootstrap pour générer des labels d'entraînement
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
            slippage_pips=half_cost, pip_size=cfg.pip_size,
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

        # Signaux primaires sur OOS
        signals_primary = _generate_model_signals(df_oos, primary_model, asset, tf)
        n_primary = int((signals_primary != 0).sum())

        # Méta-labeling sur signaux primaires
        meta_labeler = MetaLabelingRF(
            config=MetaLabelingConfig(
                n_estimators=100, max_depth=4, min_samples_leaf=10,
            ),
        )
        # Entraîner le méta-labeler sur les trades train
        meta_labeler.fit(X_primary, y_primary)

        # Backtest des signaux primaires sur OOS
        bt_oos_primary = run_deterministic_backtest(
            df=df_oos, signals=signals_primary,
            tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size,
        )
        trades_primary_oos: list[dict] = bt_oos_primary.get("trades", [])

        # Filtrer les trades OOS avec le méta-modèle
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

    # Concaténer les equity curves (chaque segment repart du capital précédent)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Backtest walk-forward mean-reversion + meta (EURUSD H4)
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_wf_meanrev(
    asset: str, tf: str, model_type: str,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Walk-forward mean-reversion RSI+BB + méta-labeling RF (EURUSD H4)."""
    print(f"\n{'='*60}")
    print(f"[WF Mean-Rev] {asset} {tf} ({model_type})")
    print(f"{'='*60}")

    from app.features.indicators import adx, atr, rsi  # noqa: E402
    from app.pipelines.walk_forward import walk_forward_meta  # noqa: E402
    from app.strategies.mean_reversion import MeanReversionRSIBB  # noqa: E402

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]

    print(f"  {len(df)} barres, {df.index.min().date()} → {df.index.max().date()}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def buy_and_hold_benchmark(
    assets: list[str],
    start: str = "2024-01-01",
    end: str | None = None,
) -> pd.Series:
    """Equity B&H equal weight des actifs retenus, frais inclus à l'achat."""
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
            # Frais d'entrée : spread/2
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
    n_iter: int = 1000,
    signal_freq: float = 0.05,
    start: str = "2024-01-01",
) -> np.ndarray:
    """Simule 1000 stratégies random sur US30 D1.

    signal_freq: probabilité de signal à chaque barre.
    direction: bernoulli 50/50.
    Mêmes coûts que le pipeline réel.
    """
    asset = "US30"
    tf = "D1"
    cfg = ASSET_CONFIGS[asset]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    df = load_asset(asset, tf)
    df = df.loc[start:]

    if df.empty:
        logger.warning("Pas de données US30 pour MC benchmark.")
        return np.array([])

    n_bars = len(df)
    sharpes = np.zeros(n_iter, dtype=float)

    for i in range(n_iter):
        rng = np.random.default_rng(seed=i)
        # Signaux aléatoires: 1=LONG, -1=SHORT, 0=pas de trade
        direction = rng.choice([1, -1], size=n_bars)
        entry_mask = rng.random(n_bars) < signal_freq
        signals = np.where(entry_mask, direction, 0)
        signals_series = pd.Series(signals, index=df.index, dtype=int)

        bt = run_deterministic_backtest(
            df=df, signals=signals_series,
            tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size,
        )
        trades_mc: list[dict] = bt.get("trades", [])
        if not trades_mc:
            sharpes[i] = 0.0
            continue

        equity_mc, _ = _trades_to_equity(trades_mc, cfg=cfg)
        if len(equity_mc) < 2:
            sharpes[i] = 0.0
            continue
        daily_ret = equity_mc.pct_change().dropna()
        sharpes[i] = float(sharpe_ratio(daily_ret)) if len(daily_ret) > 1 else 0.0

    return sharpes


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    set_global_seeds()

    print("=" * 70)
    print("PROMPT 18 — VALIDATION FINALE GO/NO-GO")
    print(f"n_trials = {N_TRIALS_CUMUL}")
    print("=" * 70)

    # ── 8a. Backtest de chaque stratégie ──────────────────────────────────
    sleeve_equities: dict[str, pd.Series] = {}
    all_trades_combined: list[pd.DataFrame] = []
    strategy_infos: list[dict[str, Any]] = []

    # Donchian + ML (4 stratégies)
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
            logger.error("Échec %s %s: %s", s["asset"], s["tf"], exc, exc_info=True)

    # Walk-forward méta-labeling (GBPUSD H4)
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
            logger.error("Échec %s %s WF: %s", s["asset"], s["tf"], exc, exc_info=True)

    print(f"\n{'='*60}")
    print(f"Sleeves backtestés avec succès : {len(sleeve_equities)}/6")
    for name in sleeve_equities:
        print(f"  ✅ {name}")
    print(f"{'='*60}")

    if len(sleeve_equities) == 0:
        logger.error("Aucun sleeve backtesté avec succès.")
        return 1

    # ── 8b. Portfolio ─────────────────────────────────────────────────────
    portfolio_equity, portfolio_returns = build_portfolio(sleeve_equities)
    portfolio_daily_returns = portfolio_equity.pct_change().dropna()

    # Trades combinés
    all_trades_df = pd.concat(all_trades_combined).sort_index() if all_trades_combined else pd.DataFrame()

    # ── 8c. validate_edge ─────────────────────────────────────────────────
    read_oos(
        prompt="18",
        hypothesis="validation_finale_portfolio",
        sharpe=float(sharpe_ratio(portfolio_daily_returns)) if len(portfolio_daily_returns) > 1 else 0.0,
        n_trades=len(all_trades_df),
    )

    report = validate_edge(
        equity=portfolio_equity,
        trades=all_trades_df if not all_trades_df.empty else pd.DataFrame({"pnl": []}),
        n_trials=N_TRIALS_CUMUL,
    )

    # ── 8d. Benchmarks ────────────────────────────────────────────────────
    print("\n── Benchmarks ──")

    # B&H equal weight
    bh_equity = buy_and_hold_benchmark(ASSETS_FOR_BENCHMARK)
    if len(bh_equity) > 1:
        bh_returns = bh_equity.pct_change().dropna()
        sr_bh = float(sharpe_ratio(bh_returns)) if len(bh_returns) > 1 else 0.0
    else:
        sr_bh = 0.0
    print(f"  B&H equal-weight Sharpe: {sr_bh:.3f}")

    # Monte Carlo random
    mc_sharpes = monte_carlo_random_benchmark(n_iter=500)  # 500 pour rapidité
    if len(mc_sharpes) > 0:
        p95_random = float(np.percentile(mc_sharpes, 95))
        p50_random = float(np.percentile(mc_sharpes, 50))
    else:
        p95_random = 0.0
        p50_random = 0.0
    print(f"  Monte Carlo P50 Sharpe: {p50_random:.3f}, P95: {p95_random:.3f}")

    sr_portfolio = float(report.metrics.get("sharpe", 0.0))
    bench_bh_ok = sr_portfolio >= sr_bh + 0.3
    bench_mc_ok = sr_portfolio > p95_random
    benches_ok = bench_bh_ok and bench_mc_ok

    print(f"  Portfolio Sharpe: {sr_portfolio:.3f}")
    print(f"  Beat B&H+0.3: {'✅' if bench_bh_ok else '❌'} (B&H={sr_bh:.3f}, Portfolio={sr_portfolio:.3f})")
    print(f"  Beat P95 random: {'✅' if bench_mc_ok else '❌'} (P95={p95_random:.3f}, Portfolio={sr_portfolio:.3f})")

    # ── 8e. Verdict ───────────────────────────────────────────────────────
    reasons = list(report.reasons)
    if not bench_bh_ok:
        reasons.append(f"Portfolio Sharpe {sr_portfolio:.2f} < B&H+0.3 ({sr_bh + 0.3:.2f})")
    if not bench_mc_ok:
        reasons.append(f"Portfolio Sharpe {sr_portfolio:.2f} <= P95 random ({p95_random:.2f})")

    go = report.go and benches_ok

    print(f"\n{'='*60}")
    print(f"VERDICT FINAL : {'✅ GO' if go else '❌ NO-GO'}")
    if reasons:
        for r in reasons:
            print(f"  ⚠️  {r}")
    print(f"{'='*60}")

    # ── 8f. Rapport JSON ──────────────────────────────────────────────────
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
        "n_trials": N_TRIALS_CUMUL,
        "date": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    out_json = Path("predictions/validation_finale.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(output, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRapport JSON sauvegardé : {out_json}")

    # ── 8g. Rapport humain ────────────────────────────────────────────────
    _write_human_report(output, sr_portfolio, sr_bh, p95_random, go)

    # ── 8h. Snooping check (ne pas lock sans confirmation) ────────────────
    print("\n⚠️  Si verdict GO :")
    print("   1. python scripts/verify_no_snooping.py")
    print("   2. from app.testing.snooping_guard import lock; lock(prompt='18')")
    print("   ⚠️  Le lock est IRRÉVERSIBLE — confirmation utilisateur requise.")

    return 0 if go else 1


def _write_human_report(
    output: dict,
    sr_portfolio: float,
    sr_bh: float,
    p95_random: float,
    go: bool,
) -> None:
    """Génère le rapport Markdown docs/v3_final_report.md."""
    metrics = output["metrics"]
    reasons = output["reasons"]

    lines = [
        "# Rapport de validation finale — Prompt 18",
        "",
        f"**Date** : {output['date']}",
        f"**n_trials cumulé** : {output['n_trials']}",
        f"**Verdict** : {'✅ GO — PRODUCTION' if go else '❌ NO-GO — RETOUR EN RECHERCHE'}",
        "",
        "## Stratégies / Sleeves retenus",
        "",
        "| Actif | TF | Modèle | Sharpe | WR | Trades | Max DD |",
        "|---|---|---|---|---|---|---|",
    ]

    for s in output["strategy_details"]:
        lines.append(
            f"| {s['asset']} | {s['tf']} | {s['model']} | "
            f"{s.get('sharpe', 0):.2f} | {s.get('wr', 0):.1%} | "
            f"{s.get('n_test_trades', 0)} | {s.get('max_dd_pct', 0):.1%} |"
        )

    lines += [
        "",
        "## Critères de la constitution",
        "",
        "| Critère | Cible | Observé | Verdict |",
        "|---|---|---|---|",
        f"| Sharpe | ≥ 1.0 | {sr_portfolio:.2f} | {'✅' if sr_portfolio >= 1.0 else '❌'} |",
        f"| DSR | > 0, p < 0.05 | {metrics['dsr']:.2f} (p={metrics['p_value']:.3f}) | {'✅' if metrics['dsr'] > 0 and metrics['p_value'] < 0.05 else '❌'} |",
        f"| Max DD | < 15% | {metrics['max_dd']:.1%} | {'✅' if abs(metrics['max_dd']) < 0.15 else '❌'} |",
        f"| WR | > 30% | {metrics['wr']:.1%} | {'✅' if metrics['wr'] > 0.30 else '❌'} |",
        f"| Trades/an | ≥ 30 | {metrics['trades_per_year']:.1f} | {'✅' if metrics['trades_per_year'] >= 30 else '❌'} |",
        "",
        "## Benchmarks",
        "",
        "| Benchmark | Cible | Observé | Verdict |",
        "|---|---|---|---|",
        f"| Beat B&H+0.3 | Sharpe > {sr_bh + 0.3:.2f} | {sr_portfolio:.2f} | {'✅' if output['benchmarks']['bh_ok'] else '❌'} |",
        f"| Beat P95 random | Sharpe > {p95_random:.2f} | {sr_portfolio:.2f} | {'✅' if output['benchmarks']['mc_ok'] else '❌'} |",
        "",
        "## Verdict",
        "",
    ]

    if go:
        lines += [
            "### ✅ GO — Passage en Phase 4 (production)",
            "",
            "Tous les critères sont satisfaits. Le portfolio peut être déployé.",
            "",
            "**Prochaine étape** : Prompt 19 — `19_h18_walk_forward_continu.md`",
        ]
    else:
        lines += [
            "### ❌ NO-GO — Itération requise",
            "",
            "**Raisons** :",
        ]
        for r in reasons:
            lines.append(f"- {r}")
        lines += [
            "",
            "**Actions recommandées** :",
        ]
        for r in reasons:
            if "Sharpe" in r and "B&H" not in r and "random" not in r:
                lines.append("- Prompt 14 (vol targeting) ou prompt 11 (méta-labeling)")
            elif "DD" in r:
                lines.append("- Prompt 15 (vol targeting) ou prompt 14 (corrélation)")
            elif "WR" in r:
                lines.append("- Prompt 10 (régime) ou prompt 11 (méta-labeling)")
            elif "Trades" in r:
                lines.append("- Prompt 16 (TF) ou prompt 08 (ajouter stratégies)")
            elif "DSR" in r:
                lines.append("- Revoir n_trials, réduire hypothèses testées")
            elif "B&H" in r:
                lines.append("- Le portfolio n'apporte pas d'edge vs buy-and-hold")
            elif "random" in r:
                lines.append("- Le portfolio ne bat pas des signaux aléatoires")

    md_path = Path("docs/v3_final_report.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport humain sauvegardé : {md_path}")


if __name__ == "__main__":
    sys.exit(main())
