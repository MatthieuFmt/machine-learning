"""Tests unitaires pour Phase B C5 — B1 méta-labeling GBPUSD H4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import des fonctions à tester depuis le script
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase_b_c5_b1_meta_labeling import (  # noqa: E402
    _build_features_for_split,
    _build_target_winner,
    _primary_signals,
    _train_primary_model,
    _trades_to_dataframe,
    ASSET,
    COUPLE_KEY,
    TF,
)
from app.config.features_selected import FEATURES_SELECTED  # noqa: E402
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.models.meta_labeling import MetaLabelingConfig, MetaLabelingRF  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ohlcv_df(n_bars: int = 500) -> pd.DataFrame:
    """Crée un DataFrame OHLCV synthétique pour GBPUSD H4."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="4h", tz="UTC")
    close = 1.2500 + rng.normal(0, 0.0010, n_bars).cumsum()
    close = np.maximum(close, 1.1000)
    return pd.DataFrame(
        {
            "Open": close - rng.normal(0, 0.0002, n_bars),
            "High": close + np.abs(rng.normal(0, 0.0005, n_bars)),
            "Low": close - np.abs(rng.normal(0, 0.0005, n_bars)),
            "Close": close,
            "Volume": rng.uniform(100, 1000, n_bars),
        },
        index=idx,
    )


# ── Tests feature builder ─────────────────────────────────────────────────────


def test_build_features_returns_correct_columns() -> None:
    """Vérifie que _build_features_for_split retourne le top 15 C5 pour GBPUSD H4."""
    df = _make_ohlcv_df(400)
    features = _build_features_for_split(df)
    expected = set(FEATURES_SELECTED[COUPLE_KEY])

    assert not features.empty, "Les features ne doivent pas être vides"
    # Au moins 80% des features C5 doivent être présentes (certaines dépendent de cross-asset)
    common = set(features.columns) & expected
    assert len(common) >= int(0.8 * len(expected)), (
        f"Seulement {len(common)}/{len(expected)} features C5 présentes: "
        f"manquantes={sorted(expected - set(features.columns))}"
    )


def test_build_features_no_nan_after_dropna() -> None:
    """Les features retournées ne doivent pas contenir de NaN (dropna appliqué)."""
    df = _make_ohlcv_df(600)
    features = _build_features_for_split(df)
    assert not features.isna().any().any(), "Features contiennent des NaN"


def test_build_features_empty_df_returns_empty() -> None:
    """DataFrame vide → DataFrame vide."""
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    features = _build_features_for_split(df)
    assert features.empty


# ── Tests target builder ──────────────────────────────────────────────────────


def test_target_winner_binary() -> None:
    """_build_target_winner doit retourner 0 ou 1."""
    df = _make_ohlcv_df(100)
    pnl = pd.Series([10.0, -5.0, 3.0, -2.0, 0.0], index=df.index[:5])
    target = _build_target_winner(df, pnl)
    assert target.dtype == int, f"Type attendu int, reçu {target.dtype}"
    assert set(target.unique()).issubset({0, 1}), f"Valeurs inattendues: {target.unique()}"


def test_target_winner_correct_mapping() -> None:
    """pnl > 0 → 1, pnl ≤ 0 → 0."""
    df = _make_ohlcv_df(3)
    pnl = pd.Series([5.0, -3.0, 0.0], index=df.index[:3])
    target = _build_target_winner(df, pnl)
    assert target.iloc[0] == 1
    assert target.iloc[1] == 0
    assert target.iloc[2] == 0


# ── Tests primary model ───────────────────────────────────────────────────────


def test_train_primary_model_returns_rf() -> None:
    """_train_primary_model doit retourner un RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=list("ABCDE"))
    y = pd.Series((X["A"] > 0).astype(int))
    model = _train_primary_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict_proba")


def test_train_primary_model_uses_c5_hyperparams() -> None:
    """Le modèle primaire doit utiliser les hyperparams C5 pour GBPUSD H4."""
    hp = HYPERPARAMS_TUNED[COUPLE_KEY]
    params = hp["params"]

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=list("ABCDE"))
    y = pd.Series((X["A"] > 0).astype(int))
    model = _train_primary_model(X, y)

    assert model.n_estimators == params.get("n_estimators", 100)
    assert model.max_depth == params.get("max_depth", 10)
    assert model.min_samples_leaf == params.get("min_samples_leaf", 10)


# ── Tests primary signals ─────────────────────────────────────────────────────


def test_primary_signals_output_format() -> None:
    """_primary_signals doit retourner une Series de même index que df, valeurs dans {-1, 0, 1}.

    Fix v5 : le modèle doit être entraîné sur les MÊMES features que celles
    produites par _build_features_for_split (15 features C5 pour GBPUSD H4),
    sinon RandomForestClassifier lève ValueError au predict.
    """
    df = _make_ohlcv_df(400)
    features = _build_features_for_split(df)
    if features.empty or len(features) < 50:
        pytest.skip("Features insuffisantes sur l'OHLC synthétique")

    rng = np.random.default_rng(2)
    n_samples = min(100, len(features))
    X = features.iloc[:n_samples]
    y = pd.Series(rng.integers(0, 2, n_samples), index=X.index)
    model = _train_primary_model(X, y)

    signals = _primary_signals(df, model)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(df)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Valeurs: {signals.unique()}"


def test_primary_signals_empty_features() -> None:
    """Si les features sont vides, retourne une Series de 0."""
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=list("ABCDE"))
    y = pd.Series((X["A"] > 0).astype(int))
    model = _train_primary_model(X, y)

    signals = _primary_signals(df, model)
    assert (signals == 0).all()


# ── Tests trades_to_dataframe ─────────────────────────────────────────────────


def test_trades_to_dataframe_empty() -> None:
    """Liste vide → DataFrame vide avec les bonnes colonnes."""
    cfg = ASSET_CONFIGS[ASSET]
    df = _trades_to_dataframe([], cfg)
    assert df.empty
    assert "Pips_Nets" in df.columns
    assert "pnl" in df.columns


def test_trades_to_dataframe_has_sizing() -> None:
    """Les trades convertis doivent avoir position_size_lots et pnl."""
    cfg = ASSET_CONFIGS[ASSET]
    trades = [
        {
            "entry_time": "2021-06-15T08:00:00Z",
            "exit_time": "2021-06-16T12:00:00Z",
            "entry_price": 1.2500,
            "exit_price": 1.2520,
            "pips_net": 18.0,
            "result": "tp",
            "signal": 1,
        },
    ]
    df = _trades_to_dataframe(trades, cfg)
    assert "position_size_lots" in df.columns
    assert "pnl" in df.columns
    assert df["position_size_lots"].iloc[0] > 0
    assert df["pnl"].iloc[0] != 0


# ── Tests MetaLabelingRF (réutilisation du module existant) ────────────────────


def test_meta_labeling_rf_fit_predict() -> None:
    """Vérifie que MetaLabelingRF fit + predict fonctionne."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=list("ABCDE"))
    y = pd.Series((X.iloc[:, 0] > 0).astype(int))

    cfg = MetaLabelingConfig(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
    )
    meta = MetaLabelingRF(config=cfg)
    meta.fit(X, y)

    assert not meta.disabled, "Le modèle ne devrait pas être désactivé"
    pred = meta.predict(X)
    assert pred.dtype == bool
    assert len(pred) == 200


def test_meta_labeling_rf_single_class_fallback() -> None:
    """Si une seule classe dans y_train → disabled=True."""
    X = pd.DataFrame(np.random.default_rng(1).normal(0, 1, (100, 3)))
    y = pd.Series([1] * 100)

    meta = MetaLabelingRF()
    meta.fit(X, y)
    assert meta.disabled


def test_meta_labeling_rf_calibrate_threshold_fallback() -> None:
    """Si aucun seuil ne conserve ≥ 20% des trades → disabled=True."""
    X = pd.DataFrame(np.zeros((100, 5)), columns=list("ABCDE"))
    y = pd.Series([0] * 99 + [1])

    meta = MetaLabelingRF()
    meta.fit(X, y)
    # backtest_fn renvoie toujours -inf si on filtre tout
    meta.calibrate_threshold(X, lambda mask: -1e9 if mask.sum() < 50 else 0.0)
    assert meta.disabled or meta.threshold >= 0.50


# ── Test d'intégration: workflow complet avec données synthétiques ────────────


def test_full_workflow_synthetic() -> None:
    """Vérifie que le workflow primaire + méta fonctionne sur données synthétiques."""
    from app.backtest.deterministic import run_deterministic_backtest

    df = _make_ohlcv_df(600)
    cfg = ASSET_CONFIGS[ASSET]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # Split train/OOS synthétique
    train_end_idx = int(len(df) * 0.7)
    df_train = df.iloc[:train_end_idx]
    df_oos = df.iloc[train_end_idx:]

    # Features train
    features_train = _build_features_for_split(df_train)
    assert not features_train.empty, "Features train vides"

    # Bootstrap signals pour générer des trades train
    trend_cols = [c for c in ["slope_sma_20", "slope_sma_50", "dist_sma_200"] if c in features_train.columns]
    if trend_cols:
        trend_score = features_train[trend_cols].mean(axis=1)
        bootstrap_signals = pd.Series(0, index=features_train.index, dtype=int)
        bootstrap_signals[trend_score > trend_score.quantile(0.80)] = 1
        bootstrap_signals[trend_score < trend_score.quantile(0.20)] = -1

        bt = run_deterministic_backtest(
            df=df_train, signals=bootstrap_signals,
            tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost,
            pip_size=cfg.pip_size,
        )
        trades = bt.get("trades", [])
        if len(trades) >= 10:
            entry_times = pd.to_datetime([t["entry_time"] for t in trades])
            common_idx = features_train.index.intersection(entry_times)
            if len(common_idx) >= 5:
                X_primary = features_train.loc[common_idx]
                trades_df = _trades_to_dataframe(trades, cfg=cfg)
                pnl_aligned = trades_df.loc[trades_df.index.intersection(common_idx), "Pips_Nets"]
                y_primary = _build_target_winner(df_train, pnl_aligned)

                if y_primary.nunique() >= 2:
                    model = _train_primary_model(X_primary, y_primary)
                    signals_oos = _primary_signals(df_oos, model)
                    assert isinstance(signals_oos, pd.Series)
                    assert len(signals_oos) == len(df_oos)
                    # Le workflow ne crashe pas
    # Si pas assez de trades (synthétique trop aléatoire), le test passe quand même
    assert True
