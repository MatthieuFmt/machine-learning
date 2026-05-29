"""Tests unitaires pour Phase B C5 Extra — BTCUSD D1 (sans méta-labeling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.run_phase_b_c5_extra_btcusd_d1 import (
    _build_target_winner,
    _generate_donchian_signals,
    _train_hgbm_model,
    _trades_to_dataframe,
)
from app.config.features_selected import FEATURES_SELECTED
from app.config.instruments import ASSET_CONFIGS


COUPLE_KEY = ("BTCUSD", "D1")
C5_FEATURES = list(FEATURES_SELECTED[COUPLE_KEY])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlc() -> pd.DataFrame:
    """Génère 500 barres D1 synthétiques avec tendance haussière volatile."""
    np.random.seed(42)
    n = 500
    base = 25000.0
    trend = np.linspace(0, 20000, n) + np.random.randn(n).cumsum() * 500
    close = base + trend
    high = close + np.abs(np.random.randn(n)) * 1000
    low = close - np.abs(np.random.randn(n)) * 1000
    open_ = close - np.random.randn(n) * 300

    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
    }, index=idx)


@pytest.fixture
def sample_trades() -> list[dict]:
    """Liste de trades synthétiques pour test."""
    return [
        {"entry_time": "2022-01-03 00:00:00+00:00", "exit_time": "2022-01-04 00:00:00+00:00",
         "signal": 1, "entry_price": 45000.0, "exit_price": 46000.0,
         "pips_net": 1000.0, "result": "win"},
        {"entry_time": "2022-01-04 00:00:00+00:00", "exit_time": "2022-01-05 00:00:00+00:00",
         "signal": -1, "entry_price": 46000.0, "exit_price": 46500.0,
         "pips_net": -500.0, "result": "loss_sl"},
        {"entry_time": "2022-01-05 00:00:00+00:00", "exit_time": "2022-01-06 00:00:00+00:00",
         "signal": 1, "entry_price": 44000.0, "exit_price": 45000.0,
         "pips_net": 1000.0, "result": "win"},
        {"entry_time": "2022-01-06 00:00:00+00:00", "exit_time": "2022-01-07 00:00:00+00:00",
         "signal": -1, "entry_price": 45500.0, "exit_price": 44500.0,
         "pips_net": 1000.0, "result": "win"},
        {"entry_time": "2022-01-07 00:00:00+00:00", "exit_time": "2022-01-08 00:00:00+00:00",
         "signal": 1, "entry_price": 45000.0, "exit_price": 44000.0,
         "pips_net": -1000.0, "result": "loss_sl"},
    ]


# ── Test 1 : Features C5 dans le superset ─────────────────────────────────────

def test_c5_features_are_defined():
    """Les 15 features C5 pour BTCUSD D1 sont définies dans FEATURES_SELECTED."""
    assert COUPLE_KEY in FEATURES_SELECTED, f"{COUPLE_KEY} absent de FEATURES_SELECTED"
    features = FEATURES_SELECTED[COUPLE_KEY]
    assert len(features) == 15, f"Attendu 15 features, reçu {len(features)}"
    assert all(isinstance(f, str) for f in features)


def test_c5_features_no_duplicates():
    """Pas de doublons dans les features C5 BTCUSD D1."""
    features = FEATURES_SELECTED[COUPLE_KEY]
    assert len(features) == len(set(features)), f"Doublons détectés dans {features}"


def test_asset_config_exists():
    """BTCUSD est dans ASSET_CONFIGS."""
    assert "BTCUSD" in ASSET_CONFIGS, "BTCUSD absent de ASSET_CONFIGS"


# ── Test 2 : Modèle produit des prédictions ───────────────────────────────────

def test_train_hgbm_produces_predictions():
    """Le modèle hgbm (defaults sklearn) entraîné produit des prédictions binaires valides."""
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    model = _train_hgbm_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    preds = model.predict(X.values)
    assert len(preds) == n_samples
    assert set(preds).issubset({0, 1})

    proba = model.predict_proba(X.values)
    assert proba.shape == (n_samples, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_hgbm_binary_target():
    """Le modèle gère correctement une target binaire (0/1)."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 15))
    y = pd.Series([0] * 25 + [1] * 25)

    model = _train_hgbm_model(X, y)
    preds = model.predict(X.values)
    unique = set(preds)
    assert len(unique) >= 1


# ── Test 3 : Backtest génère des trades ───────────────────────────────────────

def test_donchian_signals_format(sample_ohlc: pd.DataFrame):
    """Les signaux Donchian ont le bon format (1, -1, 0)."""
    signals = _generate_donchian_signals(sample_ohlc)

    assert isinstance(signals, pd.Series)
    assert len(signals) == len(sample_ohlc)
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert (signals.iloc[:20] == 0).all(), "Les 20 premières barres devraient être 0 (warmup)"


def test_donchian_signals_breakout_pattern():
    """Un breakout clair génère un signal LONG."""
    np.random.seed(42)
    n = 50
    base = 25000.0
    close = np.full(n, base)
    close[-10:] = base + np.linspace(1000, 5000, 10)
    high = close + 10.0
    low = close - 10.0
    open_ = close

    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)

    signals = _generate_donchian_signals(df)
    n_signals = int((signals != 0).sum())
    assert n_signals > 0, f"Aucun signal Donchian détecté malgré un breakout explicite"


def test_trades_to_dataframe_structure(sample_trades: list[dict]):
    """La conversion trades → DataFrame produit les colonnes attendues."""
    cfg = ASSET_CONFIGS["BTCUSD"]
    df = _trades_to_dataframe(sample_trades, cfg)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_trades)
    expected_cols = {"Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"}
    assert expected_cols.issubset(set(df.columns)), f"Colonnes manquantes : {expected_cols - set(df.columns)}"


def test_trades_to_dataframe_empty():
    """La conversion d'une liste vide retourne un DataFrame vide avec colonnes."""
    cfg = ASSET_CONFIGS["BTCUSD"]
    df = _trades_to_dataframe([], cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "Pips_Nets" in df.columns


# ── Test 4 : Métriques calculées ──────────────────────────────────────────────

def test_build_target_winner():
    """_build_target_winner produit des labels binaires corrects."""
    pnl = pd.Series([100.0, -50.0, 0.0, 25.0, -10.0])
    target = _build_target_winner(pnl)

    assert isinstance(target, pd.Series)
    assert target.dtype in (np.int32, np.int64)
    expected = pd.Series([1, 0, 0, 1, 0])
    pd.testing.assert_series_equal(target.reset_index(drop=True), expected)


def test_build_target_winner_all_wins():
    """Tous les trades gagnants → target = 1."""
    pnl = pd.Series([10.0, 20.0, 5.0])
    target = _build_target_winner(pnl)
    assert (target == 1).all()


def test_build_target_winner_all_losses():
    """Tous les trades perdants → target = 0."""
    pnl = pd.Series([-10.0, -20.0, -5.0])
    target = _build_target_winner(pnl)
    assert (target == 0).all()
