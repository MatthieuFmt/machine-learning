"""Tests anti-look-ahead et edge cases pour les 18 indicateurs du Prompt 04."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.indicators import (
    adx,
    atr,
    atr_pct,
    bbands_width,
    cci,
    compute_all_indicators,
    dist_sma,
    efficiency_ratio,
    ema,
    keltner_width,
    macd,
    mfi,
    obv,
    realized_vol,
    rsi,
    slope_sma,
    sma,
    stoch,
    williams_r,
)
from app.testing.look_ahead_validator import assert_no_look_ahead

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def _random_walk(n: int = 500, seed: int = 42) -> pd.Series:
    """Série synthétique de type random walk pour les tests."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    prices = 100 + np.cumsum(returns)
    return pd.Series(prices, name="close")


def _ohlcv_dataframe(n: int = 500) -> pd.DataFrame:
    """DataFrame OHLCV synthétique avec random walk."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = _random_walk(n, seed=42)
    close.index = dates  # alignement explicite avant arithmétique
    high = close + rng.uniform(0, 0.5, n)
    low = close - rng.uniform(0, 0.5, n)
    open_ = low + rng.uniform(0, (high - low).values)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests anti-look-ahead — chaque indicateur
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("indicator_fn, kwargs", [
    # Trend
    (sma, {"period": 20}),
    (ema, {"period": 20}),
    (dist_sma, {"period": 20}),
    (slope_sma, {"period": 20}),
    # Momentum (uni-série uniquement)
    (rsi, {"period": 14}),
    (efficiency_ratio, {"period": 20}),
    (realized_vol, {"period": 20}),
])
def test_non_look_ahead_univariate(indicator_fn, kwargs):
    """Vérifie l'absence de look-ahead pour les indicateurs uni-séries."""
    series = _random_walk(500)
    assert_no_look_ahead(
        lambda s: indicator_fn(s, **kwargs), series, n_samples=30,
    )


@pytest.mark.parametrize("indicator_fn, kwargs", [
    (stoch, {"k": 14, "d": 3}),
    (williams_r, {"period": 14}),
    (cci, {"period": 20}),
    (atr, {"period": 14}),
    (keltner_width, {"period": 20, "n_atr": 2.0}),
    (adx, {"period": 14}),
])
def test_non_look_ahead_multivariate(indicator_fn, kwargs):
    """Vérifie l'absence de look-ahead pour les indicateurs multi-séries.

    On utilise un DataFrame comme input pour assert_no_look_ahead qui
    applique .iloc[:n+1] sur l'entrée.
    """
    df = _ohlcv_dataframe(500)
    # Wrapper pour extraire les colonnes et appeler l'indicateur
    def wrapper(data: pd.DataFrame) -> pd.Series:
        return indicator_fn(data["High"], data["Low"], data["Close"], **kwargs)

    assert_no_look_ahead(wrapper, df, n_samples=30)


def test_non_look_ahead_atr_pct():
    """ATR% dépend de ATR → test combiné."""
    df = _ohlcv_dataframe(500)

    def wrapper(data: pd.DataFrame) -> pd.Series:
        a = atr(data["High"], data["Low"], data["Close"], 14)
        return atr_pct(data["Close"], a)

    assert_no_look_ahead(wrapper, df, n_samples=30)


def test_non_look_ahead_obv():
    """OBV : dépend de close et volume."""
    df = _ohlcv_dataframe(500)

    def wrapper(data: pd.DataFrame) -> pd.Series:
        return obv(data["Close"], data["Volume"])

    assert_no_look_ahead(wrapper, df, n_samples=30)


def test_non_look_ahead_mfi():
    """MFI : dépend de OHLCV."""
    df = _ohlcv_dataframe(500)

    def wrapper(data: pd.DataFrame) -> pd.Series:
        return mfi(data["High"], data["Low"], data["Close"], data["Volume"], 14)

    assert_no_look_ahead(wrapper, df, n_samples=30)


def test_non_look_ahead_bbands_width():
    """BBands width : uni-série mais testé comme les autres."""
    series = _random_walk(500)
    assert_no_look_ahead(
        lambda s: bbands_width(s, 20, 2.0), series, n_samples=30,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests : toutes les fonctions exportées sont marquées @look_ahead_safe
# ═══════════════════════════════════════════════════════════════════════════════

ALL_INDICATOR_FUNCTIONS = [
    sma, ema, dist_sma, slope_sma,
    rsi, macd, stoch, williams_r, cci,
    atr, atr_pct, bbands_width, keltner_width,
    obv, mfi,
    adx, efficiency_ratio, realized_vol,
]


@pytest.mark.parametrize("fn", ALL_INDICATOR_FUNCTIONS)
def test_indicator_is_marked_look_ahead_safe(fn):
    """Chaque indicateur doit avoir l'attribut _look_ahead_safe = True."""
    assert getattr(fn, "_look_ahead_safe", False) is True, (
        f"{fn.__name__} n'est pas décoré @look_ahead_safe"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_rsi_constant_series():
    """RSI sur série constante : avg_gain=0, avg_loss=0 → RSI NaN (Wilder).

    Une série strictement constante produit avg_gain=0 ET avg_loss=0,
    donc rs = 0/0 = NaN, donc RSI = NaN. C'est le comportement standard.
    """
    constant = pd.Series(np.ones(200))
    result = rsi(constant, 14)
    # Wilder sur série constante : tout NaN après warmup (0/0)
    assert result.iloc[30:].isna().all()


def test_stoch_flat_series():
    """Stochastique sur série plate → range=0 → retourne 50."""
    flat = pd.Series(np.ones(200))
    result = stoch(flat, flat, flat, k=14, d=3)
    assert result["stoch_k"].iloc[30:].notna().all()
    assert np.allclose(result["stoch_k"].iloc[30:], 50.0)
    assert np.allclose(result["stoch_d"].iloc[30:], 50.0)


def test_williams_r_flat_series():
    """Williams %R sur série plate → range=0 → retourne -50."""
    flat = pd.Series(np.ones(200))
    result = williams_r(flat, flat, flat, 14)
    assert result.iloc[30:].notna().all()
    assert np.allclose(result.iloc[30:], -50.0)


def test_sma_short_series():
    """SMA(period=50) sur 10 barres → tout NaN."""
    short = pd.Series(np.arange(10, dtype=float))
    result = sma(short, 50)
    assert result.isna().all()


def test_atr_wilder_no_nan_warmup():
    """ATR Wilder (ewm) — seule la 1ère barre est NaN (prev_close inconnu)."""
    df = _ohlcv_dataframe(100)
    result = atr(df["High"], df["Low"], df["Close"], 14)
    # 1ère valeur NaN, mais après warmup suffisant tout est valide
    assert np.isnan(result.iloc[0])
    assert result.iloc[30:].notna().all()


def test_macd_returns_three_columns():
    """MACD retourne exactement 3 colonnes nommées."""
    close = _random_walk(500)
    result = macd(close, 12, 26, 9)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["macd_line", "macd_signal", "macd_histogram"]
    assert len(result) == 500


def test_stoch_returns_two_columns():
    """Stochastique retourne exactement 2 colonnes nommées."""
    df = _ohlcv_dataframe(500)
    result = stoch(df["High"], df["Low"], df["Close"], 14, 3)
    assert list(result.columns) == ["stoch_k", "stoch_d"]


def test_adx_returns_three_columns():
    """ADX retourne exactement 3 colonnes nommées."""
    df = _ohlcv_dataframe(500)
    result = adx(df["High"], df["Low"], df["Close"], 14)
    assert list(result.columns) == ["adx_line", "plus_di", "minus_di"]


# ═══════════════════════════════════════════════════════════════════════════════
# compute_all_indicators
# ═══════════════════════════════════════════════════════════════════════════════


def test_compute_all_indicators_shape():
    """Même index et longueur que l'entrée."""
    df = _ohlcv_dataframe(500)
    result = compute_all_indicators(df)
    assert len(result) == 500
    assert result.index.equals(df.index)


def test_compute_all_indicators_has_volume():
    """Avec Volume > 0 → ≥ 22 colonnes (OBV + MFI inclus)."""
    df = _ohlcv_dataframe(500)
    result = compute_all_indicators(df)
    expected_cols = [
        "sma_20", "sma_50", "ema_20", "ema_50",
        "dist_sma_20", "dist_sma_50",
        "slope_sma_20", "slope_sma_50",
        "rsi_14", "macd_line", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d", "williams_r_14", "cci_20",
        "atr_14", "atr_pct", "bbands_width_20", "keltner_width_20",
        "adx_line", "plus_di", "minus_di",
        "efficiency_ratio_20", "realized_vol_20",
        "obv", "mfi_14",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_compute_all_indicators_no_volume():
    """Sans Volume → OBV/MFI absents, 25 colonnes (pas 27)."""
    df = _ohlcv_dataframe(500).drop(columns=["Volume"])
    result = compute_all_indicators(df)
    assert "obv" not in result.columns
    assert "mfi_14" not in result.columns


def test_compute_all_indicators_zero_volume():
    """Volume tout à 0 → traité comme absent."""
    df = _ohlcv_dataframe(500)
    df["Volume"] = 0
    result = compute_all_indicators(df)
    assert "obv" not in result.columns
    assert "mfi_14" not in result.columns


def test_compute_all_indicators_no_duplicate_columns():
    """Aucune colonne en double."""
    df = _ohlcv_dataframe(500)
    result = compute_all_indicators(df)
    assert len(result.columns) == len(set(result.columns))


def test_compute_all_indicators_all_float():
    """Toutes les colonnes sont numériques."""
    df = _ohlcv_dataframe(500)
    result = compute_all_indicators(df)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col]), (
            f"{col} n'est pas numérique"
        )


def test_mfi_range():
    """MFI doit être entre 0 et 100."""
    df = _ohlcv_dataframe(500)
    result = mfi(df["High"], df["Low"], df["Close"], df["Volume"], 14)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()
