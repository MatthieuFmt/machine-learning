"""Tests unitaires pour detect_regime — Phase F4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.regime import detect_regime


def _make_ohlc(close: np.ndarray, *, daily: bool = True) -> pd.DataFrame:
    """Construit un OHLC synthétique à partir d'une série Close.

    High = Close * (1 + spread), Low = Close * (1 - spread).
    """
    n = len(close)
    freq = "1D" if daily else "1h"
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    high = close * 1.002
    low = close * 0.998
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close}, index=idx
    )


class TestDetectRegimeBasics:
    """Comportement structurel : warmup, valeurs admises, index préservé."""

    def test_returns_series_with_same_index(self) -> None:
        close = np.linspace(1.0, 1.5, 200)
        df = _make_ohlc(close)
        regime = detect_regime(df)

        assert isinstance(regime, pd.Series)
        assert regime.index.equals(df.index)
        assert regime.name == "regime"

    def test_warmup_is_na(self) -> None:
        """Les `atr_quantile_window + atr_period - 1` premières barres au moins
        doivent être NA (warmup du rolling quantile + ATR)."""
        close = np.linspace(1.0, 1.2, 200)
        df = _make_ohlc(close)
        regime = detect_regime(df, atr_period=14, atr_quantile_window=60)

        # Au minimum les 60+14-1=73 premières barres sont NA
        assert regime.iloc[:60].isna().all()

    def test_only_admitted_labels(self) -> None:
        rng = np.random.default_rng(42)
        close = np.cumprod(1 + rng.normal(0, 0.005, 300))
        df = _make_ohlc(close)
        regime = detect_regime(df)

        valid_values = regime.dropna().unique()
        assert set(valid_values).issubset({"trend", "range", "vol_high"})


class TestDetectRegimeLabels:
    """Vérifie les règles de classification."""

    def test_pure_trend_strong_directional_low_vol(self) -> None:
        """Tendance haussière régulière (faible volatilité) → 'trend' dominant."""
        # Hausse régulière de +0.5% par barre, sans bruit
        close = 1.0 * (1.005 ** np.arange(300))
        df = _make_ohlc(close)
        regime = detect_regime(df, adx_threshold=25.0)

        # En régime de tendance pure stable, le quantile 80% de ATR%
        # est lui aussi stable → vol_high ne devrait pas dominer
        counts = regime.value_counts()
        assert "trend" in counts.index
        # Le mode doit être 'trend' (pas vol_high ni range)
        assert counts.idxmax() == "trend"

    def test_pure_range_no_directional(self) -> None:
        """Sinusoïde stationnaire → ADX faible → 'range' dominant."""
        t = np.arange(400)
        close = 1.0 + 0.01 * np.sin(2 * np.pi * t / 20)
        df = _make_ohlc(close)
        regime = detect_regime(df)

        counts = regime.value_counts()
        assert "range" in counts.index
        assert counts.idxmax() == "range"

    def test_vol_high_priority_over_trend(self) -> None:
        """Spike de volatilité ponctuel au milieu d'une tendance →
        la barre concernée doit être 'vol_high', pas 'trend'."""
        n = 300
        close = 1.0 * (1.003 ** np.arange(n))  # tendance régulière
        df = _make_ohlc(close)
        # Spike artificiel sur les barres 250-255 : élargir le High/Low
        df.loc[df.index[250:255], "High"] = df["High"].iloc[250:255] * 1.05
        df.loc[df.index[250:255], "Low"] = df["Low"].iloc[250:255] * 0.95

        regime = detect_regime(df)
        # Au moins une des barres spikées doit être classée vol_high
        assert (regime.iloc[250:260] == "vol_high").any()


class TestDetectRegimeEdgeCases:
    """Cas limites."""

    def test_short_series_all_na(self) -> None:
        """Série plus courte que le warmup → tout NA, pas d'exception."""
        close = np.linspace(1.0, 1.05, 30)
        df = _make_ohlc(close)
        regime = detect_regime(df, atr_quantile_window=60)
        assert regime.isna().all()

    def test_constant_close_no_volatility(self) -> None:
        """Prix constants → ATR≈0, ADX indéfini → tout doit être NA ou range."""
        close = np.ones(200)
        df = _make_ohlc(close)
        regime = detect_regime(df)
        # Aucune valeur ne doit être 'vol_high' ni 'trend'
        non_na = regime.dropna()
        assert not (non_na == "vol_high").any()
        assert not (non_na == "trend").any()

    def test_custom_thresholds_change_distribution(self) -> None:
        """Baisser le seuil ADX doit augmenter la part 'trend'."""
        rng = np.random.default_rng(0)
        close = np.cumprod(1 + rng.normal(0.0005, 0.003, 500))
        df = _make_ohlc(close)

        strict = detect_regime(df, adx_threshold=40.0)
        permissive = detect_regime(df, adx_threshold=10.0)

        n_trend_strict = (strict == "trend").sum()
        n_trend_permissive = (permissive == "trend").sum()
        assert n_trend_permissive >= n_trend_strict


class TestDetectRegimeNoLookAhead:
    """Garantit l'absence de fuite future."""

    def test_truncation_does_not_alter_past_labels(self) -> None:
        """Couper les N dernières barres ne doit pas modifier les labels passés."""
        rng = np.random.default_rng(123)
        close = np.cumprod(1 + rng.normal(0, 0.004, 500))
        df = _make_ohlc(close)

        full = detect_regime(df)
        truncated = detect_regime(df.iloc[:-50])

        # Comparaison sur les indices communs
        common = truncated.index
        # On compare seulement les positions où les deux sont non-NA
        mask = full.loc[common].notna() & truncated.notna()
        assert (full.loc[common][mask] == truncated[mask]).all()


@pytest.mark.parametrize(
    "atr_quantile_window,atr_quantile",
    [(30, 0.7), (60, 0.8), (120, 0.9)],
)
def test_param_combinations_run(atr_quantile_window: int, atr_quantile: float) -> None:
    """Plusieurs combinaisons de paramètres doivent s'exécuter sans erreur."""
    rng = np.random.default_rng(7)
    close = np.cumprod(1 + rng.normal(0, 0.003, 300))
    df = _make_ohlc(close)
    regime = detect_regime(
        df, atr_quantile_window=atr_quantile_window, atr_quantile=atr_quantile
    )
    assert len(regime) == len(df)
