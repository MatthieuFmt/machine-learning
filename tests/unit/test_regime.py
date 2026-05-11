"""Tests unitaires pour les features de regime."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.regime import (
    calc_volatilite_realisee,
    calc_range_atr_ratio,
    calc_rsi_d1_delta,
    calc_dist_sma200_d1,
)


class TestCalcVolatiliteRealisee:
    """calc_volatilite_realisee — ecart-type glissant des log-returns."""

    def test_basic_window(self) -> None:
        n = 100
        rng = np.random.default_rng(1)
        log_ret = pd.Series(rng.normal(0, 0.001, n))
        vol = calc_volatilite_realisee(log_ret, window=24)

        assert len(vol) == n
        assert vol.iloc[:23].isna().all()
        assert vol.iloc[24:].notna().all()

    def test_constant_returns(self) -> None:
        log_ret = pd.Series([0.001] * 50)
        vol = calc_volatilite_realisee(log_ret, window=10)

        assert vol.iloc[9:].max() == pytest.approx(0.0, abs=1e-12)

    def test_default_window(self) -> None:
        log_ret = pd.Series([0.0] * 30)
        vol = calc_volatilite_realisee(log_ret)
        # default window = 24
        assert len(vol) == 30
        assert vol.iloc[:23].isna().all()

    def test_returns_type(self) -> None:
        log_ret = pd.Series(np.random.default_rng(2).normal(0, 1e-4, 50))
        vol = calc_volatilite_realisee(log_ret)
        assert isinstance(vol, pd.Series)

    def test_short_series(self) -> None:
        log_ret = pd.Series([0.001, -0.002, 0.0005])
        vol = calc_volatilite_realisee(log_ret, window=5)
        assert vol.isna().all()


class TestCalcRangeAtrRatio:
    """calc_range_atr_ratio — ratio Range/ATR pour detection expansion/contraction."""

    def test_returns_positive(self) -> None:
        n = 100
        rng = np.random.default_rng(3)
        close = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum())
        high = close + 0.001
        low = close - 0.001

        ratio = calc_range_atr_ratio(high, low, close, length=14)

        assert isinstance(ratio, pd.Series)
        assert len(ratio) == n
        # After warmup, ratio should be positive
        valid = ratio.dropna()
        assert (valid > 0).all()

    def test_nan_before_warmup(self) -> None:
        n = 50
        rng = np.random.default_rng(4)
        close = pd.Series(1.1000 + rng.normal(0, 0.001, n).cumsum())
        high = close + 0.001
        low = close - 0.001

        ratio = calc_range_atr_ratio(high, low, close, length=14)
        # First 13 values should be NaN (ATR needs 14 bars)
        assert ratio.iloc[:13].isna().all()

    def test_constant_range(self) -> None:
        n = 30
        close = pd.Series(np.linspace(1.1000, 1.1100, n))
        high = close + 0.002
        low = close - 0.001

        ratio = calc_range_atr_ratio(high, low, close, length=14)
        valid = ratio.dropna()
        # Ratio should be finite
        assert np.isfinite(valid).all()


class TestCalcRsiD1Delta:
    """calc_rsi_d1_delta — variation du RSI D1 sur N jours."""

    def test_basic_diff(self) -> None:
        rsi = pd.Series([50.0, 52.0, 55.0, 53.0, 48.0, 51.0, 54.0, 56.0, 58.0, 55.0])
        delta = calc_rsi_d1_delta(rsi, diff_periods=3)

        assert len(delta) == len(rsi)
        assert delta.iloc[3] == pytest.approx(53.0 - 50.0)
        assert delta.iloc[9] == pytest.approx(55.0 - 54.0)
        assert delta.iloc[:3].isna().all()

    def test_default_periods(self) -> None:
        rsi = pd.Series(np.arange(50.0, 60.0, 1.0))
        delta = calc_rsi_d1_delta(rsi)  # diff_periods=3
        assert delta.iloc[3] == pytest.approx(3.0)

    def test_all_nan_input(self) -> None:
        rsi = pd.Series([np.nan] * 10)
        delta = calc_rsi_d1_delta(rsi)
        assert delta.isna().all()

    def test_returns_series(self) -> None:
        rsi = pd.Series([50.0, 51.0, 52.0, 53.0, 54.0])
        delta = calc_rsi_d1_delta(rsi, diff_periods=1)
        assert isinstance(delta, pd.Series)


class TestCalcDistSma200D1:
    """calc_dist_sma200_d1 — distance normalisee a la SMA200."""

    def test_basic_above_sma(self) -> None:
        n = 250
        close = pd.Series(np.linspace(1.1000, 1.2000, n))
        dist = calc_dist_sma200_d1(close, length=50)

        assert isinstance(dist, pd.Series)
        assert len(dist) == n
        # After warmup, price above SMA → dist positive (uptrend)
        valid = dist.iloc[50:]
        assert (valid > 0).all()

    def test_basic_below_sma(self) -> None:
        n = 250
        close = pd.Series(np.linspace(1.2000, 1.1000, n))
        dist = calc_dist_sma200_d1(close, length=50)

        valid = dist.iloc[50:]
        assert (valid < 0).all()

    def test_nan_before_warmup(self) -> None:
        n = 300
        close = pd.Series(np.linspace(1.1000, 1.1500, n))
        dist = calc_dist_sma200_d1(close, length=200)

        assert dist.iloc[:199].isna().all()
        assert dist.iloc[200:].notna().all()

    def test_returns_normalized(self) -> None:
        n = 250
        close = pd.Series(np.linspace(1.1000, 1.1800, n))
        dist = calc_dist_sma200_d1(close, length=100)

        # Values should be in a reasonable range
        valid = dist.dropna()
        assert (valid.abs() < 0.5).all()
