"""Tests unitaires des indicateurs techniques — < 100ms, fixtures synthétiques."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.technical import (
    calc_adx,
    calc_atr_norm,
    calc_base_features,
    calc_bb_width,
    calc_cyclical_time,
    calc_ema_distance,
    calc_log_return,
    calc_rsi,
)


@pytest.fixture
def ohlcv_mini() -> pd.DataFrame:
    """Mini DataFrame OHLCV avec une petite tendance haussière."""
    n = 200
    rng = np.random.default_rng(42)
    drift = 0.0001
    noise = rng.normal(0, 0.0005, n)
    close = 1.1000 + np.cumsum(drift + noise)
    high = close + np.abs(rng.normal(0, 0.0002, n))
    low = close - np.abs(rng.normal(0, 0.0002, n))
    open_p = close - rng.normal(0, 0.0001, n)

    return pd.DataFrame(
        {
            "Open": np.round(open_p, 5),
            "High": np.round(high, 5),
            "Low": np.round(low, 5),
            "Close": np.round(close, 5),
            "Volume": rng.integers(100, 500, n),
            "Spread": rng.integers(10, 18, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


class TestBaseFeatures:
    """Tests de calc_base_features."""

    def test_output_columns(self, ohlcv_mini):
        result = calc_base_features(ohlcv_mini, prefix="_H4")
        assert "RSI_14_H4" in result.columns
        assert "Dist_EMA_20_H4" in result.columns
        assert "Dist_EMA_50_H4" in result.columns

    def test_output_index_matches(self, ohlcv_mini):
        result = calc_base_features(ohlcv_mini)
        assert result.index.equals(ohlcv_mini.index)

    def test_rsi_bounds(self, ohlcv_mini):
        """Le RSI doit être entre 0 et 100."""
        result = calc_base_features(ohlcv_mini)
        rsi = result["RSI_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()


class TestLogReturn:
    """Tests de calc_log_return."""

    def test_first_is_nan(self, ohlcv_mini):
        lr = calc_log_return(ohlcv_mini)
        assert pd.isna(lr.iloc[0])

    def test_name(self, ohlcv_mini):
        lr = calc_log_return(ohlcv_mini)
        assert lr.name == "Log_Return"

    def test_symmetry(self, ohlcv_mini):
        """Un retour positif puis négatif de même amplitude donne des valeurs opposées."""
        df = pd.DataFrame({"Close": [1.0000, 1.0100, 1.0000]})
        lr = calc_log_return(df).dropna()
        assert lr.iloc[0] == pytest.approx(-lr.iloc[1], abs=1e-6)


class TestEMADistance:
    """Tests de calc_ema_distance."""

    def test_columns_naming(self, ohlcv_mini):
        result = calc_ema_distance(ohlcv_mini, periods=(9, 21))
        assert "Dist_EMA_9" in result.columns
        assert "Dist_EMA_21" in result.columns

    def test_zero_distance_when_flat(self):
        """Prix constant → distance EMA = 0."""
        df = pd.DataFrame({"Close": [1.1000] * 100})
        result = calc_ema_distance(df, periods=(9,)).dropna()
        assert (np.abs(result["Dist_EMA_9"]) < 0.001).all()


class TestRSI:
    """Tests de calc_rsi."""

    def test_name(self, ohlcv_mini):
        rsi = calc_rsi(ohlcv_mini)
        assert rsi.name == "RSI_14"

    def test_bounds(self, ohlcv_mini):
        rsi = calc_rsi(ohlcv_mini).dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()


class TestADX:
    """Tests de calc_adx."""

    def test_name(self, ohlcv_mini):
        adx = calc_adx(ohlcv_mini)
        assert adx.name == "ADX_14"

    def test_non_negative(self, ohlcv_mini):
        adx = calc_adx(ohlcv_mini).dropna()
        assert (adx >= 0).all()


class TestATRNorm:
    """Tests de calc_atr_norm."""

    def test_name(self, ohlcv_mini):
        atr = calc_atr_norm(ohlcv_mini)
        assert atr.name == "ATR_Norm"

    def test_positive(self, ohlcv_mini):
        atr = calc_atr_norm(ohlcv_mini).dropna()
        assert (atr > 0).all()


class TestBBWidth:
    """Tests de calc_bb_width."""

    def test_name(self, ohlcv_mini):
        bbw = calc_bb_width(ohlcv_mini)
        assert bbw.name == "BB_Width"

    def test_non_negative(self, ohlcv_mini):
        bbw = calc_bb_width(ohlcv_mini).dropna()
        assert (bbw >= 0).all()


class TestCyclicalTime:
    """Tests de calc_cyclical_time."""

    def test_columns(self, ohlcv_mini):
        result = calc_cyclical_time(ohlcv_mini)
        assert "Hour_Sin" in result.columns
        assert "Hour_Cos" in result.columns

    def test_range(self, ohlcv_mini):
        result = calc_cyclical_time(ohlcv_mini)
        assert (result["Hour_Sin"] >= -1).all() and (result["Hour_Sin"] <= 1).all()
        assert (result["Hour_Cos"] >= -1).all() and (result["Hour_Cos"] <= 1).all()

    def test_normalization(self, ohlcv_mini):
        """sin² + cos² = 1 pour chaque ligne."""
        result = calc_cyclical_time(ohlcv_mini)
        norm = result["Hour_Sin"] ** 2 + result["Hour_Cos"] ** 2
        assert (np.abs(norm - 1.0) < 1e-10).all()
