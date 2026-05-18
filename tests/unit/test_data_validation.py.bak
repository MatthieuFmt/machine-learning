"""Tests unitaires pour les schemas Pandera de validation des donnees."""

import pandas as pd
import pytest

from learning_machine_learning.data.validation import OHLCSchema, MLReadySchema, TradesSchema


class TestOHLCSchema:
    """OHLCSchema — validation des DataFrames OHLCV."""

    def test_valid_ohlcv_passes(self, ohlcv_h1_synthetic: pd.DataFrame) -> None:
        OHLCSchema.validate(ohlcv_h1_synthetic)

    def test_missing_column_fails(self) -> None:
        df = pd.DataFrame({
            "Time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "Open": [1.0, 1.01, 1.02],
            "High": [1.02, 1.03, 1.04],
            "Low": [0.99, 1.00, 1.01],
            "Close": [1.01, 1.02, 1.03],
            # Missing Volume
            "Spread": [15, 16, 14],
        })
        with pytest.raises(Exception):
            OHLCSchema.validate(df)

    def test_zero_price_fails(self) -> None:
        df = pd.DataFrame({
            "Time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "Open": [0.0, 1.01, 1.02],
            "High": [1.02, 1.03, 1.04],
            "Low": [0.99, 1.00, 1.01],
            "Close": [1.01, 1.02, 1.03],
            "Volume": [100, 200, 300],
            "Spread": [15, 16, 14],
        })
        with pytest.raises(Exception):
            OHLCSchema.validate(df)

    def test_negative_volume_fails(self) -> None:
        df = pd.DataFrame({
            "Time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "Open": [1.0, 1.01, 1.02],
            "High": [1.02, 1.03, 1.04],
            "Low": [0.99, 1.00, 1.01],
            "Close": [1.01, 1.02, 1.03],
            "Volume": [100, -5, 300],
            "Spread": [15, 16, 14],
        })
        with pytest.raises(Exception):
            OHLCSchema.validate(df)


class TestMLReadySchema:
    """MLReadySchema — validation du DataFrame pret pour le modele."""

    def test_valid_ml_ready_passes(self, ml_ready_synthetic: pd.DataFrame) -> None:
        MLReadySchema.validate(ml_ready_synthetic)

    def test_invalid_target_fails(self, ml_ready_synthetic: pd.DataFrame) -> None:
        df = ml_ready_synthetic.copy()
        df["Target"] = 5  # invalid value
        with pytest.raises(Exception):
            MLReadySchema.validate(df)

    def test_missing_target_fails(self, ml_ready_synthetic: pd.DataFrame) -> None:
        df = ml_ready_synthetic.drop(columns=["Target"])
        with pytest.raises(Exception):
            MLReadySchema.validate(df)

    def test_negative_spread_fails(self, ml_ready_synthetic: pd.DataFrame) -> None:
        df = ml_ready_synthetic.copy()
        df["Spread"] = -1.0
        with pytest.raises(Exception):
            MLReadySchema.validate(df)


class TestTradesSchema:
    """TradesSchema — validation du DataFrame de trades."""

    def test_valid_trades_passes(self, trades_synthetic: pd.DataFrame) -> None:
        TradesSchema.validate(trades_synthetic)

    def test_invalid_result_fails(self, trades_synthetic: pd.DataFrame) -> None:
        df = trades_synthetic.copy()
        df.loc[df.index[0], "result"] = "invalid_result"
        with pytest.raises(Exception):
            TradesSchema.validate(df)

    def test_negative_weight_fails(self, trades_synthetic: pd.DataFrame) -> None:
        df = trades_synthetic.copy()
        df.loc[df.index[0], "Weight"] = -0.5
        with pytest.raises(Exception):
            TradesSchema.validate(df)

    def test_missing_column_fails(self, trades_synthetic: pd.DataFrame) -> None:
        df = trades_synthetic.drop(columns=["Pips_Nets"])
        with pytest.raises(Exception):
            TradesSchema.validate(df)
