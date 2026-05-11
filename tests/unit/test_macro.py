"""Tests unitaires pour les features macro."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.macro import calc_macro_return


class TestCalcMacroReturn:
    """calc_macro_return — log-return d'un instrument macro."""

    def test_basic_return(self) -> None:
        close = pd.Series(
            [1.0, 1.01, 1.005, 1.02],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        result = calc_macro_return(close, name="TEST_Return")

        assert isinstance(result, pd.DataFrame)
        assert "TEST_Return" in result.columns
        assert len(result) == 4

        # First value should be NaN (no prior close)
        assert pd.isna(result["TEST_Return"].iloc[0])
        # Second value: log(1.01 / 1.0) = log(1.01)
        assert result["TEST_Return"].iloc[1] == pytest.approx(np.log(1.01 / 1.0))

    def test_returns_correct_shape(self) -> None:
        close = pd.Series(
            np.linspace(1.0, 1.5, 20),
            index=pd.date_range("2024-01-01", periods=20, freq="h"),
        )
        result = calc_macro_return(close, name="XAU_Return")

        assert result.shape == (20, 1)
        assert result.index.equals(close.index)

    def test_negative_prices(self) -> None:
        """Negative prices would produce NaN due to log, but function doesn't guard."""
        close = pd.Series([-1.0, 1.0, 2.0])
        with np.errstate(invalid="ignore"):
            result = calc_macro_return(close, name="BAD_Return")
        # log of negative → NaN, log of 1/-1 = log(-1) = NaN
        assert pd.isna(result["BAD_Return"].iloc[1])

    def test_empty_series(self) -> None:
        close = pd.Series([], dtype=float)
        result = calc_macro_return(close, name="EMPTY_Return")
        assert len(result) == 0
        assert "EMPTY_Return" in result.columns

    def test_single_value(self) -> None:
        close = pd.Series([1.0], index=[pd.Timestamp("2024-01-01")])
        result = calc_macro_return(close, name="SINGLE_Return")
        assert len(result) == 1
        assert pd.isna(result["SINGLE_Return"].iloc[0])

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2023-06-01", periods=5, freq="D", name="Time")
        close = pd.Series([100.0, 101.0, 99.5, 102.0, 103.5], index=idx)
        result = calc_macro_return(close, name="Z_Return")
        assert result.index.name == "Time"
        assert result.index.equals(idx)
