import numpy as np
import pandas as pd
import pytest
from app.testing.look_ahead_validator import assert_no_look_ahead


def test_safe_function_passes():
    series = pd.Series(np.arange(200, dtype=float))
    def safe(s): return s.rolling(5).mean()
    assert_no_look_ahead(safe, series, n_samples=20)


def test_leaky_function_fails():
    series = pd.Series(np.arange(200, dtype=float))
    def leaky(s): return s.shift(-1)  # utilise le futur
    with pytest.raises(AssertionError, match="Look-ahead"):
        assert_no_look_ahead(leaky, series, n_samples=20)


def test_nan_handling():
    series = pd.Series([np.nan] * 10 + list(range(190)), dtype=float)
    def safe(s): return s.rolling(3).mean()
    assert_no_look_ahead(safe, series, n_samples=20)
