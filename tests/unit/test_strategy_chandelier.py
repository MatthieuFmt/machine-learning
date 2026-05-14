"""Tests unitaires pour la stratégie Chandelier Exit (Prompt 08 / H07)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.chandelier import ChandelierExit


def _ohlcv_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """DataFrame OHLC synthétique avec tendance haussière."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 0.3, n),
            "High": close + rng.uniform(0.1, 0.8, n),
            "Low": close - rng.uniform(0.1, 0.8, n),
            "Close": close,
            "Volume": rng.uniform(100, 1000, n),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_chandelier_signals_are_valid_values() -> None:
    """Les signaux sont exclusivement -1, 0, ou +1."""
    df = _ohlcv_df()
    strat = ChandelierExit(period=22, k_atr=3.0)
    signals = strat.generate_signals(df)

    assert signals.dtype in (np.dtype("int64"), np.dtype("int32"))
    assert set(signals.unique()) <= {-1, 0, 1}


def test_chandelier_signal_length_matches_input() -> None:
    """Le nombre de signaux égale le nombre de barres OHLCV."""
    df = _ohlcv_df(n=150)
    strat = ChandelierExit(period=22, k_atr=3.0)
    signals = strat.generate_signals(df)

    assert len(signals) == len(df)


def test_chandelier_anti_look_ahead() -> None:
    """Le signal à l'instant t n'utilise que l'information ≤ t-1 (highest.shift(1))."""
    df = _ohlcv_df(n=200)
    strat = ChandelierExit(period=22, k_atr=3.0)

    full = strat.generate_signals(df)

    for n in [80, 120, 160]:
        truncated = strat.generate_signals(df.iloc[: n + 1])
        if len(truncated) > 0:
            assert truncated.iloc[-1] == full.iloc[n], (
                f"Look-ahead détecté à n={n}: "
                f"truncated[-1]={truncated.iloc[-1]}, full[n]={full.iloc[n]}"
            )


def test_chandelier_no_signal_before_warmup() -> None:
    """Pas de signal avant que ATR et highest/lowest soient calculables."""
    df = _ohlcv_df(n=200)
    strat = ChandelierExit(period=44, k_atr=3.0)
    signals = strat.generate_signals(df)

    # Les premières period+1 barres devraient être 0
    first_signal_idx = signals[signals != 0].index.min()
    assert first_signal_idx is not pd.NaT


def test_chandelier_very_wide_gives_no_signal() -> None:
    """Un k_atr énorme (k_atr=100) ne produit aucun signal (seuil trop éloigné)."""
    df = _ohlcv_df(n=200)
    strat = ChandelierExit(period=22, k_atr=100.0)
    signals = strat.generate_signals(df)

    assert (signals == 0).all(), (
        f"Attendu aucun signal avec k_atr=100, obtenu {signals.value_counts().to_dict()}"
    )
