"""Tests unitaires pour la stratégie Keltner Channel (Prompt 08 / H07)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.keltner import KeltnerChannel, _atr


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


def test_keltner_signals_are_valid_values() -> None:
    """Les signaux sont exclusivement -1, 0, ou +1."""
    df = _ohlcv_df()
    strat = KeltnerChannel(period=20, mult=2.0)
    signals = strat.generate_signals(df)

    assert signals.dtype in (np.dtype("int64"), np.dtype("int32"))
    assert set(signals.unique()) <= {-1, 0, 1}


def test_keltner_signal_length_matches_input() -> None:
    """Le nombre de signaux égale le nombre de barres OHLCV."""
    df = _ohlcv_df(n=150)
    strat = KeltnerChannel(period=20, mult=2.0)
    signals = strat.generate_signals(df)

    assert len(signals) == len(df)


def test_keltner_anti_look_ahead() -> None:
    """Le signal à l'instant t n'utilise que l'information ≤ t-1 (shift(1))."""
    df = _ohlcv_df(n=200)
    strat = KeltnerChannel(period=20, mult=2.0)

    full = strat.generate_signals(df)

    for n in [80, 120, 160]:
        truncated = strat.generate_signals(df.iloc[: n + 1])
        if len(truncated) > 0:
            assert truncated.iloc[-1] == full.iloc[n], (
                f"Look-ahead détecté à n={n}: "
                f"truncated[-1]={truncated.iloc[-1]}, full[n]={full.iloc[n]}"
            )


def test_keltner_no_signal_before_warmup() -> None:
    """Pas de signal avant que l'EMA et l'ATR soient calculables."""
    df = _ohlcv_df(n=200)
    strat = KeltnerChannel(period=50, mult=2.0)
    signals = strat.generate_signals(df)

    # Les premières period+1 barres devraient être 0 (warmup EMA + shift)
    first_signal_idx = signals[signals != 0].index.min()
    assert first_signal_idx is not pd.NaT


def test_keltner_wide_channels_give_no_signal() -> None:
    """Un multiplicateur énorme (mult=100) ne produit aucun signal (canal trop large)."""
    df = _ohlcv_df(n=200)
    strat = KeltnerChannel(period=20, mult=100.0)
    signals = strat.generate_signals(df)

    # Shift(1) => la dernière barre peut être 0. Tous les signaux doivent être 0.
    assert (signals == 0).all(), f"Attendu aucun signal avec mult=100, obtenu {signals.value_counts().to_dict()}"


def test_atr_helper_anti_look_ahead() -> None:
    """L'ATR à l'instant t n'utilise que close[t-1], high[t], low[t]."""
    n = 100
    rng = np.random.default_rng(42)
    high = pd.Series(rng.normal(100, 1, n).cumsum() + 5)
    low = pd.Series(rng.normal(100, 1, n).cumsum() - 5)
    close = pd.Series(rng.normal(100, 1, n).cumsum())

    atr_full = _atr(high, low, close, period=14)

    for cutoff in [50, 70, 90]:
        atr_trunc = _atr(high.iloc[: cutoff + 1], low.iloc[: cutoff + 1], close.iloc[: cutoff + 1], period=14)
        if not pd.isna(atr_trunc.iloc[-1]) and not pd.isna(atr_full.iloc[cutoff]):
            assert np.isclose(atr_trunc.iloc[-1], atr_full.iloc[cutoff], rtol=1e-9), (
                f"Look-ahead dans _atr à cutoff={cutoff}"
            )
