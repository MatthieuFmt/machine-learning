"""Tests unitaires pour la stratégie Dual Moving Average (Prompt 08 / H07)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.dual_ma import DualMovingAverage


def _ohlcv_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """DataFrame OHLC synthétique avec tendance haussière."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 0.3, n),
            "High": close + rng.uniform(0.1, 0.5, n),
            "Low": close - rng.uniform(0.1, 0.5, n),
            "Close": close,
            "Volume": rng.uniform(100, 1000, n),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_dual_ma_signals_are_valid_values() -> None:
    """Les signaux sont exclusivement -1, 0, ou +1."""
    df = _ohlcv_df()
    strat = DualMovingAverage(fast=5, slow=20)
    signals = strat.generate_signals(df)

    assert signals.dtype in (np.dtype("int64"), np.dtype("int32"))
    assert set(signals.unique()) <= {-1, 0, 1}


def test_dual_ma_signal_length_matches_input() -> None:
    """Le nombre de signaux égale le nombre de barres OHLCV."""
    df = _ohlcv_df(n=150)
    strat = DualMovingAverage(fast=10, slow=50)
    signals = strat.generate_signals(df)

    assert len(signals) == len(df)


def test_dual_ma_anti_look_ahead() -> None:
    """Le signal à l'instant t n'utilise que l'information ≤ t-1 (shift(1))."""
    df = _ohlcv_df(n=200)
    strat = DualMovingAverage(fast=5, slow=20)

    full = strat.generate_signals(df)

    for n in [50, 100, 150]:
        truncated = strat.generate_signals(df.iloc[: n + 1])
        # La dernière valeur (à t=n) doit être identique entre full[:n+1] et truncated
        # car le shift(1) garantit que signal[t] n'utilise que close[t-1] et antérieur
        if len(truncated) > 0:
            assert truncated.iloc[-1] == full.iloc[n], (
                f"Look-ahead détecté à n={n}: "
                f"truncated[-1]={truncated.iloc[-1]}, full[n]={full.iloc[n]}"
            )


def test_dual_ma_no_signal_before_warmup() -> None:
    """Pas de signal avant que la SMA lente soit calculable (slow > 0)."""
    df = _ohlcv_df(n=200)
    strat = DualMovingAverage(fast=5, slow=100)
    signals = strat.generate_signals(df)

    # Les premiers slow-1 signaux devraient être 0 (warmup + shift)
    # shift(1) ajoute 1 barre supplémentaire → warmup = slow, donc [slow] premières barres à 0
    first_signal_idx = signals[signals != 0].index.min()
    assert first_signal_idx is not pd.NaT
    # L'index doit être ≥ l'index de la slow-ième barre
    assert first_signal_idx >= df.index[min(100, len(df) - 1)]


def test_dual_ma_trending_market_gives_consistent_long() -> None:
    """Dans une tendance haussière forte, le signal doit être LONG (1) après warmup."""
    n = 200
    rng = np.random.default_rng(42)
    # Tendance haussière claire
    trend = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n)
    close = trend + noise

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1.0,
        },
        index=pd.DatetimeIndex(dates),
    )

    strat = DualMovingAverage(fast=10, slow=50)
    signals = strat.generate_signals(df)

    # Après warmup complet (>50 barres), > 80% des signaux doivent être LONG
    post_warmup = signals.iloc[60:]
    n_post = len(post_warmup)
    if n_post > 0:
        long_ratio = (post_warmup == 1).mean()
        assert long_ratio > 0.8, f"Attendu > 80% LONG dans tendance haussière, obtenu {long_ratio:.1%}"
