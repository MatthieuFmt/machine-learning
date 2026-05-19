"""Tests pour fix F18 — window_bars calculé via mode, pas moyenne.

Avant le fix : sur séries H1 12 ans, la moyenne des diffs incluait les gaps
weekends (48h) → typical_hours ≈ 1.5 → window_bars 30 % trop court.

Après : la mode des diffs (1.0 h pour H1) donne le bon window_bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.deterministic import run_deterministic_backtest


def _build_h1_with_weekend_gaps(n_weeks: int = 5) -> pd.DataFrame:
    """Construit un index H1 avec gaps weekends (vendredi soir → lundi matin)."""
    sessions = []
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")  # Lundi
    for w in range(n_weeks):
        start = base + pd.Timedelta(weeks=w)
        # 5 jours × 24 h = 120 bars / semaine
        for d in range(5):
            day_start = start + pd.Timedelta(days=d)
            sessions.extend(pd.date_range(day_start, periods=24, freq="h", tz="UTC"))
    idx = pd.DatetimeIndex(sessions)
    rng = np.random.default_rng(42)
    close = 1.1000 + rng.standard_normal(len(idx)).cumsum() * 0.0005
    return pd.DataFrame(
        {
            "Open": close, "High": close + 0.0005,
            "Low": close - 0.0005, "Close": close,
        },
        index=idx,
    )


def test_window_bars_correct_on_h1_with_weekend_gaps() -> None:
    """Sur H1 avec gaps weekends, window_hours=24 doit donner exactly 24 bars,
    pas ~16 bars (ce qui serait le cas avec la moyenne biaisée).
    """
    df = _build_h1_with_weekend_gaps(n_weeks=5)

    # Signal au début de la 2ᵉ semaine, prix qui chute brutalement
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[120] = -1  # début semaine 2

    # On set un TP/SL hors d'atteinte pour forcer le timeout = window_bars
    # Window = 24 h. Si fix F18 OK → 24 bars. Si biais moyen → ~16 bars.
    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=10000, sl_pips=10000,  # jamais touché
        window_hours=24,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trades = result["trades"]
    assert len(trades) == 1
    t = trades[0]
    assert t["result"] == "loss_timeout"

    # Compter les bars entre entry et exit
    entry_idx = df.index.get_loc(pd.Timestamp(t["entry_time"]))
    exit_idx = df.index.get_loc(pd.Timestamp(t["exit_time"]))
    bars_between = exit_idx - entry_idx
    assert bars_between == 24, (
        f"window_bars devrait être 24 (mode H1), observé {bars_between}. "
        f"Si < 20, le bug F18 est revenu (moyenne incluant weekends)."
    )


def test_window_bars_correct_on_d1() -> None:
    """Sur D1 sans gaps, window_hours=120 → 5 bars (= 5 jours)."""
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    close = 1.1000 + rng.standard_normal(n).cumsum() * 0.001
    df = pd.DataFrame(
        {
            "Open": close, "High": close + 0.002,
            "Low": close - 0.002, "Close": close,
        },
        index=idx,
    )
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[5] = 1

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=10000, sl_pips=10000,
        window_hours=120,  # 5 jours
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trades = result["trades"]
    assert len(trades) == 1
    entry_idx = df.index.get_loc(pd.Timestamp(trades[0]["entry_time"]))
    exit_idx = df.index.get_loc(pd.Timestamp(trades[0]["exit_time"]))
    bars_between = exit_idx - entry_idx
    assert bars_between == 5, f"D1 window 120h doit donner 5 bars, observé {bars_between}"
