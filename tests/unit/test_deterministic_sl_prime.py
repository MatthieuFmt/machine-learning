"""Tests pour run_deterministic_backtest — fix F3 SL-prime same-bar.

Avant le fix : TP prime sur les bougies same-bar (optimiste).
Après le fix : SL prime (conservateur, aligné avec simulator.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.deterministic import run_deterministic_backtest


def _build_df(rows: list[dict]) -> pd.DataFrame:
    """Construit un DataFrame OHLC à partir d'une liste de dicts."""
    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df.set_index("Time")


def test_sl_prime_same_bar_long() -> None:
    """Long : si TP et SL sont touchés dans la même barre, SL doit gagner.

    Setup :
      - Entrée à Close=1.0000 sur barre 0.
      - Barre 1 : High=1.0030 (touche TP=1.0020), Low=0.9990 (touche SL=0.9990).
      - TP=20 pips, SL=10 pips → résultat attendu = loss_sl.
    """
    df = _build_df([
        {"Time": "2024-01-01 00:00", "Open": 0.9990, "High": 1.0005, "Low": 0.9985, "Close": 1.0000},
        {"Time": "2024-01-01 01:00", "Open": 1.0000, "High": 1.0030, "Low": 0.9990, "Close": 1.0010},
        {"Time": "2024-01-01 02:00", "Open": 1.0010, "High": 1.0015, "Low": 1.0005, "Close": 1.0012},
    ])
    signals = pd.Series([1, 0, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trades = result["trades"]
    assert len(trades) == 1
    assert trades[0]["result"] == "loss_sl", (
        f"SL doit primer en same-bar (fix F3) — observé : {trades[0]['result']}"
    )
    assert trades[0]["pips_net"] == pytest.approx(-10.0)


def test_sl_prime_same_bar_short() -> None:
    """Short : SL doit primer en same-bar.

    Setup :
      - Entrée short à Close=1.0000.
      - Barre 1 : High=1.0010 (touche SL=1.0010), Low=0.9980 (touche TP=0.9980).
    """
    df = _build_df([
        {"Time": "2024-01-01 00:00", "Open": 1.0005, "High": 1.0010, "Low": 0.9995, "Close": 1.0000},
        {"Time": "2024-01-01 01:00", "Open": 1.0000, "High": 1.0010, "Low": 0.9980, "Close": 0.9990},
        {"Time": "2024-01-01 02:00", "Open": 0.9990, "High": 0.9995, "Low": 0.9985, "Close": 0.9988},
    ])
    signals = pd.Series([-1, 0, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trades = result["trades"]
    assert len(trades) == 1
    assert trades[0]["result"] == "loss_sl"
    assert trades[0]["pips_net"] == pytest.approx(-10.0)


def test_tp_only_long() -> None:
    """Sanity : si seul le TP est touché, c'est un win."""
    df = _build_df([
        {"Time": "2024-01-01 00:00", "Open": 0.9995, "High": 1.0000, "Low": 0.9990, "Close": 1.0000},
        {"Time": "2024-01-01 01:00", "Open": 1.0000, "High": 1.0025, "Low": 0.9999, "Close": 1.0020},
    ])
    signals = pd.Series([1, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trades = result["trades"]
    assert trades[0]["result"] == "win"
    assert trades[0]["pips_net"] == pytest.approx(20.0)


def test_sl_only_long() -> None:
    """Sanity : si seul le SL est touché, c'est un loss_sl."""
    df = _build_df([
        {"Time": "2024-01-01 00:00", "Open": 1.0005, "High": 1.0010, "Low": 0.9999, "Close": 1.0000},
        {"Time": "2024-01-01 01:00", "Open": 1.0000, "High": 1.0005, "Low": 0.9988, "Close": 0.9990},
    ])
    signals = pd.Series([1, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
    )
    trades = result["trades"]
    assert trades[0]["result"] == "loss_sl"
    assert trades[0]["pips_net"] == pytest.approx(-10.0)
