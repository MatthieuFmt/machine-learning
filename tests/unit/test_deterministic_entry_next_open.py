"""Tests pour le fix C2-bt — fill honnête `entry_on_next_open`.

Le signal n'étant connu qu'à la clôture de la barre i, l'entrée réaliste se fait
à l'ouverture de la barre i+1 (et non au Close de la barre de signal = look-ahead
d'exécution). La barre d'entrée est elle-même scannée (risque de gap intra-barre).
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.backtest.deterministic import run_deterministic_backtest


def _build_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df.set_index("Time")


# Gap haussier entre la clôture de la barre de signal (1.0000) et l'ouverture
# de la barre suivante (1.0010).
_GAP_DF = [
    {"Time": "2024-01-01 00:00", "Open": 0.9990, "High": 1.0000, "Low": 0.9980, "Close": 1.0000},
    {"Time": "2024-01-01 01:00", "Open": 1.0010, "High": 1.0040, "Low": 1.0005, "Close": 1.0030},
]


def test_entry_uses_next_open_not_signal_close() -> None:
    """L'entrée se fait à Open[i+1]=1.0010, pas à Close[i]=1.0000."""
    df = _build_df(_GAP_DF)
    signals = pd.Series([1, 0], index=df.index)

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        entry_on_next_open=True,
    )
    trade = result["trades"][0]
    assert trade["entry_price"] == pytest.approx(1.0010)
    # TP = 1.0010 + 0.0020 = 1.0030, touché par High=1.0040 → win.
    assert trade["result"] == "win"
    assert trade["pips_net"] == pytest.approx(20.0)


def test_legacy_and_next_open_differ_on_gap() -> None:
    """Sur un gap, legacy entre au Close[i] et next_open au Open[i+1]."""
    df = _build_df(_GAP_DF)
    signals = pd.Series([1, 0], index=df.index)
    common = dict(
        df=df, signals=signals, tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0, pip_size=0.0001,
    )
    legacy = run_deterministic_backtest(**common, entry_on_next_open=False)
    honest = run_deterministic_backtest(**common, entry_on_next_open=True)

    assert legacy["trades"][0]["entry_price"] == pytest.approx(1.0000)
    assert honest["trades"][0]["entry_price"] == pytest.approx(1.0010)


def test_no_trade_when_signal_on_last_bar() -> None:
    """Signal sur la dernière barre : aucune barre suivante → aucun trade."""
    df = _build_df(_GAP_DF)
    signals = pd.Series([0, 1], index=df.index)  # signal sur la dernière barre

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=20, sl_pips=10, window_hours=10,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        entry_on_next_open=True,
    )
    assert result["trades"] == []


def test_holding_window_counted_from_entry_bar() -> None:
    """La fenêtre de détention part de la barre d'entrée (e), pas du signal.

    D1, window_hours=120 → 5 bars. Entrée à e=idx+1 ; timeout à e+5.
    bars_between(exit, entry) doit valoir exactement 5.
    """
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    price = [1.1000] * n
    df = pd.DataFrame(
        {"Open": price, "High": [p + 0.002 for p in price],
         "Low": [p - 0.002 for p in price], "Close": price},
        index=idx,
    )
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[5] = 1

    result = run_deterministic_backtest(
        df=df, signals=signals,
        tp_pips=10000, sl_pips=10000,  # jamais touché → timeout
        window_hours=120,
        commission_pips=0.0, slippage_pips=0.0,
        pip_size=0.0001,
        entry_on_next_open=True,
    )
    trade = result["trades"][0]
    assert trade["result"] == "loss_timeout"
    entry_idx = df.index.get_loc(pd.Timestamp(trade["entry_time"]))
    exit_idx = df.index.get_loc(pd.Timestamp(trade["exit_time"]))
    assert entry_idx == 6  # entrée à la barre suivant le signal (idx 5)
    assert exit_idx - entry_idx == 5
