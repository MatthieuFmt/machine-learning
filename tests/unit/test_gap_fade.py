"""Tests unitaires pour app.strategies.gap_fade (fade du gap d'ouverture)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.gap_fade import simulate_gap_fade


def _index_config(spread: float = 1.0) -> AssetConfig:
    return AssetConfig(
        spread_pips=spread,
        slippage_pips=0.0,
        commission_pips=0.0,
        pip_size=1.0,
        pip_value_eur=1.0,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        swap_long_pips_per_night=0.0,
        swap_short_pips_per_night=0.0,
    )


def _day(date: str, sess_open: float, sess_close: float) -> dict:
    """2 barres M5 cohérentes (09:30 et 10:00) : Open de séance / Close de séance."""
    hi = max(sess_open, sess_close) + 2
    lo = min(sess_open, sess_close) - 2
    mid = (sess_open + sess_close) / 2
    return {
        f"{date} 09:30": (sess_open, hi, lo, mid),
        f"{date} 10:00": (mid, hi, lo, sess_close),
    }


def _build(days: list[dict]) -> pd.DataFrame:
    bars: dict[str, tuple] = {}
    for d in days:
        bars.update(d)
    times = sorted(bars)
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in times])
    return pd.DataFrame(
        [bars[t] for t in times], columns=["Open", "High", "Low", "Close"], index=idx
    )


# Session UTC 09:30-10:00 pour simplifier (tz="UTC").
_KW = dict(session_tz="UTC", open_time="09:30", close_time="10:00")


class TestSimulateGapFade:
    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-02 09:30", periods=3, freq="5min")  # naive
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
        with pytest.raises(ValueError, match="tz-aware"):
            simulate_gap_fade(df, _index_config(), **_KW)

    def test_gap_up_is_faded_short_and_wins_on_fill(self) -> None:
        # J1 close=100 ; J2 ouvre à 105 (gap +5 > coût 1) puis redescend à 102.
        df = _build([_day("2024-01-02", 100, 100), _day("2024-01-03", 105, 102)])
        trades = simulate_gap_fade(df, _index_config(spread=1.0), **_KW)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == -1          # gap up → short
        assert t["gap_pips"] == pytest.approx(5.0)
        assert t["entry_price"] == pytest.approx(105.0)  # open de séance
        assert t["exit_price"] == pytest.approx(102.0)   # close de séance
        # short : gain = -(102-105) = +3 ; net = 3 - cost(1) = 2
        assert t["pips_brut"] == pytest.approx(3.0)
        assert t["pips_net"] == pytest.approx(2.0)
        assert t["nights_held"] == 0

    def test_gap_down_is_faded_long(self) -> None:
        # J1 close=100 ; J2 ouvre à 95 (gap -5) puis remonte à 98.
        df = _build([_day("2024-01-02", 100, 100), _day("2024-01-03", 95, 98)])
        trades = simulate_gap_fade(df, _index_config(spread=1.0), **_KW)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1           # gap down → long
        assert t["pips_brut"] == pytest.approx(3.0)   # long : 98-95 = +3
        assert t["pips_net"] == pytest.approx(2.0)

    def test_skip_when_gap_below_cost_floor(self) -> None:
        # gap = 0.5 ≤ coût a/r (1.0) → pas de trade.
        df = _build([_day("2024-01-02", 100, 100), _day("2024-01-03", 100.5, 101)])
        trades = simulate_gap_fade(df, _index_config(spread=1.0), **_KW)
        assert trades == []

    def test_gap_that_keeps_running_loses(self) -> None:
        # J2 gap up à 105 mais CONTINUE à 110 (pas de fill) → short perd.
        df = _build([_day("2024-01-02", 100, 100), _day("2024-01-03", 105, 110)])
        trades = simulate_gap_fade(df, _index_config(spread=1.0), **_KW)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == -1
        # short : -(110-105) = -5 ; net = -5 - 1 = -6
        assert t["pips_net"] == pytest.approx(-6.0)

    def test_pnl_consistency_uses_prev_session_close(self) -> None:
        df = _build([
            _day("2024-01-02", 100, 103),   # close J1 = 103
            _day("2024-01-03", 110, 106),    # gap = 110-103 = +7 → short
        ])
        cfg = _index_config(spread=1.0)
        cost = cfg.spread_pips + 2 * (cfg.slippage_pips + cfg.commission_pips)
        trades = simulate_gap_fade(df, cfg, **_KW)
        assert len(trades) == 1
        t = trades[0]
        assert t["gap_pips"] == pytest.approx(7.0)   # vs close veille (103), pas open
        assert t["pips_net"] == pytest.approx(t["pips_brut"] - cost)
