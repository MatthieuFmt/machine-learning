"""Tests unitaires pour app.strategies.opening_range (ORB intraday)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.opening_range import simulate_orb_session, simulate_orb_trades


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


def _build(days: dict[str, dict[int, tuple[float, float, float, float]]]) -> pd.DataFrame:
    """Construit un OHLCV H1 UTC à partir de {date: {heure_utc: (O,H,L,C)}}."""
    frames = []
    for date_str, bars in days.items():
        hours = sorted(bars)
        idx = pd.DatetimeIndex(
            [pd.Timestamp(f"{date_str} {h:02d}:00", tz="UTC") for h in hours]
        )
        df = pd.DataFrame(
            [bars[h] for h in hours],
            columns=["Open", "High", "Low", "Close"],
            index=idx,
        )
        frames.append(df)
    return pd.concat(frames).sort_index()


# Journée avec cassure LONG confirmée à h10, pas de stop → sortie EOD à h15.
_DAY_LONG_EOD = {
    8: (100, 101, 99, 100),
    9: (100, 110, 90, 100),    # OR : high=110, low=90
    10: (100, 116, 100, 115),  # Close 115 > 110 → signal LONG
    11: (115, 120, 114, 118),  # barre d'ENTRÉE (Open=115)
    12: (118, 121, 116, 119),
    13: (119, 122, 117, 120),
    14: (120, 123, 118, 121),
    15: (121, 124, 119, 120),  # dernière barre de séance → exit EOD (close=120)
    16: (120, 121, 119, 120),  # hors séance (lhour > last)
}

# Journée avec cassure LONG puis retour sous l'OR_low → stop touché à l'entrée.
_DAY_STOP = {
    8: (100, 101, 99, 100),
    9: (100, 110, 90, 100),    # OR : high=110, low=90
    10: (100, 116, 100, 115),  # signal LONG
    11: (115, 118, 85, 95),    # entrée Open=115, Low=85 ≤ 90 → stop
    12: (95, 96, 94, 95),
    15: (95, 96, 94, 95),
}

# Journée sans cassure (toutes les closes dans [90, 110]).
_DAY_NONE = {
    9: (100, 110, 90, 100),
    10: (100, 112, 95, 105),   # high 112 > 110 mais CLOSE 105 ≤ 110 → pas de signal
    11: (105, 108, 96, 99),
    15: (99, 104, 93, 101),
}


class TestSimulateORB:
    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-02 08:00", periods=5, freq="1h")  # naive
        df = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx
        )
        with pytest.raises(ValueError, match="tz-aware"):
            simulate_orb_trades(df, _index_config(), session_tz="UTC",
                                or_hour_local=9, last_hour_local=15)

    def test_rejects_bad_hours(self) -> None:
        df = _build({"2024-01-02": _DAY_LONG_EOD})
        with pytest.raises(ValueError, match="last_hour_local"):
            simulate_orb_trades(df, _index_config(), session_tz="UTC",
                                or_hour_local=15, last_hour_local=9)

    def test_no_breakout_no_trade(self) -> None:
        df = _build({"2024-01-04": _DAY_NONE})
        trades = simulate_orb_trades(df, _index_config(), session_tz="UTC",
                                     or_hour_local=9, last_hour_local=15)
        assert trades == []

    def test_long_breakout_entry_next_open_eod(self) -> None:
        df = _build({"2024-01-02": _DAY_LONG_EOD})
        trades = simulate_orb_trades(df, _index_config(spread=1.0), session_tz="UTC",
                                     or_hour_local=9, last_hour_local=15)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1
        # Entrée à l'Open de la barre 11 (APRÈS le signal à 10), pas au close du signal.
        assert t["entry_price"] == pytest.approx(115.0)
        assert pd.Timestamp(t["entry_time"]).hour == 11
        assert t["exit_reason"] == "eod"
        assert t["exit_price"] == pytest.approx(120.0)
        assert t["pips_brut"] == pytest.approx(5.0)
        assert t["pips_net"] == pytest.approx(4.0)  # 5 − cost_total(=1)
        assert t["nights_held"] == 0

    def test_stop_exit(self) -> None:
        df = _build({"2024-01-03": _DAY_STOP})
        trades = simulate_orb_trades(df, _index_config(spread=1.0), session_tz="UTC",
                                     or_hour_local=9, last_hour_local=15)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(90.0)  # OR_low
        assert t["pips_brut"] == pytest.approx(-25.0)  # 90 − 115
        assert t["pips_net"] == pytest.approx(-26.0)

    def test_pnl_consistency_all_trades(self) -> None:
        df = _build({"2024-01-02": _DAY_LONG_EOD, "2024-01-03": _DAY_STOP})
        cfg = _index_config(spread=1.0)
        cost_total = cfg.spread_pips + 2 * (cfg.slippage_pips + cfg.commission_pips)
        trades = simulate_orb_trades(df, cfg, session_tz="UTC",
                                     or_hour_local=9, last_hour_local=15)
        assert len(trades) == 2
        for t in trades:
            assert t["pips_net"] == pytest.approx(t["pips_brut"] - cost_total)
            assert t["nights_held"] == 0  # intraday → zéro swap

    def test_session_tz_local_grouping(self) -> None:
        """En tz New York (hiver = UTC−5), l'OR est bien la barre 14:00 UTC = 9h ET."""
        # 2024-01-02, EST : 9h ET = 14:00 UTC ; séance jusqu'à 15h ET = 20:00 UTC.
        day = {
            13: (100, 101, 99, 100),    # 8h ET
            14: (100, 110, 90, 100),    # 9h ET → OR high=110 low=90
            15: (100, 116, 100, 115),   # 10h ET → signal LONG
            16: (115, 120, 114, 118),   # 11h ET → entrée Open=115
            17: (118, 121, 116, 119),
            18: (119, 122, 117, 120),
            19: (120, 123, 118, 121),
            20: (121, 124, 119, 122),   # 15h ET → dernière barre, exit EOD
            21: (122, 123, 121, 122),   # 16h ET → hors séance
        }
        df = _build({"2024-01-02": day})
        trades = simulate_orb_trades(
            df, _index_config(), session_tz="America/New_York",
            or_hour_local=9, last_hour_local=15,
        )
        assert len(trades) == 1
        t = trades[0]
        assert t["or_high"] == pytest.approx(110.0)
        assert t["or_low"] == pytest.approx(90.0)
        assert t["signal"] == 1
        assert t["entry_price"] == pytest.approx(115.0)  # Open 16:00 UTC = 11h ET
        assert t["exit_reason"] == "eod"
        assert t["exit_price"] == pytest.approx(122.0)


# Session intraday fine (M5) : OR = fenêtre des 15 premières minutes.
def _build_intraday(
    date_str: str, bars: dict[str, tuple[float, float, float, float]]
) -> pd.DataFrame:
    times = sorted(bars)
    idx = pd.DatetimeIndex([pd.Timestamp(f"{date_str} {t}", tz="UTC") for t in times])
    return pd.DataFrame(
        [bars[t] for t in times], columns=["Open", "High", "Low", "Close"], index=idx
    )


# OR 09:30-09:44 (3 barres M5) → high=110, low=90 ; cassure LONG à 09:45 ; sortie EOD.
_M5_LONG_EOD = {
    "09:30": (100, 110, 95, 105),
    "09:35": (105, 108, 90, 100),   # low 90 → OR_low
    "09:40": (100, 110, 98, 108),   # high 110 → OR_high
    "09:45": (108, 116, 107, 115),  # Close 115 > 110 → signal LONG
    "09:50": (115, 120, 113, 118),  # ENTRÉE Open=115
    "09:55": (118, 121, 116, 119),
    "10:00": (119, 122, 117, 120),  # dernière barre ≤ close → EOD (close=120)
}


class TestSimulateORBSession:
    def test_rejects_bad_window(self) -> None:
        df = _build_intraday("2024-01-02", _M5_LONG_EOD)
        with pytest.raises(ValueError, match="Incohérence horaire"):
            simulate_orb_session(df, _index_config(), session_tz="UTC",
                                 open_time="09:30", or_minutes=60, close_time="10:00")

    def test_or_window_aggregates_first_minutes(self) -> None:
        df = _build_intraday("2024-01-02", _M5_LONG_EOD)
        trades = simulate_orb_session(
            df, _index_config(spread=1.0), session_tz="UTC",
            open_time="09:30", or_minutes=15, close_time="10:00",
        )
        assert len(trades) == 1
        t = trades[0]
        assert t["or_high"] == pytest.approx(110.0)  # agrégé sur 3 barres M5
        assert t["or_low"] == pytest.approx(90.0)
        assert t["or_minutes"] == 15

    def test_long_breakout_entry_next_bar_eod(self) -> None:
        df = _build_intraday("2024-01-02", _M5_LONG_EOD)
        trades = simulate_orb_session(
            df, _index_config(spread=1.0), session_tz="UTC",
            open_time="09:30", or_minutes=15, close_time="10:00",
        )
        t = trades[0]
        assert t["signal"] == 1
        assert t["entry_price"] == pytest.approx(115.0)       # Open 09:50 (après signal 09:45)
        assert pd.Timestamp(t["entry_time"]).minute == 50
        assert t["exit_reason"] == "eod"
        assert t["exit_price"] == pytest.approx(120.0)
        assert t["pips_net"] == pytest.approx(4.0)            # 5 − cost(1)
        assert t["nights_held"] == 0

    def test_stop_exit(self) -> None:
        bars = dict(_M5_LONG_EOD)
        bars["09:50"] = (115, 118, 85, 95)  # Low 85 ≤ OR_low 90 → stop
        df = _build_intraday("2024-01-02", bars)
        trades = simulate_orb_session(
            df, _index_config(spread=1.0), session_tz="UTC",
            open_time="09:30", or_minutes=15, close_time="10:00",
        )
        t = trades[0]
        assert t["exit_reason"] == "stop"
        assert t["exit_price"] == pytest.approx(90.0)
        assert t["pips_net"] == pytest.approx(-26.0)          # (90−115) − 1

    def test_no_breakout_no_trade(self) -> None:
        bars = {
            "09:30": (100, 110, 95, 105),
            "09:35": (105, 108, 90, 100),
            "09:40": (100, 110, 98, 108),
            "09:45": (108, 109, 95, 104),   # close 104 dans [90,110] → pas de signal
            "09:50": (104, 107, 96, 100),
            "10:00": (100, 106, 94, 101),
        }
        df = _build_intraday("2024-01-02", bars)
        trades = simulate_orb_session(
            df, _index_config(), session_tz="UTC",
            open_time="09:30", or_minutes=15, close_time="10:00",
        )
        assert trades == []
