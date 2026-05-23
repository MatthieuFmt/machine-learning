"""Tests unitaires pour app.strategies.volatility_breakout — Phase H3 NR7."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.volatility_breakout import (
    compute_nr_days,
    simulate_nr_breakout_trades,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _flat_asset_config(
    spread: float = 0.5,
    slippage: float = 0.1,
    pip_size: float = 0.1,
    swap_long: float = -16.0,
    swap_short: float = 2.0,
) -> AssetConfig:
    """AssetConfig synthétique calé sur US500 (pip_size=0.1)."""
    return AssetConfig(
        spread_pips=spread,
        slippage_pips=slippage,
        commission_pips=0.0,
        pip_size=pip_size,
        pip_value_eur=0.092,
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        swap_long_pips_per_night=swap_long,
        swap_short_pips_per_night=swap_short,
    )


def _bar(open_: float, high: float, low: float, close: float) -> tuple[float, float, float, float]:
    """Helper pour créer une bougie OHLC. Vérifie cohérence."""
    assert high >= max(open_, close) and low <= min(open_, close), (
        f"OHLC incohérent : O={open_}, H={high}, L={low}, C={close}"
    )
    return (open_, high, low, close)


def _bar_center(close: float, range_: float, drift: float = 0.0) -> tuple[float, float, float, float]:
    """Bougie centrée sur close avec range donné. drift > 0 = haussière."""
    open_ = close - drift
    half = range_ / 2
    high = max(open_, close) + half - abs(drift) / 2
    low = min(open_, close) - half + abs(drift) / 2
    return _bar(open_, high, low, close)


def _df_from_bars(
    bars: list[tuple[float, float, float, float]],
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    idx = pd.date_range(start_date, periods=len(bars), freq="1D", tz="UTC")
    df = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 100.0
    return df


# ─────────────────────────────────────────────────────────────────────
# compute_nr_days
# ─────────────────────────────────────────────────────────────────────


class TestComputeNrDays:
    def test_basic_nr7_detected_at_day_7(self) -> None:
        """7 jours avec ranges décroissants : seul le 7ème est NR7."""
        bars = []
        for r in [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]:
            bars.append(_bar_center(close=100.0, range_=r))
        df = _df_from_bars(bars)
        result = compute_nr_days(df, lookback=7)
        assert len(result) == 7
        # Seul le jour 6 (index 6, le 7ème) est NR7 (range le plus petit sur 7)
        is_nr = result["is_nr"].tolist()
        assert is_nr == [False, False, False, False, False, False, True]
        # day_range correct
        assert result.iloc[6]["day_range"] == pytest.approx(4.0)

    def test_nr7_not_at_day_6(self) -> None:
        """6 jours seulement → aucun NR7 possible (warmup non atteint)."""
        bars = [_bar_center(close=100.0, range_=10.0 - i) for i in range(6)]
        df = _df_from_bars(bars)
        result = compute_nr_days(df, lookback=7)
        assert result["is_nr"].sum() == 0

    def test_multiple_nr7_setups(self) -> None:
        """Plusieurs NR7 successifs."""
        # 7 jours initiaux avec ranges 10..4, day 7 NR7
        # Day 8 : range 3 < min(9..4)=4 → NR7
        # Day 9 : range 2 < min(8..3)=3 → NR7
        bars = []
        for r in [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]:
            bars.append(_bar_center(close=100.0, range_=r))
        df = _df_from_bars(bars)
        result = compute_nr_days(df, lookback=7)
        is_nr = result["is_nr"].tolist()
        assert is_nr[6] is True
        assert is_nr[7] is True
        assert is_nr[8] is True

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1D")
        df = pd.DataFrame(
            {"Open": np.ones(10), "High": np.ones(10), "Low": np.ones(10), "Close": np.ones(10)},
            index=idx,
        )
        with pytest.raises(ValueError, match="tz-aware"):
            compute_nr_days(df, lookback=7)


# ─────────────────────────────────────────────────────────────────────
# simulate_nr_breakout_trades
# ─────────────────────────────────────────────────────────────────────


def _build_nr7_setup(
    nr_close: float = 100.0,
    nr_range: float = 4.0,
) -> list[tuple[float, float, float, float]]:
    """Construit 7 bougies avec le jour 6 (= jour J) en NR7.

    Ranges décroissants 10, 9, 8, 7, 6, 5, nr_range. Tous centrés sur nr_close.
    """
    bars = []
    for r in [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]:
        bars.append(_bar_center(close=nr_close, range_=r))
    bars.append(_bar_center(close=nr_close, range_=nr_range))
    return bars


class TestSimulateNrBreakoutTrades:
    """Pour tous les tests : NR7 setup sur jour 6 (idx 6, day J).
    Jour J : close=100.0, range=4.0 → High(J)≈102.0, Low(J)≈98.0.
    Day J+1 (idx 7) = jour d'exécution.
    """

    def test_no_trade_if_no_nr7(self) -> None:
        """Pas de NR7 → aucun trade."""
        cfg = _flat_asset_config()
        # 7 bougies avec ranges croissants → jour 6 = range max, pas NR7
        bars = []
        for r in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            bars.append(_bar_center(close=100.0, range_=r))
        bars.append(_bar(100.0, 110.0, 90.0, 105.0))  # day 7 (J+1)
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 0

    def test_long_breakout_via_stop_tp_hit(self) -> None:
        """NR7 + day J+1 High > High(J) seul → long @ High(J), TP touché."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()  # 7 bougies, jour 6 NR7
        # High(J) ≈ 102.0, Low(J) ≈ 98.0, range_NR = 4.0
        # Day J+1 : Open=100.0 (dans range), High=110.0 (> 102), Low=99.0 (> 98)
        # Long stop déclenché à High(J)=102.0
        # TP = 102.0 + 2×4.0 = 110.0, SL = 102.0 - 1×4.0 = 98.0
        bars.append(_bar(100.0, 110.5, 99.0, 109.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1
        assert t["exit_reason"] == "tp"
        assert t["entry_price"] == pytest.approx(102.0)
        assert t["exit_price"] == pytest.approx(110.0)
        # pips_brut = (110.0 - 102.0) / 0.1 = 80 pips
        assert t["pips_brut"] == pytest.approx(80.0)
        # cost = 0.5 + 2×0.1 = 0.7, nights_held=0 (TP) → pips_net = 80 - 0.7
        assert t["pips_net"] == pytest.approx(79.3)
        assert t["nights_held"] == 0

    def test_short_breakout_via_stop_tp_hit(self) -> None:
        """NR7 + day J+1 Low < Low(J) seul, High < High(J) → short @ Low(J), TP touché.

        Note : SL short via stop intra-bar serait à entry+range=Low(J)+range=High(J).
        Pour toucher SL il faudrait High_J+1 ≥ High(J) → cas ambigu (up_hit & down_hit) → skip.
        SL short ne peut donc être atteint que via gap down (cf test_tp_and_sl_after_gap_entry).
        """
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Short stop à Low(J)=98.0
        # TP_short = 98.0 - 2×4.0 = 90.0, SL_short = 98.0 + 1×4.0 = 102.0
        # Day J+1 : Open=100.0, High=101.5 (< 102), Low=89.0 (≤ 90 TP), Close=92.0
        bars.append(_bar(100.0, 101.5, 89.0, 92.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == -1
        assert t["exit_reason"] == "tp"
        assert t["entry_price"] == pytest.approx(98.0)
        assert t["exit_price"] == pytest.approx(90.0)
        # short : pips_brut = (entry - exit) / pip_size = (98 - 90) / 0.1 = 80
        assert t["pips_brut"] == pytest.approx(80.0)
        assert t["pips_net"] == pytest.approx(79.3)

    def test_gap_up_open_above_high(self) -> None:
        """Day J+1 Open > High(J) → entry @ Open(J+1), pas au stop."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # High(J)=102.0. Day J+1 ouvre à 103.0 (gap up), High=115, Low=102.5, Close=110
        bars.append(_bar(103.0, 115.0, 102.5, 110.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1
        assert t["entry_price"] == pytest.approx(103.0)
        # TP = 103 + 2×4 = 111, SL = 103 - 1×4 = 99
        # High = 115 ≥ 111 → TP ; Low = 102.5 > 99 → pas SL
        assert t["exit_reason"] == "tp"
        assert t["exit_price"] == pytest.approx(111.0)

    def test_gap_down_open_below_low(self) -> None:
        """Day J+1 Open < Low(J) → entry @ Open(J+1) short."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Low(J)=98.0. Day J+1 ouvre à 97.0 (gap down), High=98.5, Low=85, Close=88
        bars.append(_bar(97.0, 98.5, 85.0, 88.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == -1
        assert t["entry_price"] == pytest.approx(97.0)
        # TP_short = 97 - 2×4 = 89, SL_short = 97 + 1×4 = 101
        # Low = 85 ≤ 89 → TP ; High = 98.5 < 101 → pas SL
        assert t["exit_reason"] == "tp"
        assert t["exit_price"] == pytest.approx(89.0)

    def test_no_breakout_no_trade(self) -> None:
        """Day J+1 reste dans [Low(J), High(J)] → aucun stop touché → no trade."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Day J+1 : Open=100, High=101.5 (< 102), Low=98.5 (> 98), Close=101
        bars.append(_bar(100.0, 101.5, 98.5, 101.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 0

    def test_ambiguous_both_stops_hit_intra_skip(self) -> None:
        """Open dans range, High > High(J) ET Low < Low(J) → ambigu → skip."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Day J+1 : Open=100 (dans [98, 102]), High=103 (> 102), Low=97 (< 98)
        bars.append(_bar(100.0, 103.0, 97.0, 99.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 0

    def test_time_stop_no_tp_no_sl(self) -> None:
        """Long stop déclenché mais ni TP ni SL touché → exit @ Close(J+1)."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Long stop @ 102.0, TP=110, SL=98
        # Day J+1 : Open=100, High=104 (> 102 trigger long, < 110 TP), Low=99 (> 98 SL),
        # Close=103
        bars.append(_bar(100.0, 104.0, 99.0, 103.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1
        assert t["entry_price"] == pytest.approx(102.0)
        assert t["exit_reason"] == "time_stop"
        assert t["exit_price"] == pytest.approx(103.0)
        # nights_held=1 (time-stop, position tient 1 jour)
        assert t["nights_held"] == 1
        # pips_brut = (103 - 102) / 0.1 = 10
        # cost = 0.7, swap_long = -16 → pips_net = 10 - 0.7 - 16 = -6.7
        assert t["pips_net"] == pytest.approx(-6.7)

    def test_tp_and_sl_same_bar_sl_priority(self) -> None:
        """Long entry @ High(J), Day J+1 touche TP ET SL → SL prioritaire."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Long stop @ 102, TP=110, SL=98
        # Day J+1 : Open=100, High=112 (> TP), Low=97 (< SL), Close=105
        bars.append(_bar(100.0, 112.0, 97.0, 105.0))
        df = _df_from_bars(bars)
        # Note : Low(97) < Low(J)=98 ET High(112) > High(J)=102 → ambigu → skip
        # Donc ce test illustre que dans ce cas, le code SKIP plutôt que d'arbitrer.
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 0

    def test_tp_and_sl_after_gap_entry_sl_priority(self) -> None:
        """Entry au Open via gap up, J+1 touche TP ET SL → SL prioritaire."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        # Gap up : Open=103 > High(J)=102 → entry @ 103, TP=111, SL=99
        # High=112 (> TP), Low=98 (< SL), Close=110
        bars.append(_bar(103.0, 112.0, 98.0, 110.0))
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1
        assert t["entry_price"] == pytest.approx(103.0)
        assert t["exit_reason"] == "sl"
        assert t["exit_price"] == pytest.approx(99.0)

    def test_skip_if_no_next_day(self) -> None:
        """NR7 détecté sur dernier jour du DataFrame → pas de J+1 → skip."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()  # 7 bougies, dernière = NR7
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 0

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1D")
        df = pd.DataFrame(
            {"Open": np.ones(10), "High": np.ones(10), "Low": np.ones(10), "Close": np.ones(10)},
            index=idx,
        )
        with pytest.raises(ValueError, match="tz-aware"):
            simulate_nr_breakout_trades(df, _flat_asset_config())

    def test_metadata_fields_present(self) -> None:
        """Vérifie que tous les champs attendus sont présents dans un trade."""
        cfg = _flat_asset_config()
        bars = _build_nr7_setup()
        bars.append(_bar(100.0, 110.5, 99.0, 109.0))  # long TP
        df = _df_from_bars(bars)
        trades = simulate_nr_breakout_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        expected_keys = {
            "setup_date", "entry_time", "exit_time", "signal",
            "entry_price", "exit_price", "high_J", "low_J", "range_J",
            "tp_price", "sl_price", "pips_brut", "pips_net",
            "nights_held", "exit_reason",
        }
        assert expected_keys.issubset(set(t.keys()))
