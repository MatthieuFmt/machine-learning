"""Tests unitaires pour app.strategies.asian_range — Phase H2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.asian_range import (
    compute_tokyo_range,
    simulate_asian_range_trades,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _flat_asset_config(
    spread: float = 1.0,
    slippage: float = 0.2,
    pip_size: float = 0.01,
) -> AssetConfig:
    """AssetConfig synthétique pour tests Asian Range (intraday, swap=0)."""
    return AssetConfig(
        spread_pips=spread,
        slippage_pips=slippage,
        commission_pips=0.0,
        pip_size=pip_size,
        pip_value_eur=6.0,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        swap_long_pips_per_night=0.0,
        swap_short_pips_per_night=0.0,
    )


def _build_day(
    date: str,
    tokyo_high: float,
    tokyo_low: float,
    close_07: float,
    bars_08_22: list[tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Crée un DataFrame H1 (00:00 à 22:00 inclus = 23 barres) pour un jour UTC.

    Convention : Tokyo range = 7 barres [00:00, 06:00] (idx 0-6).
    Barre idx 7 (07:00) = signal bar dont le Close est `close_07` ;
    son High/Low n'affectent PAS le range Tokyo.

    Args:
        date: 'YYYY-MM-DD'.
        tokyo_high: High atteint pendant [00:00, 06:00] (positionné idx 2).
        tokyo_low: Low atteint pendant [00:00, 06:00] (positionné idx 3).
        close_07: Close de la barre 07:00 (= signal_price).
        bars_08_22: 15 tuples (open, high, low, close) pour les barres 08:00 à 22:00.
    """
    base = (tokyo_high + tokyo_low) / 2.0
    rows: list[tuple[float, float, float, float]] = []
    # Tokyo range : 7 barres 00:00 à 06:00
    for h in range(7):
        if h == 2:
            rows.append((base, tokyo_high, base, base))
        elif h == 3:
            rows.append((base, base, tokyo_low, base))
        else:
            rows.append((base, base, base, base))
    # Barre 07:00 (signal bar) — High/Low englobent close_07 mais hors range
    hi_07 = max(base, close_07)
    lo_07 = min(base, close_07)
    rows.append((base, hi_07, lo_07, close_07))
    rows.extend(bars_08_22)
    assert len(rows) == 23
    idx = pd.date_range(f"{date} 00:00", periods=23, freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 100.0
    return df


# ─────────────────────────────────────────────────────────────────────
# compute_tokyo_range
# ─────────────────────────────────────────────────────────────────────


class TestComputeTokyoRange:
    def test_basic_range(self) -> None:
        df = _build_day(
            "2024-01-01",
            tokyo_high=100.20, tokyo_low=99.80, close_07=100.05,
            bars_08_22=[(100.0, 100.0, 100.0, 100.0)] * 15,
        )
        ranges = compute_tokyo_range(df)
        assert len(ranges) == 1
        row = ranges.iloc[0]
        assert row["tokyo_high"] == pytest.approx(100.20)
        assert row["tokyo_low"] == pytest.approx(99.80)
        assert row["tokyo_range"] == pytest.approx(0.40)

    def test_skip_day_with_incomplete_bars(self) -> None:
        idx = pd.date_range("2024-01-01 00:00", periods=5, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [100.0] * 5,
                "Low": [100.0] * 5,
                "Close": [100.0] * 5,
            },
            index=idx,
        )
        ranges = compute_tokyo_range(df)
        assert len(ranges) == 0

    def test_multi_days(self) -> None:
        df1 = _build_day(
            "2024-01-01", 100.20, 99.80, 100.05,
            [(100.0, 100.0, 100.0, 100.0)] * 15,
        )
        df2 = _build_day(
            "2024-01-02", 101.50, 100.50, 101.00,
            [(101.0, 101.0, 101.0, 101.0)] * 15,
        )
        df = pd.concat([df1, df2])
        ranges = compute_tokyo_range(df)
        assert len(ranges) == 2
        assert ranges.iloc[0]["tokyo_range"] == pytest.approx(0.40)
        assert ranges.iloc[1]["tokyo_range"] == pytest.approx(1.00)

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=24, freq="1h")
        df = pd.DataFrame(
            {
                "Open": np.ones(24), "High": np.ones(24),
                "Low": np.ones(24), "Close": np.ones(24),
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="tz-aware"):
            compute_tokyo_range(df)


# ─────────────────────────────────────────────────────────────────────
# simulate_asian_range_trades
# ─────────────────────────────────────────────────────────────────────


class TestSimulateAsianRangeTrades:
    """Range Tokyo dans tous les tests :
        tokyo_high = 100.20, tokyo_low = 99.80, range = 0.40 (40 pips pip_size=0.01).
    """

    def test_long_breakout_tp_hit(self) -> None:
        """Close 07:00 > tokyo_high → long ; le prix atteint TP dans la barre 08:00."""
        cfg = _flat_asset_config()
        # signal = long, entry @ Open 08:00 = 100.30
        # TP = 100.30 + 1.5×0.40 = 100.90 (+60 pips brut)
        # SL = 100.30 - 0.5×0.40 = 100.10 (-20 pips brut)
        bars = [(100.30, 100.95, 100.25, 100.85)]
        bars.extend([(100.85, 100.85, 100.85, 100.85)] * 14)
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)

        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == 1
        assert t["exit_reason"] == "tp"
        assert t["entry_price"] == pytest.approx(100.30)
        assert t["exit_price"] == pytest.approx(100.90)
        assert t["pips_brut"] == pytest.approx(60.0)
        # cost_total = spread + 2×slippage = 1.0 + 0.4 = 1.4 pips
        assert t["pips_net"] == pytest.approx(58.6)

    def test_short_breakout_sl_hit(self) -> None:
        """Close 07:00 < tokyo_low → short ; le prix monte jusqu'au SL."""
        cfg = _flat_asset_config()
        # signal = short, entry @ Open 08:00 = 99.70
        # TP short = 99.70 - 1.5×0.40 = 99.10 (+60 pips brut)
        # SL short = 99.70 + 0.5×0.40 = 99.90 (-20 pips brut)
        bars = [(99.70, 99.95, 99.65, 99.85)]
        bars.extend([(99.85, 99.85, 99.85, 99.85)] * 14)
        df = _build_day("2024-01-01", 100.20, 99.80, 99.70, bars)

        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == -1
        assert t["exit_reason"] == "sl"
        assert t["entry_price"] == pytest.approx(99.70)
        assert t["exit_price"] == pytest.approx(99.90)
        # short : pips_brut = (entry - exit) / pip_size = (99.70 - 99.90) / 0.01 = -20
        assert t["pips_brut"] == pytest.approx(-20.0)
        assert t["pips_net"] == pytest.approx(-21.4)

    def test_no_signal_inside_range(self) -> None:
        """Close 07:00 ∈ [tokyo_low, tokyo_high] → pas de trade."""
        cfg = _flat_asset_config()
        bars = [(100.0, 100.0, 100.0, 100.0)] * 15
        df = _build_day("2024-01-01", 100.20, 99.80, 100.00, bars)
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 0

    def test_no_signal_at_exact_boundary(self) -> None:
        """Close 07:00 == tokyo_high : pas de breakout (strict inequality)."""
        cfg = _flat_asset_config()
        bars = [(100.20, 100.20, 100.20, 100.20)] * 15
        df = _build_day("2024-01-01", 100.20, 99.80, 100.20, bars)
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 0

    def test_time_stop_no_tp_sl(self) -> None:
        """Ni TP ni SL touchés → close au Close de la barre 22:00."""
        cfg = _flat_asset_config()
        # long, entry 100.30, TP=100.90, SL=100.10
        # toutes les barres restent dans [100.20, 100.50]
        bars = [(100.30, 100.50, 100.20, 100.40)] * 14
        bars.append((100.40, 100.50, 100.40, 100.45))
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)

        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "time_stop"
        assert t["exit_price"] == pytest.approx(100.45)
        assert t["pips_brut"] == pytest.approx(15.0)
        assert t["pips_net"] == pytest.approx(13.6)

    def test_tp_and_sl_same_bar_sl_priority(self) -> None:
        """Convention conservatrice : SL d'abord si TP et SL touchés même bougie."""
        cfg = _flat_asset_config()
        # long, entry 100.30, TP=100.90, SL=100.10
        # bar 08:00 atteint à la fois 100.95 (TP) et 100.05 (SL) → SL prioritaire
        bars = [(100.30, 100.95, 100.05, 100.50)]
        bars.extend([(100.50, 100.50, 100.50, 100.50)] * 14)
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)

        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "sl"
        assert t["exit_price"] == pytest.approx(100.10)

    def test_one_trade_max_per_day(self) -> None:
        """Test multi-jours : un trade max par jour."""
        cfg = _flat_asset_config()
        df1 = _build_day(
            "2024-01-01", 100.20, 99.80, 100.30,
            [(100.30, 100.95, 100.25, 100.85)]
            + [(100.85, 100.85, 100.85, 100.85)] * 14,
        )
        df2 = _build_day(
            "2024-01-02", 101.20, 100.80, 101.00,  # close in range, no signal
            [(101.0, 101.0, 101.0, 101.0)] * 15,
        )
        df = pd.concat([df1, df2])
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        assert trades[0]["signal"] == 1

    def test_tp_hit_later_bar(self) -> None:
        """TP atteint à la barre 12:00, après plusieurs barres dans le range."""
        cfg = _flat_asset_config()
        bars = []
        # bars 08-11 : drift haussier sans atteindre TP
        bars.extend([(100.30, 100.60, 100.25, 100.55)] * 4)
        # bar 12:00 : atteint TP=100.90
        bars.append((100.55, 100.95, 100.50, 100.92))
        # bars 13-22 : tranquilles
        bars.extend([(100.92, 100.92, 100.92, 100.92)] * 10)
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "tp"
        assert t["exit_price"] == pytest.approx(100.90)
        # entry @ 08:00, exit @ 12:00 → exit_time hour 12
        exit_ts = pd.Timestamp(t["exit_time"])
        assert exit_ts.hour == 12

    def test_short_tp_hit(self) -> None:
        """Short TP touché : Low atteint 99.10 (= entry - 1.5×range)."""
        cfg = _flat_asset_config()
        # short, entry 99.70, TP=99.10, SL=99.90
        bars = [(99.70, 99.75, 99.05, 99.15)]
        bars.extend([(99.15, 99.15, 99.15, 99.15)] * 14)
        df = _build_day("2024-01-01", 100.20, 99.80, 99.70, bars)
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["signal"] == -1
        assert t["exit_reason"] == "tp"
        assert t["exit_price"] == pytest.approx(99.10)
        # pips_brut short = (99.70 - 99.10) / 0.01 = 60
        assert t["pips_brut"] == pytest.approx(60.0)

    def test_missing_entry_bar(self) -> None:
        """Si la barre 08:00 manque, le jour est skippé."""
        cfg = _flat_asset_config()
        # On construit un jour normal puis on supprime la barre 08:00
        bars = [(100.30, 100.95, 100.25, 100.85)]
        bars.extend([(100.85, 100.85, 100.85, 100.85)] * 14)
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)
        df = df.drop(df.index[8])  # supprime 08:00
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 0

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=24, freq="1h")
        df = pd.DataFrame(
            {
                "Open": np.ones(24), "High": np.ones(24),
                "Low": np.ones(24), "Close": np.ones(24),
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="tz-aware"):
            simulate_asian_range_trades(df, _flat_asset_config())

    def test_rejects_signal_in_tokyo_range(self) -> None:
        """signal_hour_utc inclus dans range Tokyo → breakout impossible → ValueError."""
        bars = [(100.0, 100.0, 100.0, 100.0)] * 15
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)
        with pytest.raises(ValueError, match="breakout long impossible"):
            simulate_asian_range_trades(
                df, _flat_asset_config(),
                tokyo_end_hour_utc=7, signal_hour_utc=7,  # collision
            )

    def test_swap_zero_for_intraday(self) -> None:
        """Trade intraday strict : nights_held=0 → pas de coût swap appliqué."""
        cfg = AssetConfig(
            spread_pips=1.0,
            slippage_pips=0.2,
            commission_pips=0.0,
            pip_size=0.01,
            pip_value_eur=6.0,
            tp_points=20,
            sl_points=10,
            window_hours=120,
            min_lot=0.01,
            max_lot=10.0,
            swap_long_pips_per_night=-100.0,  # gros swap pour détecter une fuite
            swap_short_pips_per_night=-100.0,
        )
        bars = [(100.30, 100.95, 100.25, 100.85)]
        bars.extend([(100.85, 100.85, 100.85, 100.85)] * 14)
        df = _build_day("2024-01-01", 100.20, 99.80, 100.30, bars)
        trades = simulate_asian_range_trades(df, cfg)
        assert len(trades) == 1
        t = trades[0]
        assert t["nights_held"] == 0
        # PnL net inchangé par le swap (intraday)
        assert t["pips_net"] == pytest.approx(58.6)
