"""Tests unitaires pour app.strategies.pre_fomc_drift — Phase H1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.pre_fomc_drift import (
    _parse_et_datetime,
    simulate_pre_fomc_trades,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _synthetic_us500_h1(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    """OHLC H1 synthétique sur n_days jours, prix de base ~6000."""
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="1h", tz="UTC")
    returns = rng.normal(0, 0.001, n_hours)
    close = 6000.0 * np.cumprod(1 + returns)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.0005,
            "Low": close * 0.9995,
            "Close": close,
            "Volume": rng.uniform(100, 1000, n_hours),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


# ─────────────────────────────────────────────────────────────────────
# _parse_et_datetime
# ─────────────────────────────────────────────────────────────────────


class TestParseETDatetime:
    def test_pm_time(self) -> None:
        ts = _parse_et_datetime("2024-03-20", "1:00pm")
        assert ts is not None
        # March → DST → ET = UTC-4
        utc = ts.tz_convert("UTC")
        assert utc.hour == 17  # 13h ET + 4 = 17h UTC

    def test_am_time(self) -> None:
        ts = _parse_et_datetime("2024-01-15", "11:30am")
        assert ts is not None
        # January → standard time → ET = UTC-5
        utc = ts.tz_convert("UTC")
        assert utc.hour == 16  # 11h30 ET + 5 = 16h30 UTC
        assert utc.minute == 30

    def test_unparseable(self) -> None:
        assert _parse_et_datetime("2024-01-01", "All Day") is None
        assert _parse_et_datetime("2024-01-01", "Tentative") is None
        assert _parse_et_datetime("2024-01-01", "") is None


# ─────────────────────────────────────────────────────────────────────
# simulate_pre_fomc_trades
# ─────────────────────────────────────────────────────────────────────


class TestSimulatePreFomcTrades:
    def test_basic_long_drift_positive(self) -> None:
        """Quand le marché monte régulièrement, le PnL net doit être positif."""
        df = _synthetic_us500_h1(n_days=30, seed=1)
        # Tendance haussière forte : multiplier par croissance linéaire
        df["Close"] = df["Close"] * np.linspace(1.0, 1.10, len(df))
        df["High"] = df["Close"] * 1.0005
        df["Low"] = df["Close"] * 0.9995
        df["Open"] = df["Close"]

        # 3 FOMC events synthétiques au milieu de la période
        fomc_times = pd.DatetimeIndex([
            pd.Timestamp("2024-01-10 18:00:00", tz="UTC"),
            pd.Timestamp("2024-01-15 18:00:00", tz="UTC"),
            pd.Timestamp("2024-01-20 18:00:00", tz="UTC"),
        ])

        trades = simulate_pre_fomc_trades(
            df=df, fomc_times=fomc_times,
            spread_pips=0.5, slippage_pips=0.1, commission_pips=0.0,
            pip_size=0.1, swap_long_pips_per_night=-16.0,
        )
        assert len(trades) == 3
        # Sur une tendance forte, mean PnL doit être positif malgré le swap
        mean_pnl = np.mean([t["pips_net"] for t in trades])
        assert mean_pnl > 0

    def test_costs_applied(self) -> None:
        """Sur un marché flat, le PnL doit être ~-spread - swap (négatif)."""
        df = _synthetic_us500_h1(n_days=10, seed=2)
        # Marché flat parfait
        df["Close"] = 6000.0
        df["High"] = 6000.5
        df["Low"] = 5999.5
        df["Open"] = 6000.0

        fomc_times = pd.DatetimeIndex([
            pd.Timestamp("2024-01-05 18:00:00", tz="UTC"),
        ])

        trades = simulate_pre_fomc_trades(
            df=df, fomc_times=fomc_times,
            spread_pips=5.0, slippage_pips=1.0, commission_pips=0.0,
            pip_size=0.1, swap_long_pips_per_night=-16.0,
        )
        assert len(trades) == 1
        # Coûts attendus : spread 5 + slippage_per_side 2×1 + swap 1×(-16) = -23 pips
        assert trades[0]["pips_brut"] == pytest.approx(0.0, abs=1.0)
        assert trades[0]["pips_net"] < 0  # nettement négatif

    def test_skip_if_no_bars(self) -> None:
        """Si une date FOMC n'a aucune barre dans la fenêtre, on skip."""
        df = _synthetic_us500_h1(n_days=10, seed=3)
        # FOMC très loin de la data
        fomc_times = pd.DatetimeIndex([
            pd.Timestamp("2030-01-01 18:00:00", tz="UTC"),
        ])

        trades = simulate_pre_fomc_trades(
            df=df, fomc_times=fomc_times,
            spread_pips=0.5, slippage_pips=0.1, commission_pips=0.0,
            pip_size=0.1, swap_long_pips_per_night=0.0,
        )
        assert len(trades) == 0

    def test_entry_exit_bars_ordered(self) -> None:
        """Pour chaque trade, exit_time > entry_time."""
        df = _synthetic_us500_h1(n_days=30, seed=4)
        fomc_times = pd.DatetimeIndex([
            pd.Timestamp("2024-01-10 18:00:00", tz="UTC"),
            pd.Timestamp("2024-01-20 18:00:00", tz="UTC"),
        ])
        trades = simulate_pre_fomc_trades(
            df=df, fomc_times=fomc_times,
            spread_pips=0.5, slippage_pips=0.1, commission_pips=0.0,
            pip_size=0.1, swap_long_pips_per_night=0.0,
        )
        for t in trades:
            assert pd.Timestamp(t["exit_time"]) > pd.Timestamp(t["entry_time"])

    def test_nights_held_at_least_one(self) -> None:
        """Une fenêtre de 23h doit franchir au moins 0-1 minuit UTC."""
        df = _synthetic_us500_h1(n_days=10, seed=5)
        fomc_times = pd.DatetimeIndex([
            pd.Timestamp("2024-01-05 18:00:00", tz="UTC"),
        ])
        trades = simulate_pre_fomc_trades(
            df=df, fomc_times=fomc_times,
            spread_pips=0.5, slippage_pips=0.1, commission_pips=0.0,
            pip_size=0.1, swap_long_pips_per_night=-16.0,
        )
        assert len(trades) == 1
        # entry à 18h le jour D-1, exit à 17h le jour D → 1 nuit traversée
        assert trades[0]["nights_held"] == 1

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")  # naive
        df = pd.DataFrame({"Close": np.ones(100), "High": np.ones(100),
                           "Low": np.ones(100), "Open": np.ones(100)}, index=idx)
        with pytest.raises(ValueError, match="tz-aware"):
            simulate_pre_fomc_trades(
                df=df, fomc_times=pd.DatetimeIndex([]),
                spread_pips=0.5, slippage_pips=0.1, commission_pips=0.0,
                pip_size=0.1,
            )
