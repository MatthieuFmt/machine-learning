"""Tests stratégie MeanReversionRSIBB — pivot v4 B2.

≥ 6 tests :
1. Signal LONG quand RSI < 30 ET Close < BB_lower
2. Signal SHORT quand RSI > 70 ET Close > BB_upper
3. Pas de signal quand seul RSI < 30 (Close non sous BB_lower)
4. Cas dégénéré : série constante → 0 signal
5. Anti-look-ahead : les signaux à t n'utilisent que l'information ≤ t
6. Marqueur @look_ahead_safe présent
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.indicators import rsi
from app.strategies.mean_reversion import MeanReversionRSIBB, _bb_bands
from app.testing.look_ahead_validator import assert_no_look_ahead


def _make_df(
    close_values: list[float],
    freq: str = "4h",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Construit un DataFrame OHLC minimal à partir d'une séquence de Close."""
    n = len(close_values)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = np.asarray(close_values, dtype=float)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes + 0.0005,
            "Low": closes - 0.0005,
            "Close": closes,
            "Volume": [1000.0] * n,
        },
        index=idx,
    )


def _build_oversold_series(
    n_warmup: int = 130,
    n_crash: int = 30,
) -> list[float]:
    """Série déclenchant RSI < 30 ET Close < BB_lower.

    Crash convexe (progress**2) : la chute accélère en fin de séquence,
    garantissant |Close − SMA| / std > 2 sur les dernières barres.
    """
    rng = np.random.default_rng(42)
    warmup = [1.1000 + rng.normal(0, 0.0003) for _ in range(n_warmup)]
    crash_start = warmup[-1]
    progress = np.linspace(0, 1, n_crash)
    crash = (crash_start - 0.05 * progress**2).tolist()
    return warmup + crash


def _build_overbought_series(
    n_warmup: int = 130,
    n_rally: int = 30,
) -> list[float]:
    """Série déclenchant RSI > 70 ET Close > BB_upper.

    Rally convexe (progress**2) : la montée accélère en fin de séquence.
    """
    rng = np.random.default_rng(43)
    warmup = [1.1000 + rng.normal(0, 0.0003) for _ in range(n_warmup)]
    rally_start = warmup[-1]
    progress = np.linspace(0, 1, n_rally)
    rally = (rally_start + 0.05 * progress**2).tolist()
    return warmup + rally


class TestMeanReversionRSIBB:
    """Tests unitaires pour la stratégie mean-reversion RSI + Bollinger."""

    def test_long_signal_on_oversold_breakdown(self) -> None:
        """Krach sous BB_lower + RSI < 30 → signal LONG."""
        closes = _build_oversold_series()
        df = _make_df(closes)
        strat = MeanReversionRSIBB()
        sig = strat.generate_signals(df)

        # Vérifie que RSI < 30 sur les barres de crash
        rsi_vals = rsi(df["Close"], 14)
        crash_rsi = rsi_vals.iloc[-15:]
        assert (crash_rsi < 30).any(), (
            f"RSI pas sous 30 : min={crash_rsi.min():.1f}"
        )

        # Vérifie que Close < BB_lower sur au moins une barre de crash
        lower, _ = _bb_bands(df["Close"], 20, 2.0)
        close_vs_lower = df["Close"].iloc[-15:] < lower.iloc[-15:]
        assert close_vs_lower.any(), "Close jamais sous BB_lower"

        # Au moins un signal LONG (shifté de 1 barre)
        assert sig.iloc[-20:].max() == 1, "Aucun signal LONG détecté"

    def test_short_signal_on_overbought_breakout(self) -> None:
        """Rally au-dessus BB_upper + RSI > 70 → signal SHORT."""
        closes = _build_overbought_series()
        df = _make_df(closes)
        strat = MeanReversionRSIBB()
        sig = strat.generate_signals(df)

        rsi_vals = rsi(df["Close"], 14)
        rally_rsi = rsi_vals.iloc[-15:]
        assert (rally_rsi > 70).any(), (
            f"RSI pas au-dessus de 70 : max={rally_rsi.max():.1f}"
        )

        _, upper = _bb_bands(df["Close"], 20, 2.0)
        close_vs_upper = df["Close"].iloc[-15:] > upper.iloc[-15:]
        assert close_vs_upper.any(), "Close jamais au-dessus BB_upper"

        assert sig.iloc[-20:].min() == -1, "Aucun signal SHORT détecté"

    def test_no_signal_flat_series(self) -> None:
        """Close constant → 0 signal."""
        closes = [1.10] * 100
        df = _make_df(closes)
        strat = MeanReversionRSIBB()
        sig = strat.generate_signals(df)
        assert (sig == 0).all(), "Signal non nul sur série constante"

    def test_no_signal_rsi_only_no_bb_confirm(self) -> None:
        """RSI < 30 mais Close pas sous BB_lower → pas de signal LONG."""
        rng = np.random.default_rng(42)
        n = 250
        base = 1.10 + rng.normal(0, 0.003, n).cumsum() * 0.1
        # Déclin très progressif : RSI descend mais BB_lower suit
        base[-40:] = np.linspace(base[-41], base[-41] - 0.006, 40)
        closes = base.tolist()
        df = _make_df(closes)
        strat = MeanReversionRSIBB(rsi_long=30, rsi_short=70)
        sig = strat.generate_signals(df)
        long_bars = (sig == 1).sum()
        total_bars = len(sig)
        assert (
            long_bars / max(total_bars, 1) < 0.05
        ), f"Trop de signaux LONG : {long_bars}/{total_bars}"

    def test_anti_look_ahead_shift(self) -> None:
        """Vérifie que generate_signals ne contamine pas le passé."""
        rng = np.random.default_rng(42)
        closes = (
            1.10 + rng.normal(0, 0.005, 500).cumsum() * 0.1
        ).tolist()
        df = _make_df(closes)
        strat = MeanReversionRSIBB()
        assert_no_look_ahead(strat.generate_signals, df, n_samples=50)

    def test_marker_look_ahead_safe(self) -> None:
        """Le décorateur @look_ahead_safe doit être présent."""
        strat = MeanReversionRSIBB()
        assert getattr(
            strat.generate_signals, "_look_ahead_safe", False
        ), "generate_signals non marqué @look_ahead_safe"
