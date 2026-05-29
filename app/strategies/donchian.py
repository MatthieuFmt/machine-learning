"""Donchian Breakout — stratégie trend-following.

Breakout haussier : Close > High.rolling(N).max().shift(1).
Breakdown baissier : Close < Low.rolling(M).min().shift(1).
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """Breakout du canal Donchian.

    Paramètres:
        N: int — période du canal d'entrée (défaut 20).
        M: int — période du canal de sortie (défaut 10).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        N: int = int(self.params.get("N", 20))
        M: int = int(self.params.get("M", 10))

        high_roll = df["High"].rolling(window=N).max()
        low_roll = df["Low"].rolling(window=M).min()

        # Breakout haussier
        long_entry = df["Close"] > high_roll.shift(1)

        # Breakdown baissier
        short_entry = df["Close"] < low_roll.shift(1)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_entry] = 1
        signals[short_entry] = -1

        return signals
