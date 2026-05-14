"""Chandelier Exit — stratégie trend-following popularisée par LeBeau.

Signal LONG : Close > Highest(High, period) − k_atr * ATR(period).
Signal SHORT : Close < Lowest(Low, period) + k_atr * ATR(period).
"""

from __future__ import annotations

import pandas as pd

from learning_machine_learning_v2.strategies.base import BaseStrategy
from learning_machine_learning_v2.strategies.keltner import _atr


class ChandelierExit(BaseStrategy):
    """Breakout du Chandelier Exit (LeBeau).

    Paramètres:
        period: int — période lookback (défaut 22, ≈ 1 mois D1).
        k_atr: float — multiplicateur ATR (défaut 3.0).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period: int = int(self.params.get("period", 22))
        k_atr: float = float(self.params.get("k_atr", 3.0))

        atr_val = _atr(df["High"], df["Low"], df["Close"], period)
        highest = df["High"].rolling(window=period).max()
        lowest = df["Low"].rolling(window=period).min()

        long_trigger = highest.shift(1) - k_atr * atr_val
        short_trigger = lowest.shift(1) + k_atr * atr_val

        long_cond = df["Close"] > long_trigger
        short_cond = df["Close"] < short_trigger

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_cond] = 1
        signals[short_cond] = -1

        return signals
