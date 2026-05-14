"""Bollinger Bands — stratégie mean-reversion.

LONG quand Close < BB_lower (SMA - K*std).
SHORT quand Close > BB_upper (SMA + K*std).
"""

from __future__ import annotations

import pandas as pd

from learning_machine_learning_v2.strategies.base import BaseStrategy


class BollingerBands(BaseStrategy):
    """Bandes de Bollinger : entrée contre la bande.

    Paramètres:
        N: int — période SMA (défaut 20).
        K: float — nombre d'écarts-types (défaut 2.0).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        N: int = int(self.params.get("N", 20))
        K: float = float(self.params.get("K", 2.0))

        sma = df["Close"].rolling(window=N).mean()
        std = df["Close"].rolling(window=N).std()

        bb_upper = sma + K * std
        bb_lower = sma - K * std

        long_entry = df["Close"] < bb_lower
        short_entry = df["Close"] > bb_upper

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_entry] = 1
        signals[short_entry] = -1

        return signals
