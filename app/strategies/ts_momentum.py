"""Time-Series Momentum — stratégie momentum pure.

LONG quand return_T > 0.
SHORT quand return_T < 0.
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class TsMomentum(BaseStrategy):
    """Momentum time-series : direction = signe du rendement sur T barres.

    Paramètres:
        T: int — lookback en barres (défaut 20).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        T: int = int(self.params.get("T", 20))

        return_T = df["Close"] / df["Close"].shift(T) - 1.0

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[return_T > 0] = 1
        signals[return_T < 0] = -1

        return signals
