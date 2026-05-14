"""Dual Moving Average — stratégie trend-following avec deux SMA.

Contrairement à SmaCrossover (signal ponctuel au croisement), Dual MA reste
en position tant que SMA(fast) > SMA(slow) pour LONG, et inversement pour SHORT.
Changement de position uniquement quand la relation s'inverse.
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class DualMovingAverage(BaseStrategy):
    """Position LONG quand SMA(fast) > SMA(slow), SHORT sinon.

    Paramètres:
        fast: int — période SMA rapide (défaut 10).
        slow: int — période SMA lente (défaut 50).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast: int = int(self.params.get("fast", 10))
        slow: int = int(self.params.get("slow", 50))

        sma_fast = df["Close"].rolling(window=fast).mean()
        sma_slow = df["Close"].rolling(window=slow).mean()

        long_cond = sma_fast > sma_slow
        short_cond = sma_fast < sma_slow

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_cond] = 1
        signals[short_cond] = -1

        # shift(1) = anti-look-ahead : le signal à t n'utilise que l'info ≤ t-1
        return signals.shift(1).fillna(0).astype(int)
