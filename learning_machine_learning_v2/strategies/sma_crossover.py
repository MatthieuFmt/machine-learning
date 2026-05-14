"""SMA Crossover — stratégie trend-following.

Signal LONG quand SMA(fast) croise au-dessus de SMA(slow).
Signal SHORT quand SMA(fast) croise en-dessous de SMA(slow).
"""

from __future__ import annotations

import pandas as pd

from learning_machine_learning_v2.strategies.base import BaseStrategy


class SmaCrossover(BaseStrategy):
    """Croisement de deux SMA : fast croise slow → entrée directionnelle.

    Paramètres:
        fast: int — période SMA rapide (défaut 5).
        slow: int — période SMA lente (défaut 20).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast: int = int(self.params.get("fast", 5))
        slow: int = int(self.params.get("slow", 20))

        sma_fast = df["Close"].rolling(window=fast).mean()
        sma_slow = df["Close"].rolling(window=slow).mean()

        # Croisement haussier : fast passe au-dessus de slow
        cross_up = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))

        # Croisement baissier : fast passe en-dessous de slow
        cross_down = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[cross_up] = 1
        signals[cross_down] = -1

        return signals
