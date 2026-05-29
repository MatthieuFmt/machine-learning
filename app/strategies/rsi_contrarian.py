"""RSI Contrarian — stratégie mean-reversion.

LONG quand RSI passe sous oversold.
SHORT quand RSI passe au-dessus de overbought.
overbought = 100 - oversold.
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy


class RsiContrarian(BaseStrategy):
    """RSI contrarian : achat sur survente, vente sur surachat.

    Paramètres:
        N: int — période RSI (défaut 14).
        oversold: int — seuil de survente (défaut 30).
        overbought: int — dérivé = 100 - oversold.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        N: int = int(self.params.get("N", 14))
        oversold: int = int(self.params.get("oversold", 30))
        overbought: int = 100 - oversold

        # Calcul RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=N).mean()
        avg_loss = loss.rolling(window=N).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Entrée LONG : RSI passe sous oversold (traverse le seuil vers le bas)
        long_entry = (rsi < oversold) & (rsi.shift(1) >= oversold)

        # Entrée SHORT : RSI passe au-dessus de overbought
        short_entry = (rsi > overbought) & (rsi.shift(1) <= overbought)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_entry] = 1
        signals[short_entry] = -1

        return signals
