"""Keltner Channel — stratégie trend-following.

Breakout haussier : Close > EMA(period) + mult * ATR(period).
Breakdown baissier : Close < EMA(period) - mult * ATR(period).
Reste en position tant que la condition tient.
"""

from __future__ import annotations

import pandas as pd

from learning_machine_learning_v2.strategies.base import BaseStrategy


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """True Range → ATR (EMA smoothing)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class KeltnerChannel(BaseStrategy):
    """Breakout du canal de Keltner.

    Paramètres:
        period: int — période EMA et ATR (défaut 20).
        mult: float — multiplicateur ATR pour la largeur du canal (défaut 2.0).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period: int = int(self.params.get("period", 20))
        mult: float = float(self.params.get("mult", 2.0))

        ema = df["Close"].ewm(span=period, adjust=False).mean()
        atr_val = _atr(df["High"], df["Low"], df["Close"], period)

        upper = ema + mult * atr_val
        lower = ema - mult * atr_val

        long_cond = df["Close"] > upper
        short_cond = df["Close"] < lower

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_cond] = 1
        signals[short_cond] = -1

        return signals
