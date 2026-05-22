"""Chandelier Exit — stratégie trend-following popularisée par LeBeau.

Signal LONG : Close > Highest(High, period) − k_atr * ATR(period).
Signal SHORT : Close < Lowest(Low, period) + k_atr * ATR(period).
"""

from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy
from app.strategies.keltner import _atr


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

        import numpy as np
        n = len(df)
        trend = np.zeros(n, dtype=int)

        close_arr = df["Close"].values
        atr_arr = atr_val.values
        highest_arr = highest.values
        lowest_arr = lowest.values

        for i in range(n):
            if np.isnan(atr_arr[i]) or np.isnan(highest_arr[i]) or np.isnan(lowest_arr[i]):
                trend[i] = 0
                continue

            long_stop = highest_arr[i] - k_atr * atr_arr[i]
            short_stop = lowest_arr[i] + k_atr * atr_arr[i]

            prev_trend = trend[i - 1] if i > 0 else 0

            if prev_trend == 1:
                if close_arr[i] < long_stop:
                    trend[i] = -1
                else:
                    trend[i] = 1
            elif prev_trend == -1:
                if close_arr[i] > short_stop:
                    trend[i] = 1
                else:
                    trend[i] = -1
            else:
                if close_arr[i] > short_stop:
                    trend[i] = 1
                elif close_arr[i] < long_stop:
                    trend[i] = -1
                else:
                    trend[i] = 0

        signals = pd.Series(trend, index=df.index, dtype=int)
        # shift(1) = anti-look-ahead : le signal à t n'utilise que l'info ≤ t-1
        return signals.shift(1).fillna(0).astype(int)
