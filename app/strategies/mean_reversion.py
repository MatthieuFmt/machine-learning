"""Stratégie mean-reversion : RSI extrême + Bollinger reversal.

Pivot v4 B2 — H_new3 : EURUSD H4 mean-reversion + méta-labeling RF.
Signal LONG : RSI < rsi_long ET Close < BB_lower.
Signal SHORT : RSI > rsi_short ET Close > BB_upper.
Anti-look-ahead : .shift(1) obligatoire en fin de generate_signals.
"""
from __future__ import annotations

import pandas as pd

from app.strategies.base import BaseStrategy
from app.testing.look_ahead_validator import look_ahead_safe


@look_ahead_safe
def _bb_bands(
    close: pd.Series,
    period: int,
    mult: float,
) -> tuple[pd.Series, pd.Series]:
    """Bollinger Bands : lower = SMA - mult*std, upper = SMA + mult*std."""
    sma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return sma - mult * sd, sma + mult * sd


class MeanReversionRSIBB(BaseStrategy):
    """Stratégie mean-reversion combinant RSI extrême et Bollinger Bands.

    Signal LONG quand le prix est en condition de survente :
        RSI(14) < 30 ET Close < BB_lower(20, 2).

    Signal SHORT quand le prix est en condition de surachat :
        RSI(14) > 70 ET Close > BB_upper(20, 2).

    Le signal est shifté de 1 barre (anti-look-ahead) :
    le signal calculé sur la barre t est exécuté à l'ouverture de t+1.
    """

    name = "MeanReversionRSIBB"

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_long: float = 30.0,
        rsi_short: float = 70.0,
        bb_period: int = 20,
        bb_mult: float = 2.0,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period,
            rsi_long=rsi_long,
            rsi_short=rsi_short,
            bb_period=bb_period,
            bb_mult=bb_mult,
        )
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.bb_period = bb_period
        self.bb_mult = bb_mult

    @look_ahead_safe
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Produit pd.Series: 1=LONG, -1=SHORT, 0=FLAT, index=df.index.

        Anti-look-ahead : le signal calculé à t utilisant les indicateurs
        de t est shifté à t+1 pour exécution.
        """
        from app.features.indicators import rsi

        close = df["Close"]
        rsi_v = rsi(close, self.rsi_period)
        lower, upper = _bb_bands(close, self.bb_period, self.bb_mult)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[(rsi_v < self.rsi_long) & (close < lower)] = 1
        sig[(rsi_v > self.rsi_short) & (close > upper)] = -1

        # Anti-look-ahead : le signal calculé sur la barre t
        # s'applique à la barre t+1.
        return sig.shift(1).fillna(0).astype(int)

    def __str__(self) -> str:
        return (
            f"MeanReversionRSIBB(rsi={self.rsi_period}/"
            f"{self.rsi_long}/{self.rsi_short}, "
            f"bb={self.bb_period}/{self.bb_mult})"
        )
