"""Classe abstraite pour les stratégies déterministes H03."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Stratégie déterministe : prend un DataFrame OHLC, retourne des signaux.

    Les signaux sont : 1 = LONG, -1 = SHORT, 0 = FLAT.
    """

    def __init__(self, **params: int | float) -> None:
        self.params: dict[str, int | float] = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Retourne pd.Series: 1=LONG, -1=SHORT, 0=FLAT, index=df.index."""
        ...
