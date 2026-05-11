"""Types fondamentaux : Protocols, TypeAliases, dataclasses transverses.

Tous les composants qui traversent les frontières de package sont définis ici
pour éviter les imports circulaires.
"""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd

# ── TypeAliases ──────────────────────────────────────────────────────────────

#: DataFrame OHLCV standard (index datetime, colonnes Open/High/Low/Close/Volume/Spread)
OHLCVDataFrame: TypeAlias = pd.DataFrame

#: DataFrame ML-ready (index datetime, Target + features)
MLReadyDataFrame: TypeAlias = pd.DataFrame

#: DataFrame de trades (index datetime d'entrée, colonnes Pips_Nets/Pips_Bruts/Weight/result)
TradesDataFrame: TypeAlias = pd.DataFrame

#: NumPy array 1D de float64
FloatArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]

#: NumPy array 1D d'entiers
IntArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]


# ── Protocols ────────────────────────────────────────────────────────────────


@runtime_checkable
class WeightFunction(Protocol):
    """Protocole pour une fonction de sizing.

    Signature : (proba: np.ndarray) -> np.ndarray
    Retourne un poids ∈ [borne_inf, borne_sup] pour chaque probabilité.
    """

    def __call__(self, proba: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class SignalFilter(Protocol):
    """Protocole pour un filtre de régime.

    Chaque filtre examine le DataFrame et les masques LONG/SHORT courants,
    puis retourne les masques filtrés + un compteur de rejets.
    """

    @property
    def name(self) -> str: ...

    def apply(
        self,
        df: pd.DataFrame,
        mask_long: pd.Series,
        mask_short: pd.Series,
    ) -> tuple[pd.Series, pd.Series, int]: ...


# ── Dataclasses transverses ──────────────────────────────────────────────────


class MetricDict(dict):
    """Dict de métriques avec accès par attribut pour rétrocompatibilité.

    >>> m = MetricDict(profit_net=42.0)
    >>> m['profit_net']
    42.0
    >>> m.get('profit_net')
    42.0
    """

    pass
