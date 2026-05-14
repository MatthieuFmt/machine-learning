"""Parabolic SAR — stratégie trend-following de Wilder.

Signal LONG quand le PSAR passe en-dessous du prix (SAR < Close).
Signal SHORT quand le PSAR passe au-dessus du prix (SAR > Close).
Le PSAR s'inverse quand le prix traverse le SAR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from learning_machine_learning_v2.strategies.base import BaseStrategy


def _compute_psar(
    high: np.ndarray,
    low: np.ndarray,
    step: float = 0.02,
    af_max: float = 0.2,
) -> np.ndarray:
    """Calcule le Parabolic SAR (valeur par barre).

    Implémentation vectorisée partielle avec boucle forward obligatoire
    (PSAR est stateful par nature). Optimisé avec pré-allocation NumPy.

    Args:
        high: np.ndarray des prix hauts.
        low: np.ndarray des prix bas.
        step: Pas d'accélération (défaut 0.02).
        af_max: Facteur d'accélération max (défaut 0.2).

    Returns:
        np.ndarray de même longueur, SAR[t] → NaN si pas encore initialisé.
    """
    n = len(high)
    sar = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return sar

    # Initialisation : première barre
    # On détermine la tendance initiale via le close vs close précédent
    # Si pas possible, on suppose tendance haussière
    sar[0] = low[0]

    # Tendance : True = haussière (LONG), False = baissière (SHORT)
    trend_up = True
    af = step
    ep = high[0]  # extreme point

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if trend_up:
            # Tendance haussière
            sar_i = prev_sar + af * (ep - prev_sar)
            # SAR ne doit pas dépasser les lows des 2 barres précédentes
            sar_i = min(sar_i, low[i - 1])
            if i >= 2:
                sar_i = min(sar_i, low[i - 2])

            sar[i] = sar_i

            # Nouvel extreme point
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, af_max)

            # Inversion si prix passe sous le SAR
            if low[i] < sar_i:
                trend_up = False
                sar[i] = ep  # SAR au prochain tour = dernier EP
                ep = low[i]
                af = step
        else:
            # Tendance baissière
            sar_i = prev_sar - af * (prev_sar - ep)
            # SAR ne doit pas être en-dessous des highs des 2 barres précédentes
            sar_i = max(sar_i, high[i - 1])
            if i >= 2:
                sar_i = max(sar_i, high[i - 2])

            sar[i] = sar_i

            # Nouvel extreme point
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, af_max)

            # Inversion si prix passe au-dessus du SAR
            if high[i] > sar_i:
                trend_up = True
                sar[i] = ep
                ep = high[i]
                af = step

    return sar


class ParabolicSAR(BaseStrategy):
    """Parabolic SAR — trend-following stateful.

    Signal LONG quand SAR < Close (le SAR soutient le prix).
    Signal SHORT quand SAR > Close (le SAR résiste au prix).

    Paramètres:
        step: float — pas d'accélération (défaut 0.02).
        af_max: float — facteur d'accélération max (défaut 0.2).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        step: float = float(self.params.get("step", 0.02))
        af_max: float = float(self.params.get("af_max", 0.2))

        high_arr = df["High"].values
        low_arr = df["Low"].values
        close_arr = df["Close"].values

        sar_arr = _compute_psar(high_arr, low_arr, step=step, af_max=af_max)

        long_cond = close_arr > sar_arr
        short_cond = close_arr < sar_arr

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[long_cond & ~np.isnan(sar_arr)] = 1
        signals[short_cond & ~np.isnan(sar_arr)] = -1

        return signals
