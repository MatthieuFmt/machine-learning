"""Features de regime — volatilite, range/ATR, momentum macro, tendance long-terme.

Ces features sont calculees principalement sur D1 puis reparties sur H1 via merge_asof.
Elles ne doivent PAS etre utilisees comme features d'entrainement du modele
(risque de surapprentissage), mais servent aux filtres de regime (TrendFilter, VolFilter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def calc_volatilite_realisee(
    log_returns: pd.Series, window: int = 24
) -> pd.Series:
    """Volatilite realisee : ecart-type des log-returns sur `window` periodes.

    Args:
        log_returns: Serie de log-returns H1.
        window: Fenetre de calcul (24 = 24h).

    Returns:
        Serie de volatilite realisee.
    """
    return log_returns.rolling(window=window).std()


def calc_range_atr_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Ratio Range / ATR : detecte expansion (>1) vs contraction (<1).

    Args:
        high, low, close: Series OHLC.
        length: Periode ATR.

    Returns:
        Serie du ratio. +1e-10 evite la division par zero.
    """
    import pandas_ta as ta

    atr = ta.atr(high, low, close, length=length)
    return (high - low) / (atr + 1e-10)


def calc_rsi_d1_delta(
    rsi_d1: pd.Series, diff_periods: int = 3
) -> pd.Series:
    """Variation du RSI D1 sur `diff_periods` jours — momentum macro.

    Args:
        rsi_d1: Serie RSI calculee sur D1.
        diff_periods: Nombre de jours pour la difference.

    Returns:
        Serie de la variation du RSI.
    """
    return rsi_d1.diff(diff_periods)


def calc_dist_sma200_d1(
    close_d1: pd.Series, length: int = 200
) -> pd.Series:
    """Distance a la SMA200 D1 — vraie tendance long-terme (~9 mois).

    Utilisee par TrendFilter pour n'autoriser que les trades dans le sens
    de la tendance macro.

    Args:
        close_d1: Serie Close D1.
        length: Periode SMA (200 par defaut).

    Returns:
        Serie de la distance normalisee (Close - SMA) / Close.
    """
    import pandas_ta as ta

    sma200 = ta.sma(close_d1, length=length)
    return (close_d1 - sma200) / close_d1
