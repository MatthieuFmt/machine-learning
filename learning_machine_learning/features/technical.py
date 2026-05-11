"""Indicateurs techniques — fonctions pures, vectorisées.

Chaque fonction prend un DataFrame ou Series et retourne un DataFrame/Series
sans effet de bord. Zéro dépendance à l'instrument (pas de config hardcodée).

Utilise `pandas_ta` pour les calculs standards (RSI, EMA, ADX, ATR, BBands).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def calc_base_features(
    df: pd.DataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    """Calcule RSI, Dist_EMA_20, Dist_EMA_50 pour un DataFrame OHLCV.

    Utilisé pour les features multi-timeframe (H4, D1).

    Args:
        df: DataFrame avec colonne 'Close'.
        prefix: Suffixe ajouté aux noms de colonnes (ex: '_H4').

    Returns:
        DataFrame avec colonnes RSI_14{prefix}, Dist_EMA_20{prefix}, Dist_EMA_50{prefix}.
    """
    close = df["Close"]
    result = pd.DataFrame(index=df.index)
    result[f"RSI_14{prefix}"] = ta.rsi(close, length=14)
    result[f"Dist_EMA_20{prefix}"] = (close - ta.ema(close, length=20)) / close
    result[f"Dist_EMA_50{prefix}"] = (close - ta.ema(close, length=50)) / close
    return result


def calc_log_return(df: pd.DataFrame) -> pd.Series:
    """Log-return H1 : ln(Close_t / Close_{t-1}).

    Args:
        df: DataFrame avec colonne 'Close'.

    Returns:
        Series nommée 'Log_Return'.
    """
    return np.log(df["Close"] / df["Close"].shift(1)).rename("Log_Return")


def calc_ema_distance(
    df: pd.DataFrame,
    periods: tuple[int, ...] = (9, 21, 50),
) -> pd.DataFrame:
    """Distance relative du Close à l'EMA pour plusieurs périodes.

    Formule : (Close - EMA(period)) / Close

    Args:
        df: DataFrame avec colonne 'Close'.
        periods: Périodes d'EMA.

    Returns:
        DataFrame avec colonnes Dist_EMA_{period}.
    """
    close = df["Close"]
    result = pd.DataFrame(index=df.index)
    for p in periods:
        result[f"Dist_EMA_{p}"] = (close - ta.ema(close, length=p)) / close
    return result


def calc_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """RSI de Wilder.

    Args:
        df: DataFrame avec colonne 'Close'.
        length: Période (défaut 14).

    Returns:
        Series nommée 'RSI_14'.
    """
    return ta.rsi(df["Close"], length=length).rename(f"RSI_{length}")


def calc_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ADX (Average Directional Index).

    Args:
        df: DataFrame avec colonnes 'High', 'Low', 'Close'.
        length: Période (défaut 14).

    Returns:
        Series nommée 'ADX_14'.
    """
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=length)
    return adx_df[f"ADX_{length}"].rename(f"ADX_{length}")


def calc_atr_norm(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR normalisé par le Close.

    Formule : ATR(length) / Close

    Args:
        df: DataFrame avec colonnes 'High', 'Low', 'Close'.
        length: Période ATR (défaut 14).

    Returns:
        Series nommée 'ATR_Norm'.
    """
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=length)
    return (atr / df["Close"]).rename("ATR_Norm")


def calc_bb_width(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> pd.Series:
    """Largeur des bandes de Bollinger normalisée.

    Formule : (BB_Upper - BB_Lower) / BB_Mid

    Args:
        df: DataFrame avec colonne 'Close'.
        length: Période de la moyenne mobile.
        std: Nombre d'écarts-types.

    Returns:
        Series nommée 'BB_Width'.
    """
    bbands = ta.bbands(df["Close"], length=length, std=std)
    col_upper = [c for c in bbands.columns if "BBU" in c][0]
    col_lower = [c for c in bbands.columns if "BBL" in c][0]
    col_mid = [c for c in bbands.columns if "BBM" in c][0]
    return ((bbands[col_upper] - bbands[col_lower]) / bbands[col_mid]).rename("BB_Width")


def calc_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Encodage cyclique de l'heure (sin, cos).

    Évite la discontinuité 23h → 00h.

    Args:
        df: DataFrame avec DatetimeIndex.

    Returns:
        DataFrame avec colonnes 'Hour_Sin', 'Hour_Cos'.
    """
    hours = df.index.hour
    radians = hours * (2.0 * np.pi / 24)
    return pd.DataFrame(
        {
            "Hour_Sin": np.sin(radians),
            "Hour_Cos": np.cos(radians),
        },
        index=df.index,
    )
