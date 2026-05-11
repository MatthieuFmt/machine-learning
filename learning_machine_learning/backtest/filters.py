"""Filtres de régime — implémentations du protocole SignalFilter.

Chaque filtre est une classe indépendante testable isolément.
FilterPipeline applique une séquence ordonnée de filtres.
"""

from __future__ import annotations

import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


class TrendFilter:
    """Filtre directionnel basé sur Dist_SMA200_D1.

    LONG autorisé uniquement si Close > SMA200 (Dist > 0).
    SHORT autorisé uniquement si Close < SMA200 (Dist < 0).
    """

    name = "trend"

    def apply(
        self,
        df: pd.DataFrame,
        mask_long: pd.Series,
        mask_short: pd.Series,
    ) -> tuple[pd.Series, pd.Series, int]:
        if "Dist_SMA200_D1" not in df.columns:
            raise ValueError(
                "TrendFilter nécessite la colonne 'Dist_SMA200_D1'. "
                "Relancer le feature engineering."
            )

        trend_mask_long = df["Dist_SMA200_D1"] > 0
        trend_mask_short = df["Dist_SMA200_D1"] < 0

        rejected_long = mask_long & ~trend_mask_long
        rejected_short = mask_short & ~trend_mask_short
        n_rejected = int((rejected_long | rejected_short).sum())

        mask_long = mask_long & trend_mask_long
        mask_short = mask_short & trend_mask_short

        logger.debug("TrendFilter: %d signaux rejetés", n_rejected)
        return mask_long, mask_short, n_rejected


class VolFilter:
    """Filtre de volatilité basé sur ATR_Norm vs médiane glissante.

    Ignore les signaux si ATR_Norm > multiplier × médiane glissante.
    Protège contre les entrées en période de turbulence.
    """

    name = "vol"

    def __init__(self, window: int = 168, multiplier: float = 2.0) -> None:
        self.window = window
        self.multiplier = multiplier

    def apply(
        self,
        df: pd.DataFrame,
        mask_long: pd.Series,
        mask_short: pd.Series,
    ) -> tuple[pd.Series, pd.Series, int]:
        if "ATR_Norm" not in df.columns:
            raise ValueError(
                "VolFilter nécessite la colonne 'ATR_Norm'. "
                "Relancer le feature engineering."
            )

        atr_median = (
            df["ATR_Norm"]
            .rolling(window=self.window, min_periods=1)
            .median()
        )
        vol_threshold = atr_median * self.multiplier
        high_vol = df["ATR_Norm"] > vol_threshold

        rejected_long = mask_long & high_vol
        rejected_short = mask_short & high_vol
        n_rejected = int((rejected_long | rejected_short).sum())

        mask_long = mask_long & ~high_vol
        mask_short = mask_short & ~high_vol

        logger.debug("VolFilter: %d signaux rejetés", n_rejected)
        return mask_long, mask_short, n_rejected


class SessionFilter:
    """Filtre de session basse liquidité.

    Ignore les signaux pendant les heures de faible liquidité
    (par défaut 22h-01h GMT).
    """

    name = "session"

    def __init__(self, exclude_start: int = 22, exclude_end: int = 1) -> None:
        self.exclude_start = exclude_start
        self.exclude_end = exclude_end

    def apply(
        self,
        df: pd.DataFrame,
        mask_long: pd.Series,
        mask_short: pd.Series,
    ) -> tuple[pd.Series, pd.Series, int]:
        hours_gmt = df.index.hour

        if self.exclude_start > self.exclude_end:
            # Plage qui traverse minuit (ex: 22h → 1h)
            session_mask = (hours_gmt >= self.exclude_start) | (
                hours_gmt < self.exclude_end
            )
        else:
            session_mask = (hours_gmt >= self.exclude_start) & (
                hours_gmt < self.exclude_end
            )

        rejected_long = mask_long & session_mask
        rejected_short = mask_short & session_mask
        n_rejected = int((rejected_long | rejected_short).sum())

        mask_long = mask_long & ~session_mask
        mask_short = mask_short & ~session_mask

        logger.debug("SessionFilter: %d signaux rejetés", n_rejected)
        return mask_long, mask_short, n_rejected


class FilterPipeline:
    """Composite : applique une séquence ordonnée de filtres.

    Chaque filtre réduit les masques LONG/SHORT. L'ordre d'application
    est préservé — le premier filtre de la liste est appliqué en premier.
    """

    def __init__(self, filters: list) -> None:
        self.filters = filters

    def apply(
        self,
        df: pd.DataFrame,
        mask_long: pd.Series,
        mask_short: pd.Series,
    ) -> tuple[pd.Series, pd.Series, dict[str, int]]:
        """Applique tous les filtres séquentiellement.

        Args:
            df: DataFrame avec les colonnes nécessaires aux filtres.
            mask_long: Série booléenne des signaux LONG candidats.
            mask_short: Série booléenne des signaux SHORT candidats.

        Returns:
            Tuple (mask_long filtré, mask_short filtré, dict {nom_filtre: n_rejetés}).
        """
        all_rejected: dict[str, int] = {}

        for f in self.filters:
            mask_long, mask_short, n = f.apply(df, mask_long, mask_short)
            all_rejected[f.name] = n

        return mask_long, mask_short, all_rejected
