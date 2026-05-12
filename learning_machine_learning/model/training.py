"""Entraînement du modèle RandomForest avec split temporel + purge anti-overlap.

Conforme au split 3-étages strict : entraînement sur [début, train_end_year],
validation sur val_year, test sur test_year. La purge de `purge_hours` entre
train et OOS évite le chevauchement (López de Prado).
"""

from __future__ import annotations

from datetime import timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


# Colonnes preservees dans ml_data pour les filtres backtest mais exclues
# de l'entraînement (features_dropped avec exemption FILTER_KEEP).
_FILTER_ONLY_COLS: frozenset[str] = frozenset({"ATR_Norm", "Volatilite_Realisee_24h"})


def train_test_split_purge(
    df: pd.DataFrame,
    train_end_year: int,
    purge_hours: int = 48,
    extra_drop_cols: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split temporel avec embargo anti-overlap.

    Args:
        df: DataFrame ML-ready indexé par Time (datetime).
        train_end_year: Dernière année d'entraînement (ex: 2023).
        purge_hours: Heures d'embargo entre train et OOS.
        extra_drop_cols: Colonnes supplementaires a exclure des features
            (ex: colonnes preservees pour filtres backtest seulement).

    Returns:
        (X_train, y_train, X_cols).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index du DataFrame doit être un DatetimeIndex.")
    train_cutoff = pd.to_datetime(f"{train_end_year + 1}-01-01") - timedelta(hours=purge_hours)
    train_mask = df.index < train_cutoff

    if not train_mask.any():
        raise ValueError(f"Aucune donnée d'entraînement avant {train_cutoff}.")

    target_col = "Target"
    drop_cols = {target_col, "Spread"}
    if extra_drop_cols:
        drop_cols |= extra_drop_cols

    X_cols = [c for c in df.columns if c not in drop_cols]

    train_data = df.loc[train_mask]
    X_train = train_data[X_cols]
    y_train = train_data[target_col]

    logger.info(
        "Split train/purge : cutoff=%s, n_train=%d, purge=%dh, X_cols=%d (drop=%d)",
        train_cutoff, len(X_train), purge_hours, len(X_cols), len(drop_cols),
    )

    return X_train, y_train, X_cols


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> RandomForestClassifier:
    """Entraîne un RandomForest avec les paramètres fournis.

    Args:
        X_train: Features d'entraînement.
        y_train: Target d'entraînement.
        params: Dict kwargs pour RandomForestClassifier.

    Returns:
        Modèle entraîné.
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    n_classes = len(model.classes_)
    logger.info(
        "Modèle entraîné : n_estimators=%d, classes=%s, échantillons=%d",
        params.get("n_estimators", "?"), list(model.classes_), len(X_train),
    )

    return model
