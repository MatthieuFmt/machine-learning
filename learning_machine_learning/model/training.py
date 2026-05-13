"""Entraînement du modèle RandomForest avec split temporel + purge anti-overlap.

Conforme au split 3-étages strict : entraînement sur [début, train_end_year],
validation sur val_year, test sur test_year. La purge de `purge_hours` entre
train et OOS évite le chevauchement (López de Prado).

Support également le walk-forward retraining (v14) : fenêtre glissante de
train_months, step de step_months, purge entre chaque fold.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Callable

import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


# Colonnes preservees dans ml_data pour les filtres backtest mais exclues
# de l'entraînement (features_dropped avec exemption FILTER_KEEP).
_FILTER_ONLY_COLS: frozenset[str] = frozenset({"ATR_Norm", "Volatilite_Realisee_24h"})


# Type alias pour une factory de modèle (compatible avec train_model actuel
# et futures variantes).
ModelFactory = Callable[[pd.DataFrame, pd.Series], RandomForestClassifier]


def train_test_split_purge(
    df: pd.DataFrame,
    train_end_year: int | None = None,
    purge_hours: int = 48,
    extra_drop_cols: frozenset[str] | None = None,
    *,
    train_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split temporel avec embargo anti-overlap.

    Deux modes d'appel :
    1. Par année : `train_end_year=2023` → cutoff = 2024-01-01 - purge_hours.
    2. Par masques : `train_mask=...` + `test_mask=...` → split arbitraire
       (compatible CPCV). `test_mask` n'est utilisé que pour le logging.

    Args:
        df: DataFrame ML-ready indexé par Time (datetime).
        train_end_year: Dernière année d'entraînement (ex: 2023).
            Ignoré si `train_mask` est fourni.
        purge_hours: Heures d'embargo entre train et OOS.
        extra_drop_cols: Colonnes supplementaires a exclure des features
            (ex: colonnes preservees pour filtres backtest seulement).
        train_mask: Masque booléen d'entraînement (optionnel, prioritaire).
        test_mask: Masque booléen de test (optionnel, logging seulement).

    Returns:
        (X_train, y_train, X_cols).

    Raises:
        ValueError: Si ni train_end_year ni train_mask n'est fourni,
            ou si l'index n'est pas DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index du DataFrame doit être un DatetimeIndex.")

    # ── Résolution du masque train ──────────────────────────────────────
    if train_mask is not None:
        if not train_mask.any():
            raise ValueError("train_mask fourni mais aucune barre sélectionnée.")
        train_cutoff_str = "custom_mask"
        n_test_info = int(test_mask.sum()) if test_mask is not None else -1
    elif train_end_year is not None:
        train_cutoff = pd.to_datetime(f"{train_end_year + 1}-01-01") - timedelta(hours=purge_hours)
        train_mask = df.index < train_cutoff
        if not train_mask.any():
            raise ValueError(f"Aucune donnée d'entraînement avant {train_cutoff}.")
        train_cutoff_str = str(train_cutoff)
        n_test_info = -1
    else:
        raise ValueError(
            "train_end_year ou train_mask doit être fourni. "
            "Utiliser train_end_year pour le split par année, "
            "train_mask pour un split arbitraire (CPCV)."
        )

    if not train_mask.any():
        raise ValueError("Aucune donnée dans le masque d'entraînement.")

    target_col = "Target"
    drop_cols = {target_col, "Spread"}
    if extra_drop_cols:
        drop_cols |= extra_drop_cols

    X_cols = [c for c in df.columns if c not in drop_cols]

    train_data = df.loc[train_mask]
    X_train = train_data[X_cols]
    y_train = train_data[target_col]

    logger.info(
        "Split train/purge : cutoff=%s, n_train=%d, n_test=%d, purge=%dh, X_cols=%d (drop=%d)",
        train_cutoff_str, len(X_train), n_test_info, purge_hours, len(X_cols), len(drop_cols),
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


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> HistGradientBoostingRegressor:
    """Entraîne un HistGradientBoostingRegressor avec les paramètres fournis.

    Args:
        X_train: Features d'entraînement.
        y_train: Target continue (log-return forward).
        params: Dict kwargs pour HistGradientBoostingRegressor.

    Returns:
        Régresseur entraîné.
    """
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    logger.info(
        "Régresseur entraîné : max_iter=%d, max_depth=%s, loss=%s, échantillons=%d",
        params.get("max_iter", "?"),
        params.get("max_depth", "?"),
        params.get("loss", "?"),
        len(X_train),
    )
    return model


def walk_forward_train(
    df: pd.DataFrame,
    X_cols: list[str],
    model_factory: ModelFactory,
    train_months: int = 36,
    step_months: int = 3,
    purge_hours: int = 48,
    extra_drop_cols: frozenset[str] | None = None,
) -> Iterator[tuple[RandomForestClassifier, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Générateur walk-forward : réentraîne le modèle par fenêtre glissante.

    À chaque fold :
    1. Sélectionne les données d'entraînement sur [fold_start, fold_start + train_months).
    2. Applique un embargo de purge_hours avant la période de test.
    3. Test sur [train_end + purge, train_end + purge + step_months).
    4. Yield (modèle, train_start, train_end, test_start, test_end).

    La progression est strictement chronologique — aucun shuffle, aucun leak.

    Args:
        df: DataFrame ML-ready indexé par Time (datetime), contient 'Target'.
        X_cols: Colonnes de features utilisées pour l'entraînement.
        model_factory: Callable (X_train, y_train) -> modèle entraîné.
        train_months: Durée de la fenêtre d'entraînement en mois (défaut: 36).
        step_months: Pas d'avancement entre les folds en mois (défaut: 3).
        purge_hours: Heures d'embargo entre train et test (défaut: 48).
        extra_drop_cols: Colonnes supplémentaires à exclure des features.

    Yields:
        Tuple (model, train_start, train_end, test_start, test_end) à chaque fold.
        Les timestamps sont les bornes exactes de chaque période (inclusif début, exclusif fin).

    Raises:
        ValueError: Si l'index n'est pas un DatetimeIndex ou si les données sont insuffisantes.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index du DataFrame doit être un DatetimeIndex.")

    target_col = "Target"
    drop_cols = {target_col, "Spread"}
    if extra_drop_cols:
        drop_cols |= extra_drop_cols

    # Colonnes effectives (intersection avec df)
    effective_X_cols = [c for c in X_cols if c in df.columns and c not in drop_cols]

    data_start = df.index.min()
    data_end = df.index.max()
    fold_start = data_start

    n_folds = 0

    while True:
        train_end = fold_start + DateOffset(months=train_months)
        test_start = train_end + timedelta(hours=purge_hours)
        test_end = test_start + DateOffset(months=step_months)

        # Arrêt si la période de test dépasse les données disponibles
        if test_start >= data_end:
            break

        # Sélection train
        train_mask = (df.index >= fold_start) & (df.index < train_end)
        train_data = df.loc[train_mask]

        # Sélection test
        test_mask = (df.index >= test_start) & (df.index < test_end)
        test_data_available = df.loc[test_mask]

        # Vérifier que les deux périodes ont suffisamment de données
        if len(train_data) < 100 or len(test_data_available) < 10:
            logger.warning(
                "Fold %d ignoré : train=%d lignes, test=%d lignes (seuils min: 100/10)",
                n_folds + 1, len(train_data), len(test_data_available),
            )
            fold_start = test_start
            continue

        X_train = train_data[effective_X_cols]
        y_train = train_data[target_col]

        model = model_factory(X_train, y_train)
        n_folds += 1

        logger.info(
            "Walk-forward fold %d : train=[%s, %s) n=%d | test=[%s, %s) n=%d | purge=%dh",
            n_folds,
            fold_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
            len(train_data),
            test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"),
            len(test_data_available),
            purge_hours,
        )

        yield model, fold_start, train_end, test_start, test_end

        # Avancer la fenêtre
        fold_start = test_start

    if n_folds == 0:
        raise ValueError(
            f"Aucun fold walk-forward généré. Données: [{data_start}, {data_end}], "
            f"train_months={train_months}, step_months={step_months}"
        )

    logger.info("Walk-forward terminé : %d folds générés.", n_folds)
