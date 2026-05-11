"""Évaluation du modèle — métriques, importance des features, rapport."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Calcule accuracy et classification_report sur les données fournies.

    Args:
        model: Modèle entraîné.
        X: Features.
        y: Target vraie.

    Returns:
        Dict avec 'accuracy' et 'report' (str).
    """
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds)
    return {"accuracy": float(acc), "report": report}


def feature_importance_impurity(
    model: RandomForestClassifier,
    X_cols: list[str],
) -> pd.DataFrame:
    """Importances basées sur la réduction d'impureté (Gini).

    Args:
        model: Modèle entraîné.
        X_cols: Noms des colonnes de features.

    Returns:
        DataFrame trié avec colonnes Indicateur, Impurity_%.
    """
    fi = pd.DataFrame({
        "Indicateur": X_cols,
        "Impurity_%": np.round(model.feature_importances_ * 100, 2),
    }).sort_values(by="Impurity_%", ascending=False).reset_index(drop=True)

    return fi


def feature_importance_permutation(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Optional[pd.DataFrame]:
    """Permutation importance sur les données fournies.

    Args:
        model: Modèle entraîné.
        X: Features.
        y: Target vraie.
        n_repeats: Nombre de répétitions.
        random_state: Seed.
        n_jobs: Parallélisme.

    Returns:
        DataFrame trié ou None si X vide.
    """
    if X.empty:
        logger.warning("Permutation importance ignorée : X vide.")
        return None

    perm = permutation_importance(
        model, X, y,
        n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs,
    )

    fi = pd.DataFrame({
        "Indicateur": list(X.columns),
        "Permutation_mean": np.round(perm.importances_mean, 5),
        "Permutation_std": np.round(perm.importances_std, 5),
    }).sort_values(by="Permutation_mean", ascending=False).reset_index(drop=True)

    return fi
