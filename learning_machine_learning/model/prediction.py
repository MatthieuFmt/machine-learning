"""Génération de prédictions OOS par année."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def predict_oos(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    eval_year: int,
    X_cols: list[str],
    class_map: Optional[dict[float, int]] = None,
) -> tuple[pd.DataFrame, dict[float, int]]:
    """Prédit sur une année OOS et retourne le DataFrame enrichi.

    Args:
        model: Modèle entraîné.
        df: DataFrame ML-ready complet (index datetime).
        eval_year: Année à prédire (ex: 2024).
        X_cols: Colonnes de features à utiliser.
        class_map: Mapping classe → index dans probas. Si None, construit depuis model.

    Returns:
        (DataFrame avec Close_Reel_Direction, Prediction_Modele, Confiance_*_%, Spread,
         dict class_map pour réutilisation).
    """
    eval_start = pd.to_datetime(f"{eval_year}-01-01")
    eval_end = pd.to_datetime(f"{eval_year + 1}-01-01")
    test_data = df[(df.index >= eval_start) & (df.index < eval_end)].copy()

    if test_data.empty:
        raise ValueError(f"Aucune donnée pour l'année {eval_year}.")

    X_test = test_data[X_cols]
    y_test = test_data["Target"] if "Target" in test_data.columns else None

    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)

    if class_map is None:
        class_map = {float(cls): int(idx) for idx, cls in enumerate(model.classes_)}

    def _get_col(class_key: float) -> np.ndarray:
        """Extraction robuste : zéros si la classe est absente du modèle (B3 fix)."""
        if class_key in class_map:
            return probas[:, class_map[class_key]]
        logger.warning("Classe %.1f absente du modèle (classes=%s), colonne à zéro.", class_key, list(class_map.keys()))
        return np.zeros(len(probas), dtype=np.float64)

    proba_baisse = _get_col(-1.0)
    proba_neutre = _get_col(0.0)
    proba_hausse = _get_col(1.0)

    out = pd.DataFrame(
        {
            "Close_Reel_Direction": y_test if y_test is not None else np.nan,
            "Prediction_Modele": predictions,
            "Confiance_Baisse_%": np.round(proba_baisse * 100, 2),
            "Confiance_Neutre_%": np.round(proba_neutre * 100, 2),
            "Confiance_Hausse_%": np.round(proba_hausse * 100, 2),
        },
        index=y_test.index if y_test is not None else test_data.index,
    )

    if "Spread" in test_data.columns:
        out["Spread"] = test_data["Spread"]

    logger.info(
        "Prédictions OOS %d : %d bougies, classes=%s",
        eval_year, len(out), list(class_map.keys()),
    )

    return out, class_map
