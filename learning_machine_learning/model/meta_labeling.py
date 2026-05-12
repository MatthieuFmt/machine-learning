"""Méta-labeling binaire (López de Prado).

Entraîne un RandomForest binaire qui prédit si un trade du modèle primaire
sera rentable (Pips_Bruts > 0). Sert de filtre secondaire : seuls les trades
prédits "profitable" avec confiance > seuil sont exécutés.

Structure :
1. build_meta_labels() — extrait X_meta (features + prédiction primaire) et
   y_meta (1=profitable, 0=loss) à partir d'un backtest.
2. train_meta_model() — entraîne un RF binaire.
3. apply_meta_filter() — masque les barres où le méta-modèle prédit "loss".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def build_meta_labels(
    trades_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ml_data: pd.DataFrame,
    X_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Construit le dataset d'entraînement pour le méta-modèle.

    Pour chaque trade dans trades_df, extrait les features ml_data et la
    prédiction primaire au moment de l'entrée. Le label = 1 si le trade est
    gagnant (Pips_Bruts > 0), 0 sinon.

    Args:
        trades_df: DataFrame des trades (index=Time d'entrée, colonne Pips_Bruts).
        predictions_df: DataFrame de prédictions OOS (index=Time).
        ml_data: DataFrame ML-ready complet (index=Time).
        X_cols: Colonnes de features du modèle primaire.

    Returns:
        (X_meta, y_meta) — features + labels pour le méta-modèle.
    """
    if trades_df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Fusionner trades avec features au moment de l'entrée
    features_at_entry = ml_data[X_cols].reindex(trades_df.index)
    preds_at_entry = predictions_df[[
        "Prediction_Modele",
        "Confiance_Hausse_%",
        "Confiance_Neutre_%",
        "Confiance_Baisse_%",
    ]].reindex(trades_df.index)

    # Construire X_meta : features primaires + signal directionnel
    X_meta = features_at_entry.copy()
    X_meta["primary_pred"] = preds_at_entry["Prediction_Modele"].values
    # Confiance max (la plus haute des 3 classes)
    conf_cols = ["Confiance_Hausse_%", "Confiance_Neutre_%", "Confiance_Baisse_%"]
    valid_conf = preds_at_entry[conf_cols].fillna(0.0)
    X_meta["primary_conf_max"] = valid_conf.max(axis=1).values / 100.0

    # Label : 1 si Pips_Bruts > 0, 0 sinon
    y_meta = (trades_df["Pips_Bruts"] > 0).astype(int)

    # Supprimer les NaN éventuels (features manquantes à certaines dates)
    valid_mask = X_meta.notna().all(axis=1) & y_meta.notna()
    X_meta = X_meta.loc[valid_mask]
    y_meta = y_meta.loc[valid_mask]

    n_profitable = int(y_meta.sum())
    n_total = len(y_meta)
    logger.info(
        "Méta-labels construits : %d/%d trades profitables (%.1f%%)",
        n_profitable, n_total, 100 * n_profitable / max(n_total, 1),
    )

    return X_meta, y_meta


def train_meta_model(
    X_meta: pd.DataFrame,
    y_meta: pd.Series,
    params: dict[str, int | str] | None = None,
) -> RandomForestClassifier:
    """Entraîne un RandomForest binaire pour le méta-labeling.

    Args:
        X_meta: Features d'entraînement méta (features primaires + signal).
        y_meta: Labels binaires (1=profitable).
        params: Dict kwargs pour RandomForestClassifier.

    Returns:
        RandomForestClassifier binaire entraîné.
    """
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }

    if X_meta.empty:
        raise ValueError("X_meta est vide — impossible d'entraîner le méta-modèle.")

    model = RandomForestClassifier(**params)
    model.fit(X_meta, y_meta)

    n_class0 = int((y_meta == 0).sum())
    n_class1 = int(y_meta.sum())
    logger.info(
        "Méta-modèle entraîné : n_estimators=%d, classes=[0=%d, 1=%d], "
        "features=%d",
        params.get("n_estimators", "?"),
        n_class0, n_class1,
        X_meta.shape[1],
    )

    return model


def apply_meta_filter(
    model: RandomForestClassifier,
    df_predictions: pd.DataFrame,
    ml_data: pd.DataFrame,
    X_cols: list[str],
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Filtre les prédictions via le méta-modèle.

    Pour chaque barre avec un signal directionnel (Prediction_Modele != 0),
    le méta-modèle prédit si le trade serait profitable. Seules les barres
    avec proba_mêta > threshold sont conservées avec leur signal d'origine.
    Les barres NEUTRE (0) restent NEUTRE.

    Args:
        model: RandomForestClassifier binaire entraîné.
        df_predictions: DataFrame de prédictions OOS (index=Time).
        ml_data: DataFrame ML-ready complet.
        X_cols: Colonnes de features du modèle primaire.
        threshold: Seuil minimum de probabilité "profitable" (défaut 0.55).

    Returns:
        Copie de df_predictions avec Signal_Filtered (0, 1, -1) et colonnes
        meta_proba, meta_decision.
    """
    out = df_predictions.copy()

    # Construire X_meta pour toutes les barres
    X_meta_all = ml_data[X_cols].reindex(out.index)
    X_meta_all["primary_pred"] = out["Prediction_Modele"].values
    conf_cols = ["Confiance_Hausse_%", "Confiance_Neutre_%", "Confiance_Baisse_%"]
    X_meta_all["primary_conf_max"] = out[conf_cols].max(axis=1).values / 100.0

    # Méta-prédiction seulement sur les barres avec signal directionnel
    signal_mask = out["Prediction_Modele"] != 0
    out["meta_proba"] = np.nan
    out["meta_decision"] = 0

    if signal_mask.any():
        X_signal = X_meta_all.loc[signal_mask]
        valid_idx = X_signal.dropna().index

        if len(valid_idx) > 0:
            X_valid = X_signal.loc[valid_idx]
            proba_profitable = model.predict_proba(X_valid)
            # proba_profitable[:, 1] = P(classe=1) = P(trade profitable)
            out.loc[valid_idx, "meta_proba"] = proba_profitable[:, 1]

    # Décision : garder le signal si meta_proba > threshold
    meta_keep = out["meta_proba"] > threshold

    # Écraser Prediction_Modele pour les signaux rejetés (compatibilité simulate_trades)
    n_rejected = int((signal_mask & ~meta_keep).sum())
    out.loc[~meta_keep, "Prediction_Modele"] = 0

    n_original = int(signal_mask.sum())
    n_kept = int((out["Prediction_Modele"] != 0).sum())
    logger.info(
        "Méta-filtre (seuil=%.2f) : %d/%d signaux conservés (%d rejetés, %.1f%%)",
        threshold, n_kept, n_original, n_rejected,
        100 * n_kept / max(n_original, 1),
    )

    return out
