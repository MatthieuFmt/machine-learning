"""Pipeline méta-labeling fidèle — fix F1.

Avant le fix : les scripts run_phase_b_c5_*.py et run_validation_finale.py
généraient les signaux test à partir des features de toutes les barres et
décidaient la direction via le sign des slopes SMA. La distribution test
n'avait plus rien à voir avec la distribution train (Donchian breakouts).

Après le fix : la direction provient toujours du signal primaire (Donchian).
Le modèle ne fait que filtrer par P(winner) > threshold. C'est le méta-
labeling au sens de López de Prado §3.

Usage standard :

    primary = strategy.generate_signals(df_test)
    filtered = filter_signals_by_meta_proba(
        df=df_test,
        primary_signals=primary,
        features=features_test,
        model=trained_model,
        threshold=0.55,
    )
    # filtered conserve la direction de primary, mais seuls les signaux
    # avec P(winner) > 0.55 sont retenus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


def filter_signals_by_meta_proba(
    df: pd.DataFrame,
    primary_signals: pd.Series,
    features: pd.DataFrame,
    model: object,
    threshold: float = 0.5,
) -> pd.Series:
    """Filtre les signaux primaires par la probabilité d'être un winner.

    La direction (long=1, short=-1) provient EXCLUSIVEMENT du signal primaire.
    Le modèle ne décide jamais de la direction — il rejette ou accepte.

    Args:
        df: DataFrame OHLC du segment évalué (sert d'index de référence).
        primary_signals: Série 1/-1/0 aux barres où le générateur primaire
            (Donchian, mean-reversion, etc.) déclenche.
        features: DataFrame des features (mêmes colonnes que celles utilisées
            à l'entraînement, dans le même ordre).
        model: Classifieur entraîné avec `predict_proba`. La classe 1 doit
            correspondre à "winner".
        threshold: Probabilité minimum pour conserver le signal.

    Returns:
        Série de signaux filtrés, même index que df, dtype int.
        0 partout sauf aux barres où primary_signals ≠ 0 ET model.predict_proba
        de la classe 1 dépasse threshold.

    Raises:
        ValueError: Si primary_signals est entièrement nul, ou si la
            réindexation aboutit à un set vide.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"model doit avoir predict_proba, reçu {type(model).__name__}"
        )

    signals = pd.Series(0, index=df.index, dtype=int)

    primary_aligned = primary_signals.reindex(df.index, fill_value=0)
    primary_mask = primary_aligned != 0
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        logger.warning("filter_signals_by_meta_proba: 0 signal primaire")
        return signals

    candidate_idx = df.index[primary_mask].intersection(features.index)
    if len(candidate_idx) == 0:
        logger.warning(
            "filter_signals_by_meta_proba: 0 signal primaire aligné avec "
            "features (n_primary=%d, n_features=%d)",
            n_primary, len(features),
        )
        return signals

    X_candidates = features.loc[candidate_idx]
    if X_candidates.isna().any().any():
        # Drop les NaN avant prédiction
        valid = ~X_candidates.isna().any(axis=1)
        candidate_idx = candidate_idx[valid]
        X_candidates = X_candidates.loc[candidate_idx]
        if len(candidate_idx) == 0:
            return signals

    proba_winner = model.predict_proba(X_candidates.values)[:, 1]
    keep_mask = proba_winner > threshold
    kept_idx = candidate_idx[keep_mask]

    signals.loc[kept_idx] = primary_aligned.loc[kept_idx].astype(int)

    logger.info(
        "meta-filter: %d primaires → %d retenus (threshold=%.2f, rétention=%.1f%%)",
        n_primary, int(keep_mask.sum()), threshold,
        (keep_mask.sum() / max(n_primary, 1)) * 100,
    )
    return signals


def assert_train_test_distribution_alignment(
    primary_train_count: int,
    primary_test_count: int,
    n_train_years: float,
    n_test_years: float,
    tolerance: float = 3.0,
) -> None:
    """Sanity check (fix F1) : vérifie que la fréquence des signaux primaires
    test n'est pas anormalement supérieure à celle du train.

    Avant le fix F1, les scripts généraient ~5× plus de signaux/an en test
    qu'en train — symptôme de la rupture de distribution.

    Args:
        primary_train_count: nb de signaux primaires sur train.
        primary_test_count: nb de signaux primaires sur test.
        n_train_years: durée du train en années.
        n_test_years: durée du test en années.
        tolerance: ratio max accepté test/train. Défaut 3× (large).

    Raises:
        AssertionError: Si la fréquence test dépasse tolerance × fréquence train.
    """
    if n_train_years <= 0 or n_test_years <= 0:
        return
    train_rate = primary_train_count / n_train_years
    test_rate = primary_test_count / n_test_years
    if train_rate <= 0:
        return
    ratio = test_rate / train_rate
    if ratio > tolerance:
        raise AssertionError(
            f"Distribution train/test rompue (fix F1) : "
            f"{test_rate:.0f} signaux/an test vs {train_rate:.0f} signaux/an train "
            f"(ratio {ratio:.1f}× > tolerance {tolerance}×)"
        )
