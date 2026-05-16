"""Méta-labeling RF — filtre binaire en surcouche d'un signal déterministe.

Utilisé par B1 (H_new1) pour filtrer les trades Donchian US30 D1.
Le RF estime P(trade gagnant | features à l'entrée) et rejette les signaux
sous un seuil calibré sur train uniquement.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetaLabelingConfig:
    """Configuration du méta-modèle RF.

    Attributes:
        n_estimators: Nombre d'arbres (200 par défaut, v2 H05).
        max_depth: Profondeur max (4 = régularisation forte).
        min_samples_leaf: Échantillons minimum par feuille.
        class_weight: "balanced" pour compenser le déséquilibre winner/loser.
        random_state: Graine fixe pour reproductibilité.
        threshold_candidates: Seuils testés lors de la calibration.
        min_trade_retention: Fraction minimale de trades train conservés.
    """

    n_estimators: int = 200
    max_depth: int = 4
    min_samples_leaf: int = 10
    class_weight: str = "balanced"
    random_state: int = 42
    threshold_candidates: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
    min_trade_retention: float = 0.20


class MetaLabelingRF:
    """RandomForest pour méta-labeling binaire (winner/loser).

    Entraîné sur les barres d'entrée des trades Donchian pour prédire
    si le trade sera gagnant (Pips_Nets > 0). Le seuil de probabilité
    est calibré sur train uniquement pour maximiser le Sharpe.

    Fallback : si aucun seuil ne conserve ≥ min_trade_retention des trades
    train, le méta-modèle est désactivé (disabled=True) et tous les trades
    sont acceptés (baseline pure).
    """

    def __init__(self, config: MetaLabelingConfig | None = None) -> None:
        self.config = config or MetaLabelingConfig()
        self.model: RandomForestClassifier | None = None
        self.threshold: float = 0.50
        self.disabled: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:  # noqa: N803
        """Entraîne le RF sur les échantillons d'entrée.

        Args:
            X_train: Features aux barres d'entrée des trades train.
            y_train: Labels binaires (1 = winner, 0 = loser).
        """
        if X_train.empty or y_train.empty:
            logger.warning("MetaLabelingRF.fit: X_train ou y_train vide. Désactivation.")
            self.disabled = True
            return

        # Vérifier qu'il y a au moins 2 classes
        unique = y_train.unique()
        if len(unique) < 2:
            logger.warning(
                "MetaLabelingRF.fit: une seule classe dans y_train (%s). Désactivation.",
                unique[0],
            )
            self.disabled = True
            return

        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train.values, y_train.values)

    def calibrate_threshold(  # noqa: N803
        self,
        X_train: pd.DataFrame,  # noqa: N803
        backtest_fn: Callable[[pd.Series], float],
    ) -> None:
        """Calibre le seuil de probabilité sur train uniquement.

        Pour chaque seuil candidat, filtre les trades train et évalue
        le Sharpe via `backtest_fn`. Sélectionne le seuil maximisant
        le Sharpe, avec contrainte de rétention ≥ min_trade_retention.

        Si aucun seuil ne satisfait la contrainte → fallback :
        disabled=True, threshold=0.0 (accepte tous les trades).

        Args:
            X_train: Features aux barres d'entrée des trades train.
            backtest_fn: Callable(mask: pd.Series) -> float (Sharpe).
        """
        if self.disabled or self.model is None:
            self.disabled = True
            self.threshold = 0.0
            return

        proba = self.model.predict_proba(X_train.values)[:, 1]
        proba_series = pd.Series(proba, index=X_train.index)

        best_t: float | None = None
        best_sharpe: float = -np.inf

        for t in self.config.threshold_candidates:
            mask = proba_series > t
            retention = mask.sum() / len(X_train) if len(X_train) > 0 else 0.0
            if retention < self.config.min_trade_retention:
                logger.debug(
                    "Seuil %.2f: rétention %.1f%% < %.0f%%, ignoré.",
                    t, retention * 100, self.config.min_trade_retention * 100,
                )
                continue
            sharpe = backtest_fn(mask)
            logger.debug("Seuil %.2f: Sharpe=%.4f, rétention=%.1f%%.", t, sharpe, retention * 100)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_t = t

        if best_t is None:
            logger.warning(
                "Aucun seuil ne retient ≥ %.0f%% des trades train. "
                "Désactivation du méta-labeling (fallback baseline).",
                self.config.min_trade_retention * 100,
            )
            self.disabled = True
            self.threshold = 0.0
        else:
            # Plancher à 0.50 comme spécifié par le prompt B1
            self.threshold = max(best_t, 0.50)
            logger.info(
                "Seuil calibré: %.2f (best=%.2f, Sharpe=%.4f).",
                self.threshold, best_t, best_sharpe,
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Prédit le masque de filtrage : True = trade accepté.

        Si disabled → tous les trades acceptés (baseline pure).

        Args:
            X: Features aux barres d'entrée des trades à filtrer.

        Returns:
            np.ndarray[bool] de même longueur que X.
        """
        if self.disabled or self.model is None:
            return np.ones(len(X), dtype=bool)
        proba = self.model.predict_proba(X.values)[:, 1]
        return proba > self.threshold
