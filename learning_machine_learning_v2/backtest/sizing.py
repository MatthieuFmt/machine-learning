"""Fonctions de sizing (pondération des trades par confiance).

Implémentations du protocole WeightFunction : (proba: np.ndarray) -> np.ndarray.
Chaque fonction retourne un poids ∈ [borne_inf, borne_sup].
"""

from __future__ import annotations

import numpy as np


def weight_centered(proba: np.ndarray, seuil: float = 0.35) -> np.ndarray:
    """Pondération centrée sur un seuil ajustable.

    weight = clip(0.8 + 0.4 * (proba - seuil) / 0.10, 0.8, 1.2)
    """
    return np.clip(0.8 + 0.4 * ((proba - seuil) / 0.10), 0.8, 1.2)


def weight_linear(proba: np.ndarray, seuil: float = 0.45) -> np.ndarray:
    """Pondération linéaire entre 0.5 et 1.5."""
    excess = (proba - seuil) / 0.15
    excess = np.clip(excess, 0, 1)
    return 0.5 + excess
