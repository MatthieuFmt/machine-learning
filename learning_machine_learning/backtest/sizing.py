"""Fonctions de sizing (pondération des trades par confiance).

Implémentations du protocole WeightFunction : (proba: np.ndarray) -> np.ndarray.
Chaque fonction retourne un poids ∈ [borne_inf, borne_sup].
"""

from __future__ import annotations

import numpy as np


def weight_linear(proba: np.ndarray, seuil: float = 0.45) -> np.ndarray:
    """Pondération linéaire entre 0.5 et 1.5.

    weight = 0.5 + clip((proba - seuil) / 0.15, 0, 1)
    """
    excess = (proba - seuil) / 0.15
    excess = np.clip(excess, 0, 1)
    return 0.5 + excess


def weight_linear_v2(proba: np.ndarray, seuil: float = 0.45) -> np.ndarray:
    """Pondération linéaire douce entre 0.8 et 1.2.

    Pente plus faible que weight_linear : weight = 0.8 + 0.8 * clip((proba - seuil) / 0.10, 0, 1)
    """
    return 0.8 + 0.8 * np.clip((proba - seuil) / 0.10, 0, 1)


def weight_exp(proba: np.ndarray, seuil: float = 0.45) -> np.ndarray:
    """Pondération quadratique entre 0.5 et 1.5.

    Accélération pour hautes confiances : weight = 0.5 + clip((proba - seuil) / 0.15, 0, 1)²
    """
    z = (proba - seuil) / 0.15
    z = np.clip(z, 0, 1)
    return 0.5 + z**2


def weight_step(proba: np.ndarray, seuil: float = 0.45) -> np.ndarray:
    """Pondération par paliers : 0.5, 1.0, 1.5 selon la confiance.

    - proba < 0.50 → 0.5
    - 0.50 ≤ proba < 0.55 → 1.0
    - proba ≥ 0.55 → 1.5
    """
    weights = np.zeros_like(proba, dtype=np.float64)
    weights[proba < 0.50] = 0.5
    weights[(proba >= 0.50) & (proba < 0.55)] = 1.0
    weights[proba >= 0.55] = 1.5
    return weights


def weight_centered(proba: np.ndarray, seuil: float = 0.35) -> np.ndarray:
    """Pondération centrée sur un seuil ajustable (utilisé par le backtest principal).

    weight = clip(0.8 + 0.4 * (proba - seuil) / 0.10, 0.8, 1.2)
    """
    return np.clip(0.8 + 0.4 * ((proba - seuil) / 0.10), 0.8, 1.2)
