"""Exceptions métier du pipeline ML/trading.

Hiérarchie :
    PipelineError
    ├── DataValidationError
    ├── LookAheadError
    ├── ConfigError
    ├── ModelError
    └── BacktestError
"""

from __future__ import annotations


class PipelineError(Exception):
    """Exception de base pour toutes les erreurs du pipeline."""

    pass


class DataValidationError(PipelineError):
    """Données d'entrée invalides (colonnes manquantes, types incorrects, NaN)."""

    pass


class LookAheadError(PipelineError):
    """Détection de look-ahead bias (feature utilisant une information future)."""

    pass


class ConfigError(PipelineError):
    """Configuration incohérente ou invalide."""

    pass


class ModelError(PipelineError):
    """Erreur liée à l'entraînement ou la prédiction du modèle."""

    pass


class BacktestError(PipelineError):
    """Erreur dans la simulation de backtest."""

    pass
