"""Configuration du modèle ML (RandomForest + GBM).

Supporte à la fois RandomForestClassifier (mode classifieur) et
HistGradientBoostingRegressor (mode régression, Step 01).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Paramètres d'entraînement du modèle.

    Tous les champs sont validés dans __post_init__.
    """

    random_seed: int = 42

    # ── RandomForest Classifier ─────────────────────────────────────────
    rf_n_estimators: int = 500
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 50
    rf_class_weight: str = "balanced"
    rf_n_jobs: int = -1

    # ── Split temporel ──────────────────────────────────────────────────
    purge_hours: int = 48
    train_end_year: int = 2023
    val_year: int = 2024
    test_year: int = 2025

    # ── Step 01 — GBM Regressor ─────────────────────────────────────────
    gbm_max_iter: int = 200
    gbm_max_depth: int = 5
    gbm_min_samples_leaf: int = 50
    gbm_learning_rate: float = 0.05
    gbm_loss: str = "absolute_error"

    def __post_init__(self) -> None:
        # Validation RF
        if self.rf_n_estimators <= 0:
            raise ValueError(f"rf_n_estimators doit être > 0, reçu {self.rf_n_estimators}")
        if self.rf_max_depth <= 0:
            raise ValueError(f"rf_max_depth doit être > 0, reçu {self.rf_max_depth}")
        if self.rf_min_samples_leaf <= 0:
            raise ValueError(f"rf_min_samples_leaf doit être > 0, reçu {self.rf_min_samples_leaf}")
        if self.purge_hours <= 0:
            raise ValueError(f"purge_hours doit être > 0, reçu {self.purge_hours}")
        if not (self.train_end_year < self.val_year <= self.test_year):
            raise ValueError(
                f"Ordre des splits invalide : train_end={self.train_end_year}, "
                f"val={self.val_year}, test={self.test_year}"
            )
        # Validation GBM
        if self.gbm_max_iter <= 0:
            raise ValueError(f"gbm_max_iter doit être > 0, reçu {self.gbm_max_iter}")
        if self.gbm_max_depth <= 0:
            raise ValueError(f"gbm_max_depth doit être > 0, reçu {self.gbm_max_depth}")
        if self.gbm_min_samples_leaf <= 0:
            raise ValueError(f"gbm_min_samples_leaf doit être > 0, reçu {self.gbm_min_samples_leaf}")
        if not (0 < self.gbm_learning_rate <= 1):
            raise ValueError(
                f"gbm_learning_rate doit être dans (0, 1], reçu {self.gbm_learning_rate}"
            )
        if self.gbm_loss not in ("squared_error", "absolute_error", "poisson", "quantile", "gamma"):
            raise ValueError(
                f"gbm_loss doit être l'un des loss sklearn valides, "
                f"reçu '{self.gbm_loss}'"
            )

    @property
    def rf_params(self) -> dict[str, int | str]:
        """Retourne le dict kwargs pour RandomForestClassifier."""
        return {
            "n_estimators": self.rf_n_estimators,
            "max_depth": self.rf_max_depth,
            "min_samples_leaf": self.rf_min_samples_leaf,
            "class_weight": self.rf_class_weight,
            "n_jobs": self.rf_n_jobs,
            "random_state": self.random_seed,
        }

    @property
    def gbm_params(self) -> dict[str, int | float | str]:
        """Retourne le dict kwargs pour HistGradientBoostingRegressor."""
        return {
            "max_iter": self.gbm_max_iter,
            "max_depth": self.gbm_max_depth,
            "min_samples_leaf": self.gbm_min_samples_leaf,
            "learning_rate": self.gbm_learning_rate,
            "loss": self.gbm_loss,
            "random_state": self.random_seed,
        }

    @property
    def eval_years(self) -> list[int]:
        """Années d'évaluation OOS (jamais vues par le modèle)."""
        return [self.val_year, self.test_year]
