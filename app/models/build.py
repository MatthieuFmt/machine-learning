"""Constructeur de modèle à partir du pipeline gelé."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression

from app.config.ml_pipeline_v4 import MLPipelineConfig, get_pipeline


def build_model(cfg: MLPipelineConfig, seed: int = 42):
    """Construit un modèle sklearn-like à partir d'un MLPipelineConfig gelé."""
    if cfg.model_name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.model_params.get("n_estimators", 200),
            max_depth=cfg.model_params.get("max_depth", 4),
            min_samples_leaf=cfg.model_params.get("min_samples_leaf", 10),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if cfg.model_name == "hgbm":
        return HistGradientBoostingClassifier(
            max_iter=cfg.model_params.get("max_iter", 200),
            max_depth=cfg.model_params.get("max_depth", 5),
            learning_rate=cfg.model_params.get("learning_rate", 0.05),
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
        )
    if cfg.model_name == "stacking":
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        hgbm = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
        )
        meta = LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=1000
        )
        stacking = StackingClassifier(
            estimators=[("rf", rf), ("hgbm", hgbm)],
            final_estimator=meta,
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        return CalibratedClassifierCV(stacking, method="isotonic", cv=3)
    raise ValueError(f"Modèle inconnu : {cfg.model_name}")


def build_locked_model(asset: str, tf: str, seed: int = 42):
    """Shortcut : récupère le pipeline gelé et construit le modèle."""
    cfg = get_pipeline(asset, tf)
    return build_model(cfg, seed)
