"""Candidats de modèles pour méta-labeling (pivot v4 A7)."""
from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def build_rf(seed: int = RANDOM_STATE) -> RandomForestClassifier:
    """Random Forest baseline (v2 H05)."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def build_hgbm(seed: int = RANDOM_STATE) -> HistGradientBoostingClassifier:
    """HistGradientBoosting (équivalent sklearn de LightGBM)."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
        early_stopping=False,
    )


def build_stacking(seed: int = RANDOM_STATE) -> CalibratedClassifierCV:
    """Stacking RF + HGBM → meta-learner LogReg avec calibration isotonique."""
    rf = build_rf(seed)
    hgbm = build_hgbm(seed)
    base_estimators: list[tuple[str, Any]] = [("rf", rf), ("hgbm", hgbm)]
    meta = LogisticRegression(
        class_weight="balanced",
        random_state=seed,
        max_iter=1000,
    )
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return CalibratedClassifierCV(stacking, method="isotonic", cv=3)


CANDIDATES: dict[str, Any] = {
    "rf": build_rf,
    "hgbm": build_hgbm,
    "stacking": build_stacking,
}
