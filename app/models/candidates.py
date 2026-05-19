"""Candidats de modèles pour méta-labeling (pivot v4 A7).

Fix F4 (v5) : sklearn StackingClassifier + TimeSeriesSplit sont incompatibles
(cross_val_predict exige une partition, TimeSeriesSplit n'en est pas une car
les premiers échantillons ne sont jamais dans un test fold).

Solution : implémentation maison TimeSeriesHoldoutStacking qui :
1. Split chronologique train_base/holdout (70/30 par défaut).
2. Fit base estimators sur train_base.
3. Prédit predict_proba sur holdout → meta-features.
4. Fit final estimator (LogReg) sur ces meta-features.
5. Refit base estimators sur 100 % du train pour la prédiction finale.

→ Zéro look-ahead, équivalent au stacking sklearn dans l'esprit, sans
le bug d'incompatibilité.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
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


class TimeSeriesHoldoutStacking(BaseEstimator, ClassifierMixin):
    """Stacking chronologique sans look-ahead — fix F4.

    Remplace sklearn.ensemble.StackingClassifier qui utilise cross_val_predict
    (incompatible avec TimeSeriesSplit). Le pipeline :

    1. fit(X, y) :
       - Split chronologique : train_base = X[:n*0.7], holdout = X[n*0.7:].
       - Chaque base est fit sur train_base.
       - On prédit predict_proba sur holdout → meta_features = (n_holdout, n_bases).
       - final_estimator est fit sur (meta_features, y[n*0.7:]).
       - Chaque base est ensuite REFIT sur la totalité de X pour les prédictions.

    2. predict_proba(X) :
       - Pour chaque base : predict_proba(X)[:, 1].
       - Empilées en (len(X), n_bases) → final_estimator.predict_proba().

    Hypothèse : X et y sont déjà triés chronologiquement par l'appelant.
    Le constructeur ne réordonne PAS l'index.
    """

    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        final_estimator: Any,
        holdout_frac: float = 0.3,
    ) -> None:
        if not 0.0 < holdout_frac < 1.0:
            raise ValueError(f"holdout_frac doit être dans (0, 1), reçu {holdout_frac}")
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_frac = holdout_frac

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TimeSeriesHoldoutStacking":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n = len(X_arr)
        if n < 20:
            raise ValueError(
                f"TimeSeriesHoldoutStacking nécessite au moins 20 samples, reçu {n}"
            )
        split = int(n * (1.0 - self.holdout_frac))
        if split < 5 or n - split < 5:
            raise ValueError(
                f"Split chronologique trop déséquilibré : "
                f"train_base={split}, holdout={n - split}"
            )

        X_base, X_meta = X_arr[:split], X_arr[split:]
        y_base, y_meta = y_arr[:split], y_arr[split:]

        # 1. Fit chaque base sur train_base, prédire sur holdout
        meta_cols = []
        for _name, est in self.estimators:
            est_holdout = clone(est)
            est_holdout.fit(X_base, y_base)
            proba = est_holdout.predict_proba(X_meta)[:, 1]
            meta_cols.append(proba)
        meta_features = np.column_stack(meta_cols)

        # 2. Fit final_estimator sur meta_features
        # Si y_meta n'a qu'une classe → fallback : utiliser moyenne des bases
        if len(np.unique(y_meta)) < 2:
            self.final_estimator_ = None
        else:
            self.final_estimator_ = clone(self.final_estimator)
            self.final_estimator_.fit(meta_features, y_meta)

        # 3. Refit chaque base sur 100 % des données pour les prédictions futures
        self.fitted_bases_: list[tuple[str, Any]] = []
        for name, est in self.estimators:
            est_full = clone(est)
            est_full.fit(X_arr, y_arr)
            self.fitted_bases_.append((name, est_full))

        self.classes_ = np.unique(y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_bases_"):
            raise RuntimeError("fit() doit être appelé avant predict_proba()")
        X_arr = np.asarray(X)
        meta_cols = [
            est.predict_proba(X_arr)[:, 1] for _, est in self.fitted_bases_
        ]
        meta_features = np.column_stack(meta_cols)
        if self.final_estimator_ is None:
            # Fallback : moyenne des proba des bases
            avg = meta_features.mean(axis=1)
            return np.column_stack([1.0 - avg, avg])
        return self.final_estimator_.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


def build_stacking(seed: int = RANDOM_STATE) -> TimeSeriesHoldoutStacking:
    """Stacking RF + HGBM → meta-learner LogReg, sans look-ahead (fix F4).

    Utilise TimeSeriesHoldoutStacking : split chronologique 70/30 pour
    générer les meta-features, puis refit complet pour la prédiction.

    L'ancien design (StackingClassifier + CalibratedClassifierCV avec
    cv=5 KFold) avait du look-ahead silencieux : pour générer la
    meta-feature du fold k, sklearn entraînait les bases sur k+1...K
    → utilisation de données du futur. Voir audit_v4_findings.md §F4.
    """
    rf = build_rf(seed)
    hgbm = build_hgbm(seed)
    meta = LogisticRegression(
        class_weight="balanced",
        random_state=seed,
        max_iter=1000,
    )
    return TimeSeriesHoldoutStacking(
        estimators=[("rf", rf), ("hgbm", hgbm)],
        final_estimator=meta,
        holdout_frac=0.3,
    )


CANDIDATES: dict[str, Any] = {
    "rf": build_rf,
    "hgbm": build_hgbm,
    "stacking": build_stacking,
}
