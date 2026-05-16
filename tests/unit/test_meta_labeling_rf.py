"""Tests unitaires pour MetaLabelingRF — B1 pivot v4.

3 tests requis par la spec B1 :
  1. test_fit_predict_proba — sanity check basique
  2. test_calibrate_threshold_falls_back_if_all_strict — fallback si aucun
     seuil ne retient ≥20% des trades
  3. test_class_weight_balanced — vérifie que class_weight="balanced"
     est bien passé au RF
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.models.meta_labeling import MetaLabelingConfig, MetaLabelingRF


class TestMetaLabelingRF:
    """Suite de tests pour MetaLabelingRF."""

    @pytest.fixture
    def xy_balanced(self) -> tuple[pd.DataFrame, pd.Series]:
        """Génère 200 échantillons avec 2 classes équilibrées, 7 features."""
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame(  # noqa: N806
            rng.randn(n, 7),
            columns=[f"feat_{i}" for i in range(7)],
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        y = pd.Series(  # noqa: N806
            np.concatenate([np.ones(n // 2), np.zeros(n // 2)]),
            index=X.index,
        ).astype(int)
        # Shuffle
        idx = rng.permutation(n)
        return X.iloc[idx], y.iloc[idx]

    def test_fit_predict_proba(self, xy_balanced: tuple[pd.DataFrame, pd.Series]):
        """Vérifie que fit + predict produisent des probas ∈ [0,1]."""
        X, y = xy_balanced  # noqa: N806
        meta = MetaLabelingRF()
        meta.fit(X, y)

        assert meta.model is not None, "Le modèle RF doit être entraîné"
        assert not meta.disabled, "Le modèle ne doit pas être désactivé"

        # predict retourne un masque booléen
        mask = meta.predict(X)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(X)

        # Au moins quelques trades acceptés (threshold ≤ 0.5 par défaut)
        assert mask.sum() > 0, "Au moins quelques trades doivent être acceptés"

        # Vérifier que predict_proba fonctionne bien
        proba = meta.model.predict_proba(X.values)[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_calibrate_threshold_falls_back_if_all_strict(
        self, xy_balanced: tuple[pd.DataFrame, pd.Series]
    ):
        """Si aucun seuil ne retient ≥ 20 % des trades train → fallback disabled.

        On force tous les seuils très hauts (0.95, 0.96, 0.97, 0.98)
        via un config custom, ce qui empêche de passer la contrainte de
        rétention. Résultat attendu : disabled=True, threshold=0.0.
        """
        X, y = xy_balanced  # noqa: N806

        config = MetaLabelingConfig(
            threshold_candidates=(0.95, 0.96, 0.97, 0.98),
            min_trade_retention=0.20,
        )
        meta = MetaLabelingRF(config=config)
        meta.fit(X, y)
        assert not meta.disabled, "Doit être actif après fit"

        # Simuler un backtest_fn qui retourne toujours 1.0
        # Même avec Sharpe=1.0, si la rétention est < 20%, le seuil est ignoré
        meta.calibrate_threshold(X, lambda mask: 1.0)

        # Vérifier le fallback
        assert meta.disabled, (
            "Meta doit être désactivé quand aucun seuil ne passe "
            "la contrainte de rétention"
        )
        assert meta.threshold == 0.0  # type: ignore[unreachable]

        # predict doit accepter tous les trades (baseline pure)
        mask = meta.predict(X)
        assert mask.all(), (
            "En mode disabled, tous les trades doivent être acceptés"
        )

    def test_class_weight_balanced(
        self, xy_balanced: tuple[pd.DataFrame, pd.Series]
    ):
        """Vérifie que class_weight='balanced' est bien passé au RF.

        On inspecte l'attribut class_weight du modèle après fit.
        """
        X, y = xy_balanced  # noqa: N806
        meta = MetaLabelingRF()
        meta.fit(X, y)

        assert meta.model is not None
        assert isinstance(meta.model, RandomForestClassifier)

        # Vérifier que class_weight est bien 'balanced'
        assert meta.model.class_weight == "balanced", (
            f"class_weight attendu='balanced', reçu={meta.model.class_weight}"
        )

        # Vérifier que les autres hyperparams correspondent au config
        assert meta.model.n_estimators == meta.config.n_estimators
        assert meta.model.max_depth == meta.config.max_depth
        assert meta.model.min_samples_leaf == meta.config.min_samples_leaf
        assert meta.model.random_state == meta.config.random_state

    def test_fit_single_class_disables(self):
        """Si y_train n'a qu'une seule classe → désactivation."""
        X = pd.DataFrame(  # noqa: N806
            np.random.randn(50, 3),
            columns=["a", "b", "c"],
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
        )
        y = pd.Series(np.ones(50, dtype=int), index=X.index)  # noqa: N806

        meta = MetaLabelingRF()
        meta.fit(X, y)
        assert meta.disabled, "Doit être désactivé avec une seule classe"

    def test_fit_empty_disables(self):
        """Si X_train ou y_train est vide → désactivation."""
        X = pd.DataFrame(columns=["a", "b"])  # noqa: N806
        y = pd.Series(dtype=int)  # noqa: N806

        meta = MetaLabelingRF()
        meta.fit(X, y)
        assert meta.disabled, "Doit être désactivé avec données vides"
