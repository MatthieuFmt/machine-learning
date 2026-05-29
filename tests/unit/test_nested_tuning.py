"""Tests nested CPCV tuning pivot v4 A8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.models.nested_tuning import (
    TuningResult,
    _expand_grid,
    nested_cpcv_tuning,
)


def _rf_factory(params: dict, seed: int) -> RandomForestClassifier:
    """RandomForest factory pour les tests."""
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 50)),
        max_depth=int(params.get("max_depth", 3)),
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def _make_xy_pnl(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Crée un dataset synthétique avec signal exploitable."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(10)]
    X = pd.DataFrame(rng.normal(0, 1, (n, 10)), columns=cols)
    logit = X.iloc[:, :3].sum(axis=1) + rng.normal(0, 0.3, n)
    y = pd.Series((logit > 0).astype(int))
    pnl = pd.Series(np.where(y == 1, rng.uniform(50, 200, n), rng.uniform(-150, -50, n)))
    return X, y, pnl


# ═══════════════════════════════════════════════════════════════════════════
# 1. _expand_grid produit le produit cartésien
# ═══════════════════════════════════════════════════════════════════════════


def test_expand_grid_produces_cartesian() -> None:
    """_expand_grid doit produire le produit cartésien des listes de valeurs."""
    g = {"a": [1, 2], "b": [3, 4, 5]}
    combos = _expand_grid(g)
    assert len(combos) == 6  # 2 × 3
    assert all("a" in c and "b" in c for c in combos)

    # Vérifier que toutes les combinaisons sont présentes
    keys_set = {(c["a"], c["b"]) for c in combos}
    expected = {(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)}
    assert keys_set == expected


def test_expand_grid_single_value() -> None:
    """_expand_grid avec une seule valeur par axe : 1 combo."""
    g = {"lr": [0.05], "depth": [5]}
    combos = _expand_grid(g)
    assert len(combos) == 1
    assert combos[0] == {"lr": 0.05, "depth": 5}


# ═══════════════════════════════════════════════════════════════════════════
# 2. nested_cpcv_tuning tourne sans erreur
# ═══════════════════════════════════════════════════════════════════════════


def test_nested_tuning_runs() -> None:
    """nested_cpcv_tuning tourne sur données synthétiques et retourne TuningResult."""
    X, y, pnl = _make_xy_pnl(n=300)
    r = nested_cpcv_tuning(
        model_factory=_rf_factory,
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 4]},
        threshold_grid=[0.50, 0.55],
        X=X,
        y=y,
        pnl=pnl,
        outer_k=3,
        inner_k=2,
        embargo_pct=0.01,
        seed=42,
    )
    assert isinstance(r, TuningResult)
    assert r.best_threshold >= 0.50
    assert isinstance(r.best_params, dict)
    assert len(r.outer_fold_results) >= 1
    assert r.n_combos_evaluated == 4 * 2  # 2×2 combos × 2 thresholds


# ═══════════════════════════════════════════════════════════════════════════
# 3. Seuil < 0.50 lève AssertionError (leçon H04 v2)
# ═══════════════════════════════════════════════════════════════════════════


def test_threshold_below_05_raises() -> None:
    """Un seuil < 0.50 doit lever AssertionError."""
    X, y, pnl = _make_xy_pnl()
    with pytest.raises(AssertionError, match="Seuil < 0.50 interdit"):
        nested_cpcv_tuning(
            model_factory=_rf_factory,
            param_grid={"n_estimators": [50]},
            threshold_grid=[0.40],
            X=X,
            y=y,
            pnl=pnl,
            outer_k=3,
            inner_k=2,
        )


def test_threshold_mixed_with_below_05_raises() -> None:
    """Même si un seul seuil est < 0.50 dans la liste, doit lever."""
    X, y, pnl = _make_xy_pnl()
    with pytest.raises(AssertionError, match="Seuil < 0.50 interdit"):
        nested_cpcv_tuning(
            model_factory=_rf_factory,
            param_grid={"n_estimators": [50]},
            threshold_grid=[0.50, 0.45, 0.55],
            X=X,
            y=y,
            pnl=pnl,
            outer_k=3,
            inner_k=2,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Reproductibilité : même seed → même résultat
# ═══════════════════════════════════════════════════════════════════════════


def test_reproducible() -> None:
    """Deux appels avec le même seed doivent donner le même best_params et threshold."""
    X, y, pnl = _make_xy_pnl()
    kwargs = dict(
        model_factory=_rf_factory,
        param_grid={"n_estimators": [50, 100]},
        threshold_grid=[0.50, 0.55],
        X=X,
        y=y,
        pnl=pnl,
        outer_k=3,
        inner_k=2,
        seed=42,
    )
    r1 = nested_cpcv_tuning(**kwargs)
    r2 = nested_cpcv_tuning(**kwargs)
    assert r1.best_params == r2.best_params
    assert r1.best_threshold == r2.best_threshold
    assert r1.sharpe_outer_mean == r2.sharpe_outer_mean


# ═══════════════════════════════════════════════════════════════════════════
# 5. n_combos_evaluated correct
# ═══════════════════════════════════════════════════════════════════════════


def test_n_combos_correct() -> None:
    """n_combos_evaluated doit être le produit des axes × nombre de seuils."""
    X, y, pnl = _make_xy_pnl()
    grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 4]}  # 3 × 2 = 6 combos
    thresholds = [0.50, 0.55, 0.60]  # 3 thresholds
    r = nested_cpcv_tuning(
        model_factory=_rf_factory,
        param_grid=grid,
        threshold_grid=thresholds,
        X=X,
        y=y,
        pnl=pnl,
        outer_k=3,
        inner_k=2,
        seed=42,
    )
    assert r.n_combos_evaluated == 3 * 2 * 3  # = 18


# ═══════════════════════════════════════════════════════════════════════════
# 6. Structure de TuningResult
# ═══════════════════════════════════════════════════════════════════════════


def test_tuning_result_fields() -> None:
    """TuningResult doit avoir tous les champs attendus avec les bons types."""
    X, y, pnl = _make_xy_pnl(n=300)
    r = nested_cpcv_tuning(
        model_factory=_rf_factory,
        param_grid={"max_depth": [3, 4]},
        threshold_grid=[0.50],
        X=X,
        y=y,
        pnl=pnl,
        outer_k=3,
        inner_k=2,
        seed=42,
    )
    assert isinstance(r.best_params, dict)
    assert isinstance(r.best_threshold, float)
    assert isinstance(r.sharpe_outer_mean, float)
    assert isinstance(r.sharpe_outer_std, float)
    assert isinstance(r.wr_outer_mean, float)
    assert isinstance(r.n_kept_outer_mean, float)
    assert isinstance(r.n_combos_evaluated, int)
    assert isinstance(r.outer_fold_results, list)
    assert len(r.outer_fold_results) >= 1
    # Chaque outer fold doit avoir les clés attendues
    fold = r.outer_fold_results[0]
    assert "best_params" in fold
    assert "best_threshold" in fold
    assert "outer_sharpe" in fold
    assert "outer_wr" in fold
    assert "outer_n_kept" in fold
    assert "inner_best_score" in fold


# ═══════════════════════════════════════════════════════════════════════════
# 7. Pas de fuite outer → inner
# ═══════════════════════════════════════════════════════════════════════════


def test_no_data_leak_outer_to_inner() -> None:
    """Les timestamps outer_test ne doivent jamais apparaître dans outer_train (purge)."""
    n = 300
    X, y, pnl = _make_xy_pnl(n=n)
    from app.models.cpcv_evaluation import _purged_kfold_indices

    outer_splits = _purged_kfold_indices(n, k=3, embargo_pct=0.02)  # 2% embargo → ~6 barres

    for outer_train_idx, outer_test_idx in outer_splits:
        train_set = set(outer_train_idx.tolist())
        test_set = set(outer_test_idx.tolist())
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Overlap train/test de {len(overlap)} indices — "
            f"la purge est censée les isoler"
        )

        # Vérifier embargo : pas d'indice adjacent au test dans le train
        embargo = max(1, int(n * 0.02))
        test_min = outer_test_idx.min()
        test_max = outer_test_idx.max()
        # Les indices train ne doivent pas être dans [test_min - embargo, test_max + embargo]
        purge_zone = set(range(
            max(0, test_min - embargo),
            min(n, test_max + embargo + 1),
        ))
        train_in_purge = train_set & purge_zone
        assert len(train_in_purge) == 0, (
            f"{len(train_in_purge)} indices train dans la zone de purge "
            f"[{max(0, test_min - embargo)}, {min(n, test_max + embargo)}]"
        )
