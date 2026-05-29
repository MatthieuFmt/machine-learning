"""Tests sélection de modèle pivot v4 A7."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.candidates import CANDIDATES, build_hgbm, build_rf, build_stacking
from app.models.cpcv_evaluation import (
    _compute_sharpe_per_trade,
    _purged_kfold_indices,
    evaluate_model_cpcv,
)


def _make_dataset(n: int = 500, n_features: int = 15, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=cols)
    logit = X.iloc[:, :5].sum(axis=1) + rng.normal(0, 0.3, n)
    y = pd.Series((logit > 0).astype(int))
    pnl = pd.Series(np.where(y == 1, rng.uniform(50, 200, n), rng.uniform(-150, -50, n)))
    return X, y, pnl


def _make_imbalanced_dataset(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Crée un dataset déséquilibré (10% winners, 90% losers)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, 10)), columns=[f"f{i}" for i in range(10)])
    logit = X.iloc[:, :3].sum(axis=1) + rng.normal(0, 0.3, n)
    # Seuil haut → peu de winners
    threshold = float(np.quantile(logit, 0.90))
    y = pd.Series((logit > threshold).astype(int))
    pnl = pd.Series(np.where(y == 1, rng.uniform(50, 200, n), rng.uniform(-150, -50, n)))
    return X, y, pnl


# ═══════════════════════════════════════════════════════════════════════════
# Tests fit + predict
# ═══════════════════════════════════════════════════════════════════════════


def test_rf_fits_and_predicts() -> None:
    """RF fit et predict_proba sur données synthétiques."""
    X, y, _ = _make_dataset()
    rf = build_rf()
    rf.fit(X.values, y.values)
    proba = rf.predict_proba(X.values)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_hgbm_fits_and_predicts() -> None:
    """HGBM fit et predict_proba sur données synthétiques."""
    X, y, _ = _make_dataset()
    hgbm = build_hgbm()
    hgbm.fit(X.values, y.values)
    proba = hgbm.predict_proba(X.values)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_stacking_fits_and_predicts() -> None:
    """Stacking fit et predict_proba — proba ∈ [0, 1]."""
    X, y, _ = _make_dataset(n=300)
    st = build_stacking()
    st.fit(X.values, y.values)
    proba = st.predict_proba(X.values)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


# ═══════════════════════════════════════════════════════════════════════════
# Tests CPCV / embargo
# ═══════════════════════════════════════════════════════════════════════════


def test_purged_kfold_embargo() -> None:
    """CPCV ne fait pas de leak temporel : embargo respecté entre train et test."""
    splits = _purged_kfold_indices(n=1000, k=5, embargo_pct=0.02)
    embargo = 20  # 1000 * 0.02
    for train_idx, test_idx in splits:
        test_min, test_max = test_idx.min(), test_idx.max()
        purge_min = max(0, test_min - embargo)
        purge_max = min(1000, test_max + embargo)
        # Aucun train index ne doit tomber dans la zone de purge
        overlap = (train_idx >= purge_min) & (train_idx <= purge_max)
        assert not np.any(overlap), f"Train index in purge zone: {train_idx[overlap]}"


def test_compute_sharpe_per_trade_kept_too_few() -> None:
    """Sharpe filtré = 0 si < 5 trades gardés (seuil non atteint)."""
    proba = np.array([0.4, 0.45, 0.48])
    pnl = np.array([100.0, -100.0, 50.0])
    sr, wr, n_kept = _compute_sharpe_per_trade(proba, pnl, threshold=0.50)
    assert sr == 0.0
    assert n_kept == 0


def test_cpcv_runs_on_each_candidate() -> None:
    """Les 3 candidats s'exécutent via CPCV sans erreur sur données synthétiques."""
    X, y, pnl = _make_dataset(n=500)
    for name, builder in CANDIDATES.items():
        r = evaluate_model_cpcv(
            model_builder=builder,
            X=X,
            y=y,
            pnl=pnl,
            model_name=name,
            n_splits=3,
        )
        assert r.model_name == name
        # Au moins quelques folds ont produit un résultat
        assert len(r.fold_sharpes) >= 1
        assert len(r.fold_wrs) >= 1


def test_class_weight_balanced_on_imbalanced_data() -> None:
    """Test que class_weight='balanced' fonctionne sur dataset déséquilibré (10/90)."""
    X, y, pnl = _make_imbalanced_dataset(n=500)
    # Vérifier l'équilibre
    winner_pct = y.mean()
    assert 0.05 < winner_pct < 0.15, f"Expected ~10% winners, got {winner_pct:.1%}"

    # Chaque modèle doit réussir à fit sans erreur
    for builder_name in ["rf", "hgbm"]:
        builder = CANDIDATES[builder_name]
        model = builder()
        model.fit(X.values, y.values)
        proba = model.predict_proba(X.values)[:, 1]
        # Le modèle ne doit pas toujours prédire la classe majoritaire
        # (proba moyenne pas trop proche de 0)
        assert proba.mean() > 0.03, (
            f"{builder_name}: mean proba {proba.mean():.4f} — "
            f"suspect de toujours prédire la classe majoritaire"
        )
