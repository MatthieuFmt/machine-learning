"""Tuning hyperparams + seuil via nested CPCV (pivot v4 A8).

Nested cross-validation :
- Outer CV : évalue la performance attendue (honnête — jamais vu pendant le tuning).
- Inner CV : sélectionne les meilleurs hyperparams sur le train de chaque outer fold.

Référence : López de Prado, Advances in Financial ML, chapitre 12.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from app.models.cpcv_evaluation import _compute_sharpe_per_trade, _purged_kfold_indices


@dataclass
class TuningResult:
    """Résultat du nested CPCV tuning."""

    best_params: dict
    best_threshold: float
    sharpe_outer_mean: float
    sharpe_outer_std: float
    wr_outer_mean: float
    n_kept_outer_mean: float
    outer_fold_results: list[dict]
    n_combos_evaluated: int


def _expand_grid(param_grid: dict) -> list[dict]:
    """Expand un grid {k: [v1, v2], k2: [w1]} en liste de dicts (produit cartésien)."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def nested_cpcv_tuning(
    model_factory: Callable[[dict, int], Any],
    param_grid: dict,
    threshold_grid: list[float],
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    outer_k: int = 5,
    inner_k: int = 3,
    embargo_pct: float = 0.01,
    seed: int = 42,
) -> TuningResult:
    """Nested CPCV : outer pour évaluation honnête, inner pour sélection hyperparams.

    Args:
        model_factory: fonction (params: dict, seed: int) -> modèle sklearn-like
                       (doit exposer fit + predict_proba).
        param_grid: dict de listes pour chaque hyperparamètre.
        threshold_grid: liste de seuils candidats (plancher 0.50 imposé).
        X, y, pnl: données train (features, cible binaire, PnL brut par trade).
        outer_k: nombre de folds pour l'évaluation externe.
        inner_k: nombre de folds pour la sélection interne.
        embargo_pct: pourcentage d'embargo entre folds (0.01 = 1%).
        seed: graine aléatoire pour la reproductibilité.

    Returns:
        TuningResult avec best_params (vote majoritaire), best_threshold,
        Sharpe outer non biaisé ± std.

    Raises:
        AssertionError: si un seuil < 0.50 est fourni (leçon H04 v2).
        RuntimeError: si aucun outer fold ne produit de résultat.
    """
    assert all(t >= 0.50 for t in threshold_grid), "Seuil < 0.50 interdit (leçon H04)"

    n = len(X)
    outer_splits = _purged_kfold_indices(n, k=outer_k, embargo_pct=embargo_pct)
    combos = _expand_grid(param_grid)
    n_combos = len(combos)

    outer_results: list[dict] = []
    best_combo_votes: dict[tuple, int] = {}
    best_threshold_votes: dict[float, int] = {}

    for outer_train_idx, outer_test_idx in outer_splits:
        if len(outer_train_idx) < 60:
            continue

        X_outer_tr = X.iloc[outer_train_idx]
        y_outer_tr = y.iloc[outer_train_idx]
        pnl_outer_tr = pnl.iloc[outer_train_idx]
        X_outer_te = X.iloc[outer_test_idx]
        pnl_outer_te = pnl.iloc[outer_test_idx]

        # ── Inner CV sur outer_train ──────────────────────────────────────
        inner_splits = _purged_kfold_indices(
            len(X_outer_tr), k=inner_k, embargo_pct=embargo_pct,
        )
        best_inner_score = -np.inf
        best_inner_params = combos[0]
        best_inner_threshold = 0.50

        for combo in combos:
            for threshold in threshold_grid:
                inner_sharpes: list[float] = []
                for in_tr, in_te in inner_splits:
                    if len(in_tr) < 30 or len(in_te) < 10:
                        continue
                    model = model_factory(combo, seed)
                    model.fit(
                        X_outer_tr.iloc[in_tr].values,
                        y_outer_tr.iloc[in_tr].values,
                    )
                    proba = model.predict_proba(X_outer_tr.iloc[in_te].values)[:, 1]
                    sr, _, _ = _compute_sharpe_per_trade(
                        proba, pnl_outer_tr.iloc[in_te].values, threshold,
                    )
                    inner_sharpes.append(sr)

                if not inner_sharpes:
                    continue

                inner_mean = float(np.mean(inner_sharpes))
                if inner_mean > best_inner_score:
                    best_inner_score = inner_mean
                    best_inner_params = combo
                    best_inner_threshold = threshold

        # ── Évaluation outer (honnête : jamais vu pendant le tuning) ──────
        model = model_factory(best_inner_params, seed)
        model.fit(X_outer_tr.values, y_outer_tr.values)
        proba_te = model.predict_proba(X_outer_te.values)[:, 1]
        outer_sr, outer_wr, outer_n_kept = _compute_sharpe_per_trade(
            proba_te, pnl_outer_te.values, best_inner_threshold,
        )

        outer_results.append({
            "best_params": best_inner_params,
            "best_threshold": best_inner_threshold,
            "outer_sharpe": outer_sr,
            "outer_wr": outer_wr,
            "outer_n_kept": outer_n_kept,
            "inner_best_score": best_inner_score,
        })

        # Vote majoritaire
        combo_key = tuple(sorted(best_inner_params.items()))
        best_combo_votes[combo_key] = best_combo_votes.get(combo_key, 0) + 1
        best_threshold_votes[best_inner_threshold] = (
            best_threshold_votes.get(best_inner_threshold, 0) + 1
        )

    if not outer_results:
        raise RuntimeError("Aucun outer fold n'a produit de résultat")

    # Sélection finale : params le plus voté + threshold le plus voté
    final_combo = max(best_combo_votes.items(), key=lambda kv: kv[1])[0]
    final_params = dict(final_combo)
    final_threshold = max(best_threshold_votes.items(), key=lambda kv: kv[1])[0]

    outer_sharpes = [r["outer_sharpe"] for r in outer_results]
    outer_wrs = [r["outer_wr"] for r in outer_results]
    outer_n_kepts = [r["outer_n_kept"] for r in outer_results]

    return TuningResult(
        best_params=final_params,
        best_threshold=final_threshold,
        sharpe_outer_mean=float(np.mean(outer_sharpes)),
        sharpe_outer_std=float(np.std(outer_sharpes)),
        wr_outer_mean=float(np.mean(outer_wrs)),
        n_kept_outer_mean=float(np.mean(outer_n_kepts)),
        outer_fold_results=outer_results,
        n_combos_evaluated=n_combos * len(threshold_grid),
    )
