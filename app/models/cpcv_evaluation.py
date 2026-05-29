"""Évaluation CPCV d'un modèle de méta-labeling sur train uniquement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CPCVResult:
    """Résultat d'une évaluation CPCV pour un modèle."""

    model_name: str
    sharpe_mean: float
    sharpe_std: float
    sharpe_ratio_stability: float  # std / |mean|
    wr_mean: float
    n_kept_mean: float
    fold_sharpes: list[float]
    fold_wrs: list[float]


def _purged_kfold_indices(
    n: int,
    k: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged k-fold splits with embargo (López de Prado)."""
    embargo = max(1, int(n * embargo_pct))
    fold_size = n // k
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        test_idx = np.arange(test_start, test_end)
        train_mask = np.ones(n, dtype=bool)
        purge_start = max(0, test_start - embargo)
        purge_end = min(n, test_end + embargo)
        train_mask[purge_start:purge_end] = False
        train_idx = np.where(train_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def _compute_sharpe_per_trade(
    proba: np.ndarray,
    pnl: np.ndarray,
    threshold: float = 0.50,
) -> tuple[float, float, int]:
    """Sharpe per-trade × √n_trades annualized, sur les trades filtrés."""
    keep = proba > threshold
    if keep.sum() < 5:
        return 0.0, 0.0, int(keep.sum())
    pnl_kept = pnl[keep]
    wr = float((pnl_kept > 0).mean())
    if pnl_kept.std() == 0:
        return 0.0, wr, int(keep.sum())
    sr_per_trade = float(pnl_kept.mean() / pnl_kept.std())
    sr_annualized = sr_per_trade * np.sqrt(float(keep.sum()))
    return sr_annualized, wr, int(keep.sum())


def evaluate_model_cpcv(
    model_builder: Any,
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    model_name: str,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    threshold: float = 0.50,
    seed: int = 42,
) -> CPCVResult:
    """Évalue un modèle via CPCV. Retourne le Sharpe filtré moyen + std.

    Args:
        model_builder: fonction(seed) qui retourne un modèle sklearn-like (fit/predict_proba).
        X: features train.
        y: cible binaire train.
        pnl: PnL brut par trade train (pour calcul Sharpe filtré).
        model_name: identifiant pour log.
        n_splits, embargo_pct: paramètres CPCV.
        threshold: seuil de filtrage (fixé à 0.50 pour A7).
        seed: graine aléatoire.

    Returns:
        CPCVResult avec les métriques agrégées.
    """
    n = len(X)
    splits = _purged_kfold_indices(n, k=n_splits, embargo_pct=embargo_pct)
    fold_sharpes: list[float] = []
    fold_wrs: list[float] = []
    fold_n_kept: list[int] = []

    for train_idx, test_idx in splits:
        if len(train_idx) < 30 or len(test_idx) < 10:
            continue
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        pnl_te = pnl.iloc[test_idx].values

        model = model_builder(seed)
        model.fit(X_tr.values, y_tr.values)
        proba = model.predict_proba(X_te.values)[:, 1]
        sr, wr, n_kept = _compute_sharpe_per_trade(proba, pnl_te, threshold)
        fold_sharpes.append(sr)
        fold_wrs.append(wr)
        fold_n_kept.append(n_kept)

    if not fold_sharpes:
        return CPCVResult(model_name, 0.0, 0.0, float("inf"), 0.0, 0.0, [], [])

    sr_mean = float(np.mean(fold_sharpes))
    sr_std = float(np.std(fold_sharpes))
    stability = sr_std / (abs(sr_mean) + 1e-9)

    return CPCVResult(
        model_name=model_name,
        sharpe_mean=sr_mean,
        sharpe_std=sr_std,
        sharpe_ratio_stability=stability,
        wr_mean=float(np.mean(fold_wrs)),
        n_kept_mean=float(np.mean(fold_n_kept)),
        fold_sharpes=fold_sharpes,
        fold_wrs=fold_wrs,
    )
