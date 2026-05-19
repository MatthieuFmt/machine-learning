"""Tests pour fix F15 — bootstrap stationnaire (Politis-Romano).

Vérifie :
1. Les indices générés respectent la propriété de bloc (≠ iid pur).
2. block_size=1 reproduit le bootstrap iid.
3. Sur une série autocorrélée, la variance du Sharpe bootstrap est plus
   grande qu'avec un bootstrap iid → IC plus larges, plus honnêtes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.edge_validation import (
    _stationary_bootstrap_indices,
    bootstrap_sharpe,
)


def test_indices_in_bounds() -> None:
    """Tous les indices ∈ [0, n)."""
    rng = np.random.default_rng(42)
    n = 200
    idx = _stationary_bootstrap_indices(n, avg_block_size=10, rng=rng)
    assert idx.shape == (n,)
    assert (idx >= 0).all() and (idx < n).all()


def test_block_size_1_behaves_iid() -> None:
    """avg_block_size=1 → p_restart=1 → indices iid uniformes.

    Avec block_size=1, on tire un nouvel index aléatoire à chaque pas
    (sauf le premier qui est déjà aléatoire). On vérifie l'absence
    de séquences consécutives.
    """
    rng = np.random.default_rng(42)
    n = 1000
    idx = _stationary_bootstrap_indices(n, avg_block_size=1, rng=rng)
    diffs = np.diff(idx)
    # Avec n=1000 indices iid, la proba d'avoir deux indices consécutifs
    # (diff = 1 par hasard) est ~1/n. Donc < 5 % au total.
    n_consecutive = int((diffs == 1).sum())
    assert n_consecutive < n * 0.05


def test_block_size_large_preserves_runs() -> None:
    """avg_block_size grand → longues séquences consécutives (blocs)."""
    rng = np.random.default_rng(42)
    n = 1000
    idx = _stationary_bootstrap_indices(n, avg_block_size=50, rng=rng)
    diffs = np.diff(idx)
    # Au moins 70 % des transitions devraient être diff=1 (intra-bloc)
    pct_consecutive = float((diffs == 1).mean())
    assert pct_consecutive > 0.7, (
        f"Avec block_size=50, attendu > 70 % de transitions consécutives, "
        f"observé {pct_consecutive:.1%}"
    )


def test_bootstrap_sharpe_returns_valid_pvalue() -> None:
    """Sanity : bootstrap_sharpe retourne un Sharpe moyen et p-value ∈ [0, 1]."""
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.standard_normal(500) * 0.01 + 0.001)
    mean_boot, p_val = bootstrap_sharpe(returns, n_iter=200, seed=42, block_size=10)
    assert np.isfinite(mean_boot)
    assert 0.0 <= p_val <= 1.0


def test_bootstrap_sharpe_pvalue_low_on_positive_signal() -> None:
    """Sur une série à mean positif clair, p(Sharpe ≤ 0) doit être faible."""
    rng = np.random.default_rng(99)
    # Returns avec drift positif clair : mean 0.005, std 0.01 → Sharpe annualisé ~7.9
    returns = pd.Series(rng.standard_normal(500) * 0.01 + 0.005)
    _, p_val = bootstrap_sharpe(returns, n_iter=500, seed=42, block_size=10)
    assert p_val < 0.05


def test_bootstrap_sharpe_block_wider_ci_than_iid() -> None:
    """Sur une série autocorrélée, block bootstrap donne une variance Sharpe
    plus grande (IC plus large) que l'iid.

    Construction : returns AR(1) avec phi=0.5 → forte autocorrélation.
    """
    rng = np.random.default_rng(123)
    n = 500
    phi = 0.5
    noise = rng.standard_normal(n) * 0.01
    returns = np.zeros(n)
    returns[0] = noise[0]
    for i in range(1, n):
        returns[i] = phi * returns[i - 1] + noise[i]
    series = pd.Series(returns)

    # Bootstrap iid (block_size=1) vs block (size=20)
    rng_iid = np.random.default_rng(42)
    rng_blk = np.random.default_rng(42)

    boot_iid = np.empty(300)
    for i in range(300):
        idx = _stationary_bootstrap_indices(n, avg_block_size=1, rng=rng_iid)
        s = returns[idx]
        std = float(np.std(s, ddof=1))
        boot_iid[i] = float(np.mean(s) / std * np.sqrt(252)) if std > 0 else 0.0

    boot_blk = np.empty(300)
    for i in range(300):
        idx = _stationary_bootstrap_indices(n, avg_block_size=20, rng=rng_blk)
        s = returns[idx]
        std = float(np.std(s, ddof=1))
        boot_blk[i] = float(np.mean(s) / std * np.sqrt(252)) if std > 0 else 0.0

    var_iid = float(np.var(boot_iid, ddof=1))
    var_blk = float(np.var(boot_blk, ddof=1))
    # Le block bootstrap doit donner une variance >= iid (à 10 % près minimum)
    assert var_blk >= 0.9 * var_iid, (
        f"Block bootstrap variance {var_blk:.4f} doit être >= iid {var_iid:.4f}"
    )
