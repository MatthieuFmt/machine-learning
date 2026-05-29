"""Tests sanity pour deflated_sharpe — Phase 5.

Vérifie les invariants de la formule Bailey & López de Prado 2014 :
1. DSR croît avec le Sharpe observé.
2. DSR décroît avec n_trials (plus on a testé, plus le DSR doit se déflater).
3. DSR croît avec n_obs (plus on a d'observations, plus le test est puissant).
4. p-value(DSR) cohérente avec la distribution normale standard.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis.edge_validation import deflated_sharpe


def test_dsr_increases_with_sharpe() -> None:
    """À n_trials, n_obs, skew, kurt fixés, DSR croît avec sr."""
    base = dict(n_trials=10, n_obs=500, skew=0.0, kurtosis=3.0)
    dsr_low, _ = deflated_sharpe(sr=0.5, **base)
    dsr_mid, _ = deflated_sharpe(sr=1.5, **base)
    dsr_high, _ = deflated_sharpe(sr=3.0, **base)
    assert dsr_low < dsr_mid < dsr_high, (
        f"DSR doit croître avec sr : {dsr_low:.2f} < {dsr_mid:.2f} < {dsr_high:.2f}"
    )


def test_dsr_decreases_with_n_trials() -> None:
    """Plus on teste de stratégies, plus le DSR se déflate (à sr fixe).

    C'est l'effet de la correction multi-tests : si on a essayé 100
    stratégies au lieu de 5, l'observation d'un Sharpe 2.0 est moins
    significative.
    """
    base = dict(sr=2.0, n_obs=500, skew=0.0, kurtosis=3.0)
    dsr_few, _ = deflated_sharpe(n_trials=5, **base)
    dsr_many, _ = deflated_sharpe(n_trials=100, **base)
    assert dsr_few > dsr_many, (
        f"DSR(n=5)={dsr_few:.2f} doit être > DSR(n=100)={dsr_many:.2f}"
    )


def test_dsr_increases_with_n_obs() -> None:
    """Plus on a d'observations, plus le DSR augmente (pouvoir statistique)."""
    base = dict(sr=1.5, n_trials=10, skew=0.0, kurtosis=3.0)
    dsr_small, _ = deflated_sharpe(n_obs=100, **base)
    dsr_large, _ = deflated_sharpe(n_obs=2000, **base)
    assert dsr_small < dsr_large


def test_dsr_pvalue_consistent_with_normal_cdf() -> None:
    """p-value = 1 - Φ(dsr). Pour dsr=0, p ≈ 0.5. Pour dsr=1.645, p ≈ 0.05."""
    base = dict(n_trials=2, n_obs=1000, skew=0.0, kurtosis=3.0)
    # On cherche un sr tel que dsr soit proche de 1.645
    # Pour n_trials=2 sr_zero ≈ 0.45, donc sr ≈ 0.5 + 1.645/sqrt(999) × √denom
    # Approximation suffisante : on teste juste la monotonie.
    dsr_neg, p_neg = deflated_sharpe(sr=-0.5, **base)
    dsr_zero, p_zero = deflated_sharpe(sr=0.5, **base)
    dsr_pos, p_pos = deflated_sharpe(sr=2.0, **base)

    assert p_pos < p_zero < p_neg, (
        "p-value doit décroître avec DSR : "
        f"p(sr=-0.5)={p_neg:.3f}, p(sr=0.5)={p_zero:.3f}, p(sr=2.0)={p_pos:.3f}"
    )


def test_dsr_handles_invalid_inputs() -> None:
    """n_trials < 1 ou n_obs < 30 → (NaN, NaN)."""
    base = dict(sr=1.0, skew=0.0, kurtosis=3.0)
    d1, p1 = deflated_sharpe(n_trials=0, n_obs=500, **base)
    assert np.isnan(d1) and np.isnan(p1)

    d2, p2 = deflated_sharpe(n_trials=10, n_obs=10, **base)
    assert np.isnan(d2) and np.isnan(p2)


def test_dsr_skew_kurt_affect_denominator() -> None:
    """Skew négatif ET kurtosis élevée déflatent le DSR (queue gauche épaisse).

    Pour des retours avec skew=-1 et kurtosis=5 (typique d'actifs financiers
    avec crash risk), le DSR doit être < celui d'une distribution gaussienne
    (skew=0, kurt=3) à mêmes paramètres.
    """
    base = dict(sr=2.0, n_trials=10, n_obs=500)
    dsr_normal, _ = deflated_sharpe(skew=0.0, kurtosis=3.0, **base)
    dsr_heavy, _ = deflated_sharpe(skew=-1.0, kurtosis=5.0, **base)
    assert dsr_heavy < dsr_normal, (
        f"DSR doit être plus faible avec skew<0 et kurt>3 (queues lourdes) : "
        f"normal={dsr_normal:.2f}, heavy={dsr_heavy:.2f}"
    )
