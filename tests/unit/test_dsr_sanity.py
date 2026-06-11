"""Tests sanity pour deflated_sharpe — Phase 5.

Vérifie les invariants de la formule Bailey & López de Prado 2014 :
1. DSR croît avec le Sharpe observé.
2. DSR décroît avec n_trials (plus on a testé, plus le DSR doit se déflater).
3. DSR croît avec n_obs (plus on a d'observations, plus le test est puissant).
4. p-value(DSR) cohérente avec la distribution normale standard.
"""
from __future__ import annotations

import numpy as np

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
    """p-value = 1 - Φ(dsr), strictement décroissante avec le Sharpe par-période.

    Valeurs PAR PÉRIODE réalistes (sémantique canonique post-fix 2026-06-09) :
    sur 1000 observations, SR/période −0.05 / 0.01 / 0.10 → z ≈ −2.1 / −0.2 / +2.6.
    """
    base = dict(n_trials=2, n_obs=1000, skew=0.0, kurtosis=3.0)
    dsr_neg, p_neg = deflated_sharpe(sr=-0.05, **base)
    dsr_zero, p_zero = deflated_sharpe(sr=0.01, **base)
    dsr_pos, p_pos = deflated_sharpe(sr=0.10, **base)

    assert p_pos < p_zero < p_neg, (
        "p-value doit décroître avec DSR : "
        f"p(sr=-0.05)={p_neg:.3f}, p(sr=0.01)={p_zero:.3f}, p(sr=0.10)={p_pos:.3f}"
    )
    assert dsr_neg < dsr_zero < dsr_pos


def test_dsr_orb_regression_per_period_not_annualized() -> None:
    """Régression du bug « ORB US500 M5 : DSR +11.29 (p=0.000) » (2026-06-09).

    Profil réel : ~2 828 trades sur 11 ans (~1/jour), Sharpe/trade ≈ 0.011
    (≈ 0.17 annualisé). En per-période (canonique), z ≈ 0.6 → bruit. L'ancien
    chemin (Sharpe ANNUALISÉ avec n_obs = nb de trades) donnait z > 5 → faux edge.
    """
    sr_per_trade = 0.17 / np.sqrt(252)  # ≈ 0.0107
    z, p = deflated_sharpe(
        sr=sr_per_trade, n_trials=1, n_obs=2828, skew=0.5, kurtosis=5.0
    )
    assert z < 1.0, f"z={z:.2f} : un Sharpe/trade de 0.011 ne peut pas être significatif"
    assert p > 0.15

    # Démonstration de l'artefact : nourrir le Sharpe annualisé explose le z.
    z_bug, p_bug = deflated_sharpe(
        sr=0.17, n_trials=1, n_obs=2828, skew=0.5, kurtosis=5.0
    )
    assert z_bug > 5.0
    assert p_bug < 1e-6


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
