"""Puissance statistique de la fenêtre OOS.

Verrouille le constat central de l'audit 2026-08-22 : les critères
d'acceptation (Sharpe >= 1.0 ET DSR > 0 a p < 0.05) et la quantité de données
OOS disponibles sont mathématiquement incompatibles.
"""
from __future__ import annotations

import pytest

from app.analysis.edge_validation import (
    oos_power_report,
    required_observations_for_dsr,
)

TRADING_DAYS = 252


def test_required_obs_grows_with_n_trials() -> None:
    """Chaque hypothèse testée durcit DÉFINITIVEMENT le seuil de la suivante."""
    reqs = [required_observations_for_dsr(1.0, n) for n in (1, 46, 88, 1500)]
    assert reqs == sorted(reqs), "le seuil doit croître avec n_trials"


@pytest.mark.parametrize(
    ("n_trials", "expected_years"),
    [(1, 2.7), (88, 17.1), (1500, 25.1)],
)
def test_years_of_oos_needed_for_sharpe_one(n_trials: int, expected_years: float) -> None:
    """Chiffres cités dans l'audit — vérifiés par le calcul, pas par la mémoire."""
    years = required_observations_for_dsr(1.0, n_trials) / TRADING_DAYS
    assert years == pytest.approx(expected_years, abs=0.1)


def test_virgin_window_is_underpowered() -> None:
    """La fenêtre 2026 (~95 barres D1) ne peut porter aucun verdict positif."""
    rep = oos_power_report(n_obs=95, n_trials=88)
    assert rep["has_power"] is False
    assert rep["min_detectable_sharpe"] > 5.0, (
        "sur 95 observations il faut un Sharpe absurde pour franchir le DSR"
    )


def test_thirty_trades_per_year_yields_nan_dsr_on_virgin_window() -> None:
    """30 trades/an -> ~12 trades sur 0.38 an -> n_obs < 31 -> DSR = NaN."""
    rep = oos_power_report(n_obs=12, n_trials=88, periods_per_year=30)
    assert "INEXPLOITABLE" in str(rep["verdict"])


def test_even_one_trial_needs_far_more_than_the_virgin_window() -> None:
    """Le constat ne dépend PAS de la convention de comptage de n_trials.

    Même à n_trials=1 — la comptabilité la plus généreuse possible — il faut
    ~2.7 ans, contre 0.38 an disponible : un facteur ~7.
    """
    required = required_observations_for_dsr(1.0, n_trials=1)
    available = 0.378 * TRADING_DAYS
    assert required / available > 6.0


def test_sufficient_window_reports_power() -> None:
    """Contrôle positif : une fenêtre assez longue est bien déclarée suffisante."""
    rep = oos_power_report(n_obs=5000, n_trials=88)
    assert rep["has_power"] is True
    assert "SUFFISANTE" in str(rep["verdict"])


def test_negative_target_sharpe_is_unreachable() -> None:
    assert required_observations_for_dsr(0.0, 10) == float("inf")
