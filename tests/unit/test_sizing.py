"""Tests unitaires pour backtest.sizing — 5 fonctions de pondération."""

import numpy as np
from numpy.testing import assert_array_almost_equal

from learning_machine_learning.backtest.sizing import (
    weight_linear,
    weight_linear_v2,
    weight_exp,
    weight_step,
    weight_centered,
)


# ── Helpers ───────────────────────────────────────────────

PROBAS = np.array([0.35, 0.45, 0.50, 0.60, 0.75], dtype=np.float64)


def assert_bounds(weights: np.ndarray, lo: float, hi: float) -> None:
    """Vérifie que tous les poids sont dans [lo, hi]."""
    assert (weights >= lo).all(), f"Poids < {lo} détecté : {weights}"
    assert (weights <= hi).all(), f"Poids > {hi} détecté : {weights}"


def assert_monotonic(weights: np.ndarray) -> None:
    """Vérifie que les poids sont non-décroissants avec la proba."""
    diffs = np.diff(weights)
    assert (diffs >= -1e-10).all(), f"Non monotone : {weights}"


def assert_output_shape(weights: np.ndarray, probas: np.ndarray) -> None:
    """Vérifie que la sortie a la même forme que l'entrée."""
    assert weights.shape == probas.shape
    assert weights.dtype == np.float64


# ── weight_linear ─────────────────────────────────────────

class TestWeightLinear:
    def test_bounds(self) -> None:
        w = weight_linear(PROBAS, seuil=0.45)
        assert_bounds(w, 0.5, 1.5)

    def test_monotonic(self) -> None:
        w = weight_linear(PROBAS, seuil=0.45)
        assert_monotonic(w)

    def test_at_threshold(self) -> None:
        """Au seuil exact : weight = 0.5."""
        w = weight_linear(np.array([0.45]), seuil=0.45)
        assert_array_almost_equal(w, [0.5])

    def test_at_cap(self) -> None:
        """Au cap : proba >= seuil + 0.15 → weight = 1.5."""
        w = weight_linear(np.array([0.60, 0.90]), seuil=0.45)
        assert_array_almost_equal(w, [1.5, 1.5])

    def test_below_threshold(self) -> None:
        """En-dessous du seuil : weight = 0.5 (plancher)."""
        w = weight_linear(np.array([0.30, 0.40]), seuil=0.45)
        assert_array_almost_equal(w, [0.5, 0.5])

    def test_output_shape(self) -> None:
        assert_output_shape(weight_linear(PROBAS), PROBAS)

    def test_scalar_array(self) -> None:
        w = weight_linear(np.array([0.52]), seuil=0.45)
        assert 0.5 < w[0] < 1.5

    def test_all_same_proba(self) -> None:
        """Toutes les probas identiques → tous les poids identiques."""
        w = weight_linear(np.array([0.50, 0.50, 0.50]), seuil=0.45)
        assert np.allclose(w, w[0])


# ── weight_linear_v2 ──────────────────────────────────────

class TestWeightLinearV2:
    def test_bounds(self) -> None:
        w = weight_linear_v2(PROBAS, seuil=0.45)
        assert_bounds(w, 0.8, 1.6)  # 0.8 + 0.8 = 1.6 max

    def test_monotonic(self) -> None:
        w = weight_linear_v2(PROBAS, seuil=0.45)
        assert_monotonic(w)

    def test_at_threshold(self) -> None:
        w = weight_linear_v2(np.array([0.45]), seuil=0.45)
        assert_array_almost_equal(w, [0.8])

    def test_at_cap(self) -> None:
        """proba >= seuil + 0.10 → weight = 1.6."""
        w = weight_linear_v2(np.array([0.55, 0.90]), seuil=0.45)
        assert_array_almost_equal(w, [1.6, 1.6])

    def test_below_threshold(self) -> None:
        w = weight_linear_v2(np.array([0.30]), seuil=0.45)
        assert_array_almost_equal(w, [0.8])


# ── weight_exp ────────────────────────────────────────────

class TestWeightExp:
    def test_bounds(self) -> None:
        w = weight_exp(PROBAS, seuil=0.45)
        assert_bounds(w, 0.5, 1.5)

    def test_monotonic(self) -> None:
        w = weight_exp(PROBAS, seuil=0.45)
        assert_monotonic(w)

    def test_at_threshold(self) -> None:
        w = weight_exp(np.array([0.45]), seuil=0.45)
        assert_array_almost_equal(w, [0.5])

    def test_at_cap(self) -> None:
        w = weight_exp(np.array([0.60, 0.90]), seuil=0.45)
        assert_array_almost_equal(w, [1.5, 1.5])

    def test_convexity(self) -> None:
        """weight_exp ≤ weight_linear pour les mêmes probas (z² ≤ z sur [0,1])."""
        for p in [0.50, 0.52, 0.55, 0.58]:
            w_exp = weight_exp(np.array([p]), seuil=0.45)
            w_lin = weight_linear(np.array([p]), seuil=0.45)
            assert w_exp[0] <= w_lin[0] + 1e-10, f"Échec convexité à proba={p}"
            # Égalité aux extrêmes
            if p <= 0.45:
                assert w_exp[0] == pytest.approx(w_lin[0])
            if p >= 0.60:
                assert w_exp[0] == pytest.approx(w_lin[0])


# ── weight_step ───────────────────────────────────────────

class TestWeightStep:
    def test_bounds(self) -> None:
        w = weight_step(PROBAS, seuil=0.45)
        assert_bounds(w, 0.5, 1.5)

    def test_monotonic(self) -> None:
        w = weight_step(PROBAS, seuil=0.45)
        assert_monotonic(w)

    def test_discrete_levels(self) -> None:
        """Les paliers sont bien 0.5, 1.0, 1.5."""
        probas = np.array([0.40, 0.52, 0.60])
        w = weight_step(probas, seuil=0.45)
        assert_array_almost_equal(w, [0.5, 1.0, 1.5])

    def test_all_levels_present(self) -> None:
        """Tous les niveaux de poids apparaissent avec des probas variées."""
        probas = np.linspace(0.35, 0.75, 100)
        w = weight_step(probas, seuil=0.45)
        unique = np.unique(w)
        assert 0.5 in unique
        assert 1.0 in unique
        assert 1.5 in unique


# ── weight_centered ───────────────────────────────────────

class TestWeightCentered:
    def test_bounds(self) -> None:
        w = weight_centered(PROBAS, seuil=0.35)
        assert_bounds(w, 0.8, 1.2)

    def test_monotonic(self) -> None:
        w = weight_centered(PROBAS, seuil=0.35)
        assert_monotonic(w)

    def test_at_threshold(self) -> None:
        """Au seuil : weight = 0.8."""
        w = weight_centered(np.array([0.35]), seuil=0.35)
        assert_array_almost_equal(w, [0.8])

    def test_at_cap(self) -> None:
        """proba >= seuil + 0.10 → weight = 1.2."""
        w = weight_centered(np.array([0.45, 0.90]), seuil=0.35)
        assert_array_almost_equal(w, [1.2, 1.2])

    def test_below_threshold(self) -> None:
        """proba < seuil → weight = 0.8 (plancher)."""
        w = weight_centered(np.array([0.20, 0.30]), seuil=0.35)
        assert_array_almost_equal(w, [0.8, 0.8])

    def test_custom_threshold(self) -> None:
        """Avec seuil=0.50, weight=0.8 à proba=0.50 et 1.2 à proba=0.60."""
        w_low = weight_centered(np.array([0.50]), seuil=0.50)
        w_high = weight_centered(np.array([0.60]), seuil=0.50)
        assert_array_almost_equal(w_low, [0.8])
        assert_array_almost_equal(w_high, [1.2])
