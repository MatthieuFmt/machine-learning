"""Tests pour fix F6 — Monte Carlo benchmark multi-asset représentatif.

Vérifie :
1. _estimate_sleeve_trade_rate clamp et calcule correctement.
2. monte_carlo_random_benchmark accepte plusieurs sleeves.
3. La distribution Sharpe MC reste dans une plage raisonnable post-fix F3.

Note : ces tests utilisent les vraies données via load_asset() — ils sont
donc plus lents que les autres unit tests. Si load_asset échoue, on skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_validation_finale import (  # noqa: E402
    _estimate_sleeve_trade_rate,
    monte_carlo_random_benchmark,
)


def test_estimate_sleeve_trade_rate_clamp_low() -> None:
    """Trade rate négatif/zéro → clamp à 0.001."""
    rate = _estimate_sleeve_trade_rate(
        asset="GBPUSD", tf="D1",
        n_trades=0,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    assert rate >= 0.001


def test_estimate_sleeve_trade_rate_clamp_high() -> None:
    """Trade rate > 0.5 → clamp à 0.5."""
    rate = _estimate_sleeve_trade_rate(
        asset="GBPUSD", tf="D1",
        n_trades=10_000_000,
        start=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    assert rate <= 0.5


def test_monte_carlo_empty_sleeves_returns_empty_array() -> None:
    """Si aucun sleeve fourni → array vide."""
    sharpes = monte_carlo_random_benchmark(
        sleeve_specs=[],
        sleeve_trade_rates={},
        n_iter=5,
    )
    assert isinstance(sharpes, np.ndarray)
    assert len(sharpes) == 0


def test_monte_carlo_single_sleeve_smoke() -> None:
    """Smoke test : 1 sleeve, 3 itérations → renvoie un array de longueur 3."""
    try:
        sharpes = monte_carlo_random_benchmark(
            sleeve_specs=[{"asset": "GBPUSD", "tf": "D1"}],
            sleeve_trade_rates={"GBPUSD_D1": 0.05},
            n_iter=3,
        )
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"Données indisponibles pour smoke test : {exc}")
    if len(sharpes) == 0:
        pytest.skip("Données chargées mais Monte Carlo vide")
    assert len(sharpes) == 3
    assert np.all(np.isfinite(sharpes))


def test_monte_carlo_p95_drops_after_f3_fix() -> None:
    """Vérification qualitative : P95 < 6 après F3 (au lieu de 9.96 avant).

    Sur 100 iterations × signal_freq=0.05 × GBPUSD D1 (~500 bars), le Sharpe
    random doit converger vers ~0 (sans edge), avec P95 < 6 plutôt que ~10.
    """
    try:
        sharpes = monte_carlo_random_benchmark(
            sleeve_specs=[{"asset": "GBPUSD", "tf": "D1"}],
            sleeve_trade_rates={"GBPUSD_D1": 0.05},
            n_iter=100,
        )
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"Données indisponibles : {exc}")

    if len(sharpes) == 0:
        pytest.skip("Monte Carlo vide")
    p95 = float(np.percentile(sharpes, 95))
    p50 = float(np.percentile(sharpes, 50))
    assert -2.0 <= p50 <= 2.0, (
        f"P50 doit être proche de 0 sur des signaux random, observé {p50:.2f}"
    )
    assert p95 < 6.0, (
        f"P95 random devrait passer < 6 après fix F3 (TP-prime → SL-prime), "
        f"observé {p95:.2f}. Si > 6, F3 a peut-être régressé."
    )
