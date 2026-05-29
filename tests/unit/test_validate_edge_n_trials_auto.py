"""Test C4 — validate_edge(n_trials=None) lit n_trials depuis le registre anti-snooping.

Le DSR ne doit JAMAIS s'appuyer sur une constante en dur : par défaut, le nombre
d'essais provient de snooping_guard.n_trials_from_history().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import app.testing.snooping_guard as guard
from app.analysis.edge_validation import validate_edge


def _synthetic_edge(seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    daily = rng.normal(0.0008, 0.01, n)  # léger drift positif
    equity = pd.Series(1.0 + np.cumsum(daily), index=idx)
    pnl = rng.normal(5.0, 50.0, 120)
    trades = pd.DataFrame({"pnl": pnl})
    return equity, trades


def test_validate_edge_auto_n_trials_matches_explicit(monkeypatch) -> None:
    """n_trials=None doit produire le même rapport que n_trials=<valeur registre>."""
    monkeypatch.setattr(guard, "n_trials_from_history", lambda *a, **k: 7)

    equity, trades = _synthetic_edge()
    auto = validate_edge(equity, trades, n_trials=None)
    explicit = validate_edge(equity, trades, n_trials=7)

    assert auto.metrics["dsr"] == explicit.metrics["dsr"] or (
        np.isnan(auto.metrics["dsr"]) and np.isnan(explicit.metrics["dsr"])
    )
    assert auto.metrics["p_value"] == explicit.metrics["p_value"] or (
        np.isnan(auto.metrics["p_value"]) and np.isnan(explicit.metrics["p_value"])
    )
    assert auto.go == explicit.go


def test_validate_edge_auto_differs_when_history_changes(monkeypatch) -> None:
    """Plus d'essais enregistrés → DSR plus sévère (n_trials plus grand)."""
    equity, trades = _synthetic_edge(seed=1)

    monkeypatch.setattr(guard, "n_trials_from_history", lambda *a, **k: 1)
    few = validate_edge(equity, trades, n_trials=None)

    monkeypatch.setattr(guard, "n_trials_from_history", lambda *a, **k: 100)
    many = validate_edge(equity, trades, n_trials=None)

    # Avec plus d'essais, le DSR ne peut pas être plus favorable.
    if not (np.isnan(few.metrics["dsr"]) or np.isnan(many.metrics["dsr"])):
        assert many.metrics["dsr"] <= few.metrics["dsr"]
