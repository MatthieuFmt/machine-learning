"""Tests du harnais de recherche d'edge honnête (app.research.edge_harness)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.testing.snooping_guard as guard
from app.config.instruments import AssetConfig
from app.research.edge_harness import (
    evaluate_oos,
    run_honest_backtest,
    screen_candidates,
)

_CFG = AssetConfig(
    spread_pips=0.7,
    slippage_pips=0.3,
    commission_pips=0.0,
    pip_size=0.0001,
    pip_value_eur=10.0,
    tp_points=20,
    sl_points=10,
    window_hours=120,
)


@pytest.fixture(autouse=True)
def _isolate_lock(tmp_path, monkeypatch):
    """Isole le registre anti-snooping dans tmp_path (pas de pollution du repo)."""
    monkeypatch.setattr(guard, "LOCK_PATH", tmp_path / "lock.json")


def _uptrend_df(n: int = 400, drift: float = 0.0004, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 1.1000 + np.arange(n) * drift + rng.normal(0, 0.0002, n)
    high = close + 0.0008
    low = close - 0.0008
    open_ = close - drift  # ouverture proche du close précédent
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_run_honest_backtest_uses_next_open_and_costs() -> None:
    """Le backtest honnête tourne et applique des coûts non nuls."""
    df = _uptrend_df()
    signals = pd.Series(1, index=df.index)  # long permanent
    res = run_honest_backtest(df, signals, _CFG, tp_pips=20, sl_pips=10)
    assert res["total_trades"] > 0
    # Coût round-trip = total_cost_pips appliqué (PnL d'un win < tp brut).
    wins = [t for t in res["trades"] if t["result"] == "win"]
    if wins:
        assert wins[0]["pips_net"] == pytest.approx(20 - _CFG.total_cost_pips, abs=1e-6)


def test_evaluate_oos_records_one_read_and_resolves_n_trials() -> None:
    """evaluate_oos journalise exactement une lecture OOS et résout n_trials."""
    df = _uptrend_df()
    signals = pd.Series(1, index=df.index)
    oos_start = pd.Timestamp("2021-01-01", tz="UTC")

    assert guard.n_trials_from_history(min_floor=0) == 0  # registre vierge
    res = evaluate_oos(
        df, signals, _CFG,
        asset="EURUSD", timeframe="D1", label="long_perma",
        tp_pips=20, sl_pips=10, oos_start=oos_start,
    )
    assert res.oos_trades > 0
    assert res.is_trades > 0
    assert guard.n_trials_from_history(min_floor=0) == 1  # une lecture enregistrée
    assert res.n_trials == 1
    assert isinstance(res.go, bool)


def test_evaluate_oos_no_read_when_disabled() -> None:
    """record_read=False ne journalise rien (utile pour la sélection IS)."""
    df = _uptrend_df()
    signals = pd.Series(1, index=df.index)
    evaluate_oos(
        df, signals, _CFG,
        asset="EURUSD", timeframe="D1", label="x",
        tp_pips=20, sl_pips=10, oos_start=pd.Timestamp("2021-01-01", tz="UTC"),
        record_read=False,
    )
    assert guard.n_trials_from_history(min_floor=0) == 0


def test_screen_selects_best_and_reads_oos_once() -> None:
    """screen_candidates choisit le meilleur sur l'IS et ne lit l'OOS qu'une fois."""
    df = _uptrend_df()
    candidates = {
        "long": pd.Series(1, index=df.index),   # gagnant en uptrend
        "short": pd.Series(-1, index=df.index),  # perdant en uptrend
    }
    res = screen_candidates(
        df, candidates, _CFG,
        asset="EURUSD", timeframe="D1",
        tp_sl_grid=[(20, 10), (30, 15)],
        oos_start=pd.Timestamp("2021-01-01", tz="UTC"),
    )
    assert res.label.startswith("long")  # le long est sélectionné en uptrend
    assert guard.n_trials_from_history(min_floor=0) == 1  # un seul regard OOS
    assert res.oos_trades > 0


def test_screen_no_candidates_returns_nogo_without_read() -> None:
    """Sans trade in-sample, NO-GO explicite et aucune lecture OOS."""
    df = _uptrend_df(n=50)
    candidates = {"flat": pd.Series(0, index=df.index)}  # aucun signal
    res = screen_candidates(
        df, candidates, _CFG,
        asset="EURUSD", timeframe="D1",
        tp_sl_grid=[(20, 10)],
        oos_start=pd.Timestamp("2020-02-01", tz="UTC"),
    )
    assert res.go is False
    assert guard.n_trials_from_history(min_floor=0) == 0
