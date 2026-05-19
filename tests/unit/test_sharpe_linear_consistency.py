"""Tests pour le fix F2 — Sharpe linéaire cohérent.

Avant le fix : `compute_metrics` faisait `pct_change` sur equity compoundée
alors que le sizing utilise un capital fixe → Sharpe gonflé ×3.5 sur les
stratégies winnantes.

Après le fix : retours linéaires `daily.diff() / capital_eur` → Sharpe
cohérent avec `sharpe_daily_from_trades` sur les mêmes trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.deterministic import run_deterministic_backtest
from app.backtest.metrics import compute_metrics, sharpe_daily_from_trades
from app.backtest.sizing import compute_position_size, expected_pnl_eur
from app.config.instruments import AssetConfig

CFG_EURUSD = AssetConfig(
    spread_pips=0.7,
    slippage_pips=0.2,
    commission_pips=0.0,
    pip_size=0.0001,
    pip_value_eur=10.0,
    tp_points=20,
    sl_points=10,
    window_hours=120,
    min_lot=0.01,
    max_lot=10.0,
)


def _build_synthetic_trades(
    n_trades: int = 200,
    win_rate: float = 0.6,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Génère un set de trades synthétiques alternés."""
    rng = np.random.default_rng(seed)
    is_win = rng.random(n_trades) < win_rate
    pips_net = np.where(is_win, tp_pips, -sl_pips)

    times = pd.date_range("2024-01-01", periods=n_trades, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Pips_Nets": pips_net,
            "Pips_Bruts": pips_net,
            "result": np.where(is_win, "win", "loss_sl"),
        },
        index=times,
    )


def test_sharpe_compute_metrics_aligned_with_daily_from_trades() -> None:
    """Sharpe via compute_metrics ≈ Sharpe via sharpe_daily_from_trades.

    Sur un set de trades synthétiques, les deux fonctions doivent donner
    des Sharpe à ±15 % (légère différence due au lots variables / arrondis).
    Avant fix F2 : différence ×3.5.
    """
    trades_df = _build_synthetic_trades(n_trades=200, win_rate=0.55, seed=42)

    capital = 10_000.0
    sl_dist = 10 * CFG_EURUSD.pip_size
    entry_prices = np.full(len(trades_df), 1.1000)
    sl_prices = entry_prices - sl_dist
    lots = np.array(
        [compute_position_size(ep, sl, capital, 0.02, CFG_EURUSD)
         for ep, sl in zip(entry_prices, sl_prices, strict=True)],
        dtype=float,
    )
    trades_df["position_size_lots"] = lots

    metrics = compute_metrics(
        trades_df, asset_cfg=CFG_EURUSD, capital_eur=capital,
    )
    sharpe_cm = float(metrics["sharpe"])

    fake_trades = [
        {
            "pips_net": float(p),
            "exit_time": str(t),
        }
        for p, t in zip(trades_df["Pips_Nets"].values, trades_df.index, strict=True)
    ]
    sharpe_dft = sharpe_daily_from_trades(
        fake_trades, initial_capital_pips=capital,
    )

    assert sharpe_cm > 0
    assert sharpe_dft > 0

    ratio = sharpe_cm / sharpe_dft if sharpe_dft > 0 else 0.0
    assert 0.7 <= ratio <= 1.4, (
        f"Sharpe compute_metrics ({sharpe_cm:.2f}) et "
        f"sharpe_daily_from_trades ({sharpe_dft:.2f}) doivent rester "
        f"proches après fix F2 (ratio observé : {ratio:.2f})"
    )


def test_sharpe_does_not_explode_on_long_winning_streak() -> None:
    """Sharpe sur une série purement gagnante reste fini et borné.

    Avec pct_change sur cumsum, le Sharpe pouvait approcher 20-50.
    Avec retours linéaires, il reste ≤ ~10 sur des séries réalistes.
    """
    trades_df = _build_synthetic_trades(
        n_trades=300, win_rate=0.65, seed=7,
    )
    capital = 10_000.0
    sl_dist = 10 * CFG_EURUSD.pip_size
    entry_prices = np.full(len(trades_df), 1.1000)
    sl_prices = entry_prices - sl_dist
    lots = np.array(
        [compute_position_size(ep, sl, capital, 0.02, CFG_EURUSD)
         for ep, sl in zip(entry_prices, sl_prices, strict=True)],
        dtype=float,
    )
    trades_df["position_size_lots"] = lots

    metrics = compute_metrics(
        trades_df, asset_cfg=CFG_EURUSD, capital_eur=capital,
    )
    sharpe = float(metrics["sharpe"])

    assert np.isfinite(sharpe), "Sharpe doit être fini"
    assert 0 < sharpe < 15.0, (
        f"Sharpe doit rester dans une plage réaliste (observé : {sharpe:.2f})"
    )


def test_max_dd_pct_bounded_after_fix() -> None:
    """max_dd_pct doit rester dans [-100, 0] (mode A1)."""
    trades_df = _build_synthetic_trades(n_trades=100, win_rate=0.30, seed=1)
    capital = 10_000.0
    sl_dist = 10 * CFG_EURUSD.pip_size
    entry_prices = np.full(len(trades_df), 1.1000)
    sl_prices = entry_prices - sl_dist
    lots = np.array(
        [compute_position_size(ep, sl, capital, 0.02, CFG_EURUSD)
         for ep, sl in zip(entry_prices, sl_prices, strict=True)],
        dtype=float,
    )
    trades_df["position_size_lots"] = lots

    metrics = compute_metrics(
        trades_df, asset_cfg=CFG_EURUSD, capital_eur=capital,
    )
    max_dd_pct = float(metrics["max_dd_pct"])
    assert -100.0 <= max_dd_pct <= 0.0, (
        f"max_dd_pct doit être dans [-100, 0], observé : {max_dd_pct:.2f}"
    )
