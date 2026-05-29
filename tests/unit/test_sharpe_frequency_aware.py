"""Tests pour le fix E4 — Sharpe annualisé routé par fréquence.

Le resample daily + ffill insère des jours à retour nul (jours sans trade) qui
écrasent la volatilité et GONFLENT le Sharpe pour les stratégies basse-fréquence.
`sharpe_daily_from_trades(frequency_aware=True)` route l'annualisation :
    ≥ 100 trades/an → daily, 30-99 → weekly, < 30 → per-trade × √(trades/an).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import sharpe_daily_from_trades


def _trades(pnls: list[float], dates: list[str]) -> list[dict]:
    return [{"pips_net": p, "exit_time": d} for p, d in zip(pnls, dates, strict=True)]


def test_low_frequency_routes_per_trade() -> None:
    """Stratégie basse-fréquence (< 30 trades/an) → annualisation per-trade.

    12 trades étalés sur ~3 ans (≈ 4 trades/an). Le Sharpe fréquence-aware doit
    valoir EXACTEMENT mean/std × √(trades/an) — pas le daily-ffill × √252 qui
    s'appuie sur des centaines de jours à retour nul fantômes (E4).
    """
    dates = [f"{year}-{month:02d}-15" for year in (2021, 2022, 2023) for month in (1, 4, 7, 10)]
    pnls = [20.0, -10.0, 20.0, 20.0, -10.0, 20.0, 20.0, -10.0, 20.0, 20.0, -10.0, 20.0]
    trades = _trades(pnls, dates)
    capital = 10_000.0

    aware = sharpe_daily_from_trades(trades, initial_capital_pips=capital, frequency_aware=True)

    # Reproduction indépendante de la formule per-trade.
    per_trade = np.array(pnls) / capital
    et = pd.to_datetime(dates)
    years = (et.max() - et.min()).total_seconds() / (365.25 * 86400)
    tpy = len(pnls) / years
    expected = per_trade.mean() / np.std(per_trade) * np.sqrt(tpy)

    assert aware == pytest.approx(expected, rel=1e-9)
    # Le routage change bien le résultat vs daily-ffill pur.
    legacy_daily = sharpe_daily_from_trades(
        trades, initial_capital_pips=capital, frequency_aware=False
    )
    assert aware != pytest.approx(legacy_daily, rel=1e-3)


def test_high_frequency_routes_daily_unchanged() -> None:
    """Haute fréquence (≥100 trades/an) : route daily, identique au legacy."""
    dates = pd.date_range("2024-01-01", periods=250, freq="D").strftime("%Y-%m-%d").tolist()
    pnls = [20.0 if i % 3 else -10.0 for i in range(250)]
    trades = _trades(pnls, dates)

    aware = sharpe_daily_from_trades(trades, frequency_aware=True)
    legacy = sharpe_daily_from_trades(trades, frequency_aware=False)

    assert aware == legacy  # même chemin (daily √252)


def test_empty_and_all_loss_return_zero() -> None:
    """Garde-fous : pas de trades ou que des pertes → 0.0."""
    assert sharpe_daily_from_trades([]) == 0.0
    losing = _trades([-10.0, -10.0, -10.0], ["2024-01-01", "2024-06-01", "2024-12-01"])
    assert sharpe_daily_from_trades(losing) == 0.0
