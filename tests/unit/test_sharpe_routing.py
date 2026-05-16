"""Tests du routing Sharpe selon la fréquence des trades — Pivot v4 A3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.metrics import sharpe_annualized
from app.config.instruments import AssetConfig

# US30 config from v4 ASSET_CONFIGS
US30 = AssetConfig(
    spread_pips=1.5,
    slippage_pips=0.3,
    pip_size=1.0,
    pip_value_eur=0.92,
    tp_points=200,
    sl_points=100,
)


def _build_trades(
    n: int, span_years: float, mean_pip: float, std_pip: float, seed: int = 42
) -> tuple[pd.Series, pd.DataFrame]:
    """Build synthetic equity + trades_df for testing Sharpe routing."""
    rng = np.random.default_rng(seed)
    pips = rng.normal(mean_pip, std_pip, n)
    total_minutes = int(span_years * 365.25 * 24 * 60)
    freq_minutes = max(total_minutes // max(n, 1), 1)
    timestamps = pd.date_range("2024-01-01", periods=n, freq=f"{freq_minutes}min")
    capital = 10_000.0
    lots = 2.17
    pnl_eur = pips * lots * 0.92
    equity = pd.Series(capital + pnl_eur.cumsum(), index=timestamps)
    trades_df = pd.DataFrame(
        {
            "Pips_Nets": pips,
            "Pips_Bruts": pips,
            "position_size_lots": [lots] * n,
        },
        index=timestamps,
    )
    return equity, trades_df


def test_daily_method_high_frequency():
    """250 trades en 1 an → method='daily', Sharpe non nul."""
    equity, trades = _build_trades(n=250, span_years=1.0, mean_pip=1.0, std_pip=10.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert method == "daily"
    assert sr != 0.0


def test_weekly_method_mid_frequency():
    """50 trades/an → method='weekly'."""
    equity, trades = _build_trades(n=50, span_years=1.0, mean_pip=1.0, std_pip=10.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert method == "weekly"
    assert sr != 0.0


def test_per_trade_method_low_frequency():
    """15 trades/an → method='per_trade'."""
    equity, trades = _build_trades(n=15, span_years=1.0, mean_pip=5.0, std_pip=20.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert method == "per_trade"
    assert sr != 0.0


def test_single_trade_returns_zero():
    """1 trade → 0.0 (pas NaN)."""
    equity, trades = _build_trades(n=1, span_years=1.0, mean_pip=1.0, std_pip=1.0)
    sr, method = sharpe_annualized(equity, trades, US30)
    assert sr == 0.0
    assert not np.isnan(sr)


def test_equity_plate_returns_zero():
    """Trades tous à 0 € → Sharpe = 0."""
    timestamps = pd.date_range("2024-01-01", periods=50, freq="D")
    equity = pd.Series([10_000.0] * 50, index=timestamps)
    trades = pd.DataFrame(
        {
            "Pips_Nets": [0.0] * 50,
            "Pips_Bruts": [0.0] * 50,
            "position_size_lots": [1.0] * 50,
        },
        index=timestamps,
    )
    sr, _ = sharpe_annualized(equity, trades, US30)
    assert sr == 0.0


def test_methods_cohere_high_frequency():
    """Sur stratégie hyper-fréquente, daily et weekly donnent des valeurs proches."""
    equity, trades = _build_trades(n=2000, span_years=1.0, mean_pip=0.5, std_pip=5.0)
    sr_daily, _ = sharpe_annualized(equity, trades, US30)
    # Forcer weekly avec 80 trades/an
    equity_w, trades_w = _build_trades(
        n=80, span_years=1.0, mean_pip=0.5, std_pip=5.0
    )
    sr_weekly, _ = sharpe_annualized(equity_w, trades_w, US30)
    # Sanity check large : les deux méthodes ne doivent pas diverger de > 5
    assert abs(sr_daily - sr_weekly) < 5.0
