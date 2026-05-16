"""Tests unitaires pour backtest.metrics — sharpe_ratio, max_drawdown, buy_and_hold_pips, compute_metrics."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.backtest.metrics import (
    sharpe_ratio,
    max_drawdown,
    buy_and_hold_pips,
    compute_metrics,
    _pips_to_return,
)


# ── _pips_to_return ───────────────────────────────────────

class TestPipsToReturn:
    def test_positive_pips(self) -> None:
        assert _pips_to_return(100.0, pip_value_eur=1.0, initial_capital=10_000) == 0.01

    def test_negative_pips(self) -> None:
        assert _pips_to_return(-200.0, pip_value_eur=1.0, initial_capital=10_000) == -0.02

    def test_array(self) -> None:
        result = _pips_to_return(np.array([100, 200, -50]), initial_capital=10_000)
        expected = np.array([0.01, 0.02, -0.005])
        np.testing.assert_array_almost_equal(result, expected)

    def test_custom_pip_value(self) -> None:
        assert _pips_to_return(100.0, pip_value_eur=10.0, initial_capital=10_000) == 0.1


# ── sharpe_ratio ──────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_sharpe(self) -> None:
        """Returns strictement positifs → Sharpe > 0."""
        returns = np.array([0.01, 0.02, 0.015, 0.03, 0.01])
        assert sharpe_ratio(returns) > 0

    def test_negative_sharpe(self) -> None:
        """Returns strictement négatifs → Sharpe < 0."""
        returns = np.array([-0.01, -0.02, -0.015])
        assert sharpe_ratio(returns) < 0

    def test_flat_returns_zero(self) -> None:
        """Returns constants → Sharpe = 0 (volatilité nulle)."""
        assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0

    def test_insufficient_data(self) -> None:
        """Moins de 2 observations → Sharpe = 0."""
        assert sharpe_ratio(np.array([0.01])) == 0.0
        assert sharpe_ratio(np.array([])) == 0.0

    def test_annualization(self) -> None:
        """Vérifie le facteur sqrt(annual_factor)."""
        returns = np.array([0.01, -0.01, 0.02, -0.005])
        s1 = sharpe_ratio(returns, annual_factor=1.0)
        s252 = sharpe_ratio(returns, annual_factor=252.0)
        assert pytest.approx(s252) == s1 * np.sqrt(252)

    def test_zero_mean(self) -> None:
        """Returns centrés → Sharpe ≈ 0."""
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        result = sharpe_ratio(returns)
        assert abs(result) < 1e-10

    def test_pandas_series(self) -> None:
        """Fonctionne avec pd.Series."""
        returns = pd.Series([0.01, 0.02, 0.015, -0.01, 0.005])
        assert isinstance(sharpe_ratio(returns), float)


# ── max_drawdown ──────────────────────────────────────────

class TestMaxDrawdown:
    def test_empty(self) -> None:
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_no_drawdown(self) -> None:
        """PnL monotone croissant → DD = 0."""
        pnl = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert max_drawdown(pnl) == 0.0

    def test_simple_drawdown(self) -> None:
        """Pic à 10, descente à 5 → DD = -5."""
        pnl = pd.Series([0.0, 10.0, 8.0, 5.0, 7.0])
        assert max_drawdown(pnl) == -5.0

    def test_multiple_drawdowns(self) -> None:
        """Deux drawdowns : -3 puis -7 → DD max = -7."""
        pnl = pd.Series([0.0, 5.0, 2.0, 3.0, 12.0, 5.0, 8.0])
        assert max_drawdown(pnl) == -7.0

    def test_all_negative(self) -> None:
        """PnL constamment négatif."""
        pnl = pd.Series([0.0, -1.0, -3.0, -6.0])
        assert max_drawdown(pnl) == -6.0

    def test_float_return(self) -> None:
        assert isinstance(max_drawdown(pd.Series([1.0, 2.0, 0.5])), float)


# ── buy_and_hold_pips ─────────────────────────────────────

class TestBuyAndHoldPips:
    def test_positive_bh(self) -> None:
        df = pd.DataFrame({"Close": [1.1000, 1.1050]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
        assert buy_and_hold_pips(df, pip_size=0.0001) == pytest.approx(50.0)

    def test_negative_bh(self) -> None:
        df = pd.DataFrame({"Close": [1.1000, 1.0950]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
        assert buy_and_hold_pips(df, pip_size=0.0001) == pytest.approx(-50.0)

    def test_empty_df(self) -> None:
        assert buy_and_hold_pips(pd.DataFrame(), pip_size=0.0001) == 0.0

    def test_none_df(self) -> None:
        assert buy_and_hold_pips(None, pip_size=0.0001) == 0.0

    def test_single_row(self) -> None:
        df = pd.DataFrame({"Close": [1.1000]}, index=pd.date_range("2024-01-01", periods=1, freq="D"))
        assert buy_and_hold_pips(df) == 0.0

    def test_custom_pip_size(self) -> None:
        """BTC-like : pip_size = 1.0."""
        df = pd.DataFrame({"Close": [50000.0, 51000.0]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
        assert buy_and_hold_pips(df, pip_size=1.0) == 1000.0


# ── compute_metrics ───────────────────────────────────────

class TestComputeMetrics:
    def test_empty_trades(self) -> None:
        """DataFrame vide → métriques par défaut."""
        empty = pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "Weight", "result"])
        metrics = compute_metrics(empty)
        assert metrics["trades"] == 0
        assert metrics["profit_net"] == 0.0
        assert metrics["win_rate"] == 0.0
        assert metrics["sharpe"] == 0.0

    def test_basic_trades(self, trades_synthetic: pd.DataFrame) -> None:
        """Vérifie les métriques de base sur les trades synthétiques."""
        metrics = compute_metrics(trades_synthetic)
        assert metrics["trades"] == 10
        assert metrics["profit_net"] > 0  # somme positive
        assert 0 < metrics["win_rate"] < 100
        assert metrics["dd"] <= 0  # drawdown ≤ 0

    def test_all_wins(self) -> None:
        """100% de win rate — trades étalés sur plusieurs jours pour Sharpe > 0."""
        trades = pd.DataFrame(
            {
                "Pips_Nets": [2.0, 3.0, 1.5, 4.0, 2.5],
                "Pips_Bruts": [2.0, 3.0, 1.5, 4.0, 2.5],
                "Weight": [1.0] * 5,
                "result": ["win"] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        metrics = compute_metrics(trades)
        assert metrics["win_rate"] == 100.0
        assert metrics["dd"] == 0.0
        assert metrics["sharpe"] > 0

    def test_all_losses(self) -> None:
        """0% de win rate."""
        trades = pd.DataFrame(
            {
                "Pips_Nets": [-1.0, -2.0, -1.5],
                "Pips_Bruts": [-1.0, -2.0, -1.5],
                "Weight": [1.0] * 3,
                "result": ["loss_sl"] * 3,
            },
            index=pd.date_range("2024-01-01", periods=3, freq="3h"),
        )
        metrics = compute_metrics(trades)
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_net"] < 0

    def test_bh_benchmark(
        self, ohlcv_h1_synthetic: pd.DataFrame, trades_synthetic: pd.DataFrame
    ) -> None:
        """Avec df fourni, calcule le benchmark B&H."""
        metrics = compute_metrics(trades_synthetic, df=ohlcv_h1_synthetic)
        assert "bh_pips" in metrics
        assert "alpha_pips" in metrics
        assert metrics["alpha_pips"] == metrics["profit_net"] - metrics["bh_pips"]

    def test_annee_preserved(self) -> None:
        """L'année est passée dans les métriques."""
        empty = pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "Weight", "result"])
        metrics = compute_metrics(empty, annee=2024)
        assert metrics["annee"] == 2024

    def test_dict_keys_complete(self, trades_synthetic: pd.DataFrame) -> None:
        """Toutes les clés attendues sont présentes."""
        metrics = compute_metrics(trades_synthetic)
        expected_keys = {
            "annee", "profit_net", "dd", "trades", "win_rate",
            "sharpe", "sharpe_method", "sharpe_per_trade",
            "total_return_pct", "max_dd_pct",
            "bh_pips", "bh_return_pct", "alpha_pips", "alpha_return_pct",
        }
        assert set(metrics.keys()) == expected_keys

    def test_sharpe_per_trade_single_trade(self) -> None:
        """Sharpe per-trade = 0 pour un seul trade."""
        trades = pd.DataFrame(
            {"Pips_Nets": [3.0], "Pips_Bruts": [3.0], "Weight": [1.0], "result": ["win"]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        metrics = compute_metrics(trades)
        assert metrics["sharpe_per_trade"] == 0.0


