"""Tests unitaires pour edge_validation.py — couvre cas nominal, dégénéré, 0 trade,
returns constants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.analysis.edge_validation import (
    _compute_sharpe_from_returns,
    _expected_max_sr,
    validate_edge,
)
from learning_machine_learning.config.backtest import BacktestConfig


@pytest.fixture
def backtest_cfg() -> BacktestConfig:
    """Configuration standard : TP=30, SL=10, commission=0.5, slippage=1.0."""
    return BacktestConfig(tp_pips=30.0, sl_pips=10.0, commission_pips=0.5, slippage_pips=1.0)


class TestComputeSharpeFromReturns:
    def test_positive_returns(self) -> None:
        returns = np.array([1.0, 2.0, 1.5, 2.5, 1.0], dtype=np.float64)
        sr = _compute_sharpe_from_returns(returns)
        assert sr > 0.0

    def test_zero_std(self) -> None:
        returns = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        sr = _compute_sharpe_from_returns(returns)
        assert sr == 0.0

    def test_single_element(self) -> None:
        returns = np.array([5.0], dtype=np.float64)
        sr = _compute_sharpe_from_returns(returns)
        assert sr == 0.0  # std=0 → Sharpe=0


class TestExpectedMaxSR:
    def test_single_trial(self) -> None:
        em = _expected_max_sr(n_trials=1, n_observations=100)
        assert em == 0.0

    def test_zero_observations(self) -> None:
        em = _expected_max_sr(n_trials=10, n_observations=0)
        assert em == 0.0

    def test_typical_case(self) -> None:
        em = _expected_max_sr(n_trials=10, n_observations=100)
        assert 0.1 < em < 1.0  # ordre de grandeur attendu


class TestValidateEdgeNominal:
    """Cas nominal : trades avec edge clairement positif."""

    def test_strong_edge(self, backtest_cfg: BacktestConfig) -> None:
        rng = np.random.default_rng(42)
        # 200 trades, moyenne +2.0 pips, écart-type 5.0 → Sharpe ≈ 0.4
        pnl = rng.normal(loc=2.0, scale=5.0, size=200)
        trades_df = pd.DataFrame({"Pips_Nets": pnl})

        result = validate_edge(trades_df, backtest_cfg, n_trials_searched=10)

        assert result["n_trades"] == 200
        # Breakeven WR = (10 + 1.5) / (10 + 30) = 11.5/40 = 28.75%
        assert result["breakeven"]["wr_pct"] == pytest.approx(28.75, abs=0.01)
        assert result["breakeven"]["observed_wr_pct"] > result["breakeven"]["wr_pct"]
        # p-value bootstrap < 0.05 (edge fort)
        assert result["bootstrap_sharpe"]["p_value_gt_0"] < 0.05
        # DSR positif
        assert not np.isnan(result["deflated_sharpe"]["dsr"])
        assert result["deflated_sharpe"]["dsr"] > 0.0
        # t-test significatif
        assert result["t_statistic"]["p_value"] < 0.05

    def test_no_edge(self, backtest_cfg: BacktestConfig) -> None:
        rng = np.random.default_rng(123)
        # 200 trades, moyenne 0.0, écart-type 5.0 → Sharpe proche de 0
        pnl = rng.normal(loc=0.0, scale=5.0, size=200)
        trades_df = pd.DataFrame({"Pips_Nets": pnl})

        result = validate_edge(trades_df, backtest_cfg, n_trials_searched=10)

        assert result["n_trades"] == 200
        # Sans edge, l'observed Sharpe doit être proche de 0 (|SR| < 0.15)
        assert abs(result["bootstrap_sharpe"]["observed"]) < 0.15
        # Le CI 95% doit contenir 0
        assert result["bootstrap_sharpe"]["ci_95_lower"] <= 0.0 <= result["bootstrap_sharpe"]["ci_95_upper"]

    def test_negative_edge(self, backtest_cfg: BacktestConfig) -> None:
        rng = np.random.default_rng(456)
        # Trades perdants en moyenne
        pnl = rng.normal(loc=-2.0, scale=5.0, size=200)
        trades_df = pd.DataFrame({"Pips_Nets": pnl})

        result = validate_edge(trades_df, backtest_cfg, n_trials_searched=10)

        assert result["n_trades"] == 200
        assert result["bootstrap_sharpe"]["observed"] < 0.0
        assert result["bootstrap_sharpe"]["p_value_gt_0"] > 0.90


class TestValidateEdgeDegenerate:
    """Distribution dégénérée : écart-type nul."""

    def test_constant_returns(self, backtest_cfg: BacktestConfig) -> None:
        pnl = np.full(100, 3.0, dtype=np.float64)  # Tous les trades gagnent 3 pips
        trades_df = pd.DataFrame({"Pips_Nets": pnl})

        result = validate_edge(trades_df, backtest_cfg, n_trials_searched=10)

        assert result["n_trades"] == 100
        assert result["bootstrap_sharpe"]["observed"] == 0.0
        # WR observé = 100%
        assert result["breakeven"]["observed_wr_pct"] == 100.0
        # t-test : std=0 → p_value=1.0
        assert result["t_statistic"]["p_value"] == 1.0


class TestValidateEdgeZeroTrades:
    """0 trade : toutes les métriques doivent être NaN."""

    def test_empty_trades(self, backtest_cfg: BacktestConfig) -> None:
        trades_df = pd.DataFrame({"Pips_Nets": pd.Series([], dtype=np.float64)})

        result = validate_edge(trades_df, backtest_cfg, n_trials_searched=10)

        assert result["n_trades"] == 0
        assert np.isnan(result["breakeven"]["wr_pct"])
        assert np.isnan(result["bootstrap_sharpe"]["observed"])
        assert np.isnan(result["deflated_sharpe"]["dsr"])
        assert np.isnan(result["t_statistic"]["t_stat"])


class TestValidateEdgeSingleTrade:
    """Cas particulier : 1 seul trade."""

    def test_single_winning_trade(self, backtest_cfg: BacktestConfig) -> None:
        trades_df = pd.DataFrame({"Pips_Nets": [30.0]})

        result = validate_edge(trades_df, backtest_cfg, n_trials_searched=10)

        assert result["n_trades"] == 1
        assert result["breakeven"]["observed_wr_pct"] == 100.0


class TestDSRWithTrials:
    """DSR doit diminuer quand n_trials augmente."""

    def test_more_trials_reduces_dsr(self, backtest_cfg: BacktestConfig) -> None:
        rng = np.random.default_rng(789)
        pnl = rng.normal(loc=1.0, scale=5.0, size=100)
        trades_df = pd.DataFrame({"Pips_Nets": pnl})

        result_1 = validate_edge(trades_df, backtest_cfg, n_trials_searched=1)
        result_50 = validate_edge(trades_df, backtest_cfg, n_trials_searched=50)

        # DSR avec 50 trials doit être plus bas qu'avec 1 trial
        assert (
            result_50["deflated_sharpe"]["dsr"]
            < result_1["deflated_sharpe"]["dsr"]
        )
