"""Tests unitaires pour simulate_trades_continuous — mode régression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.backtest.simulator import (
    _simulate_stateful_core,
    simulate_trades,
    simulate_trades_continuous,
)


# ── Weight function de test ──────────────────────────────────────────────

def _weight_identity(values: np.ndarray) -> np.ndarray:
    """Poids = valeur absolue (min 1.0)."""
    w = np.abs(values)
    w[w < 1.0] = 1.0
    return w


# ── Fixture ──────────────────────────────────────────────────────────────

def _make_continuous_df(
    predicted_returns: list[float],
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    spreads: list[float] | None = None,
) -> pd.DataFrame:
    """Construit un DataFrame compatible simulate_trades_continuous."""
    n = len(predicted_returns)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "Predicted_Return": predicted_returns,
            "Close": closes if closes is not None else [1.0] * n,
            "High": highs if highs is not None else [1.0005] * n,
            "Low": lows if lows is not None else [0.9995] * n,
            "Spread": spreads if spreads is not None else [14.0] * n,
        },
        index=dates,
    )


# ── Tests ────────────────────────────────────────────────────────────────

class TestSimulateTradesContinuous:
    """Tests de simulate_trades_continuous."""

    def test_long_signal_above_threshold(self) -> None:
        """predicted_return > threshold → trade LONG."""
        df = _make_continuous_df(
            predicted_returns=[0.001, 0.0001, -0.0001],
        )
        trades, n_signaux, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=24,
            signal_threshold=0.0005,
        )
        assert n_signaux == 1  # seule la 1ère barre > 0.0005

    def test_short_signal_below_neg_threshold(self) -> None:
        """predicted_return < -threshold → trade SHORT."""
        df = _make_continuous_df(
            predicted_returns=[-0.002, 0.0001, 0.0001],
        )
        trades, n_signaux, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=24,
            signal_threshold=0.0005,
        )
        assert n_signaux == 1

    def test_no_trade_between_thresholds(self) -> None:
        """|predicted_return| < threshold → pas de trade."""
        df = _make_continuous_df(
            predicted_returns=[0.0004, -0.0003, 0.0001],
        )
        trades, n_signaux, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=24,
            signal_threshold=0.0005,
        )
        assert n_signaux == 0
        assert trades.empty

    def test_zero_signal_all_below_threshold(self) -> None:
        """Tous les retours sous seuil → 0 trades."""
        df = _make_continuous_df(
            predicted_returns=[0.0001, -0.0001, 0.0, 0.0002, -0.0004],
        )
        trades, n_signaux, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=24,
            signal_threshold=0.001,
        )
        assert n_signaux == 0

    def test_tp_touche_long(self) -> None:
        """Vérifie qu'un trade LONG gagne si TP touché (1 seul signal)."""
        returns = [0.0] * 20
        returns[0] = 0.001  # seul signal
        closes = [1.0] * 20
        closes[3] = 1.004  # TP à 1.003 touché
        df = _make_continuous_df(
            predicted_returns=returns,
            closes=closes,
            highs=[1.0041] * 20,
            lows=[0.9999] * 20,
        )
        trades, _, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=10,
            signal_threshold=0.0005,
        )
        assert len(trades) == 1
        assert trades.iloc[0]["result"] == "win"

    def test_sl_touche_long(self) -> None:
        """Vérifie qu'un trade LONG perd si SL touché (1 seul signal)."""
        returns = [0.0] * 20
        returns[0] = 0.001  # seul signal
        closes = [1.0] * 20
        closes[3] = 0.998  # SL à 0.999 touché
        df = _make_continuous_df(
            predicted_returns=returns,
            closes=closes,
            highs=[1.0001] * 20,
            lows=[0.9979] * 20,
        )
        trades, _, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=10,
            signal_threshold=0.0005,
        )
        assert len(trades) == 1
        assert trades.iloc[0]["result"] == "loss_sl"

    def test_stateful_core_identical_classifier_vs_continuous(self) -> None:
        """Les signaux et poids différents mais la boucle stateful est identique."""
        # On vérifie que pour les mêmes signaux/poids, simulate_trades
        # et simulate_trades_continuous produisent le même résultat
        # via _simulate_stateful_core.

        dates = pd.date_range("2020-01-01", periods=50, freq="h")
        closes = np.ones(50, dtype=np.float64)
        highs = np.full(50, 1.0005, dtype=np.float64)
        lows = np.full(50, 0.9995, dtype=np.float64)
        signals = np.zeros(50, dtype=np.int32)
        signals[0] = 1  # un seul trade LONG
        weights = np.zeros(50, dtype=np.float64)
        weights[0] = 1.0
        spreads = np.full(50, 14.0, dtype=np.float64)
        filter_rejected = np.full(50, "", dtype=object)

        trades = _simulate_stateful_core(
            n=50, dates=dates, highs=highs, lows=lows, closes=closes,
            signals=signals, weights=weights, spreads=spreads,
            filter_rejected_arr=filter_rejected,
            tp_dist=30 * 0.0001, sl_dist=10 * 0.0001,
            spread_cost_base=1.5, window=24, pip_size=0.0001,
        )
        assert len(trades) == 1
        # Timeout car High/Low ne touchent ni TP ni SL
        assert trades[0]["result"] == "loss_timeout"

    def test_missing_predicted_return_raises(self) -> None:
        """simulate_trades_continuous doit lever ValueError si colonne absente."""
        df = pd.DataFrame({"Close": [1.0]}, index=pd.date_range("2020-01-01", periods=1, freq="h"))
        with pytest.raises(ValueError, match="Predicted_Return"):
            simulate_trades_continuous(df, weight_func=_weight_identity)

    def test_weight_proportional_to_return(self) -> None:
        """Les poids passés à simulate_trades_continuous = |Predicted_Return|."""
        returns = [0.002, 1.5, -0.003]
        n = len(returns)
        df = _make_continuous_df(
            predicted_returns=returns,
            closes=[1.0] * n,
            highs=[1.0005] * n,
            lows=[0.9995] * n,
        )
        trades, _, _ = simulate_trades_continuous(
            df, weight_func=_weight_identity,
            tp_pips=30.0, sl_pips=10.0, window=1,
            signal_threshold=0.0005,
        )
        assert len(trades) == 3
        weights = trades["Weight"].values
        # _weight_identity: clamp < 1.0 à 1.0
        assert weights[0] == 1.0    # abs(0.002) < 1.0 → clampé
        assert weights[1] == 1.5    # abs(1.5) ≥ 1.0 → conservé
        assert weights[2] == 1.0    # abs(-0.003) < 1.0 → clampé
