"""Tests unitaires pour app.strategies.crypto_trend.tsmom_daily_returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.crypto_trend import tsmom_daily_returns


def _crypto_config(swap_long: float = -16.0, total_cost: float = 60.0) -> AssetConfig:
    # pip_size=1 (style BTC), coûts/ swap paramétrables pour les tests.
    return AssetConfig(
        spread_pips=total_cost / 2.0,
        slippage_pips=total_cost / 2.0,
        commission_pips=0.0,
        pip_size=1.0,
        pip_value_eur=0.92,
        tp_points=2000,
        sl_points=1000,
        window_hours=120,
        swap_long_pips_per_night=swap_long,
        swap_short_pips_per_night=-2.0,
    )


def _series(vals: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.DataFrame({"Open": vals, "High": vals, "Low": vals, "Close": vals},
                        index=idx)


class TestTsmomDailyReturns:
    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=10, freq="D")  # naive
        df = pd.DataFrame({"Close": np.arange(1, 11, dtype=float)}, index=idx)
        with pytest.raises(ValueError, match="tz-aware"):
            tsmom_daily_returns(df, _crypto_config(), lookback=5)

    def test_uptrend_goes_long(self) -> None:
        # Prix strictement croissants → tendance up → position +1 après warmup.
        df = _series([100 * (1.01 ** i) for i in range(40)])
        net, gross, position = tsmom_daily_returns(
            _df := df, _crypto_config(swap_long=0.0, total_cost=0.0), lookback=5
        )
        assert (position.iloc[10:] == 1).all()           # long en tendance up
        assert gross.iloc[10:].mean() > 0                # gross positif

    def test_downtrend_goes_short_and_profits(self) -> None:
        # Prix décroissants → short → gross positif (on gagne en baisse).
        df = _series([100 * (0.99 ** i) for i in range(40)])
        net, gross, position = tsmom_daily_returns(
            df, _crypto_config(swap_long=0.0, total_cost=0.0), lookback=5
        )
        assert (position.iloc[10:] == -1).all()
        assert gross.iloc[10:].mean() > 0

    def test_no_look_ahead(self) -> None:
        """La position en t ne dépend que des prix jusqu'à t-1."""
        rng = np.random.default_rng(0)
        prices = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 60)))
        df1 = _series(prices)
        _, _, pos1 = tsmom_daily_returns(df1, _crypto_config(), lookback=10)
        k = 40
        prices2 = list(prices)
        prices2[k] *= 1.5  # choc en k
        df2 = _series(prices2)
        _, _, pos2 = tsmom_daily_returns(df2, _crypto_config(), lookback=10)
        # Les positions jusqu'à k inclus ne bougent pas (position[k] utilise ≤ k-1).
        pd.testing.assert_series_equal(pos1.iloc[: k + 1], pos2.iloc[: k + 1])

    def test_swap_reduces_net_vs_gross(self) -> None:
        # Tendance up, swap long négatif → net < gross sur les jours détenus.
        df = _series([100 * (1.01 ** i) for i in range(40)])
        net, gross, position = tsmom_daily_returns(
            df, _crypto_config(swap_long=-16.0, total_cost=0.0), lookback=5
        )
        held = position != 0
        assert net[held].sum() < gross[held].sum()

    def test_flip_costs_charged_on_reversal(self) -> None:
        # V : baisse puis hausse → 1 retournement short→long → coût ponctuel.
        down = [100 * (0.99 ** i) for i in range(20)]
        up = [down[-1] * (1.01 ** i) for i in range(1, 21)]
        df = _series(down + up)
        net_costly, _, _ = tsmom_daily_returns(df, _crypto_config(total_cost=60.0, swap_long=0.0), lookback=5)
        net_free, _, _ = tsmom_daily_returns(df, _crypto_config(total_cost=0.0, swap_long=0.0), lookback=5)
        # Avec coûts, le rendement cumulé est inférieur (au moins un flip payé).
        assert net_costly.sum() < net_free.sum()
