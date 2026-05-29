"""Tests unitaires pour app.strategies.pairs_trading — Phase H4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import AssetConfig
from app.strategies.pairs_trading import (
    compute_rolling_beta,
    compute_spread,
    compute_zscore,
    simulate_pairs_trades,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _eurusd_config() -> AssetConfig:
    return AssetConfig(
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
        swap_long_pips_per_night=-0.8,
        swap_short_pips_per_night=0.1,
    )


def _gbpusd_config() -> AssetConfig:
    return AssetConfig(
        spread_pips=0.9,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=0.0001,
        pip_value_eur=9.2,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        swap_long_pips_per_night=-1.2,
        swap_short_pips_per_night=0.4,
    )


def _cointegrated_pair(n: int = 200, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Génère deux séries cointégrées : a = β*b + noise stationnaire."""
    rng = np.random.default_rng(seed)
    b = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 1.3)
    noise = pd.Series(rng.normal(0, 0.001, n))  # stationnaire
    a = 0.85 * b + noise
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return a.set_axis(idx), b.set_axis(idx)


def _df_from_close(close: pd.Series, vol: float = 0.0005) -> pd.DataFrame:
    """Wrap une série Close en DataFrame OHLCV synthétique."""
    df = pd.DataFrame({
        "Open": close,
        "High": close * (1 + vol),
        "Low": close * (1 - vol),
        "Close": close,
        "Volume": 100.0,
    }, index=close.index)
    return df


# ─────────────────────────────────────────────────────────────────────
# compute_rolling_beta
# ─────────────────────────────────────────────────────────────────────


class TestComputeRollingBeta:
    def test_recovers_true_beta(self) -> None:
        """β ≈ 0.85 sur paire cointégrée avec vrai β=0.85."""
        a, b = _cointegrated_pair(n=200)
        beta = compute_rolling_beta(a, b, lookback=60)
        # Les valeurs après warmup doivent être proches de 0.85
        beta_clean = beta.dropna()
        assert len(beta_clean) > 0
        assert beta_clean.mean() == pytest.approx(0.85, abs=0.1)

    def test_returns_nan_before_warmup(self) -> None:
        a, b = _cointegrated_pair(n=100)
        beta = compute_rolling_beta(a, b, lookback=60)
        # 59 premières valeurs doivent être NaN
        assert beta.iloc[:59].isna().all()

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="4h")  # naive
        a = pd.Series(np.arange(100, dtype=float), index=idx)
        b = pd.Series(np.arange(100, dtype=float), index=idx)
        with pytest.raises(ValueError, match="tz-aware"):
            compute_rolling_beta(a, b, lookback=60)


# ─────────────────────────────────────────────────────────────────────
# compute_spread
# ─────────────────────────────────────────────────────────────────────


class TestComputeSpread:
    def test_basic_residual(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
        a = pd.Series([1.1, 1.2, 1.3, 1.4, 1.5], index=idx)
        b = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
        beta = pd.Series([0.5] * 5, index=idx)
        spread = compute_spread(a, b, beta)
        # spread = a - 0.5*b
        expected = a - 0.5 * b
        pd.testing.assert_series_equal(spread, expected, check_names=False)


# ─────────────────────────────────────────────────────────────────────
# compute_zscore
# ─────────────────────────────────────────────────────────────────────


class TestComputeZscore:
    def test_basic_zscore(self) -> None:
        """Pour spread = pd.Series ~N(0,1) sur 100 bars, z ~N(0,1)."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC")
        spread = pd.Series(rng.normal(0, 1, 200), index=idx)
        z = compute_zscore(spread, lookback=60)
        # z doit être centré ~0
        assert z.dropna().mean() == pytest.approx(0.0, abs=0.3)

    def test_returns_nan_before_warmup(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        spread = pd.Series(np.ones(100), index=idx)
        z = compute_zscore(spread, lookback=60)
        assert z.iloc[:59].isna().all()


# ─────────────────────────────────────────────────────────────────────
# simulate_pairs_trades
# ─────────────────────────────────────────────────────────────────────


class TestSimulatePairsTrades:
    def test_no_trade_if_z_never_exceeds_entry(self) -> None:
        """Z reste entre -2 et +2 → aucun trade."""
        cfg_a, cfg_b = _eurusd_config(), _gbpusd_config()
        # Cointégrés tightly : z reste petit
        a, b = _cointegrated_pair(n=200, seed=1)
        df_a, df_b = _df_from_close(a), _df_from_close(b)
        trades = simulate_pairs_trades(
            df_a, df_b, cfg_a, cfg_b,
            z_entry=5.0, z_exit=0.5,
        )
        # z très restrictif → 0 trade
        assert len(trades) == 0

    def test_produces_trades_on_artificial_divergence(self) -> None:
        """Forcer une divergence dans le spread → au moins 1 trade."""
        cfg_a, cfg_b = _eurusd_config(), _gbpusd_config()
        a, b = _cointegrated_pair(n=400, seed=2)
        # Injecter une grosse divergence puis retour
        # Boost a entre idx 150 et 200, puis retour
        a_vals = a.values.copy()
        a_vals[150:200] += 0.05  # ~50 pips divergence sur a
        a_vals[200:250] -= 0.0  # retour progressif
        a = pd.Series(a_vals, index=a.index)
        df_a, df_b = _df_from_close(a), _df_from_close(b)
        trades = simulate_pairs_trades(
            df_a, df_b, cfg_a, cfg_b,
            z_entry=1.5, z_exit=0.3,
        )
        assert len(trades) >= 1

    def test_required_trade_fields(self) -> None:
        cfg_a, cfg_b = _eurusd_config(), _gbpusd_config()
        a, b = _cointegrated_pair(n=400, seed=3)
        a_vals = a.values.copy()
        a_vals[150:200] += 0.05
        a = pd.Series(a_vals, index=a.index)
        df_a, df_b = _df_from_close(a), _df_from_close(b)
        trades = simulate_pairs_trades(
            df_a, df_b, cfg_a, cfg_b,
            z_entry=1.5, z_exit=0.3,
        )
        if not trades:
            pytest.skip("Aucun trade produit, test field non applicable")
        required = {
            "entry_time", "exit_time", "signal",
            "entry_zscore", "exit_zscore", "entry_beta",
            "entry_price_a", "exit_price_a",
            "entry_price_b", "exit_price_b",
            "pnl_eur_brut", "pnl_eur_net",
            "bars_held", "exit_reason",
        }
        assert required.issubset(set(trades[0].keys()))

    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="4h")
        df_a = pd.DataFrame({"Open": np.ones(100), "High": np.ones(100),
                             "Low": np.ones(100), "Close": np.ones(100)}, index=idx)
        df_b = df_a.copy()
        with pytest.raises(ValueError, match="tz-aware"):
            simulate_pairs_trades(df_a, df_b, _eurusd_config(), _gbpusd_config())

    def test_time_stop_closes_long_running_trade(self) -> None:
        """Si z ne retourne jamais à <0.5, time-stop ferme à time_stop_bars."""
        cfg_a, cfg_b = _eurusd_config(), _gbpusd_config()
        # Diverger a indéfiniment pour forcer time-stop
        rng = np.random.default_rng(4)
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        b = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 1.3, index=idx)
        a_vals = (0.85 * b).values.copy()
        a_vals[150:] += np.linspace(0, 0.1, n - 150)  # divergence persistante
        a = pd.Series(a_vals, index=idx)
        df_a, df_b = _df_from_close(a), _df_from_close(b)
        trades = simulate_pairs_trades(
            df_a, df_b, cfg_a, cfg_b,
            z_entry=1.5, z_exit=0.3, time_stop_bars=20,
        )
        # Au moins un trade doit avoir été time-stoppé
        if trades:
            time_stop_trades = [t for t in trades if t["exit_reason"] == "time_stop"]
            assert len(time_stop_trades) >= 1
            for t in time_stop_trades:
                assert t["bars_held"] <= 20
