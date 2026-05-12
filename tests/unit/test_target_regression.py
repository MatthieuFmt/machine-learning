"""Tests unitaires des 3 nouvelles fonctions cible — forward_return, directional_clean, cost_aware_v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.triple_barrier import (
    compute_cost_aware_target_v2,
    compute_directional_clean_target,
    compute_forward_return_target,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_ohlcv(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Construit un DataFrame OHLC synthétique."""
    n = len(closes)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "Close": closes,
            "High": highs if highs is not None else closes,
            "Low": lows if lows is not None else closes,
        },
        index=dates,
    )


# ── compute_forward_return_target ────────────────────────────────────────

class TestForwardReturnTarget:
    """Tests de compute_forward_return_target."""

    def test_basic_up_move(self) -> None:
        """Hausse de ~1% → log return ≈ 0.00995."""
        closes = [1.0, 1.01]
        df = _make_ohlcv(closes)
        result = compute_forward_return_target(df, horizon_hours=1)
        expected = np.array([np.log(1.01 / 1.0), np.nan], dtype=np.float64)
        np.testing.assert_allclose(result[:-1], expected[:-1], rtol=1e-5)
        assert np.isnan(result[-1])

    def test_basic_down_move(self) -> None:
        """Baisse de 1% → log return ≈ -0.01005."""
        closes = [1.0, 0.99]
        df = _make_ohlcv(closes)
        result = compute_forward_return_target(df, horizon_hours=1)
        np.testing.assert_allclose(result[0], np.log(0.99), rtol=1e-5)

    def test_last_horizon_nan(self) -> None:
        """Les `horizon` dernières barres doivent être NaN."""
        closes = list(range(1, 11))
        df = _make_ohlcv([float(c) for c in closes])
        horizon = 3
        result = compute_forward_return_target(df, horizon_hours=horizon)
        assert not np.isnan(result[0])
        assert all(np.isnan(result[-horizon:]))

    def test_horizon_equals_length(self) -> None:
        """Si horizon >= n, tout est NaN."""
        df = _make_ohlcv([1.0, 2.0, 3.0])
        result = compute_forward_return_target(df, horizon_hours=3)
        assert all(np.isnan(result))

    def test_vectorized_matches_loop(self) -> None:
        """Cohérence avec un calcul boucle Python naïf."""
        closes = np.array([1.0, 1.02, 0.98, 1.05, 1.03], dtype=np.float64)
        df = _make_ohlcv(closes.tolist())
        horizon = 2
        result = compute_forward_return_target(df, horizon_hours=horizon)
        # Calcul manuel
        expected = np.full(len(closes), np.nan, dtype=np.float64)
        for i in range(len(closes) - horizon):
            expected[i] = np.log(closes[i + horizon] / closes[i])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_invalid_horizon_raises(self) -> None:
        """horizon_hours=0 doit lever ValueError."""
        df = _make_ohlcv([1.0, 2.0])
        with pytest.raises(ValueError):
            compute_forward_return_target(df, horizon_hours=0)


# ── compute_directional_clean_target ──────────────────────────────────────

class TestDirectionalCleanTarget:
    """Tests de compute_directional_clean_target."""

    def test_noise_below_threshold_yields_zero(self) -> None:
        """Petit mouvement sous le seuil de bruit ATR → 0."""
        n = 50
        closes = [1.0 + 0.00001 * i for i in range(n)]
        highs = [1.0002 + 0.00001 * i for i in range(n)]
        lows = [0.9998 + 0.00001 * i for i in range(n)]
        df = _make_ohlcv(closes, highs, lows)
        result = compute_directional_clean_target(
            df, horizon_hours=1, noise_threshold_atr=5.0, atr_period=14,
        )
        # Avec un seuil très élevé, aucun signal ne passe
        valid = result[~np.isnan(result)]
        assert all(valid == 0.0)

    def test_signal_above_threshold(self) -> None:
        """Grand mouvement au-dessus du seuil → ±1."""
        n = 50
        # Gros saut à la barre 20
        closes = [1.0] * n
        closes[21] = 1.005  # +0.5% à la barre suivant l'entrée
        highs = [1.0005] * n
        lows = [0.9995] * n
        df = _make_ohlcv(closes, highs, lows)
        result = compute_directional_clean_target(
            df, horizon_hours=1, noise_threshold_atr=0.1, atr_period=14,
        )
        # La barre 20 doit avoir signal = 1 (hausse significative)
        assert result[20] == 1.0

    def test_short_signal(self) -> None:
        """Baisse au-dessus du seuil → -1."""
        n = 50
        closes = [1.0] * n
        closes[21] = 0.995  # -0.5%
        highs = [1.0005] * n
        lows = [0.9995] * n
        df = _make_ohlcv(closes, highs, lows)
        result = compute_directional_clean_target(
            df, horizon_hours=1, noise_threshold_atr=0.1, atr_period=14,
        )
        assert result[20] == -1.0

    def test_last_bars_nan(self) -> None:
        """Les `horizon_hours` dernières barres doivent être NaN (forward return)."""
        closes = [1.0 + 0.0001 * i for i in range(30)]
        highs = [1.0005 + 0.0001 * i for i in range(30)]
        lows = [0.9995 + 0.0001 * i for i in range(30)]
        df = _make_ohlcv(closes, highs, lows)
        horizon = 3
        result = compute_directional_clean_target(
            df, horizon_hours=horizon, atr_period=14,
        )
        # Les `horizon_hours` dernières barres n'ont pas de forward return → NaN
        assert all(np.isnan(result[-horizon:]))
        # Les barres avant horizon doivent être valides (après warmup ATR)
        # Note: les atr_period premières barres peuvent être NaN (warmup ATR)
        valid_mid = result[14:-horizon]
        assert not np.isnan(valid_mid).any()


# ── compute_cost_aware_target_v2 ─────────────────────────────────────────

class TestCostAwareTargetV2:
    """Tests de compute_cost_aware_target_v2."""

    def test_tp_profitable_long(self) -> None:
        """TP touché, PnL > k_atr * ATR → +1 (après warmup ATR)."""
        closes = [1.0] * 50
        closes[5] = 1.004  # TP touché à +0.004
        highs = [1.0041] * 50
        lows = [0.9999] * 50
        df = _make_ohlcv(closes, highs, lows)
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=24,
            friction_pips=1.5, k_atr=0.01,
        )
        # Index 0 = NaN (warmup ATR), index 14 = valide (ATR stable)
        assert result[14] == 1.0

    def test_sl_hit_yields_minus_one(self) -> None:
        """SL touché → -1 (après warmup ATR)."""
        closes = [1.0] * 50
        closes[5] = 0.998  # SL touché
        highs = [1.0001] * 50
        lows = [0.9979] * 50
        df = _make_ohlcv(closes, highs, lows)
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=24,
            friction_pips=1.5, k_atr=0.01,
        )
        assert result[14] == -1.0

    def test_timeout_zero(self) -> None:
        """Timeout sans toucher TP/SL → 0."""
        closes = [1.0] * 50
        # Aucun High/Low suffisant pour toucher TP ou SL
        highs = [1.001] * 50  # TP à 1.003 pas touché
        lows = [0.999] * 50   # SL à 0.999 pas touché
        df = _make_ohlcv(closes, highs, lows)
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=5,
            friction_pips=1.5, k_atr=0.01,
        )
        assert result[0] == 0.0

    def test_last_window_nan(self) -> None:
        """Les `window` dernières barres sont NaN."""
        n = 30
        closes = [1.0] * n
        highs = [1.0002] * n
        lows = [0.9998] * n
        df = _make_ohlcv(closes, highs, lows)
        window = 5
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=window,
        )
        assert all(np.isnan(result[-window:]))
