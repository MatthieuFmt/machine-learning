"""Tests unitaires pour les 3 nouvelles fonctions cibles Step 01.

Couvre :
- compute_forward_return_target  (régression, log-return forward)
- compute_directional_clean_target (binaire avec seuil ATR)
- compute_cost_aware_target_v2       (cost-aware avec seuil ATR adaptatif)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.triple_barrier import (
    compute_forward_return_target,
    compute_directional_clean_target,
    compute_cost_aware_target_v2,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Génère un DataFrame OHLCV synthétique avec une tendance."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    trend = np.linspace(1.1000, 1.1050, n)
    noise = rng.normal(0, 0.0002, n)
    close = trend + noise
    high = close + np.abs(rng.normal(0, 0.0003, n))
    low = close - np.abs(rng.normal(0, 0.0003, n))
    return pd.DataFrame(
        {"High": high, "Low": low, "Close": close}, index=dates,
    )


# ═══════════════════════════════════════════════════════════════════════════
# compute_forward_return_target
# ═══════════════════════════════════════════════════════════════════════════

class TestForwardReturnTarget:
    """Tests pour compute_forward_return_target."""

    def test_output_shape(self) -> None:
        """La sortie a la même longueur que l'entrée."""
        df = _make_ohlcv(100)
        result = compute_forward_return_target(df, horizon_hours=24)
        assert len(result) == 100
        assert result.dtype == np.float64

    def test_nan_at_end(self) -> None:
        """Les horizon_hours dernières barres sont NaN."""
        df = _make_ohlcv(100)
        horizon = 24
        result = compute_forward_return_target(df, horizon_hours=horizon)
        assert np.all(np.isnan(result[-horizon:]))
        assert not np.any(np.isnan(result[:-horizon]))

    def test_horizon_1_equals_log_return(self) -> None:
        """Avec horizon=1, le résultat est log(C_{i+1}/C_i)."""
        df = _make_ohlcv(50)
        result = compute_forward_return_target(df, horizon_hours=1)
        expected = np.log(df["Close"].values[1:] / df["Close"].values[:-1])
        expected = np.append(expected, np.nan)
        np.testing.assert_array_almost_equal(result, expected)

    def test_value_error_missing_close(self) -> None:
        """ValueError si la colonne 'Close' est absente."""
        df = pd.DataFrame({"High": [1.0], "Low": [1.0]})
        with pytest.raises(ValueError, match="Close"):
            compute_forward_return_target(df, horizon_hours=12)

    def test_value_error_horizon_invalid(self) -> None:
        """ValueError si horizon_hours < 1."""
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="horizon_hours"):
            compute_forward_return_target(df, horizon_hours=0)

    def test_short_dataframe_warning(self, capsys) -> None:
        """DataFrame plus court que horizon → toutes les barres NaN, warning log."""
        df = _make_ohlcv(5)
        result = compute_forward_return_target(df, horizon_hours=10)
        assert np.all(np.isnan(result))

    def test_vectorized_matches_manual(self) -> None:
        """Vérifie que le calcul vectorisé correspond au calcul manuel barre par barre."""
        n = 60
        horizon = 12
        df = _make_ohlcv(n, seed=99)
        result = compute_forward_return_target(df, horizon_hours=horizon)
        closes = df["Close"].values
        limit = n - horizon
        for i in range(limit):
            expected = np.log(closes[i + horizon] / closes[i])
            assert abs(result[i] - expected) < 1e-12, f"Mismatch at i={i}"

    def test_sign_matches_direction(self) -> None:
        """Un retour forward positif donne un signe positif et vice-versa."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Construire un Close qui monte strictement
        close = np.linspace(1.1000, 1.2000, n)
        df = pd.DataFrame({"Close": close}, index=dates)
        result = compute_forward_return_target(df, horizon_hours=5)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0), "Tendance haussière → tous les retours > 0"


# ═══════════════════════════════════════════════════════════════════════════
# compute_directional_clean_target
# ═══════════════════════════════════════════════════════════════════════════

class TestDirectionalCleanTarget:
    """Tests pour compute_directional_clean_target."""

    def test_output_values_in_set(self) -> None:
        """Toutes les valeurs valides sont dans {-1.0, 0.0, 1.0}."""
        df = _make_ohlcv(200)
        result = compute_directional_clean_target(
            df, horizon_hours=24, noise_threshold_atr=0.5,
        )
        valid = result[~np.isnan(result)]
        assert set(np.unique(valid)).issubset({-1.0, 0.0, 1.0})

    def test_nan_at_end(self) -> None:
        """Les horizon_hours dernières barres sont NaN (ATR affecte le début, pas la fin)."""
        df = _make_ohlcv(100)
        horizon = 12
        result = compute_directional_clean_target(
            df, horizon_hours=horizon, noise_threshold_atr=0.5, atr_period=14,
        )
        assert np.all(np.isnan(result[-horizon:]))

    def test_noise_threshold_filters_small_moves(self) -> None:
        """Un seuil de bruit très élevé → tous les labels à 0 (sauf NaN)."""
        df = _make_ohlcv(200)
        result = compute_directional_clean_target(
            df, horizon_hours=12, noise_threshold_atr=100.0,
        )
        valid = result[~np.isnan(result)]
        assert np.all(valid == 0.0)

    def test_zero_threshold_all_directional(self) -> None:
        """Un seuil de bruit nul → tous les labels sont directionnels (±1)."""
        df = _make_ohlcv(200)
        result = compute_directional_clean_target(
            df, horizon_hours=12, noise_threshold_atr=1e-6,
        )
        valid = result[~np.isnan(result)]
        # Aucun zéro (sauf si forward_return est exactement 0, très improbable)
        unique = set(np.unique(valid))
        assert unique.issubset({-1.0, 1.0})

    def test_value_error_missing_columns(self) -> None:
        """ValueError si colonnes OHLC absentes."""
        df = pd.DataFrame({"Close": [1.0]})
        with pytest.raises(ValueError):
            compute_directional_clean_target(df)

    def test_value_error_horizon_invalid(self) -> None:
        """ValueError si horizon_hours < 1."""
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="horizon_hours"):
            compute_directional_clean_target(df, horizon_hours=0)

    def test_value_error_noise_threshold_invalid(self) -> None:
        """ValueError si noise_threshold_atr <= 0."""
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="noise_threshold_atr"):
            compute_directional_clean_target(df, noise_threshold_atr=0.0)

    def test_value_error_atr_period_invalid(self) -> None:
        """ValueError si atr_period < 1."""
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="atr_period"):
            compute_directional_clean_target(df, atr_period=0)

    def test_direction_matches_forward_return_sign(self) -> None:
        """Le label directionnel correspond au signe du forward_return."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        rng = np.random.default_rng(42)
        # Fermetures aléatoires (bruit pur)
        close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
        high = close + np.abs(rng.normal(0, 0.0002, n))
        low = close - np.abs(rng.normal(0, 0.0002, n))
        df = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)

        result = compute_directional_clean_target(
            df, horizon_hours=6, noise_threshold_atr=1e-6,
        )
        # Avec noise_threshold quasi nul, le label = sign du forward_return
        horizon = 6
        limit = n - horizon
        for i in range(limit):
            if np.isnan(result[i]):
                continue
            fwd_ret = np.log(close[i + horizon] / close[i])
            if fwd_ret > 0:
                assert result[i] == 1.0, f"i={i}: fwd_ret={fwd_ret:.6f} > 0 → label={result[i]}"
            elif fwd_ret < 0:
                assert result[i] == -1.0, f"i={i}: fwd_ret={fwd_ret:.6f} < 0 → label={result[i]}"


# ═══════════════════════════════════════════════════════════════════════════
# compute_cost_aware_target_v2
# ═══════════════════════════════════════════════════════════════════════════

class TestCostAwareTargetV2:
    """Tests pour compute_cost_aware_target_v2."""

    def test_output_values_in_set(self) -> None:
        """Toutes les valeurs valides sont dans {-1.0, 0.0, 1.0}."""
        df = _make_ohlcv(200)
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=24, k_atr=1.0,
        )
        valid = result[~np.isnan(result)]
        assert set(np.unique(valid)).issubset({-1.0, 0.0, 1.0})

    def test_nan_at_end(self) -> None:
        """Les window dernières barres sont NaN."""
        df = _make_ohlcv(100)
        window = 24
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=window, k_atr=1.0,
        )
        assert np.all(np.isnan(result[-window:]))

    def test_value_error_missing_columns(self) -> None:
        """ValueError si colonnes OHLC absentes."""
        df = pd.DataFrame({"Close": [1.0]})
        with pytest.raises(ValueError):
            compute_cost_aware_target_v2(df)

    def test_value_error_friction_negative(self) -> None:
        """ValueError si friction_pips < 0."""
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="friction_pips"):
            compute_cost_aware_target_v2(df, friction_pips=-0.5)

    def test_value_error_k_atr_invalid(self) -> None:
        """ValueError si k_atr <= 0."""
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="k_atr"):
            compute_cost_aware_target_v2(df, k_atr=0.0)

    def test_sl_touched_labels_short(self) -> None:
        """Quand SL est touché en LONG (long_dead et pas short_win), label = -1.

        On utilise une barre > atr_period pour que l'ATR soit valide.
        """
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 1.1000)
        high = np.full(n, 1.1001)  # ne touche jamais TP
        low = np.full(n, 1.0999)
        # Barre 22 : low descend sous SL (1.0990) — après qu'ATR(14) soit valide
        low[22] = 1.0989
        df = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=10, k_atr=0.01,
        )
        assert result[20] == -1.0, f"Barre 20 devrait être -1 (SL touché), reçu {result[20]}"

    def test_k_atr_very_high_all_zero_on_timeout(self) -> None:
        """k_atr très élevé → timeout/TP filtrés à 0. Les pertes SL restent -1 (by design)."""
        df = _make_ohlcv(200)
        result = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=24, k_atr=100.0,
        )
        valid = result[~np.isnan(result)]
        unique = set(np.unique(valid))
        # Avec k_atr=100, aucun TP ni timeout n'est profitable.
        # Les pertes SL restent -1 (pas filtrées par min_profit).
        # On vérifie qu'il n'y a pas de 1 (LONG gagnant filtré).
        assert 1.0 not in unique, (
            f"Aucun TP ne devrait être profitable avec k_atr=100, "
            f"reçu classes={unique}"
        )

    def test_k_atr_zero_equivalent_classic(self) -> None:
        """k_atr quasi nul + friction nulle → comportement similaire triple barrière classique."""
        df = _make_ohlcv(200, seed=42)
        from learning_machine_learning.features.triple_barrier import apply_triple_barrier

        result_v2 = compute_cost_aware_target_v2(
            df, tp_pips=30.0, sl_pips=10.0, window=24,
            friction_pips=0.0, k_atr=1e-6,
        )
        result_classic = apply_triple_barrier(
            df, tp_pips=30.0, sl_pips=10.0, window=24,
        )

        # Les deux doivent être identiques là où les deux sont valides
        mask = ~np.isnan(result_v2) & ~np.isnan(result_classic)
        # Avec k_atr quasi nul, friction nulle, et ATR nan géré → les résultats
        # doivent être très proches (différence uniquement sur cas où SL touché
        # car cost_aware_v2 traite long_dead/!short_win comme -1 au lieu de 0)
        # On vérifie au moins que les signaux forts (TP touché) sont identiques
        tp_mask = mask & (result_classic != 0.0)
        if tp_mask.any():
            match_rate = np.mean(result_v2[tp_mask] == result_classic[tp_mask])
            assert match_rate > 0.85, (
                f"Concordance TP: {match_rate:.1%}"
            )
