"""Tests unitaires du labelling cost-aware (v15) — < 100ms.

Valide que apply_triple_barrier_cost_aware :
1. Filtre les TP non rentables apres friction.
2. Resout les timeouts via PnL sur Close.
3. Respecte la convention de signature (NaN sur les window dernieres barres).
4. Leve ValueError si friction_pips ou min_profit_pips sont negatifs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.triple_barrier import (
    apply_triple_barrier,
    apply_triple_barrier_cost_aware,
)


class TestCostAwareLabeling:
    """Tests de la fonction apply_triple_barrier_cost_aware."""

    @staticmethod
    def _make_ohlcv(
        closes: list[float],
        highs: list[float] | None = None,
        lows: list[float] | None = None,
    ) -> pd.DataFrame:
        """Fabrique un DataFrame OHLCV minimal.

        Si highs/lows non fournis, utilise Close +/- 2% (range large, safe).
        """
        n = len(closes)
        h = highs if highs is not None else [c * 1.02 for c in closes]
        l = lows if lows is not None else [c * 0.98 for c in closes]
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        return pd.DataFrame({"High": h, "Low": l, "Close": closes}, index=idx)

    def test_tp_profitable_long(self) -> None:
        """Un TP touche avec profit net >= min_profit doit etre labellise 1."""
        entry = 1.1000
        closes = [
            entry,          # 0: entry
            entry + 0.0005,  # 1: +5p
            entry + 0.0010,  # 2: +10p
            entry + 0.0025,  # 3: +25p → high touche TP 20p
            entry + 0.0030,  # 4
            entry + 0.0035,  # 5
        ]
        # Highs: legerement au-dessus du Close pour toucher le TP
        highs = [
            c + 0.00005 for c in closes
        ]
        # Lows: proches du Close, pas de SL accidentel
        lows = [
            c - 0.00005 for c in closes
        ]
        df = self._make_ohlcv(closes, highs=highs, lows=lows)
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=20.0, sl_pips=10.0, window=4,
            pip_size=0.0001, friction_pips=1.5, min_profit_pips=3.0,
        )
        assert targets[0] == 1.0, f"TP profitable LONG doit etre 1.0, recu {targets[0]}"

    def test_tp_not_profitable_after_friction(self) -> None:
        """Un TP touche mais profit net < min_profit → label 0."""
        entry = 1.1000
        closes = [
            entry,
            entry + 0.0015,  # +15p → TP touche
            entry + 0.0010,
            entry + 0.0005,
            entry + 0.0005,
            entry + 0.0005,
        ]
        highs = [c + 0.00005 for c in closes]
        lows = [c - 0.00005 for c in closes]
        df = self._make_ohlcv(closes, highs=highs, lows=lows)
        # TP=15p, friction=1.5p → net=13.5p, min_profit=20p → pas profitable
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=15.0, sl_pips=10.0, window=4,
            pip_size=0.0001, friction_pips=1.5, min_profit_pips=20.0,
        )
        assert targets[0] == 0.0, (
            f"TP non profitable apres friction doit etre 0.0, recu {targets[0]}"
        )

    def test_timeout_pnl_profitable_long(self) -> None:
        """Timeout : le prix de cloture donne un PnL net >= min_profit → label 1."""
        entry = 1.1000
        closes = [
            entry,
            entry + 0.0002,  # +2p
            entry + 0.0008,  # +8p → Close a i+window, PnL = 8-1.5=6.5p >= 3p ✓
            entry + 0.0010,  # +10p
        ]
        highs = [c + 0.00001 for c in closes]
        lows = [c - 0.00001 for c in closes]
        df = self._make_ohlcv(closes, highs=highs, lows=lows)
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=50.0, sl_pips=50.0, window=2,
            pip_size=0.0001, friction_pips=1.5, min_profit_pips=3.0,
        )
        assert targets[0] == 1.0, (
            f"Timeout profitable LONG doit etre 1.0, recu {targets[0]}"
        )

    def test_timeout_pnl_not_profitable(self) -> None:
        """Timeout : PnL net < min_profit → label 0."""
        entry = 1.1000
        closes = [
            entry,
            entry + 0.0001,  # +1p
            entry + 0.0002,  # +2p
            entry + 0.0003,  # +3p → Close, PnL = 3-1.5=1.5p < 3p
        ]
        highs = [c + 0.00001 for c in closes]
        lows = [c - 0.00001 for c in closes]
        df = self._make_ohlcv(closes, highs=highs, lows=lows)
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=50.0, sl_pips=50.0, window=2,
            pip_size=0.0001, friction_pips=1.5, min_profit_pips=3.0,
        )
        assert targets[0] == 0.0, (
            f"Timeout non profitable doit etre 0.0, recu {targets[0]}"
        )

    def test_sl_hit_both_directions_gives_zero(self) -> None:
        """Quand SL est touche dans les deux directions → label 0 (pas de trade)."""
        entry = 1.1000
        # Le prix descend (SL LONG touche) puis monte (SL SHORT touche)
        closes = [
            entry,
            entry - 0.0020,  # -20p → SL LONG 10p touche
            entry - 0.0010,
            entry + 0.0020,  # +20p → SL SHORT 10p touche
            entry + 0.0030,
            entry + 0.0040,
        ]
        highs = [c + 0.00005 for c in closes]
        lows = [c - 0.00005 for c in closes]
        df = self._make_ohlcv(closes, highs=highs, lows=lows)
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=50.0, sl_pips=10.0, window=5,
            pip_size=0.0001, friction_pips=1.5, min_profit_pips=3.0,
        )
        assert targets[0] == 0.0, (
            f"SL des deux cotes doit etre 0.0, recu {targets[0]}"
        )

    def test_sl_only_long_gives_zero(self) -> None:
        """SL LONG touche, SHORT timeout non profitable → label 0."""
        entry = 1.1000
        closes = [
            entry,
            entry - 0.0012,  # -12p → SL LONG 10p touche
            entry - 0.0004,  # Close=-4p, PnL short = 4-1.5=2.5p < 3p → pas profitable
            entry - 0.0005,
        ]
        highs = [c + 0.00005 for c in closes]
        lows = [c - 0.00005 for c in closes]
        df = self._make_ohlcv(closes, highs=highs, lows=lows)
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=50.0, sl_pips=10.0, window=2,
            pip_size=0.0001, friction_pips=1.5, min_profit_pips=3.0,
        )
        assert targets[0] == 0.0, (
            f"SL LONG seul doit etre 0.0, recu {targets[0]}"
        )

    def test_nan_on_last_bars(self) -> None:
        """Les window dernieres barres doivent etre NaN."""
        n = 10
        closes = [1.1000] * n
        df = self._make_ohlcv(closes)
        targets = apply_triple_barrier_cost_aware(
            df, tp_pips=20.0, sl_pips=10.0, window=4,
            pip_size=0.0001,
        )
        assert np.isnan(targets[-4:]).all(), "Les 4 dernieres barres doivent etre NaN"
        assert not np.isnan(targets[:6]).any(), "Les 6 premieres barres ne doivent pas etre NaN"

    def test_negative_friction_raises(self) -> None:
        """friction_pips negatif doit lever ValueError."""
        df = self._make_ohlcv([1.1000] * 10)
        with pytest.raises(ValueError, match="friction_pips"):
            apply_triple_barrier_cost_aware(df, friction_pips=-1.0)

    def test_negative_min_profit_raises(self) -> None:
        """min_profit_pips negatif doit lever ValueError."""
        df = self._make_ohlcv([1.1000] * 10)
        with pytest.raises(ValueError, match="min_profit_pips"):
            apply_triple_barrier_cost_aware(df, min_profit_pips=-1.0)

    def test_cost_aware_vs_classic_shape(self) -> None:
        """Les deux fonctions doivent retourner des arrays de meme forme."""
        n = 100
        rng = np.random.default_rng(42)
        base = 1.1000
        closes = base + np.cumsum(rng.normal(0, 0.0005, n))
        highs = closes + np.abs(rng.normal(0, 0.0002, n))
        lows = closes - np.abs(rng.normal(0, 0.0002, n))
        df = pd.DataFrame(
            {"High": highs, "Low": lows, "Close": closes},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )
        classic = apply_triple_barrier(df, tp_pips=20.0, sl_pips=10.0, window=24)
        cost_aware = apply_triple_barrier_cost_aware(
            df, tp_pips=20.0, sl_pips=10.0, window=24,
        )
        assert classic.shape == cost_aware.shape
        assert classic.dtype == cost_aware.dtype

    def test_cost_aware_trades_ratio_reasonable(self) -> None:
        """Le cost-aware peut ajouter des trades (timeouts profitables) mais < 4x."""
        n = 200
        rng = np.random.default_rng(123)
        base = 1.1000
        closes = base + np.cumsum(rng.normal(0, 0.0003, n))
        highs = closes + np.abs(rng.normal(0, 0.0001, n))
        lows = closes - np.abs(rng.normal(0, 0.0001, n))
        df = pd.DataFrame(
            {"High": highs, "Low": lows, "Close": closes},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )
        classic = apply_triple_barrier(df, tp_pips=15.0, sl_pips=10.0, window=12)
        cost_aware = apply_triple_barrier_cost_aware(
            df, tp_pips=15.0, sl_pips=10.0, window=12,
            friction_pips=1.5, min_profit_pips=3.0,
        )
        classic_trades = np.sum(np.abs(classic[~np.isnan(classic)]))
        cost_aware_trades = np.sum(np.abs(cost_aware[~np.isnan(cost_aware)]))
        # Le cost-aware peut ajouter des trades (timeouts profitables)
        # mais ne doit pas multiplier le total par plus de 4x
        assert cost_aware_trades <= classic_trades * 4.0, (
            f"Cost-aware ({cost_aware_trades}) ne doit pas exceder 4x classic ({classic_trades})"
        )
        # Meme nombre de barres labellisables dans les deux cas
        assert np.sum(~np.isnan(classic)) == np.sum(~np.isnan(cost_aware)), (
            "Meme nombre de barres labellisables"
        )
