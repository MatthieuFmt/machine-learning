"""Tests unitaires pour backtest.filters — TrendFilter, VolFilter, SessionFilter, FilterPipeline."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.backtest.filters import (
    TrendFilter,
    VolFilter,
    SessionFilter,
    FilterPipeline,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def df_basic() -> pd.DataFrame:
    """DataFrame 24h avec colonnes Dist_SMA200_D1, ATR_Norm."""
    n = 24
    index = pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time")
    return pd.DataFrame(
        {
            "Dist_SMA200_D1": [0.005, -0.003, 0.002, -0.004] * 6,
            "ATR_Norm": [0.003] * n,
        },
        index=index,
    )


@pytest.fixture
def signals(df_basic: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Masques LONG/SHORT uniformes (toutes les barres sont des signaux)."""
    n = len(df_basic)
    mask_long = pd.Series(True, index=df_basic.index)
    mask_short = pd.Series(True, index=df_basic.index)
    return mask_long, mask_short


# ── TrendFilter ───────────────────────────────────────────

class TestTrendFilter:
    def test_rejects_long_when_below_sma(self, df_basic: pd.DataFrame) -> None:
        """LONG rejeté si Dist_SMA200_D1 <= 0."""
        filt = TrendFilter()
        mask_long = pd.Series(True, index=df_basic.index)
        mask_short = pd.Series(False, index=df_basic.index)

        ml, ms, n = filt.apply(df_basic, mask_long, mask_short)

        # Seules les barres avec Dist > 0 survivent
        assert ml.sum() == (df_basic["Dist_SMA200_D1"] > 0).sum()
        assert n > 0  # au moins quelques rejets

    def test_rejects_short_when_above_sma(self, df_basic: pd.DataFrame) -> None:
        """SHORT rejeté si Dist_SMA200_D1 >= 0."""
        filt = TrendFilter()
        mask_long = pd.Series(False, index=df_basic.index)
        mask_short = pd.Series(True, index=df_basic.index)

        ml, ms, n = filt.apply(df_basic, mask_long, mask_short)

        assert ms.sum() == (df_basic["Dist_SMA200_D1"] < 0).sum()
        assert n > 0

    def test_no_reject_when_all_agree(self) -> None:
        """Aucun rejet si LONG uniquement sur Dist>0 et SHORT sur Dist<0."""
        filt = TrendFilter()
        index = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({"Dist_SMA200_D1": [0.01] * 5 + [-0.01] * 5}, index=index)

        mask_long = pd.Series([True] * 5 + [False] * 5, index=index)
        mask_short = pd.Series([False] * 5 + [True] * 5, index=index)

        ml, ms, n = filt.apply(df, mask_long, mask_short)
        assert n == 0
        assert ml.sum() == 5
        assert ms.sum() == 5

    def test_raises_without_column(self) -> None:
        """ValueError si Dist_SMA200_D1 absente."""
        filt = TrendFilter()
        df = pd.DataFrame({"ATR_Norm": [0.003]}, index=pd.date_range("2024-01-01", periods=1, freq="h"))
        mask = pd.Series(True, index=df.index)
        with pytest.raises(ValueError, match="Dist_SMA200_D1"):
            filt.apply(df, mask, mask)


# ── VolFilter ─────────────────────────────────────────────

class TestVolFilter:
    def test_rejects_high_vol(self) -> None:
        """Rejette si ATR_Norm > 2 × médiane glissante."""
        filt = VolFilter(window=5, multiplier=2.0)
        n = 10
        index = pd.date_range("2024-01-01", periods=n, freq="h")
        # Premières valeurs normales, puis spike
        atr = [0.002] * 8 + [0.020, 0.020]  # spike bien au-dessus de 2× médiane
        df = pd.DataFrame({"ATR_Norm": atr}, index=index)

        mask_long = pd.Series(True, index=index)
        mask_short = pd.Series(True, index=index)

        ml, ms, n_rej = filt.apply(df, mask_long, mask_short)

        # Les 2 spikes doivent être rejetés
        assert n_rej >= 1  # au moins 1 rejeté (les deux barres du spike)
        # Les barres normales survivent
        assert ml.iloc[0]  # première barre OK
        assert ms.iloc[0]

    def test_no_reject_normal_vol(self) -> None:
        """Aucun rejet si volatilité normale."""
        filt = VolFilter(window=5, multiplier=2.0)
        index = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({"ATR_Norm": [0.002] * 10}, index=index)

        mask = pd.Series(True, index=index)
        ml, ms, n = filt.apply(df, mask, mask)
        assert n == 0
        assert ml.all()
        assert ms.all()

    def test_raises_without_column(self) -> None:
        """ValueError si ATR_Norm absente."""
        filt = VolFilter()
        df = pd.DataFrame({"Dist_SMA200_D1": [0.01]}, index=pd.date_range("2024-01-01", periods=1, freq="h"))
        mask = pd.Series(True, index=df.index)
        with pytest.raises(ValueError, match="ATR_Norm"):
            filt.apply(df, mask, mask)


# ── SessionFilter ─────────────────────────────────────────

class TestSessionFilter:
    def test_rejects_during_low_liquidity(self) -> None:
        """Rejette les signaux entre 22h et 1h GMT (1h exclus, donc 22, 23, 0)."""
        filt = SessionFilter(exclude_start=22, exclude_end=1)
        hours = [20, 21, 22, 23, 0, 1, 2, 3]
        index = pd.date_range("2024-01-01 20:00", periods=len(hours), freq="h")
        df = pd.DataFrame({"Close": np.ones(len(hours))}, index=index)

        mask = pd.Series(True, index=index)
        ml, ms, n = filt.apply(df, mask, mask)
        assert n == 3  # 22, 23, 0 (1h est exclu car end exclusif)
        assert ml.iloc[0] and ml.iloc[1]  # 20h, 21h survivent
        assert not ml.iloc[2]  # 22h rejeté
        assert ml.iloc[5]  # 1h survit (1 < 1 est False)

    def test_simple_range_not_crossing_midnight(self) -> None:
        """Plage horaire simple (ex: 2h-5h)."""
        filt = SessionFilter(exclude_start=2, exclude_end=5)
        index = pd.date_range("2024-01-01 00:00", periods=8, freq="h")
        df = pd.DataFrame({"Close": np.ones(8)}, index=index)

        mask = pd.Series(True, index=index)
        ml, ms, n = filt.apply(df, mask, mask)
        assert n == 3  # 2, 3, 4 (5 est exclu car end exclusif)
        assert not ml.iloc[2] and not ml.iloc[3] and not ml.iloc[4]

    def test_no_exclusion_outside_range(self) -> None:
        """Aucun rejet en dehors de la plage exclue."""
        filt = SessionFilter(exclude_start=22, exclude_end=1)
        # Heures de marché : 8h-16h GMT
        index = pd.date_range("2024-01-01 08:00", periods=9, freq="h")
        df = pd.DataFrame({"Close": np.ones(9)}, index=index)

        mask = pd.Series(True, index=index)
        ml, ms, n = filt.apply(df, mask, mask)
        assert n == 0
        assert ml.all()


# ── FilterPipeline ────────────────────────────────────────

class TestFilterPipeline:
    def test_applies_filters_in_order(self) -> None:
        """Les filtres sont appliqués séquentiellement, les rejets s'accumulent."""
        index = pd.date_range("2024-01-01 22:00", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "Dist_SMA200_D1": [0.01, 0.01, -0.01, -0.01, 0.01],
                "ATR_Norm": [0.003, 0.003, 0.003, 0.003, 0.003],
            },
            index=index,
        )
        pipeline = FilterPipeline([TrendFilter(), SessionFilter(exclude_start=22, exclude_end=1)])

        mask_long = pd.Series(True, index=index)
        mask_short = pd.Series(True, index=index)

        ml, ms, rejected, reasons = pipeline.apply(df, mask_long, mask_short)

        assert "trend" in rejected
        assert "session" in rejected
        assert rejected["trend"] + rejected["session"] >= 1
        assert isinstance(reasons, pd.Series)

    def test_empty_pipeline_no_effect(self, df_basic: pd.DataFrame) -> None:
        """Pipeline vide = aucun rejet."""
        pipeline = FilterPipeline([])
        mask = pd.Series(True, index=df_basic.index)
        ml, ms, rejected, reasons = pipeline.apply(df_basic, mask, mask)
        assert ml.all()
        assert ms.all()
        assert rejected == {}
        assert (reasons == "").all()

    def test_n_rejected_never_exceeds_signal_count(self, df_basic: pd.DataFrame) -> None:
        """Le nombre de rejets par filtre ≤ nombre de signaux."""
        pipeline = FilterPipeline([TrendFilter(), VolFilter()])
        mask = pd.Series(True, index=df_basic.index)  # 24 signaux
        _, _, rejected, _ = pipeline.apply(df_basic, mask, mask)
        for n in rejected.values():
            assert 0 <= n <= 24
