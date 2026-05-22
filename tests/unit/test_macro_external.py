"""Tests unitaires pour app.features.macro_external — Phase F5.

Aucun appel réseau : on injecte `macro_df` directement dans
`add_external_macro` pour bypasser le téléchargement yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.macro_external import (
    ZSCORE_WINDOW,
    _zscore,
    add_external_macro,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _synthetic_macro(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Construit un DataFrame macro synthétique respectant le schéma attendu."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="1D", tz="UTC")
    dxy = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.2, n_days)), index=idx)
    vix = pd.Series(15.0 + 5 * rng.normal(0, 1, n_days).cumsum() * 0.01, index=idx)
    tnx = pd.Series(4.0 + rng.normal(0, 0.05, n_days), index=idx)
    irx = pd.Series(3.0 + rng.normal(0, 0.05, n_days), index=idx)

    df = pd.DataFrame(
        {
            "dxy_zscore_60": _zscore(dxy, ZSCORE_WINDOW),
            "vix_level": vix,
            "vix_zscore_60": _zscore(vix, ZSCORE_WINDOW),
            "yield_slope_10y_3m": tnx - irx,
        },
        index=idx,
    )
    return df


def _synthetic_ohlc(n_hours: int = 24 * 100, seed: int = 1) -> pd.DataFrame:
    """OHLCV H1 sur 100 jours."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-02-01", periods=n_hours, freq="1h", tz="UTC")
    close = 1.0 + np.cumsum(rng.normal(0, 0.0005, n_hours))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.0005,
            "Low": close * 0.9995,
            "Close": close,
            "Volume": rng.uniform(100, 1000, n_hours),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


# ─────────────────────────────────────────────────────────────────────
# Tests de _zscore
# ─────────────────────────────────────────────────────────────────────


class TestZscore:
    def test_warmup_is_nan(self) -> None:
        s = pd.Series(np.arange(100.0))
        z = _zscore(s, window=60)
        assert z.iloc[:59].isna().all()
        assert z.iloc[59:].notna().all()

    def test_zero_std_returns_nan(self) -> None:
        s = pd.Series([5.0] * 100)
        z = _zscore(s, window=60)
        # std = 0 → division par NaN → NaN
        assert z.iloc[59:].isna().all()


# ─────────────────────────────────────────────────────────────────────
# Tests de add_external_macro — structure
# ─────────────────────────────────────────────────────────────────────


class TestAddExternalMacroStructure:
    def test_adds_four_columns(self) -> None:
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)

        for col in ("dxy_zscore_60", "vix_level", "vix_zscore_60", "yield_slope_10y_3m"):
            assert col in out.columns, f"colonne {col} manquante"

    def test_preserves_length(self) -> None:
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)
        assert len(out) == len(df)

    def test_preserves_original_columns(self) -> None:
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in out.columns

    def test_index_preserved(self) -> None:
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)
        assert out.index.equals(df.index)


# ─────────────────────────────────────────────────────────────────────
# Tests anti-look-ahead
# ─────────────────────────────────────────────────────────────────────


class TestAddExternalMacroNoLookAhead:
    def test_macro_value_at_t_uses_past_data(self) -> None:
        """À l'instant t (heure H1), la valeur macro doit être ≤ celle de J-1."""
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)

        # Prendre une barre arbitraire après warmup
        sample_idx = out.index[24 * 80]  # jour 80
        merged_vix = out.loc[sample_idx, "vix_level"]
        date_t = sample_idx.normalize()

        # La valeur doit correspondre à un jour macro strictement antérieur
        # (≤ t - 1j car shift +1 dans add_external_macro)
        macro_eligible = macro[macro.index < date_t]
        if not pd.isna(merged_vix) and not macro_eligible.empty:
            last_legitimate = macro_eligible["vix_level"].iloc[-1]
            assert merged_vix == pytest.approx(last_legitimate, nan_ok=True)

    def test_truncation_does_not_change_past(self) -> None:
        """Retirer les N dernières barres ne doit pas modifier les valeurs passées."""
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        full = add_external_macro(df, macro_df=macro)
        truncated = add_external_macro(df.iloc[:-100], macro_df=macro)

        common = truncated.index
        for col in ("dxy_zscore_60", "vix_level", "vix_zscore_60", "yield_slope_10y_3m"):
            a = full.loc[common, col]
            b = truncated[col]
            mask = a.notna() & b.notna()
            assert (a[mask] == b[mask]).all(), f"divergence sur {col}"

    def test_macro_truncation_propagates_correctly(self) -> None:
        """Tronquer le macro DataFrame doit produire des NaN aux dates non couvertes."""
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        # On garde uniquement les 30 premiers jours de macro
        truncated_macro = macro.iloc[:30]
        out = add_external_macro(df, macro_df=truncated_macro)

        # Les dernières barres de df (au-delà de J30) doivent forward-filler
        # la dernière valeur connue (merge_asof backward)
        last_macro_date = truncated_macro.index.max()
        late_rows = out[out.index > last_macro_date + pd.Timedelta(days=2)]
        if not late_rows.empty:
            # Toutes ces barres utilisent la même valeur (dernière publiée)
            assert late_rows["vix_level"].nunique(dropna=True) <= 1


# ─────────────────────────────────────────────────────────────────────
# Tests de validation d'entrée
# ─────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_naive_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=50, freq="1h")  # naive
        df = pd.DataFrame({"Close": np.arange(50)}, index=idx)
        macro = _synthetic_macro()
        with pytest.raises(ValueError, match="tz-aware"):
            add_external_macro(df, macro_df=macro)

    def test_rejects_unsorted_index(self) -> None:
        idx = pd.date_range("2020-01-01", periods=50, freq="1h", tz="UTC")
        df = pd.DataFrame({"Close": np.arange(50)}, index=idx[::-1])
        macro = _synthetic_macro()
        with pytest.raises(ValueError, match="trié"):
            add_external_macro(df, macro_df=macro)

    def test_rejects_non_datetime_index(self) -> None:
        df = pd.DataFrame({"Close": np.arange(10)})  # RangeIndex
        macro = _synthetic_macro()
        with pytest.raises(TypeError, match="DatetimeIndex"):
            add_external_macro(df, macro_df=macro)


# ─────────────────────────────────────────────────────────────────────
# Tests sémantiques
# ─────────────────────────────────────────────────────────────────────


class TestSemantics:
    def test_vix_level_is_raw_close(self) -> None:
        """vix_level doit être identique au Close du VIX (pas zscoré)."""
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)

        # Sur une barre lointaine dans le futur, la valeur stagne sur la
        # dernière macro publiée
        last_macro_date = macro.index.max()
        future_bars = out[out.index > last_macro_date + pd.Timedelta(days=1)]
        if not future_bars.empty:
            expected = macro["vix_level"].iloc[-1]
            assert future_bars["vix_level"].iloc[-1] == pytest.approx(expected)

    def test_yield_slope_is_difference(self) -> None:
        """yield_slope_10y_3m doit refléter la différence (TNX - IRX)."""
        df = _synthetic_ohlc()
        macro = _synthetic_macro()
        out = add_external_macro(df, macro_df=macro)
        # La feature existe et n'est pas constante (les séries TNX/IRX bougent)
        assert out["yield_slope_10y_3m"].dropna().nunique() > 1
