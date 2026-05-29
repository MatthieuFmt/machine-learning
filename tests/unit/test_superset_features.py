"""Tests du superset de features pivot v4 A5."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.superset import build_superset
from app.testing.look_ahead_validator import assert_no_look_ahead


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """500 barres OHLCV synthétiques reproductibles."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    close = 40_000 + rng.normal(0, 100, n).cumsum()
    open_ = close + rng.normal(0, 20, n)
    high = np.maximum(close, open_) + rng.uniform(10, 50, n)
    low = np.minimum(close, open_) - rng.uniform(10, 50, n)
    volume = rng.uniform(1000, 10_000, n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


# ══════════════════════════════════════════════════════════════════════
# Structural tests
# ══════════════════════════════════════════════════════════════════════


def test_superset_min_60_columns(synthetic_df: pd.DataFrame) -> None:
    """Le superset doit avoir ≥ 60 colonnes."""
    feat = build_superset(synthetic_df)
    assert feat.shape[1] >= 60, f"Attendu ≥ 60, obtenu {feat.shape[1]}"


def test_superset_unique_columns(synthetic_df: pd.DataFrame) -> None:
    """Tous les noms de colonnes doivent être uniques."""
    feat = build_superset(synthetic_df)
    assert feat.columns.is_unique, f"Doublons: {feat.columns[feat.columns.duplicated()].tolist()}"


def test_superset_dtypes_float64(synthetic_df: pd.DataFrame) -> None:
    """Toutes les colonnes doivent être en float64."""
    feat = build_superset(synthetic_df)
    non_float = feat.dtypes[feat.dtypes != np.float64]
    assert non_float.empty, f"Dtypes non-float64: {non_float.to_dict()}"


def test_superset_no_nan_after_warmup(synthetic_df: pd.DataFrame) -> None:
    """Aucun NaN après warmup 250 barres (hors cross-asset optionnel)."""
    feat = build_superset(synthetic_df)
    after_warmup = feat.iloc[250:]
    nan_cols = after_warmup.columns[after_warmup.isna().any()].tolist()
    # cross-asset features may legitimately be NaN if unavailable
    allowed_prefixes = ("usdchf_", "xauusd_", "btcusd_")
    forbidden = [c for c in nan_cols if not c.startswith(allowed_prefixes)]
    assert not forbidden, f"NaN après warmup: {forbidden}"


def test_superset_anti_look_ahead(synthetic_df: pd.DataFrame) -> None:
    """Vérifie l'absence de look-ahead sur un aggregat de features."""

    def feat_close_zscore(df: pd.DataFrame) -> pd.Series:
        return build_superset(df)["close_zscore_20"]

    assert_no_look_ahead(feat_close_zscore, synthetic_df, n_samples=20)


# ══════════════════════════════════════════════════════════════════════
# Category presence tests
# ══════════════════════════════════════════════════════════════════════


def test_category_trend(synthetic_df: pd.DataFrame) -> None:
    """Présence des features de tendance."""
    feat = build_superset(synthetic_df)
    expected = [
        "sma_20", "sma_50", "sma_200",
        "dist_sma_20", "dist_sma_50", "dist_sma_200",
        "ema_12", "ema_26",
        "dist_ema_12", "dist_ema_26",
        "slope_sma_20", "slope_sma_50",
    ]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_momentum(synthetic_df: pd.DataFrame) -> None:
    """Présence des features momentum/RSI/MACD."""
    feat = build_superset(synthetic_df)
    expected = ["rsi_7", "rsi_14", "rsi_21", "macd", "macd_signal", "macd_hist"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_oscillators(synthetic_df: pd.DataFrame) -> None:
    """Présence des oscillateurs (Stoch K/D, Williams, CCI, MFI)."""
    feat = build_superset(synthetic_df)
    expected = ["stoch_k_14", "stoch_d_14", "williams_r_14", "cci_20", "mfi_14"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_price_action(synthetic_df: pd.DataFrame) -> None:
    """Présence des features price action."""
    feat = build_superset(synthetic_df)
    expected = [
        "body_to_range_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
        "gap_overnight", "consecutive_up", "consecutive_down",
        "range_atr_ratio", "inside_bar", "outside_bar", "doji",
    ]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_stats(synthetic_df: pd.DataFrame) -> None:
    """Présence des features statistiques."""
    feat = build_superset(synthetic_df)
    expected = [
        "close_zscore_20", "close_zscore_60", "atr_zscore_60",
        "volume_zscore_20",
        "return_percentile_20", "return_percentile_60",
        "vol_percentile_60",
        "skew_returns_20", "kurt_returns_20",
        "autocorr_returns_lag1_20",
    ]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_regime(synthetic_df: pd.DataFrame) -> None:
    """Présence des features régime de marché."""
    feat = build_superset(synthetic_df)
    expected = [
        "efficiency_ratio_20", "trend_strength",
        "dist_sma_200_abs_atr", "regime_trending_binary",
        "vol_regime_low", "vol_regime_mid", "vol_regime_high",
    ]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_sessions(synthetic_df: pd.DataFrame) -> None:
    """Présence des features de session + temps cyclique."""
    feat = build_superset(synthetic_df)
    expected = [
        "session_tokyo", "session_london", "session_ny",
        "session_overlap_london_ny",
        "day_sin", "day_cos", "month_sin", "month_cos",
    ]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"
