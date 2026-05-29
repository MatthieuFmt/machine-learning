"""Tests unitaires pour app.strategies.nr7_meta — Phase H3 Étape 2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.nr7_meta import (
    FEATURE_NAMES,
    HGB_PARAMS,
    build_features_at_entry,
    cv_select_threshold,
    filter_trades,
    train_meta_model,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _synthetic_us500_d1(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """OHLCV D1 synthétique sur n_days jours, ~6000 + bruit."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="1D", tz="UTC")
    returns = rng.normal(0.0005, 0.01, n_days)
    close = 6000.0 * np.cumprod(1 + returns)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "Close": close,
            "Volume": rng.uniform(100, 1000, n_days),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def _synthetic_macro(n_days: int = 350, start: str = "2022-12-01") -> pd.DataFrame:
    """Macro daily synthétique (VIX, DXY zscore, yield slope)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n_days, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "vix_level": rng.uniform(12, 25, n_days),
            "vix_zscore_60": rng.normal(0, 1, n_days),
            "dxy_zscore_60": rng.normal(0, 1, n_days),
            "yield_slope_10y_3m": rng.normal(0.5, 0.5, n_days),
        },
        index=idx,
    )
    return df


def _make_trade(
    setup_date: str,
    entry_time: str,
    signal: int = 1,
    range_J: float = 100.0,
    pips_net: float = 50.0,
) -> dict:
    """Crée un trade dict synthétique compatible nr7_meta."""
    return {
        "setup_date": setup_date,
        "entry_time": entry_time,
        "signal": signal,
        "range_J": range_J,
        "pips_net": pips_net,
        "high_J": 6100.0,
        "low_J": 6000.0,
        "tp_price": 6200.0,
        "sl_price": 5950.0,
        "exit_reason": "tp" if pips_net > 0 else "sl",
    }


# ─────────────────────────────────────────────────────────────────────
# build_features_at_entry
# ─────────────────────────────────────────────────────────────────────


class TestBuildFeaturesAtEntry:
    def test_basic_returns_correct_shape(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        trades = [
            _make_trade("2023-10-01", "2023-10-02T00:00:00+00:00"),
            _make_trade("2023-10-05", "2023-10-06T00:00:00+00:00", signal=-1),
            _make_trade("2023-10-10", "2023-10-11T00:00:00+00:00"),
        ]
        X = build_features_at_entry(df_d1, df_macro, trades)
        assert len(X) == 3
        assert list(X.columns) == FEATURE_NAMES

    def test_skips_setup_outside_df(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=100)  # 2023-01 → 2023-04
        df_macro = _synthetic_macro()
        trades = [
            _make_trade("2030-01-01", "2030-01-02T00:00:00+00:00"),  # hors df
        ]
        X = build_features_at_entry(df_d1, df_macro, trades)
        assert len(X) == 0

    def test_signal_direction_encoded_as_float(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        trades = [
            _make_trade("2023-10-01", "2023-10-02T00:00:00+00:00", signal=1),
            _make_trade("2023-10-05", "2023-10-06T00:00:00+00:00", signal=-1),
        ]
        X = build_features_at_entry(df_d1, df_macro, trades)
        assert X["signal_direction"].iloc[0] == pytest.approx(1.0)
        assert X["signal_direction"].iloc[1] == pytest.approx(-1.0)

    def test_day_of_week_computed(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        # 2023-10-02 = lundi (0)
        # 2023-10-06 = vendredi (4)
        trades = [
            _make_trade("2023-10-01", "2023-10-02T00:00:00+00:00"),
            _make_trade("2023-10-05", "2023-10-06T00:00:00+00:00"),
        ]
        X = build_features_at_entry(df_d1, df_macro, trades)
        assert X["day_of_week"].iloc[0] == 0
        assert X["day_of_week"].iloc[1] == 4

    def test_range_NR_atr20_ratio_computed(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        trades = [_make_trade("2023-10-01", "2023-10-02T00:00:00+00:00", range_J=50.0)]
        X = build_features_at_entry(df_d1, df_macro, trades)
        # ratio devrait être un float fini, range_J/ATR_20 ~ 0.5-2.0 typique
        val = X["range_NR_atr20_ratio"].iloc[0]
        assert not np.isnan(val)
        assert val > 0

    def test_handles_macro_warmup_early_date(self) -> None:
        """Trade avant la première date macro → vix/dxy/yield = NaN."""
        df_d1 = _synthetic_us500_d1(n_days=300)
        # Macro commence le 2023-12-01 mais le trade est le 2023-10-01
        df_macro = _synthetic_macro(n_days=100, start="2023-12-01")
        trades = [_make_trade("2023-10-01", "2023-10-02T00:00:00+00:00")]
        X = build_features_at_entry(df_d1, df_macro, trades)
        assert len(X) == 1
        assert np.isnan(X["vix_level"].iloc[0])


# ─────────────────────────────────────────────────────────────────────
# train_meta_model
# ─────────────────────────────────────────────────────────────────────


class TestTrainMetaModel:
    def test_fits_without_error(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame(
            rng.normal(0, 1, (n, len(FEATURE_NAMES))),
            columns=FEATURE_NAMES,
        )
        # Label corrélé à signal_direction pour avoir un edge artificiel
        y = pd.Series((X["signal_direction"] > 0).astype(int).values)
        model = train_meta_model(X, y)
        assert model is not None
        # predict_proba donne des probas valides
        probas = model.predict_proba(X.values)
        assert probas.shape == (n, 2)
        assert np.all((probas >= 0) & (probas <= 1))


# ─────────────────────────────────────────────────────────────────────
# cv_select_threshold
# ─────────────────────────────────────────────────────────────────────


class TestCvSelectThreshold:
    def test_returns_valid_threshold(self) -> None:
        rng = np.random.default_rng(42)
        n = 150
        X = pd.DataFrame(
            rng.normal(0, 1, (n, len(FEATURE_NAMES))),
            columns=FEATURE_NAMES,
        )
        # Bias : returns 1 si feature[0] > 0 (signal apprenable)
        y = pd.Series((X["vix_level"] > 0).astype(int).values)
        # PnL aléatoire mais corrélé à y
        pnls = pd.Series(rng.normal(0, 50, n) + 50 * y.values)
        best, stats = cv_select_threshold(X, y, pnls, n_splits=3)
        assert 0.40 <= best <= 0.70
        assert "per_threshold" in stats
        assert all(0.40 <= t <= 0.70 for t in stats["per_threshold"].keys())


# ─────────────────────────────────────────────────────────────────────
# filter_trades
# ─────────────────────────────────────────────────────────────────────


class TestFilterTrades:
    def test_threshold_zero_takes_all(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        trades = [
            _make_trade("2023-10-01", "2023-10-02T00:00:00+00:00"),
            _make_trade("2023-10-05", "2023-10-06T00:00:00+00:00", signal=-1),
            _make_trade("2023-10-10", "2023-10-11T00:00:00+00:00"),
        ]
        X = build_features_at_entry(df_d1, df_macro, trades)
        y = pd.Series([1, 0, 1], index=X.index)
        model = train_meta_model(X.fillna(0), y)
        kept, probas = filter_trades(trades, X.fillna(0), model, threshold=0.0)
        assert len(kept) == 3
        assert len(probas) == 3

    def test_threshold_one_takes_none(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        trades = [
            _make_trade("2023-10-01", "2023-10-02T00:00:00+00:00"),
            _make_trade("2023-10-05", "2023-10-06T00:00:00+00:00"),
        ]
        X = build_features_at_entry(df_d1, df_macro, trades)
        y = pd.Series([1, 0], index=X.index)
        model = train_meta_model(X.fillna(0), y)
        kept, probas = filter_trades(trades, X.fillna(0), model, threshold=1.01)
        assert len(kept) == 0

    def test_empty_trades_returns_empty(self) -> None:
        df_d1 = _synthetic_us500_d1(n_days=300)
        df_macro = _synthetic_macro()
        X = build_features_at_entry(df_d1, df_macro, [])
        # Forge un modèle pour le filter (sur features synthétiques)
        rng = np.random.default_rng(42)
        X_fake = pd.DataFrame(
            rng.normal(0, 1, (50, len(FEATURE_NAMES))),
            columns=FEATURE_NAMES,
        )
        y_fake = pd.Series(rng.integers(0, 2, 50))
        model = train_meta_model(X_fake, y_fake)
        kept, probas = filter_trades([], X, model, threshold=0.5)
        assert len(kept) == 0
        assert len(probas) == 0


# ─────────────────────────────────────────────────────────────────────
# Constantes & params
# ─────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_feature_names_count(self) -> None:
        assert len(FEATURE_NAMES) == 11

    def test_hgb_params_have_seed(self) -> None:
        assert "random_state" in HGB_PARAMS
        assert HGB_PARAMS["random_state"] == 42
