"""Tests unitaires pour app.strategies.pre_fomc_meta — Phase H1 Étape 2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.pre_fomc_meta import (
    FEATURE_NAMES,
    build_features_at_entry,
    cv_select_threshold,
    filter_trades,
    train_meta_model,
)


def _synthetic_us500_h1(n_days: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    returns = rng.normal(0.00001, 0.001, n_hours)
    close = 6000.0 * np.cumprod(1 + returns)
    df = pd.DataFrame(
        {"Open": close, "High": close * 1.0005, "Low": close * 0.9995, "Close": close},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def _synthetic_macro(n_days: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-09-01", periods=n_days, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "vix_level": 15.0 + 3 * rng.normal(0, 1, n_days),
            "vix_zscore_60": rng.normal(0, 1, n_days),
            "dxy_zscore_60": rng.normal(0, 1, n_days),
            "yield_slope_10y_3m": rng.normal(1.5, 0.5, n_days),
        },
        index=idx,
    )


class TestBuildFeatures:
    def test_returns_one_row_per_fomc(self) -> None:
        df = _synthetic_us500_h1()
        macro = _synthetic_macro()
        fomc = pd.DatetimeIndex([
            pd.Timestamp("2020-04-01 18:00", tz="UTC"),
            pd.Timestamp("2020-06-15 18:00", tz="UTC"),
            pd.Timestamp("2020-08-01 18:00", tz="UTC"),
        ])
        X = build_features_at_entry(df, macro, fomc)
        assert len(X) == 3
        assert set(X.columns) >= set(FEATURE_NAMES)

    def test_first_event_has_nan_days_since(self) -> None:
        df = _synthetic_us500_h1()
        macro = _synthetic_macro()
        fomc = pd.DatetimeIndex([
            pd.Timestamp("2020-04-01 18:00", tz="UTC"),
            pd.Timestamp("2020-06-15 18:00", tz="UTC"),
        ])
        X = build_features_at_entry(df, macro, fomc)
        # Premier event = pas de FOMC précédent
        assert pd.isna(X["days_since_last_fomc"].iloc[0])
        # Second event = ~75 jours
        assert X["days_since_last_fomc"].iloc[1] == 75.0

    def test_skips_event_without_bar(self) -> None:
        df = _synthetic_us500_h1(n_days=10)
        macro = _synthetic_macro()
        fomc = pd.DatetimeIndex([
            pd.Timestamp("2030-01-01 18:00", tz="UTC"),  # hors data
        ])
        X = build_features_at_entry(df, macro, fomc)
        assert len(X) == 0


class TestTrainAndFilter:
    def test_model_fits_and_filters(self) -> None:
        df = _synthetic_us500_h1(n_days=400, seed=2)
        macro = _synthetic_macro(n_days=420, seed=3)

        rng = np.random.default_rng(0)
        fomc = pd.DatetimeIndex(sorted(set([
            pd.Timestamp("2020-03-15 18:00", tz="UTC") + pd.Timedelta(days=int(d))
            for d in rng.integers(0, 200, 50)
        ])))

        X = build_features_at_entry(df, macro, fomc)
        X = X.dropna()
        if len(X) < 20:
            pytest.skip("Pas assez d'events synthétiques pour le test")

        # Labels aléatoires mais avec class balance > 0
        rng2 = np.random.default_rng(7)
        y = pd.Series(rng2.integers(0, 2, len(X)), index=X.index)
        if y.sum() < 3 or (1 - y).sum() < 3:
            pytest.skip("class balance trop déséquilibré")

        model = train_meta_model(X, y)
        # Modèle doit produire des probabilités
        proba = model.predict_proba(X[FEATURE_NAMES].values)[:, 1]
        assert proba.shape == (len(X),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_filter_keeps_high_proba_trades(self) -> None:
        df = _synthetic_us500_h1(n_days=400, seed=4)
        macro = _synthetic_macro(n_days=420, seed=5)
        rng = np.random.default_rng(2)
        fomc = pd.DatetimeIndex(sorted(set([
            pd.Timestamp("2020-03-15 18:00", tz="UTC") + pd.Timedelta(days=int(d))
            for d in rng.integers(0, 200, 50)
        ])))
        X = build_features_at_entry(df, macro, fomc)
        X = X.dropna()
        if len(X) < 20:
            pytest.skip("pas assez d'events")
        rng2 = np.random.default_rng(9)
        y = pd.Series(rng2.integers(0, 2, len(X)), index=X.index)
        if y.sum() < 3 or (1 - y).sum() < 3:
            pytest.skip("class balance déséquilibré")

        model = train_meta_model(X, y)

        # Trades fictifs alignés sur X
        trades = [
            {"fomc_time": ts.isoformat(), "pips_net": 10.0 * (i % 2 - 0.5)}
            for i, ts in enumerate(X.index)
        ]

        kept_low, _ = filter_trades(trades, X, model, threshold=0.0)
        kept_high, _ = filter_trades(trades, X, model, threshold=0.99)
        assert len(kept_high) <= len(kept_low)


class TestCVThresholdSelection:
    def test_returns_threshold_in_grid(self) -> None:
        df = _synthetic_us500_h1(n_days=500, seed=10)
        macro = _synthetic_macro(n_days=520, seed=11)
        rng = np.random.default_rng(3)
        fomc = pd.DatetimeIndex(sorted(set([
            pd.Timestamp("2020-03-15 18:00", tz="UTC") + pd.Timedelta(days=int(d))
            for d in rng.integers(0, 350, 80)
        ])))
        X = build_features_at_entry(df, macro, fomc)
        X = X.dropna()
        if len(X) < 30:
            pytest.skip("pas assez d'events")

        rng2 = np.random.default_rng(13)
        y = pd.Series(rng2.integers(0, 2, len(X)), index=X.index)
        if y.sum() < 5 or (1 - y).sum() < 5:
            pytest.skip("class balance déséquilibré")

        pnls = pd.Series(rng2.normal(20.0, 50.0, len(X)), index=X.index)

        best, stats = cv_select_threshold(X, y, pnls)
        assert best in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
        assert "per_threshold" in stats
