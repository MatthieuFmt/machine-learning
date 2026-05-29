"""Tests unitaires pour app.pipelines.walk_forward_rolling — Pivot v4 B3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.instruments import ASSET_CONFIGS
from app.pipelines.walk_forward_rolling import RollingSegment, walk_forward_rolling
from app.strategies.donchian import DonchianBreakout


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """DataFrame OHLC synthétique de 6 ans (2019-2025), 1 barre/jour."""
    rng = np.random.default_rng(0)
    n = 365 * 6
    idx = pd.date_range("2019-01-01", periods=n, freq="D", tz="UTC")
    price = 40_000 + rng.normal(0, 50, n).cumsum()
    high_adj = rng.uniform(50, 200, n)
    low_adj = rng.uniform(50, 200, n)
    return pd.DataFrame(
        {
            "Open": price,
            "Close": price,
            "High": price + high_adj,
            "Low": price - low_adj,
            "Volume": 1.0,
        },
        index=idx,
    )


def _simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature builder minimal pour les tests."""
    return pd.DataFrame({"close": df["Close"]}, index=df.index)


def _simple_target(_df: pd.DataFrame, pnl: pd.Series) -> pd.Series:
    """Target builder minimal."""
    return (pnl > 0).astype(int)


def _make_strat():
    """Donchian(20, 20) standard."""
    return DonchianBreakout(N=20, M=20)


def test_returns_tuple(synthetic_df: pd.DataFrame) -> None:
    """Le walk_forward_rolling retourne bien (DataFrame, list[RollingSegment])."""
    trades, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["US30"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
    )
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(segments, list)
    if segments:
        assert all(isinstance(s, RollingSegment) for s in segments)


def test_train_window_max_3_years(synthetic_df: pd.DataFrame) -> None:
    """La fenêtre train ne dépasse pas train_window_years * 365 + 30 jours de marge."""
    _, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["US30"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
    )
    for s in segments:
        delta_days = (s.train_end - s.train_start).days
        assert delta_days <= 3 * 365 + 30, (
            f"Train > 3 ans : {delta_days} jours "
            f"({s.train_start.date()} → {s.train_end.date()})"
        )


def test_no_temporal_leak(synthetic_df: pd.DataFrame) -> None:
    """train_end < oos_start et embargo ≥ 2 jours."""
    _, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["US30"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
        embargo_days=2,
    )
    for s in segments:
        assert s.train_end < s.oos_start, (
            f"Leak : train_end {s.train_end.date()} >= oos_start {s.oos_start.date()}"
        )
        gap = (s.oos_start - s.train_end).days
        assert gap >= 2, (
            f"Embargo non respecté : {gap} jours "
            f"(train_end={s.train_end.date()}, oos_start={s.oos_start.date()})"
        )


def test_oos_after_test_start(synthetic_df: pd.DataFrame) -> None:
    """Tous les segments OOS commencent ≥ test_start."""
    _, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["US30"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
    )
    test_start_ts = pd.Timestamp("2024-01-01")
    for s in segments:
        assert s.oos_start >= test_start_ts, (
            f"OOS avant test_start: {s.oos_start.date()}"
        )


def test_no_overlap_between_segments(synthetic_df: pd.DataFrame) -> None:
    """Les segments OOS ne se chevauchent pas."""
    _, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["US30"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
    )
    prev_oos_end: pd.Timestamp | None = None
    for s in segments:
        if prev_oos_end is not None:
            assert s.oos_start > prev_oos_end, (
                f"Chevauchement segments: prev_end={prev_oos_end.date()}, "
                f"cur_start={s.oos_start.date()}"
            )
        prev_oos_end = s.oos_end


def test_xauusd_config(synthetic_df: pd.DataFrame) -> None:
    """Le walk_forward_rolling fonctionne avec la config XAUUSD."""
    trades, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["XAUUSD"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
    )
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(segments, list)


def test_empty_oos_returns_empty_dataframe(synthetic_df: pd.DataFrame) -> None:
    """Si test_start est après la fin des données, retourne un DataFrame vide."""
    trades, segments = walk_forward_rolling(
        df=synthetic_df,
        strat=_make_strat(),
        cfg=ASSET_CONFIGS["US30"],
        feature_builder=_simple_features,
        target_builder=_simple_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2030-01-01",
    )
    assert trades.empty or len(segments) == 0
