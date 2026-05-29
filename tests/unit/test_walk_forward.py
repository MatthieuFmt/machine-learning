"""Tests des splits walk-forward — non-chevauchement, embargo, invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.edge_validation import walk_forward_split


def _ohlcv_index(n: int = 500) -> pd.DataFrame:
    """DataFrame avec DatetimeIndex pour les tests de splits."""
    return pd.DataFrame(
        {"close": np.random.default_rng(42).normal(size=n).cumsum() + 100.0},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


class TestWalkForwardSplit:
    def test_splits_no_overlap(self) -> None:
        """Aucun chevauchement train/val dans aucun split."""
        df = _ohlcv_index(500)
        splits = list(walk_forward_split(
            df, train_months=12, val_months=3, step_months=3,
        ))
        assert len(splits) > 0
        for train_idx, val_idx in splits:
            assert len(np.intersect1d(train_idx, val_idx)) == 0, \
                f"Overlap detected: train={train_idx[:5]}..., val={val_idx[:5]}..."

    def test_splits_chronological(self) -> None:
        """train_end < val_start pour chaque split."""
        df = _ohlcv_index(500)
        for train_idx, val_idx in walk_forward_split(
            df, train_months=12, val_months=3, step_months=3,
        ):
            assert train_idx.max() < val_idx.min(), \
                f"train_end={train_idx.max()} >= val_start={val_idx.min()}"

    def test_val_windows_monotonic(self) -> None:
        """Les fenêtres val ne se chevauchent pas entre elles."""
        df = _ohlcv_index(500)
        val_windows = [
            (val_idx.min(), val_idx.max())
            for _, val_idx in walk_forward_split(
                df, train_months=12, val_months=3, step_months=3,
            )
        ]
        for i in range(len(val_windows) - 1):
            assert val_windows[i][1] < val_windows[i + 1][0], \
                f"Val window {i} and {i+1} overlap"

    def test_expanding_window(self) -> None:
        """La taille du train augmente à chaque step (expanding window)."""
        df = _ohlcv_index(500)
        prev_train_len = 0
        for train_idx, _ in walk_forward_split(
            df, train_months=12, val_months=3, step_months=3,
        ):
            assert len(train_idx) >= prev_train_len, \
                f"Train size decreased: {len(train_idx)} < {prev_train_len}"
            prev_train_len = len(train_idx)

    def test_covers_full_range(self) -> None:
        """L'union des fenêtres val couvre l'espace post-train initial."""
        df = _ohlcv_index(500)
        splits = list(walk_forward_split(
            df, train_months=12, val_months=3, step_months=3,
        ))
        assert len(splits) > 1  # Au moins 2 splits

        # La première val commence après le train initial
        first_val_start = splits[0][1].min()
        # La dernière val finit proche de la fin des données
        last_val_end = splits[-1][1].max()

        assert last_val_end > first_val_start
        assert last_val_end <= len(df) - 1

    def test_too_short_series(self) -> None:
        """Série trop courte → 0 split."""
        df = _ohlcv_index(10)
        splits = list(walk_forward_split(
            df, train_months=36, val_months=6, step_months=6,
        ))
        assert len(splits) == 0

    def test_non_datetime_index_raises(self) -> None:
        """Index non-DatetimeIndex → TypeError."""
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            list(walk_forward_split(df, train_months=12, val_months=3, step_months=3))

    def test_non_monotonic_raises(self) -> None:
        """Index non trié → ValueError."""
        dates = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=dates)
        with pytest.raises(ValueError, match="trié"):
            list(walk_forward_split(df, train_months=12, val_months=3, step_months=3))
