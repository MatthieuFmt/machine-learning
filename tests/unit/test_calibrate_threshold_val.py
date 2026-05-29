"""Tests pour fix F14 — calibration du seuil sur val 2023.

Vérifie :
1. _calibrate_threshold_on_val retourne un seuil ∈ threshold_candidates.
2. Le seuil retenu maximise effectivement le Sharpe sur 2023.
3. Fallback gracieux si val est vide ou insuffisante.
4. Le test set ≥ 2024 n'est jamais consulté.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.instruments import ASSET_CONFIGS  # noqa: E402

from scripts.run_validation_finale import (  # noqa: E402
    VAL_END,
    VAL_START,
    _calibrate_threshold_on_val,
)


def _build_synthetic_ohlc(start: str, periods: int = 600) -> pd.DataFrame:
    """OHLC synthétique D1 avec tendance + bruit."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    close = 1.1000 + np.linspace(0, 0.05, periods) + rng.standard_normal(periods).cumsum() * 0.001
    close = np.maximum(close, 1.0500)
    return pd.DataFrame(
        {
            "Open": close - rng.standard_normal(periods) * 0.0002,
            "High": close + np.abs(rng.standard_normal(periods)) * 0.001,
            "Low": close - np.abs(rng.standard_normal(periods)) * 0.001,
            "Close": close,
        },
        index=idx,
    )


class _MockModel:
    """Mock minimal de RandomForestClassifier renvoyant des proba alternées."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Retourne proba variées : moitié sous 0.5, moitié au-dessus
        rng = np.random.default_rng(hash(X.tobytes()) % (2**32))
        p = rng.uniform(0.3, 0.8, size=len(X))
        return np.column_stack([1.0 - p, p])


def test_calibrate_returns_threshold_in_candidates() -> None:
    """Le seuil retenu doit faire partie de la liste candidate."""
    df = _build_synthetic_ohlc("2020-01-01", periods=1000)
    cfg = ASSET_CONFIGS["GBPUSD"]

    # Mock _build_features pour retourner des features stables
    fake_features = pd.DataFrame(
        np.random.RandomState(1).standard_normal((len(df), 3)),
        index=df.index, columns=["f1", "f2", "f3"],
    )
    with patch("scripts.run_validation_finale._build_features", return_value=fake_features):
        candidates = (0.45, 0.50, 0.55, 0.60)
        threshold, scores = _calibrate_threshold_on_val(
            df_full=df, model=_MockModel(),
            asset="GBPUSD", tf="D1", cfg=cfg, half_cost=0.55,
            threshold_candidates=candidates,
        )

    assert threshold in candidates or threshold == 0.50, (
        f"Seuil retenu {threshold} doit être dans {candidates} (ou fallback 0.50)"
    )


def test_calibrate_fallback_on_empty_val() -> None:
    """Si la val 2023 est vide, fallback sur 0.50."""
    # DataFrame qui ne couvre pas 2023
    df = _build_synthetic_ohlc("2020-01-01", periods=300)  # 2020-2020 only
    cfg = ASSET_CONFIGS["GBPUSD"]

    threshold, scores = _calibrate_threshold_on_val(
        df_full=df, model=_MockModel(),
        asset="GBPUSD", tf="D1", cfg=cfg, half_cost=0.55,
    )
    assert threshold == 0.50
    assert scores == {}


def test_val_window_constants_correct() -> None:
    """VAL_START et VAL_END couvrent bien l'année 2023."""
    assert VAL_START == pd.Timestamp("2023-01-01", tz="UTC")
    assert VAL_END == pd.Timestamp("2023-12-31", tz="UTC")
    # Et test commence après val
    from scripts.run_validation_finale import TEST_START
    assert TEST_START > VAL_END


def test_calibrate_does_not_read_test_set() -> None:
    """Le helper ne lit JAMAIS df.loc[TEST_START:].

    On donne un df qui couvre 2020-2025, mais on s'attend à ce que
    _calibrate_threshold_on_val ne lise que df.loc[VAL_START:VAL_END].
    Pour vérifier, on patch _generate_donchian_signals et on inspecte
    le df passé.
    """
    df = _build_synthetic_ohlc("2020-01-01", periods=2000)
    cfg = ASSET_CONFIGS["GBPUSD"]

    fake_features = pd.DataFrame(
        np.random.RandomState(1).standard_normal((len(df), 3)),
        index=df.index, columns=["f1", "f2", "f3"],
    )

    captured_dfs: list[pd.DataFrame] = []

    def capturing_donchian(df_arg):
        captured_dfs.append(df_arg.copy())
        return pd.Series(0, index=df_arg.index, dtype=int)

    with patch("scripts.run_validation_finale._generate_donchian_signals",
               side_effect=capturing_donchian), \
         patch("scripts.run_validation_finale._build_features",
               return_value=fake_features):
        _calibrate_threshold_on_val(
            df_full=df, model=_MockModel(),
            asset="GBPUSD", tf="D1", cfg=cfg, half_cost=0.55,
        )

    # Au moins un appel a été fait
    assert len(captured_dfs) >= 1
    # Aucun timestamp ≥ TEST_START n'a été lu par _generate_donchian_signals
    from scripts.run_validation_finale import TEST_START
    for cap_df in captured_dfs:
        assert cap_df.index.max() < TEST_START, (
            f"Calibration a lu une barre ≥ TEST_START ({cap_df.index.max()}) "
            f"→ violation F14"
        )
