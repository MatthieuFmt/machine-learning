"""Tests pour fix F7 — cross_asset_features anti look-ahead.

Avant le fix : `reindex(method="ffill")` sans shift pouvait laisser une
barre H1 du jour J récupérer la close D1 du même jour J (donnée du futur
si la convention CSV est end-of-day).

Après : `shift(1)` avant le ffill garantit que la valeur D1 du jour J
n'est visible qu'à partir de J+1 00:00.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.features.superset import cross_asset_features


def _make_macro_d1(n_days: int = 30) -> pd.DataFrame:
    """Construit un mock DataFrame D1 avec Close monotone croissant."""
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    close = np.arange(1.0, 1.0 + 0.1 * n_days, 0.1)[:n_days]
    return pd.DataFrame(
        {
            "Open": close, "High": close + 0.005,
            "Low": close - 0.005, "Close": close,
        },
        index=idx,
    )


def test_cross_asset_shift_1_no_same_day_leak() -> None:
    """L'H1 du jour J ne doit JAMAIS voir la close D1 du jour J.

    Construction : un D1 strictement croissant. Pour chaque H1 du jour J,
    la valeur doit correspondre à un log-return D1 calculé sur des données
    se terminant au plus tard au jour J-1.
    """
    df_macro = _make_macro_d1(n_days=30)
    # Index H1 de 7 jours
    h1_idx = pd.date_range(
        "2024-01-10", periods=7 * 24, freq="h", tz="UTC",
    )

    # Mock load_asset/discover_assets pour ne pas dépendre du système de fichiers
    with patch("app.data.loader.load_asset", return_value=df_macro), \
         patch("app.data.registry.discover_assets",
               return_value={"USDCHF": ["D1"], "XAUUSD": ["D1"]}):
        out = cross_asset_features(h1_idx, asset="EURUSD")

    assert "usdchf_return_5" in out.columns

    # log_return_5 attendu : log(close[J-1] / close[J-6])
    # Avec close croissant linéairement, return est positif et stable.
    # Pour H1 du 2024-01-10, la valeur doit être log(close[2024-01-09]/close[2024-01-04])
    # = log(1.9 / 1.4) ≈ 0.3055

    val_2024_01_10 = out.loc["2024-01-10 00:00:00+00:00", "usdchf_return_5"]
    expected = float(np.log(df_macro.loc["2024-01-09", "Close"] / df_macro.loc["2024-01-04", "Close"]))

    assert val_2024_01_10 == pytest.approx(expected, rel=1e-6), (
        f"H1 du 10/01 doit voir le D1 du 09/01 (passé), pas du 10/01. "
        f"Observé {val_2024_01_10:.6f}, attendu {expected:.6f}"
    )


def test_cross_asset_value_constant_within_day() -> None:
    """Toutes les H1 d'un même jour J reçoivent la même valeur D1 (la veille)."""
    df_macro = _make_macro_d1(n_days=30)
    h1_idx = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")

    with patch("app.data.loader.load_asset", return_value=df_macro), \
         patch("app.data.registry.discover_assets",
               return_value={"USDCHF": ["D1"]}):
        out = cross_asset_features(h1_idx, asset="EURUSD")

    values = out["usdchf_return_5"].dropna().unique()
    assert len(values) == 1, (
        f"Toutes les H1 du même jour doivent avoir la même valeur D1 ffill, "
        f"observé {len(values)} valeurs distinctes : {values}"
    )


def test_cross_asset_self_excluded() -> None:
    """Si asset = USDCHF, usdchf_return_5 doit être NaN."""
    df_macro = _make_macro_d1(n_days=30)
    h1_idx = pd.date_range("2024-01-10", periods=24, freq="h", tz="UTC")

    with patch("app.data.loader.load_asset", return_value=df_macro), \
         patch("app.data.registry.discover_assets",
               return_value={"USDCHF": ["D1"]}):
        out = cross_asset_features(h1_idx, asset="USDCHF")

    assert out["usdchf_return_5"].isna().all(), (
        "asset=USDCHF doit donner NaN sur usdchf_return_5 (auto-référence interdite)"
    )
