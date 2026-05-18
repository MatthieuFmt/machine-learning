"""Smoke test : build_superset() retourne ≥ 60 features sur tous les couples disponibles."""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.data.loader import load_asset
from app.data.registry import discover_assets
from app.features.superset import build_superset

CUTOFF_TRAIN = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# Paramètres : 21 couples cibles (7 actifs × 3 TF)
_AVAILABLE = discover_assets()
_COUPLES = [
    (asset, tf)
    for asset in ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "US30", "USDCHF", "XAUUSD"]
    for tf in ["D1", "H4", "H1"]
    if asset in _AVAILABLE and tf in _AVAILABLE.get(asset, [])
]


@pytest.mark.parametrize("asset,tf", _COUPLES)
def test_superset_min_60_features(asset: str, tf: str) -> None:
    try:
        df = load_asset(asset, tf)
    except (DataValidationError, Exception) as exc:
        pytest.skip(f"{asset}/{tf} : données invalides — {type(exc).__name__}: {exc}")
    df_train = df.loc[:CUTOFF_TRAIN]
    if len(df_train) < 250:
        pytest.skip(f"{asset}/{tf} : train trop court ({len(df_train)} barres)")
    feat = build_superset(df_train, asset=asset)
    assert feat.shape[1] >= 60, f"{asset}/{tf} : {feat.shape[1]} features < 60"


@pytest.mark.parametrize("asset,tf", _COUPLES)
def test_superset_no_nan_after_warmup(asset: str, tf: str) -> None:
    try:
        df = load_asset(asset, tf)
    except (DataValidationError, Exception) as exc:
        pytest.skip(f"{asset}/{tf} : données invalides — {type(exc).__name__}: {exc}")
    df_train = df.loc[:CUTOFF_TRAIN]
    if len(df_train) < 300:
        pytest.skip(f"{asset}/{tf} : train trop court")
    feat = build_superset(df_train, asset=asset)
    after_warmup = feat.iloc[250:]
    nan_cols = after_warmup.columns[after_warmup.isna().any()].tolist()
    allowed_prefixes = ("usdchf_", "xauusd_", "btcusd_")
    allowed_exact = {
        "mfi_14", "body_to_range_ratio", "upper_shadow_ratio",
        "lower_shadow_ratio", "volume_zscore_20", "range_atr_ratio",
    }
    forbidden = [
        c for c in nan_cols
        if not c.startswith(allowed_prefixes) and c not in allowed_exact
    ]
    assert not forbidden, f"{asset}/{tf} : NaN après warmup sur {forbidden}"
