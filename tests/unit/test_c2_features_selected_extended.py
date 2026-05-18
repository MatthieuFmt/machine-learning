"""Vérifie l'extension de FEATURES_SELECTED en C2 sans régression A6."""
from __future__ import annotations

import pandas as pd
import pytest

from app.config.features_selected import FEATURES_SELECTED
from app.data.loader import load_asset
from app.features.superset import build_superset

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# Entrées d'origine de A6 — doivent rester intactes
_A6_ORIGINAL = {
    ("US30", "D1"),
    ("EURUSD", "H4"),
    ("XAUUSD", "D1"),
}


def test_a6_original_entries_preserved() -> None:
    for key in _A6_ORIGINAL:
        assert key in FEATURES_SELECTED, f"{key} (A6) doit rester dans FEATURES_SELECTED"
        assert len(FEATURES_SELECTED[key]) == 15


@pytest.mark.parametrize("key", list(FEATURES_SELECTED.keys()))
def test_each_entry_has_15_unique_features(key: tuple[str, str]) -> None:
    feats = FEATURES_SELECTED[key]
    assert len(feats) == 15, f"{key} : {len(feats)} features ≠ 15"
    assert len(set(feats)) == 15, f"{key} : doublons détectés"


@pytest.mark.parametrize("key", list(FEATURES_SELECTED.keys()))
def test_features_exist_in_superset(key: tuple[str, str]) -> None:
    asset, tf = key
    try:
        df = load_asset(asset, tf)
    except Exception:
        pytest.skip(f"{asset}/{tf} : données indisponibles")
    df_train = df.loc[:CUTOFF]
    if len(df_train) < 300:
        pytest.skip(f"{asset}/{tf} : train trop court")
    feat = build_superset(df_train, asset=asset)
    available = set(feat.columns)
    missing = [f for f in FEATURES_SELECTED[key] if f not in available]
    assert not missing, f"{asset}/{tf} : features absentes du superset : {missing}"
