"""Tests d'intégrité du pipeline lock étendu (C5)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.config.ml_pipeline_v4 import (
    LOCKED_COUPLES,
    PIPELINE_VERSION,
    get_pipeline,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _PROJECT_ROOT / "TEST_SET_LOCK.json"


def test_pipeline_version_extended() -> None:
    assert PIPELINE_VERSION == "v4.1.0-extended"


def test_a9_original_couples_present() -> None:
    """Les 3 couples A9 doivent rester dans LOCKED_COUPLES."""
    for key in [("US30", "D1"), ("EURUSD", "H4"), ("XAUUSD", "D1")]:
        assert key in LOCKED_COUPLES


@pytest.mark.parametrize("key", sorted(LOCKED_COUPLES))
def test_each_couple_loadable(key: tuple[str, str]) -> None:
    asset, tf = key
    cfg = get_pipeline(asset, tf)
    assert cfg.asset == asset
    assert cfg.tf == tf
    assert cfg.version == PIPELINE_VERSION
    assert 0.50 <= cfg.threshold <= 0.80
    assert len(cfg.features) == 15
    assert cfg.model_name in ("rf", "hgbm", "stacking")


def test_test_set_lock_has_pipeline_section() -> None:
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    pl = data["pipeline_locked"]
    assert pl["pipeline_version"] == "v4.1.0-extended"
    assert "phase_a_extended_completed_at" in pl


def test_checksums_match_current_files() -> None:
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    stored = data["pipeline_locked"]["config_checksums"]
    for stored_path, expected_hash in stored.items():
        rel = stored_path.replace("\\", "/")
        full = _PROJECT_ROOT / rel
        actual = hashlib.sha256(full.read_bytes()).hexdigest()
        assert actual == expected_hash, f"Drift sur {rel}"


def test_unknown_couple_raises() -> None:
    with pytest.raises(KeyError):
        get_pipeline("INEXISTANT", "D1")
