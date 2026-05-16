"""Test d'intégrité du pipeline gelé (pivot v4 A9).

Si ce test échoue : un fichier config a été modifié après A9 = data snooping potentiel.
Pour résoudre : soit revert les modifications, soit re-faire un nouveau pivot complet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.testing.snooping_guard import LOCK_PATH

CONFIG_FILES: list[Path] = [
    Path("app/config/features_selected.py"),
    Path("app/config/model_selected.py"),
    Path("app/config/hyperparams_tuned.py"),
    Path("app/config/ml_pipeline_v4.py"),
]


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_locked_section_present() -> None:
    """TEST_SET_LOCK.json doit contenir une section pipeline_locked après A9."""
    assert LOCK_PATH.exists(), "TEST_SET_LOCK.json absent — A9 pas exécuté"
    state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert "pipeline_locked" in state, "Pas de section pipeline_locked — A9 pas exécuté"


@pytest.mark.parametrize("config_path", CONFIG_FILES)
def test_config_checksum_unchanged(config_path: Path) -> None:
    """Chaque config doit avoir le même SHA256 qu'au moment du lock."""
    state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if "pipeline_locked" not in state:
        pytest.skip("Pipeline pas encore locked (A9 non exécuté)")

    expected = state["pipeline_locked"]["config_checksums"].get(str(config_path))
    if expected is None:
        pytest.skip(f"{config_path} pas dans les checksums lockés")
    actual = _sha256_of_file(config_path)
    assert actual == expected, (
        f"⚠️ {config_path} a été MODIFIÉ depuis le pipeline lock.\n"
        f"Expected SHA256 : {expected}\n"
        f"Actual SHA256   : {actual}\n"
        f"= data snooping potentiel. Revert le fichier ou refaire un nouveau pivot."
    )


def test_pipeline_version_locked() -> None:
    """La version dans TEST_SET_LOCK.json doit correspondre à PIPELINE_VERSION."""
    state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if "pipeline_locked" not in state:
        pytest.skip("Pipeline pas encore locked")
    from app.config.ml_pipeline_v4 import PIPELINE_VERSION

    assert state["pipeline_locked"]["pipeline_version"] == PIPELINE_VERSION
