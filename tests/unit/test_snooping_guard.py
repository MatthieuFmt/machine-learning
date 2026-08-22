"""Tests du registre anti-data-snooping.

⚠️ ISOLATION : ces tests doivent écrire dans un registre JETABLE. Ils
utilisaient `monkeypatch.chdir`, ce qui ne fonctionnait QUE parce que
`LOCK_PATH` était relatif au CWD — le bug même qui laissait un screen lancé
depuis un autre dossier créer un second registre en silence. Depuis que
`LOCK_PATH` est ancré à la racine du dépôt, l'isolation DOIT passer par un
monkeypatch explicite de l'attribut, sinon la suite de tests écraserait le
vrai TEST_SET_LOCK.json (gitignoré, donc irrécupérable).
"""
from __future__ import annotations

import json

import pytest

from app.testing import snooping_guard as guard
from app.testing.snooping_guard import TestSetSnoopingError


@pytest.fixture
def isolated_lock(tmp_path, monkeypatch):
    """Redirige le registre vers un fichier temporaire."""
    path = tmp_path / "TEST_SET_LOCK.json"
    monkeypatch.setattr(guard, "LOCK_PATH", path)
    return path


def test_lifecycle(isolated_lock):
    assert not guard.is_locked()
    guard.check_unlocked()  # no-op tant que non verrouillé
    guard.read_oos("07", "H06", sharpe=1.2, n_trades=40)
    assert isolated_lock.exists()
    state = json.loads(isolated_lock.read_text(encoding="utf-8"))
    assert state["n_reads"] == 1
    guard.lock("18")
    assert guard.is_locked()
    with pytest.raises(TestSetSnoopingError):
        guard.check_unlocked()


def test_read_oos_blocked_after_lock(isolated_lock):
    """Le verrou doit BLOQUER la lecture OOS, pas seulement la journaliser.

    Avant correction, `check_unlocked()` n'était appelé par aucun screen : on
    pouvait relire l'OOS indéfiniment après le verrou. Le « garde » était un
    carnet de bord, pas un garde.
    """
    guard.read_oos("07", "H06", sharpe=1.2, n_trades=40)
    guard.lock("18")
    with pytest.raises(TestSetSnoopingError):
        guard.read_oos("99", "post-lock", sharpe=9.9, n_trades=50)
    # La tentative bloquée ne doit RIEN ajouter au registre.
    state = json.loads(isolated_lock.read_text(encoding="utf-8"))
    assert state["n_reads"] == 1


def test_lock_path_is_repo_anchored_not_cwd_relative(monkeypatch):
    """Le registre doit être unique quel que soit le répertoire courant."""
    monkeypatch.delenv("TEST_SET_LOCK_PATH", raising=False)
    import importlib

    reloaded = importlib.reload(guard)
    try:
        assert reloaded.LOCK_PATH.is_absolute(), (
            "LOCK_PATH relatif au CWD : un screen lancé ailleurs forkerait le "
            "registre et sous-évaluerait n_trials."
        )
        assert reloaded.LOCK_PATH.name == "TEST_SET_LOCK.json"
        assert (reloaded.LOCK_PATH.parent / "app").is_dir(), (
            "LOCK_PATH doit être ancré à la racine du dépôt"
        )
    finally:
        importlib.reload(guard)


def test_n_trials_counters_disagree_by_design(isolated_lock):
    """Les deux compteurs doivent rester distincts et documentés.

    `n_trials_from_history` = lectures brutes (conservateur).
    `n_unique_hypotheses`   = (prompt, hypothesis) dédupliqués.
    Le harness utilise le second, un `validate_edge(n_trials=None)` nu le
    premier — d'où deux DSR différents pour le même run si on ne fixe pas
    explicitement n_trials.
    """
    guard.read_oos("s", "A", sharpe=1.0, n_trades=40)
    guard.read_oos("s", "A", sharpe=1.0, n_trades=40)  # même hypothèse relancée
    guard.read_oos("s", "B", sharpe=1.0, n_trades=40)
    assert guard.n_trials_from_history() == 3
    assert guard.n_unique_hypotheses() == 2
