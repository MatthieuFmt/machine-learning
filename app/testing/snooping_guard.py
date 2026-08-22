"""Anti-data-snooping mécanique. Étape critique pour la validité statistique."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

# Racine du dépôt : app/testing/snooping_guard.py -> parents[2].
# ⚠️ LOCK_PATH était relatif au CWD : lancer un screen depuis un autre dossier
#    créait SILENCIEUSEMENT un second registre, donc un n_trials sous-évalué et
#    un DSR trop favorable. Le registre doit être unique, quel que soit le CWD.
#    Surcharge possible via TEST_SET_LOCK_PATH (tests, environnements isolés).
_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(os.environ.get("TEST_SET_LOCK_PATH") or (_REPO_ROOT / "TEST_SET_LOCK.json"))

# Début de l'ancien test set. ⚠️ BRÛLÉ : 88 lectures enregistrées, 45 hypothèses
# distinctes. Conservé pour lecture de l'historique, PAS comme fenêtre valide.
TEST_START = "2024-01-01"

# Seule fenêtre encore vierge. Voir `oos_power_warning` : elle est trop courte
# pour porter un verdict (~0.38 an, alors qu'un DSR non-NaN exige ≥ 31 trades).
VIRGIN_OOS_START = "2026-01-01"


class TestSetSnoopingError(Exception):
    """Levée si une modification post-lock est tentée."""


def _load() -> dict:
    if LOCK_PATH.exists():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {"locked": False, "n_reads": 0, "read_history": []}


def _save(state: dict) -> None:
    LOCK_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def is_locked() -> bool:
    return _load().get("locked", False)


def read_oos(
    prompt: str,
    hypothesis: str,
    sharpe: float,
    n_trades: int | None = None,
) -> None:
    """À appeler à CHAQUE lecture du test set OOS.

    Raises:
        TestSetSnoopingError: si le registre est verrouillé. Lire l'OOS après
            le verrou EST exactement le data-snooping que le verrou interdit —
            `check_unlocked()` existait mais n'était appelé par AUCUN screen,
            donc le « garde » n'était qu'un carnet de bord.
    """
    check_unlocked()
    state = _load()
    state["n_reads"] += 1
    state["read_history"].append({
        "prompt": prompt,
        "hypothesis": hypothesis,
        "timestamp": datetime.now(UTC).isoformat(),
        "sharpe": sharpe,
        "n_trades": n_trades,
    })
    _save(state)


def lock(prompt: str) -> None:
    """À appeler une seule fois, au prompt 18 (validation finale GO)."""
    state = _load()
    state["locked"] = True
    state["locked_at"] = datetime.now(UTC).isoformat()
    state["locked_by_prompt"] = prompt
    _save(state)


def check_unlocked() -> None:
    """À appeler en haut de tout script qui modifie une stratégie/feature/seuil."""
    if is_locked():
        raise TestSetSnoopingError(
            "TEST_SET_LOCK.json est verrouillé. Modifier la stratégie après lecture finale = "
            "data snooping. Pour itérer, il faut un nouveau split temporel (split ≥ 2026)."
        )


def get_history() -> list[dict]:
    return _load().get("read_history", [])


def n_trials_from_history(min_floor: int = 1) -> int:
    """Retourne n_trials = nb de lectures OOS loggées (fix F5).

    Bailey & López de Prado définissent N comme le nombre de configurations
    testées sur le test set. Chaque appel à `read_oos()` correspond à une
    configuration consultée → n_reads est la borne basse stricte.

    Args:
        min_floor: Plancher minimum (au cas où l'historique n'aurait pas
            été correctement maintenu pendant les premières itérations).

    Returns:
        max(n_reads, min_floor).
    """
    state = _load()
    n_reads = int(state.get("n_reads", 0))
    return max(n_reads, min_floor)


def n_unique_hypotheses() -> int:
    """Compte les hypothèses uniques (clé (prompt, hypothesis)).

    Alternative moins conservatrice à `n_trials_from_history` : si plusieurs
    `read_oos` partagent le même (prompt, hypothesis), on les compte comme
    une seule configuration testée.

    Returns:
        Nombre d'hypothèses distinctes dans read_history.
    """
    history = _load().get("read_history", [])
    keys = {(h.get("prompt"), h.get("hypothesis")) for h in history}
    return len(keys)
