"""Anti-data-snooping mécanique. Étape critique pour la validité statistique."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

LOCK_PATH = Path("TEST_SET_LOCK.json")
TEST_START = "2024-01-01"


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
    """À appeler à CHAQUE lecture du test set OOS (≥ 2024)."""
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
