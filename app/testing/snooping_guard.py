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
