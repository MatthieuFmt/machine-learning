"""Scanne le code source pour détecter des accès au test set après lock.

À ajouter dans `make verify` et la CI."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LOCK_PATH = Path("TEST_SET_LOCK.json")
SUSPICIOUS_PATTERNS = [
    re.compile(r'\.loc\[\s*["\']?(2024|2025|2026)'),
    re.compile(r'df\[df\.index\s*>=?\s*["\']?(2024|2025|2026)'),
    re.compile(r'index\.year\s*[><=!]+\s*202[4-9]'),
]
SCAN_DIRS = ["app", "scripts"]
ALLOWED_FILES = {"app/testing/snooping_guard.py"}


def main() -> int:
    if not LOCK_PATH.exists():
        print("TEST_SET_LOCK.json absent : pas de scan nécessaire.")
        return 0
    state = json.loads(LOCK_PATH.read_text())
    if not state.get("locked"):
        print("Test set non verrouillé : pas de scan nécessaire.")
        return 0

    offenders: list[tuple[Path, int, str]] = []
    for dir_name in SCAN_DIRS:
        for py in Path(dir_name).rglob("*.py"):
            if str(py).replace("\\", "/") in ALLOWED_FILES:
                continue
            for i, line in enumerate(py.read_text(encoding="utf-8").readlines(), 1):
                for pat in SUSPICIOUS_PATTERNS:
                    if pat.search(line):
                        offenders.append((py, i, line.strip()))

    if offenders:
        print("❌ Accès suspects au test set détectés après verrouillage :")
        for p, i, line in offenders:
            print(f"  {p}:{i}  {line}")
        return 1
    print("✅ Aucun accès suspect au test set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
