"""Pivot v4 C5 — Pipeline lock étendu (extension A9 multi-actifs).

⚠️ 0 n_trial consommé. Aucune lecture du test set 2024+.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.ml_pipeline_v4 import (
    LOCKED_COUPLES,
    PIPELINE_VERSION,
    get_pipeline,
    list_locked_couples,
)

CONFIG_FILES = [
    "app/config/features_selected.py",
    "app/config/model_selected.py",
    "app/config/hyperparams_tuned.py",
    "app/config/ml_pipeline_v4.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock_path = _PROJECT_ROOT / "TEST_SET_LOCK.json"
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))

    now_iso = datetime.now(timezone.utc).isoformat()
    couples = list_locked_couples()

    checksums: dict[str, str] = {}
    for rel in CONFIG_FILES:
        full = _PROJECT_ROOT / rel
        # Use backslash path separator for Windows consistency with existing A9 checksums
        checksums[rel.replace("/", "\\")] = _sha256(full)

    pl = lock_data.setdefault("pipeline_locked", {})
    pl["pipeline_version"] = PIPELINE_VERSION
    pl["configured_pairs"] = [{"asset": a, "tf": tf} for (a, tf) in couples]
    pl["config_checksums"] = checksums
    pl.setdefault("locked_at", now_iso)  # ne pas écraser l'horodatage A9
    pl["phase_a_extended_completed_at"] = now_iso

    lock_path.write_text(
        json.dumps(lock_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Sanity check : tous les couples doivent être loadables
    failures: list[str] = []
    for (asset, tf) in couples:
        try:
            cfg = get_pipeline(asset, tf)
            assert cfg.version == PIPELINE_VERSION
        except Exception as exc:
            failures.append(f"{asset}/{tf} : {exc}")

    if failures:
        print("ÉCHEC : couples non chargeables")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"Pipeline lock étendu OK (version {PIPELINE_VERSION})")
    print(f"  {len(couples)} couples figés :")
    for (a, tf) in couples:
        cfg = get_pipeline(a, tf)
        print(
            f"    {a}/{tf}: {cfg.model_name} threshold={cfg.threshold} "
            f"sharpe_outer={cfg.expected_sharpe_outer:.2f}"
        )
    print("  Checksums :")
    for f, h in checksums.items():
        print(f"    {f} : {h[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
