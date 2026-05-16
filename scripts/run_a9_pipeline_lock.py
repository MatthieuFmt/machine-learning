"""Pivot v4 A9 — Pipeline lock + checksum dans TEST_SET_LOCK.json."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config.ml_pipeline_v4 import (
    PIPELINE_VERSION,
    all_configured_pairs,
    get_pipeline,
)
from app.testing.snooping_guard import LOCK_PATH

CONFIG_FILES: list[Path] = [
    Path("app/config/features_selected.py"),
    Path("app/config/model_selected.py"),
    Path("app/config/hyperparams_tuned.py"),
    Path("app/config/ml_pipeline_v4.py"),
]


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    # Vérifier que tous les configs existent et sont cohérents
    pairs = all_configured_pairs()
    if not pairs:
        print("ERREUR : aucun (asset, tf) configuré. Re-exécuter A6/A7/A8.")
        return 1

    print(f"Pipelines configurés : {pairs}")
    for asset, tf in pairs:
        cfg = get_pipeline(asset, tf)
        print(
            f"  {asset} {tf}: {cfg.model_name} | "
            f"{len(cfg.features)} features | "
            f"threshold={cfg.threshold} | "
            f"expected_sharpe_outer={cfg.expected_sharpe_outer:.3f}"
        )

    # Calculer SHA256 de chaque fichier config
    checksums: dict[str, str] = {}
    for p in CONFIG_FILES:
        if not p.exists():
            print(f"ERREUR : fichier config manquant : {p}")
            return 1
        checksums[str(p)] = _sha256_of_file(p)
        print(f"  SHA256 {p} : {checksums[str(p)][:16]}...")

    # Écrire dans TEST_SET_LOCK.json
    state: dict
    if LOCK_PATH.exists():
        state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    else:
        state = {"locked": False, "n_reads": 0, "read_history": []}

    state["pipeline_locked"] = {
        "locked_at": datetime.now(UTC).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "configured_pairs": [{"asset": a, "tf": t} for (a, t) in pairs],
        "config_checksums": checksums,
    }
    LOCK_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[OK] Pipeline gele. {len(pairs)} configurations enregistrees dans {LOCK_PATH}.")
    print("[!] Toute modification de config = echec test_pipeline_integrity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
