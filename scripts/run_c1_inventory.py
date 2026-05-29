"""Pivot v4 C1 — Inventaire des 21 couples (actif, TF) du projet.

⚠️ Train ≤ 2022-12-31 uniquement. Test set 2024+ JAMAIS lu.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.features_selected import FEATURES_SELECTED
from app.data.loader import load_asset
from app.data.registry import discover_assets
from app.features.superset import build_superset

CUTOFF_TRAIN = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# ── Suppression des logs parasites (gaps_anormaux, dropna, gaps_normaux, load_asset) ──
_LOADER_LOGGER = logging.getLogger("app.data.loader")
_LOADER_LOGGER.setLevel(logging.WARNING)


def main() -> int:
    available = discover_assets()  # {asset: [tf, ...]}
    target_assets = ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "US30", "USDCHF", "XAUUSD"]
    target_tfs = ["D1", "H4", "H1"]

    inventory: list[dict] = []
    for asset in target_assets:
        for tf in target_tfs:
            entry: dict = {"asset": asset, "tf": tf}
            if asset not in available or tf not in available.get(asset, []):
                entry.update({
                    "available": False,
                    "status": "data_missing",
                    "n_bars_total": 0,
                    "n_bars_train": 0,
                    "n_features_superset": 0,
                })
                inventory.append(entry)
                continue

            try:
                df = load_asset(asset, tf)
                df_train = df.loc[:CUTOFF_TRAIN]
                feat = build_superset(df_train, asset=asset)
                entry.update({
                    "available": True,
                    "first_date": str(df.index[0]),
                    "last_date": str(df.index[-1]),
                    "n_bars_total": int(len(df)),
                    "n_bars_train": int(len(df_train)),
                    "n_features_superset": int(feat.shape[1]),
                    "status": (
                        "existing_pipeline"
                        if (asset, tf) in FEATURES_SELECTED
                        else "new_phase_c"
                    ),
                })
            except Exception as exc:
                entry.update({
                    "available": False,
                    "status": "load_error",
                    "error": str(exc)[:300],
                    "error_type": type(exc).__qualname__,
                    "error_traceback": traceback.format_exc(),
                })
            inventory.append(entry)

    out_path = _PROJECT_ROOT / "predictions" / "c1_couples_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Tableau console ────────────────────────────────────────────────────
    print(f"{'Asset':<8} {'TF':<4} {'Avail':<6} {'NBars':<8} {'NTrain':<8} {'NFeat':<6} {'Status':<20} {'Erreur'}")
    print("-" * 130)
    for e in inventory:
        error_snippet = ""
        if e["status"] == "load_error":
            err_type = e.get("error_type", "")
            err_msg = e.get("error", "")
            error_snippet = f"[{err_type}] {err_msg[:80]}"
        print(
            f"{e['asset']:<8} {e['tf']:<4} {str(e['available']):<6} "
            f"{e.get('n_bars_total', 0):<8} {e.get('n_bars_train', 0):<8} "
            f"{e.get('n_features_superset', 0):<6} {e['status']:<20} {error_snippet}"
        )

    # ── Résumé ─────────────────────────────────────────────────────────────
    n_new = sum(1 for e in inventory if e["status"] == "new_phase_c")
    n_existing = sum(1 for e in inventory if e["status"] == "existing_pipeline")
    n_missing = sum(1 for e in inventory if e["status"] == "data_missing")
    n_load_error = sum(1 for e in inventory if e["status"] == "load_error")
    print()
    print(f"Résumé : {n_existing} déjà dans pipeline, {n_new} nouveaux pour Phase C, "
          f"{n_missing} data_missing, {n_load_error} load_error.")
    print(f"→ {n_new} couples à traiter en C2 (ranking).")

    # ── Détail des load_error ──────────────────────────────────────────────
    if n_load_error > 0:
        print()
        print("=" * 80)
        print(" DÉTAIL DES LOAD_ERROR")
        print("=" * 80)
        for e in inventory:
            if e["status"] == "load_error":
                print(f"\n── {e['asset']}/{e['tf']} ──")
                print(f"  Type    : {e.get('error_type', '?')}")
                print(f"  Message : {e.get('error', '?')}")
                tb = e.get("error_traceback", "")
                if tb:
                    # Afficher seulement les 3 dernières lignes pertinentes
                    tb_lines = tb.strip().split("\n")
                    relevant = [ln for ln in tb_lines if "DataValidationError" in ln or "raise" in ln or "Error" in ln]
                    if not relevant:
                        relevant = tb_lines[-3:]  # fallback: 3 dernières lignes
                    print("  Trace   :")
                    for line in relevant[:5]:
                        print(f"           {line.strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
