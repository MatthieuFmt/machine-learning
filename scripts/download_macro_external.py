#!/usr/bin/env python3
"""Pré-télécharge les séries macro externes (DXY, VIX, ^TNX, ^IRX) — Phase F5.

Stocke chaque série dans data/raw/macro/<NAME>.csv.

Usage :
    python scripts/download_macro_external.py
    python scripts/download_macro_external.py --refresh    # force re-download
    python scripts/download_macro_external.py --check      # affiche le résumé sans télécharger
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.features.macro_external import (
    MACRO_CACHE_DIR,
    SYMBOLS,
    build_macro_dataframe,
    load_macro_series,
)


def cmd_check() -> int:
    """Affiche l'état du cache local sans déclencher de download."""
    print(f"Cache : {MACRO_CACHE_DIR.resolve()}")
    if not MACRO_CACHE_DIR.exists():
        print("  (répertoire absent — rien en cache)")
        return 0
    for name in SYMBOLS:
        path = MACRO_CACHE_DIR / f"{name}.csv"
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {name:<5} → {path.name} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {name:<5} → absent")
    return 0


def cmd_download(refresh: bool) -> int:
    """Télécharge les 4 séries macro (ou recharge depuis cache si présent)."""
    print(f"[F5] Téléchargement des séries macro (refresh={refresh})…")
    for name, symbol in SYMBOLS.items():
        try:
            series = load_macro_series(name, refresh=refresh)
            print(
                f"  ✓ {name:<5} ({symbol:<12}) "
                f"n={len(series):>5}  "
                f"{series.index.min().date()} → {series.index.max().date()}"
            )
        except Exception as exc:
            print(f"  ✗ {name:<5} ({symbol:<12}) ÉCHEC : {exc}")
            return 1

    # Test de construction du DataFrame agrégé
    print("\nConstruction du DataFrame macro agrégé…")
    df = build_macro_dataframe(refresh=False)
    print(f"  Shape : {df.shape}")
    print(f"  Colonnes : {list(df.columns)}")
    print(f"  Période : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  NaN par colonne :")
    for col, n_nan in df.isna().sum().items():
        print(f"    {col:<25} {n_nan}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download même si le cache existe")
    parser.add_argument("--check", action="store_true",
                        help="Affiche l'état du cache sans télécharger")
    args = parser.parse_args()

    if args.check:
        return cmd_check()
    return cmd_download(args.refresh)


if __name__ == "__main__":
    sys.exit(main())
