"""Diagnostic des CSV BTCUSD bruts — validation et copie vers cleaned-data/.

Usage : python scripts/inspect_btc_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLS = {"Time", "Open", "High", "Low", "Close"}
TIMEFRAMES: list[str] = ["H1", "H4", "D1"]
SRC_DIR = Path("data")
DST_DIR = Path("cleaned-data")


def inspect_and_copy(tf: str) -> bool:
    """Valide BTCUSD_{tf}.csv et le copie dans cleaned-data/ si OK."""
    src = SRC_DIR / f"BTCUSD_{tf}.csv"
    dst = DST_DIR / f"BTCUSD_{tf}_cleaned.csv"

    if not src.exists():
        print(f"[SKIP] {src} — fichier absent")
        return False

    try:
        df = pd.read_csv(src, sep="\t", index_col=False)
    except Exception as e:
        print(f"[ERREUR] Lecture {src} : {e}")
        return False

    # Renommer la 7eme colonne en "Spread" si elle existe sans nom
    if len(df.columns) >= 7 and df.columns[6] != "Spread":
        df.columns = ["Time", "Open", "High", "Low", "Close", "Volume", "Spread"]
        print("    [INFO] Colonne 7 detectee -> renommee en 'Spread'")

    print(f"\n-- {src} --")
    print(f"    Lignes : {len(df)} | Colonnes : {list(df.columns)}")

    # Verifier colonnes obligatoires
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"[ERREUR] Colonnes manquantes : {missing}")
        return False

    # Parser les dates
    try:
        df["Time"] = pd.to_datetime(df["Time"])
    except Exception as e:
        print(f"[ERREUR] Parse datetime : {e}")
        return False

    # Trier par Time si necessaire
    if not df["Time"].is_monotonic_increasing:
        print("    [INFO] Time en ordre decroissant — tri ascendant applique")
        df = df.sort_values("Time").reset_index(drop=True)

    # Plage de dates
    print(f"    Debut : {df['Time'].min()} | Fin : {df['Time'].max()}")

    # Taux de NaN par colonne
    nan_pct = df.isna().mean() * 100
    for col, pct in nan_pct.items():
        if pct > 0:
            print(f"    NaN {col}: {pct:.2f}%")

    if "Volume" in df.columns:
        print(f"    Volume min={df['Volume'].min()} max={df['Volume'].max()}")

    if "Spread" in df.columns:
        print(f"    Spread min={df['Spread'].min()} max={df['Spread'].max()}")

    # Copier dans cleaned-data
    DST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"[OK] Copie -> {dst}")

    return True


def main() -> None:
    print("=== Diagnostic CSV BTCUSD ===")
    results: dict[str, bool] = {}
    for tf in TIMEFRAMES:
        results[tf] = inspect_and_copy(tf)

    print("\n=== Resume ===")
    for tf, ok in results.items():
        status = "OK" if ok else "ECHEC"
        print(f"  BTCUSD_{tf}: {status}")

    all_ok = all(results.values())
    if not all_ok:
        print("\nWARNING: Certains fichiers sont absents ou invalides.")
        sys.exit(1)


if __name__ == "__main__":
    main()
