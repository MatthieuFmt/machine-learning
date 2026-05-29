"""Inspecte les CSV XAUUSD H4/D1 — validation préalable au pipeline v2.

Usage : python scripts/inspect_xauusd_csv.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"Time", "Open", "High", "Low", "Close"}
OPTIONAL_COLS = {"Volume", "Spread"}
CSV_FILES = ["data/XAUUSD_H4.csv", "data/XAUUSD_D1.csv"]
CLEANED_DIR = Path("cleaned-data")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def inspect_csv(path: str) -> pd.DataFrame | None:
    """Charge et valide un CSV XAUUSD.

    Returns:
        DataFrame parsé ou None si absent/invalide.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.error("Fichier introuvable : %s", file_path.absolute())
        return None

    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["Time", "Open", "High", "Low", "Close", "Volume", "Spread"],
            skiprows=1,
        )
    except Exception as exc:
        logger.error("Erreur lecture %s : %s", file_path, exc)
        return None

    cols = set(df.columns)
    missing = REQUIRED_COLS - cols
    if missing:
        logger.error(
            "%s : colonnes obligatoires manquantes : %s",
            file_path.name,
            sorted(missing),
        )
        return None

    has_volume = bool(OPTIONAL_COLS & cols)

    # Parsing du temps
    try:
        df["Time"] = pd.to_datetime(df["Time"])
    except Exception as exc:
        logger.error("%s : échec parsing colonne Time : %s", file_path.name, exc)
        return None

    df = df.set_index("Time").sort_index()

    # Monotonie
    if not df.index.is_monotonic_increasing:
        logger.error(
            "%s : index temporel NON monotone — doublons détectés", file_path.name
        )
        dups = df.index.duplicated().sum()
        logger.info("  → %d doublons, conservation du dernier", dups)
        df = df[~df.index.duplicated(keep="last")]
        if not df.index.is_monotonic_increasing:
            logger.error("  → toujours non monotone après dédup")
            return None

    # Métriques
    n_bars = len(df)
    date_start = df.index.min().strftime("%Y-%m-%d")
    date_end = df.index.max().strftime("%Y-%m-%d")
    ohlc_cols = list(REQUIRED_COLS & cols - {"Time"})
    nan_rate = df[ohlc_cols].isna().mean().mean() * 100 if ohlc_cols else 0.0

    logger.info("─" * 60)
    logger.info("Fichier : %s", file_path.name)
    logger.info("  Barres    : %d", n_bars)
    logger.info("  Période   : %s → %s", date_start, date_end)
    logger.info("  NaN rate  : %.2f%%", nan_rate)
    if has_volume:
        vol_zero_rate = (df["Volume"] == 0).mean() * 100
        logger.info("  Vol=0     : %.2f%%", vol_zero_rate)
    logger.info("  Colonnes  : %s", sorted(cols))
    logger.info("  VALIDE ✓")

    return df


def save_cleaned(df: pd.DataFrame, name: str) -> None:
    """Sauvegarde le DataFrame nettoyé dans cleaned-data/."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    dest = CLEANED_DIR / name
    df.to_csv(dest, index=True)
    logger.info("  → Sauvegardé : %s", dest)


def main() -> int:
    setup_logging()
    logger.info("=== Inspection CSV XAUUSD H4 / D1 ===")

    any_valid = False
    for csv_path in CSV_FILES:
        df = inspect_csv(csv_path)
        if df is not None:
            any_valid = True
            tf = "D1" if "D1" in csv_path else "H4"
            cleaned_name = f"XAUUSD_{tf}_cleaned.csv"
            save_cleaned(df, cleaned_name)

    if not any_valid:
        logger.error("=" * 60)
        logger.error("DONNÉES XAUUSD INDISPONIBLES")
        logger.error("Aucun CSV XAUUSD trouvé dans data/")
        logger.error("Vérifier data/ ou fournir les CSV.")
        logger.error("=" * 60)
        return 1

    logger.info("=== Inspection terminée avec succès ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
