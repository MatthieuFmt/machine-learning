"""Télécharge les actifs H06 manquants via yfinance.

Format de sortie : data/raw/<ASSET>/<ASSET>_D1.csv
Structure compatible avec load_asset() : Time,Open,High,Low,Close,Volume

Usage :
    python scripts/download_h06_missing_assets.py           # force re-download
    python scripts/download_h06_missing_assets.py --force   # idem
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ajouter la racine projet au PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mapping actif → ticker yfinance
# BUND indisponible sur yfinance (delisté sur tous les tickers testés)
ASSET_TICKERS: dict[str, str] = {
    "GER30": "^GDAXI",    # DAX
    "USOIL": "CL=F",       # WTI Crude Oil futures
}

# Période de téléchargement : max historique
START_DATE = "2000-01-01"
END_DATE = "2026-01-01"

# Colonnes OHLCV PascalCase attendues par load_asset()
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def download_asset(asset: str, ticker: str, data_root: Path, force: bool = False) -> bool:
    """Télécharge et sauvegarde un actif au format D1 (PascalCase OHLCV)."""
    asset_dir = data_root / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    output_path = asset_dir / f"{asset}_D1.csv"

    if output_path.exists() and not force:
        print(f"[SKIP] {asset} : déjà présent → {output_path}")
        return True

    print(f"[DOWNLOAD] {asset} ← ticker={ticker} ...")
    try:
        # auto_adjust=False préserve les colonnes PascalCase
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        print(f"[ERROR] {asset} : yfinance échec → {exc}")
        return False

    if df is None or df.empty:
        print(f"[ERROR] {asset} : DataFrame vide — ticker {ticker} invalide ?")
        return False

    # Aplatir le MultiIndex si présent
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normaliser les noms de colonnes (PascalCase)
    rename: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower()
        for ohlc in OHLCV_COLS:
            if col_lower == ohlc.lower():
                rename[col] = ohlc
                break

    missing = [c for c in OHLCV_COLS if c not in rename.values()]
    if missing:
        print(f"[ERROR] {asset} : colonnes OHLCV manquantes {missing} (colonnes dispo: {df.columns.tolist()})")
        return False

    df = df.rename(columns=rename)[OHLCV_COLS].copy()

    # Index → colonne Time (sans timezone)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Time"
    df = df.reset_index()

    # Time en YYYY-MM-DD HH:MM:SS
    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Supprimer les lignes avec OHLCV NaN (weekends, holidays)
    df = df.dropna(subset=OHLCV_COLS)

    df.to_csv(output_path, index=False, sep="\t")
    n_rows = len(df)
    date_range = f"{df['Time'].iloc[0]} → {df['Time'].iloc[-1]}"
    print(f"[OK] {asset} : {n_rows} barres D1, {date_range} → {output_path}")
    return True


def main() -> None:
    data_root = _PROJECT_ROOT / "data" / "raw"
    force = "--force" in sys.argv

    success = 0
    fail = 0
    for asset, ticker in ASSET_TICKERS.items():
        if download_asset(asset, ticker, data_root, force=force):
            success += 1
        else:
            fail += 1

    print(f"\nRésultat : {success} succès, {fail} échecs sur {len(ASSET_TICKERS)} actifs.")

    if fail > 0:
        print("\n⚠️  Actifs manquants. Vérifier les tickers dans ASSET_TICKERS.")


if __name__ == "__main__":
    main()
