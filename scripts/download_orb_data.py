#!/usr/bin/env python3
"""Téléchargeur intraday FIN (M5/M15/M1) par tranches annuelles — pour l'ORB.

Le téléchargeur principal (``download_dukascopy_full.py``) plafonne à H1 et fetch
en un seul appel (limit 200k), insuffisant pour du M5 (~72k bougies/an, ~1 M sur
14 ans). Ici on découpe ANNÉE PAR ANNÉE, on concatène, puis on valide via
``load_asset`` (qui sait déjà lire M5/M15).

⚠️ À lancer sur TA machine (le cloud n'a pas accès à Dukascopy).
⚠️ Les indices CFD M5 peuvent contenir des trous de flux : si la validation
   signale des « gaps anormaux », le fichier est CONSERVÉ (l'ORB n'utilise que
   les barres de séance).

USAGE :
    python scripts/download_orb_data.py --asset US500 --tf M5 --start 2015 --end 2026
    python scripts/download_orb_data.py --asset US30  --tf M5
    python scripts/download_orb_data.py --asset GER30 --tf M5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dukascopy_python  # noqa: E402

from app.core.exceptions import DataValidationError  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from scripts.download_dukascopy_full import MAPPINGS  # noqa: E402


def _resolve_interval(tf: str):
    """Résout la constante d'intervalle dukascopy_python (noms tolérants)."""
    candidates = {
        "M1": ("INTERVAL_MIN_1", "INTERVAL_MINUTE_1"),
        "M5": ("INTERVAL_MIN_5", "INTERVAL_MINUTE_5"),
        "M15": ("INTERVAL_MIN_15", "INTERVAL_MINUTE_15"),
        "M30": ("INTERVAL_MIN_30", "INTERVAL_MINUTE_30"),
    }
    for name in candidates.get(tf, ()):
        if hasattr(dukascopy_python, name):
            return getattr(dukascopy_python, name)
    raise SystemExit(
        f"Timeframe {tf} introuvable dans dukascopy_python "
        f"(constantes dispo : {[c for c in dir(dukascopy_python) if c.startswith('INTERVAL')]})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Téléchargeur intraday fin par année.")
    ap.add_argument("--asset", default="US500")
    ap.add_argument("--tf", default="M5")
    ap.add_argument("--start", type=int, default=2015, help="Année de début (incluse).")
    ap.add_argument("--end", type=int, default=2026, help="Année de fin (incluse).")
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    symbol = MAPPINGS.get(args.asset)
    if not symbol:
        raise SystemExit(f"Actif inconnu : {args.asset} (voir MAPPINGS).")
    interval = _resolve_interval(args.tf)

    frames: list[pd.DataFrame] = []
    for year in range(args.start, args.end + 1):
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        print(f"[*] {args.asset} ({symbol}) {args.tf} {year} …")
        try:
            df = dukascopy_python.fetch(
                instrument=symbol,
                interval=interval,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start,
                end=end,
                max_retries=args.max_retries,
                limit=500_000,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [-] {year} échec : {exc}")
            continue
        if df is None or df.empty:
            print(f"    [-] {year} : aucune donnée.")
            continue
        frames.append(df)
        print(f"    [+] {len(df)} bougies.")

    if not frames:
        raise SystemExit("Aucune donnée téléchargée — abandon.")

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full.index.name = "Time"
    full.columns = [c.title() for c in full.columns]

    ohlc = ["Open", "High", "Low", "Close"]
    n_before = len(full)
    full = full[(full[ohlc] > 0).all(axis=1)]
    if len(full) < n_before:
        print(f"[!] {n_before - len(full)} barres à prix ≤ 0 supprimées.")

    out_dir = ROOT / "data" / "raw" / args.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob(f"*_{args.tf}.csv"):
        print(f"[~] Suppression ancien fichier : {existing.name}")
        existing.unlink()
    out_file = out_dir / f"{args.asset}_{args.tf}.csv"
    full.to_csv(out_file, sep="\t", index=True)
    print(f"[+] {len(full)} bougies → {out_file}")

    try:
        load_asset(args.asset, args.tf)
        print("[+] Validé via load_asset.")
    except DataValidationError as exc:
        print(f"[!] Validation : {exc}")
        print("    Fichier CONSERVÉ — gaps intraday probables ; l'ORB n'utilise "
              "que la séance. Si le screen plante au chargement, dis-le moi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
