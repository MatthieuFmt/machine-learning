"""Diagnostic des 2 gaps anormaux dans data/raw/USDJPY/USDJPY_H1.csv.

But : afficher les timestamps précis et les durées des gaps non-explicables
par le calendrier de jours fériés, pour patcher app/config/calendar.py.

Lecture directe du CSV (sans passer par load_asset qui crash sur les gaps).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.calendar import is_market_open, is_normal_gap  # noqa: E402

ASSET = "USDJPY"
TF = "H1"
MAX_GAP_HOURS = 2 * 24  # tolérance H1 (cf loader.py)


def main() -> int:
    csv_path = Path("data") / "raw" / ASSET / f"{ASSET}_{TF}.csv"
    if not csv_path.exists():
        print(f"❌ Fichier introuvable : {csv_path}")
        return 1

    # Lecture adaptative (cohérente avec loader.py)
    import csv as _csv
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = _csv.reader(f, delimiter="\t")
        header = next(reader)
        first = next(reader)
    n_headers, n_data = len(header), len(first)

    if n_data > n_headers:
        col_names = (["Open", "High", "Low", "Close", "Volume", "Spread"]
                     if (n_data == 7 and n_headers == 6)
                     else [f"Col_{i}" for i in range(n_data - 1)])
        df = pd.read_csv(csv_path, sep="\t", index_col=0, names=col_names, skiprows=1)
    else:
        df = pd.read_csv(csv_path, sep="\t")

    if "Time" in df.columns or "time" in df.columns:
        time_col = "Time" if "Time" in df.columns else "time"
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

    print(f"USDJPY H1 chargé : {len(df)} barres")
    print(f"  Période : {df.index.min()} → {df.index.max()}")

    # ── Détection gaps ───────────────────────────────────────────────────
    gaps_hours = df.index.to_series().diff().dt.total_seconds() / 3600.0
    big_gaps = gaps_hours[gaps_hours > MAX_GAP_HOURS]
    print(f"\nGaps > {MAX_GAP_HOURS}h : {len(big_gaps)} au total")

    n_normal = 0
    n_abnormal = 0
    abnormal_list: list[dict] = []

    for ts in big_gaps.index:
        idx = df.index.get_loc(ts)
        prev_ts = df.index[idx - 1]
        gap_h = float(gaps_hours.loc[ts])
        is_normal = is_normal_gap(
            ASSET, prev_ts.to_pydatetime(), ts.to_pydatetime()
        )
        if is_normal:
            n_normal += 1
        else:
            n_abnormal += 1
            # Lister tous les jours non-couverts par calendar entre prev_ts et ts
            from datetime import timedelta
            cur = prev_ts.to_pydatetime() + timedelta(days=1)
            cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)
            end = ts.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
            uncovered_days = []
            while cur < end:
                if is_market_open(ASSET, cur):
                    uncovered_days.append(cur.date())
                cur += timedelta(days=1)
            abnormal_list.append({
                "prev_ts": prev_ts,
                "next_ts": ts,
                "gap_hours": gap_h,
                "uncovered_days": uncovered_days,
            })

    print(f"  Normaux (weekend/holiday)  : {n_normal}")
    print(f"  Anormaux (data manquante)  : {n_abnormal}")

    if abnormal_list:
        print("\n" + "═" * 70)
        print("DÉTAIL DES GAPS ANORMAUX")
        print("═" * 70)
        for i, g in enumerate(abnormal_list, 1):
            print(f"\nGap #{i} :")
            print(f"  Dernière barre : {g['prev_ts']}")
            print(f"  Barre suivante : {g['next_ts']}")
            print(f"  Durée          : {g['gap_hours']:.1f}h "
                  f"({g['gap_hours']/24:.1f} jours)")
            print(f"  Jours 'market open' selon calendar : {g['uncovered_days']}")

        print("\n" + "═" * 70)
        print("ACTIONS POSSIBLES")
        print("═" * 70)
        print("Option 1 — Ajouter ces jours dans XTB_HOLIDAYS[\"USDJPY\"] (calendar.py)")
        print("  si ce sont des jours fériés JP non couverts.")
        print("Option 2 — Ajouter une règle de gap historique dans is_normal_gap()")
        print("  si ce sont des lacunes Dukascopy connues.")

    return 0 if n_abnormal == 0 else 0  # exit 0 pour debug (toujours)


if __name__ == "__main__":
    sys.exit(main())
