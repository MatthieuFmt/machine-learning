#!/usr/bin/env python3
"""Vérifie le calendrier FOMC local contre une référence indépendante.

POURQUOI
--------
Tout le résultat pre-FOMC repose sur les dates lues dans
`data/raw/economic_calendar/<ANNÉE>.csv` (scrape Forex Factory). Si ce scrape a
des dates manquantes, décalées ou en double, le backtest mesure autre chose que
ce qu'on croit — et **aucun test statistique ne peut détecter ça**. C'est le
même angle mort que les coûts estimés : une donnée jamais recoupée.

Ce script compare les dates du calendrier local à une liste de référence
indépendante et signale : manquants, en trop, doublons.

SOURCE DE LA RÉFÉRENCE
----------------------
- **2010-2018 (72 dates) : VÉRIFIÉES** — extraites de `FOMCscrape` (scrape du
  site de la Fed, `Scheduled == 1`), archivé dans `data/vendor/`.
  https://github.com/tobiasi/FOMCscrape
- **2019-2026 : À VÉRIFIER** — le jeu ci-dessus s'arrête en 2018 et
  federalreserve.gov est inaccessible depuis cet environnement. Ces dates
  proviennent de sources publiques secondaires. Les traiter comme *indicatives*
  jusqu'à recoupement manuel sur
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

USAGE :
    python scripts/verify_fomc_calendar.py
    python scripts/verify_fomc_calendar.py --data-root data/raw --from-year 2010
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Dates d'ANNONCE (dernier jour de réunion), format ISO ────────────────────
# Bloc VÉRIFIÉ : scrape du site de la Fed via tobiasi/FOMCscrape (Scheduled==1).
FOMC_VERIFIED_2010_2018: tuple[str, ...] = (
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
)

# Bloc INDICATIF (⚠️ non recoupé sur federalreserve.gov depuis cet environnement).
FOMC_UNVERIFIED_2019_2026: tuple[str, ...] = (
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 : issu d'une recherche web du 2026-08-01 (réunions sur 2 jours,
    # la date retenue est le SECOND jour = jour de l'annonce).
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
)


def load_local_fomc_dates(calendar_root: Path, event_name: str) -> pd.DatetimeIndex:
    """Dates (jour, sans heure) des événements `event_name` du calendrier local."""
    rows: list[pd.DataFrame] = []
    for path in sorted(calendar_root.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["date", "event"])
        except Exception:  # noqa: BLE001 — CSV sans ces colonnes : on saute
            continue
        rows.append(df[df["event"] == event_name])
    if not rows:
        return pd.DatetimeIndex([])
    dates = pd.to_datetime(pd.concat(rows, ignore_index=True)["date"], errors="coerce")
    return pd.DatetimeIndex(dates.dropna().dt.normalize())


def main() -> int:
    p = argparse.ArgumentParser(description="Vérifie le calendrier FOMC local.")
    p.add_argument("--data-root", default=Path("data/raw"), type=Path)
    p.add_argument("--event", default="FOMC Statement")
    p.add_argument("--from-year", default=2010, type=int)
    p.add_argument("--to-year", default=2026, type=int)
    args = p.parse_args()

    cal_root = args.data_root / "economic_calendar"
    if not cal_root.exists():
        print(f"❌ Calendrier introuvable : {cal_root}")
        print("   Ce script tourne sur la machine où vivent les CSV.")
        return 1

    ref_all = list(FOMC_VERIFIED_2010_2018) + list(FOMC_UNVERIFIED_2019_2026)
    ref = pd.DatetimeIndex(sorted(pd.to_datetime(ref_all)))
    ref = ref[(ref.year >= args.from_year) & (ref.year <= args.to_year)]
    verified_cut = pd.Timestamp("2019-01-01")

    local = load_local_fomc_dates(cal_root, args.event)
    local = local[(local.year >= args.from_year) & (local.year <= args.to_year)]

    print("=" * 74)
    print(f"VÉRIFICATION DU CALENDRIER — « {args.event} », {args.from_year}-{args.to_year}")
    print(f"  référence : {len(ref)} dates  ({(ref < verified_cut).sum()} vérifiées "
          f"+ {(ref >= verified_cut).sum()} indicatives)")
    print(f"  local     : {len(local)} entrées ({local.nunique()} dates uniques)")
    print("=" * 74)

    dupes = local[local.duplicated()].unique()
    missing = ref.difference(local)
    extra = pd.DatetimeIndex(local.unique()).difference(ref)

    def show(title: str, idx, warn: bool) -> None:
        mark = "⚠️ " if (warn and len(idx)) else ("✅ " if not len(idx) else "ℹ️  ")
        print(f"\n{mark}{title} : {len(idx)}")
        for d in idx[:20]:
            flag = "" if pd.Timestamp(d) < verified_cut else "   (zone indicative)"
            print(f"     {pd.Timestamp(d).date()}{flag}")
        if len(idx) > 20:
            print(f"     … et {len(idx) - 20} autres")

    show("Dates de référence ABSENTES du calendrier local", missing, warn=True)
    show("Dates locales ABSENTES de la référence", extra, warn=True)
    show("Doublons dans le calendrier local", pd.DatetimeIndex(dupes), warn=True)

    print("\n" + "=" * 74)
    n_ver_missing = (missing < verified_cut).sum() if len(missing) else 0
    if n_ver_missing == 0 and not len(dupes):
        print("✅ Aucune anomalie sur la période VÉRIFIÉE (2010-2018).")
    else:
        print(f"🚨 {n_ver_missing} date(s) manquante(s) sur la période VÉRIFIÉE + "
              f"{len(dupes)} doublon(s) → le backtest pre-FOMC est FAUSSÉ tant que")
        print("   ce n'est pas corrigé. Aucun test statistique ne détecte ce genre d'erreur.")
    print("\n⚠️ Écarts ≥ 2019 : la référence elle-même n'est pas recoupée. Vérifier sur")
    print("   https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm avant de conclure.")
    print("=" * 74)
    return 0 if n_ver_missing == 0 and not len(dupes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
