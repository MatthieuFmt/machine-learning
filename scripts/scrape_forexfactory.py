#!/usr/bin/env python3
"""Scraper du calendrier économique Forex Factory (2010-2025).

Parcourt semaine par semaine, parse la grille HTML, sauvegarde par année
dans data/raw/economic_calendar/.

Usage:
    python scripts/scrape_forexfactory.py [--start 2010] [--end 2025]
    python scripts/scrape_forexfactory.py --start 2024 --end 2024  # test
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import cloudscraper
from bs4 import BeautifulSoup, Tag

# ── Configuration ──────────────────────────────────────────────────────────

BASE_URL = "https://www.forexfactory.com/calendar"
OUTPUT_DIR = Path("data/raw/economic_calendar")
CSV_COLUMNS = ["date", "time", "currency", "event", "impact", "actual", "forecast", "previous"]

MIN_DELAY_SEC = 3.0
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

MONTH_ABBRS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
MONTH_ABBRS_LONG = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_MAP: dict[str, int] = {}
for _i, _abbr in enumerate(MONTH_ABBRS):
    _MONTH_MAP[_abbr] = _i + 1
    _MONTH_MAP[MONTH_ABBRS_LONG[_i]] = _i + 1

TARGET_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF"}

_scraper: cloudscraper.CloudScraper | None = None


def _get_scraper() -> cloudscraper.CloudScraper:
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
    return _scraper


# ── Helpers ────────────────────────────────────────────────────────────────

def _mondays_between(start_year: int, end_year: int) -> list[str]:
    weeks: list[str] = []
    d = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    while d <= end:
        month_abbr = MONTH_ABBRS[d.month - 1]
        weeks.append(f"{month_abbr}{d.day}.{d.year}")
        d += timedelta(days=7)
    return weeks


def _parse_impact_from_icon(icon_span: Tag | None) -> str:
    """Extrait l'impact depuis la classe CSS de l'icône FF (icon--ff-impact-*)."""
    if icon_span is None:
        return "Low"
    classes = icon_span.get("class")
    if not classes or not isinstance(classes, list):
        return "Low"
    class_str = " ".join(classes).lower()
    if "red" in class_str:
        return "High"
    if "yel" in class_str:
        return "Medium"
    if "gra" in class_str:
        return "Low"
    return "Low"


def _parse_date(raw_date: str, week_str: str) -> str:
    """Convertit 'MonJan 1' + 'jan1.2024' → '2024-01-01'."""
    if not raw_date or not raw_date.strip():
        return ""
    text = raw_date.strip()
    # Extraire l'abbréviation du mois (3 premières lettres après le jour)
    # Format attendu: "MonJan 1", "TueDec 31", etc.
    # Le jour de semaine fait toujours 3 lettres, le mois aussi
    month_abbr = text[3:6].lower() if len(text) >= 6 else ""
    day_str = text.split()[-1] if " " in text else text[6:].strip()

    if month_abbr not in _MONTH_MAP:
        return raw_date  # fallback

    month = _MONTH_MAP[month_abbr]
    day = int(day_str) if day_str.isdigit() else 1
    year = int(week_str.split(".")[-1]) if "." in week_str else 2000

    return f"{year}-{month:02d}-{day:02d}"


def _scrape_week(week_str: str) -> list[dict[str, str]]:
    """Scrape une semaine. Retourne une liste d'événements."""
    url = f"{BASE_URL}?week={week_str}"
    events: list[dict[str, str]] = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            scraper = _get_scraper()
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  [ERREUR] {week_str} — {e}", file=sys.stderr)
                return events
            time.sleep(MIN_DELAY_SEC * attempt)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.select_one("table.calendar__table")
        if table is None:
            break  # page sans données

        rows = table.find_all("tr")
        current_date = ""
        current_time = ""

        DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

        for row in rows:
            row_classes = row.get("class") or []

            if "subhead" in row_classes or "borderfix" in row_classes:
                continue

            # Day breaker : met à jour current_date
            if "calendar__row--day-breaker" in row_classes:
                date_cell = row.find("td")
                if date_cell:
                    current_date = date_cell.get_text(strip=True).replace("\xa0", " ")
                continue

            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            is_no_grid = "calendar__row--no-grid" in row_classes
            is_grey = "calendar__row--grey" in row_classes

            if not is_grey and not is_no_grid:
                continue

            # Déterminer la structure : 11 cellules = [date, time, currency, ...]
            #                        10 cellules = [time, currency, ...] (date héritée)
            #                        no-grid   = [currency, ...] ou [-, currency, ...]
            n = len(cells)

            if is_no_grid:
                # Hérite date et heure
                currency_idx = 0 if cells[0].get_text(strip=True) else 1
                currency = cells[currency_idx].get_text(strip=True).upper()
                impact_idx = currency_idx + 1
                event_idx = impact_idx + 1
                event_cell = cells[event_idx] if event_idx < n else None
                icon_span = cells[impact_idx].select_one("span.icon") if impact_idx < n else None
            else:
                # Ligne standard ou avec date héritée
                # Détecter si cells[0] contient une date (jour de semaine)
                cell0_text = cells[0].get_text(strip=True)
                first_word = cell0_text.split()[0] if cell0_text else ""

                if first_word in DAYS:
                    # Format 11 cellules: [date, time, currency, icon, event, ...]
                    current_date = cell0_text
                    current_time = cells[1].get_text(strip=True)
                    currency = cells[2].get_text(strip=True).upper()
                    impact_idx = 2 if cells[2].select_one("span.icon") else 3
                    event_idx = 3 if impact_idx == 2 else 4
                    event_cell = cells[event_idx] if event_idx < n else None
                    icon_span = cells[impact_idx].select_one("span.icon") if impact_idx < n else None
                else:
                    # Format 10 cellules: [time, currency, icon, event, ...]
                    current_time = cell0_text
                    currency = cells[1].get_text(strip=True).upper()
                    impact_idx = 1 if cells[1].select_one("span.icon") else 2
                    event_idx = 2 if impact_idx == 1 else 3
                    event_cell = cells[event_idx] if event_idx < n else None
                    icon_span = cells[impact_idx].select_one("span.icon") if impact_idx < n else None

            if currency not in TARGET_CURRENCIES:
                continue

            if event_cell is None:
                continue

            event_text = event_cell.get_text(" ", strip=True)
            if not event_text:
                continue

            impact = _parse_impact_from_icon(icon_span)

            # Actual / Forecast / Previous : 3 avant-dernières cellules
            n = len(cells)
            actual = cells[n - 3].get_text(strip=True).replace("\xa0", "") if n >= 3 else ""
            forecast = cells[n - 2].get_text(strip=True).replace("\xa0", "") if n >= 4 else ""
            previous = cells[n - 1].get_text(strip=True).replace("\xa0", "") if n >= 5 else ""

            # Nettoyer les valeurs vides / non numériques
            # On garde tout tel quel, le calendar_loader fera le parsing numérique

            events.append({
                "date": _parse_date(current_date, week_str),
                "time": current_time,
                "currency": currency,
                "event": event_text,
                "impact": impact,
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
            })

        break  # succès

    return events


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scraper le calendrier économique Forex Factory")
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    weeks = _mondays_between(args.start, args.end)
    print(f"Semaines a scraper : {len(weeks)} ({args.start}-{args.end})")

    if args.dry_run:
        for w in weeks[:10]:
            print(f"  {BASE_URL}?week={w}")
        print(f"  ... et {len(weeks) - 10} autres")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    year_events: dict[int, list[dict[str, str]]] = {y: [] for y in range(args.start, args.end + 1)}
    total_events = 0

    for i, week_str in enumerate(weeks):
        if i % 50 == 0:
            pct = 100.0 * i / len(weeks) if len(weeks) > 0 else 0
            print(f"Progression : {i}/{len(weeks)} ({pct:.0f}%) - {total_events} evenements")

        events = _scrape_week(week_str)

        for evt in events:
            year = int(week_str.split(".")[-1])
            if year in year_events:
                year_events[year].append(evt)
                total_events += 1

        time.sleep(MIN_DELAY_SEC)

    for year, evts in sorted(year_events.items()):
        if not evts:
            continue
        output_path = OUTPUT_DIR / f"{year}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(evts)
        print(f"  {year}.csv : {len(evts)} evenements -> {output_path}")

    print(f"\nTermine : {total_events} evenements, "
          f"{len([y for y, e in year_events.items() if e])} annees.")


if __name__ == "__main__":
    main()
