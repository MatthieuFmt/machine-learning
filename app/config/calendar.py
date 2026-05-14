"""Jours fériés XTB et logique de gap normal vs anormal."""
from __future__ import annotations

from datetime import date, datetime, timedelta

# Source : https://www.xtb.com/fr/horaires-de-trading
# Complété au fil de l'eau. Format : date(year, month, day)
XTB_HOLIDAYS: dict[str, list[date]] = {
    "USA30IDXUSD": [
        date(2024, 1, 1), date(2024, 7, 4), date(2024, 11, 28), date(2024, 12, 25),
        date(2025, 1, 1), date(2025, 7, 4), date(2025, 11, 27), date(2025, 12, 25),
    ],
    "US30": [  # alias
        date(2024, 1, 1), date(2024, 7, 4), date(2024, 11, 28), date(2024, 12, 25),
        date(2025, 1, 1), date(2025, 7, 4), date(2025, 11, 27), date(2025, 12, 25),
    ],
    "XAUUSD": [
        date(2024, 1, 1), date(2024, 12, 25),
        date(2025, 1, 1), date(2025, 12, 25),
    ],
    "EURUSD": [
        date(2024, 1, 1), date(2024, 12, 25),
        date(2025, 1, 1), date(2025, 12, 25),
    ],
    "GBPUSD": [
        date(2024, 1, 1), date(2024, 12, 25),
        date(2025, 1, 1), date(2025, 12, 25),
    ],
    "USDCHF": [
        date(2024, 1, 1), date(2024, 12, 25),
        date(2025, 1, 1), date(2025, 12, 25),
    ],
    "BTCUSD": [],
    "ETHUSD": [],
}


def _resolve_asset(asset: str) -> str:
    """Résout les alias d'actifs vers la clé XTB_HOLIDAYS."""
    if asset in XTB_HOLIDAYS:
        return asset
    if asset == "US30":
        return "USA30IDXUSD"
    return asset


def is_market_open(asset: str, ts: datetime) -> bool:
    """True si le marché est ouvert à l'instant ts (UTC)."""
    if ts.weekday() >= 5:
        return False
    holidays = XTB_HOLIDAYS.get(_resolve_asset(asset), [])
    return ts.date() not in holidays


def is_normal_gap(asset: str, t1: datetime, t2: datetime) -> bool:
    """True si le gap entre t1 (exclu) et t2 (inclus) est explicable par weekend/holiday.

    Vérifie que TOUS les jours de t1+1d à t2-1d sont des jours fermés.
    """
    if t2 <= t1:
        return True
    cur = t1 + timedelta(days=1)
    cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    end = t2.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur < end:
        if is_market_open(asset, cur):
            return False
        cur += timedelta(days=1)
    return True
