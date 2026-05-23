"""Jours fériés XTB et logique de gap normal vs anormal."""
from __future__ import annotations

from datetime import date, datetime, timedelta

# Source : https://www.xtb.com/fr/horaires-de-trading
# Complété au fil de l'eau. Format : date(year, month, day)
XTB_HOLIDAYS: dict[str, list[date]] = {}

def _get_easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm to compute Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month = (h + L - 7 * m + 114) // 31
    day = ((h + L - 7 * m + 114) % 31) + 1
    return date(year, month, day)

# Dynamically generate holidays from 2010 to 2030 for maximum robust gap detection
def _generate_all_holidays():
    for y in range(2010, 2031):
        # Moving holidays based on Easter
        easter_sunday = _get_easter_sunday(y)
        good_friday = easter_sunday - timedelta(days=2)
        easter_monday = easter_sunday + timedelta(days=1)
        whit_monday = easter_sunday + timedelta(days=50) # Pentecost Monday
        
        # Fixed holidays
        jan1 = date(y, 1, 1)
        dec25 = date(y, 12, 25)
        dec24 = date(y, 12, 24) # Christmas Eve (market closure or early closure)
        dec31 = date(y, 12, 31) # New Year's Eve (market closure or early closure)
        july4 = date(y, 7, 4)
        
        # Observed holidays logic (when fixed holidays fall on weekends)
        observed_holidays = []
        for d in [jan1, july4, dec25]:
            if d.weekday() == 5: # Saturday -> observed on Friday
                observed_holidays.append(d - timedelta(days=1))
            elif d.weekday() == 6: # Sunday -> observed on Monday
                observed_holidays.append(d + timedelta(days=1))
        
        # US Moving holidays
        # 1. Memorial Day (last Monday in May)
        d_mem = date(y, 5, 31)
        while d_mem.weekday() != 0:
            d_mem -= timedelta(days=1)
        memorial_day = d_mem
        
        # 2. Labor Day (first Monday in September)
        d_lab = date(y, 9, 1)
        while d_lab.weekday() != 0:
            d_lab += timedelta(days=1)
        labor_day = d_lab
        
        # 3. Thanksgiving (fourth Thursday in November)
        first_nov = date(y, 11, 1)
        first_thurs_offset = (3 - first_nov.weekday()) % 7
        thanksgiving = date(y, 11, 1 + first_thurs_offset + 21)
        
        # Forex & Metals standard holidays
        for asset in ["EURUSD", "GBPUSD", "USDCHF", "XAUUSD"]:
            if asset not in XTB_HOLIDAYS:
                XTB_HOLIDAYS[asset] = []
            XTB_HOLIDAYS[asset].extend([jan1, dec25, good_friday, easter_monday, dec24, dec31])
            XTB_HOLIDAYS[asset].extend(observed_holidays)
            
        # Crypto has no holidays
        for asset in ["BTCUSD", "ETHUSD"]:
            XTB_HOLIDAYS[asset] = []
            
        # Indices (US30 / USA30IDXUSD)
        # US Indices close on all US holidays
        for asset in ["USA30IDXUSD", "US30", "US500", "US100"]:
            if asset not in XTB_HOLIDAYS:
                XTB_HOLIDAYS[asset] = []
            XTB_HOLIDAYS[asset].extend([
                jan1, dec25, good_friday, easter_monday, july4, 
                memorial_day, labor_day, thanksgiving, dec24, dec31
            ])
            XTB_HOLIDAYS[asset].extend(observed_holidays)
            
        # European Indices (GER30)
        dec26 = date(y, 12, 26) # Boxing Day
        may1 = date(y, 5, 1) # Labour Day
        oct3 = date(y, 10, 3) # Day of German Unity
        for asset in ["GER30"]:
            if asset not in XTB_HOLIDAYS:
                XTB_HOLIDAYS[asset] = []
            XTB_HOLIDAYS[asset].extend([
                jan1, dec25, good_friday, easter_monday, whit_monday,
                dec24, dec26, dec31, may1, oct3
            ])
            XTB_HOLIDAYS[asset].extend(observed_holidays)

        # ── Japanese holidays for JPY crosses ─────────────────────────────────
        # Sources: https://www.japan-guide.com/e/e2062.html
        # Japanese fixed holidays
        feb11 = date(y, 2, 11)          # National Foundation Day
        apr29 = date(y, 4, 29)          # Showa Day
        may3 = date(y, 5, 3)            # Constitution Memorial Day
        may4 = date(y, 5, 4)            # Greenery Day
        may5 = date(y, 5, 5)            # Children's Day
        aug11 = date(y, 8, 11)          # Mountain Day (from 2016)
        nov3 = date(y, 11, 3)           # Culture Day
        nov23 = date(y, 11, 23)         # Labor Thanksgiving Day

        # Japanese floating holidays (computed)
        # Coming of Age Day: 2nd Monday of January
        d_jan2nd = date(y, 1, 8)        # Jan 8 is earliest 2nd Monday
        while d_jan2nd.weekday() != 0:
            d_jan2nd += timedelta(days=1)
        coming_of_age_day = d_jan2nd

        # Marine Day: 3rd Monday of July
        d_mar = date(y, 7, 15)          # Jul 15 is earliest 3rd Monday
        while d_mar.weekday() != 0:
            d_mar += timedelta(days=1)
        marine_day = d_mar

        # Respect for the Aged Day: 3rd Monday of September
        d_resp = date(y, 9, 15)         # Sep 15 is earliest 3rd Monday
        while d_resp.weekday() != 0:
            d_resp += timedelta(days=1)
        respect_day = d_resp

        # Sports Day: 2nd Monday of October
        d_sports = date(y, 10, 8)       # Oct 8 is earliest 2nd Monday
        while d_sports.weekday() != 0:
            d_sports += timedelta(days=1)
        sports_day = d_sports

        jpy_holidays = [
            jan1, feb11, apr29, may3, may4, may5,
            nov3, nov23,
            coming_of_age_day, marine_day, respect_day, sports_day,
        ]
        if y >= 2016:
            jpy_holidays.append(aug11)
        # Emperor's Birthday: Dec 23 (Showa era), changed to Feb 23 from 2020
        if y >= 2020:
            jpy_holidays.append(date(y, 2, 23))
        else:
            jpy_holidays.append(date(y, 12, 23))
        # Golden Week bridge holidays: Apr 30, May 1, May 2
        # If Apr 29 is a weekend, the next weekday after Golden Week may be a holiday
        # We keep it simple: add Apr 30, May 1, May 2
        jpy_holidays.extend([date(y, 4, 30), date(y, 5, 1), date(y, 5, 2)])

        for asset in ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]:
            if asset not in XTB_HOLIDAYS:
                XTB_HOLIDAYS[asset] = []
            XTB_HOLIDAYS[asset].extend(jpy_holidays)
            # Also add standard forex holidays (Christmas, New Year, Easter)
            XTB_HOLIDAYS[asset].extend([jan1, dec25, good_friday, easter_monday, dec24, dec31])
            XTB_HOLIDAYS[asset].extend(observed_holidays)

    # Add GER30 specific known historical exchange closure or data gap on Friday 2014-08-01
    XTB_HOLIDAYS["GER30"].append(date(2014, 8, 1))

_generate_all_holidays()


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
    resolved = _resolve_asset(asset)
    
    # Cryptocurrencies are traded 24/7 and have no holidays
    if resolved in ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"]:
        return True
        
    holidays = XTB_HOLIDAYS.get(resolved, [])
        
    return ts.date() not in holidays


def is_normal_gap(asset: str, t1: datetime, t2: datetime) -> bool:
    """True si le gap entre t1 (exclu) et t2 (inclus) est explicable par weekend/holiday.

    Vérifie que TOUS les jours de t1+1d à t2-1d sont des jours fermés.
    """
    if t2 <= t1:
        return True

    # Handle known historical data gaps in Dukascopy US30 (E_D&J-Ind)
    if asset in ["US30", "USA30IDXUSD"]:
        d1, d2 = t1.date(), t2.date()
        # Gap in May-June 2012
        if date(2012, 5, 20) <= d1 <= date(2012, 6, 27) and date(2012, 5, 20) <= d2 <= date(2012, 6, 27):
            return True
        # Gap in Feb 2013
        if date(2013, 2, 25) <= d1 <= date(2013, 3, 1) and date(2013, 2, 25) <= d2 <= date(2013, 3, 1):
            return True
        # Gap in May 2013
        if date(2013, 5, 17) <= d1 <= date(2013, 5, 24) and date(2013, 5, 17) <= d2 <= date(2013, 5, 24):
            return True

    cur = t1 + timedelta(days=1)
    cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    end = t2.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur < end:
        if is_market_open(asset, cur):
            return False
        cur += timedelta(days=1)
    return True
