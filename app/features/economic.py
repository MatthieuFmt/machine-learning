"""Features issues du calendrier économique — vectorisé avec searchsorted.

Principes :
    - Zéro boucle Python sur les barres de prix (searchsorted O(E × log B)).
    - Les features forward (hours_to_next_event) sont connues à t car les
      annonces sont publiées des jours à l'avance. Risque résiduel : dates
      corrigées rétrospectivement si le scrape est postérieur aux events.
      Ce risque est documenté et ces features NE sont PAS utilisées comme
      target — uniquement comme contexte de régime.
    - Sentinelle NaN (pas -1) quand aucun événement pertinent n'existe.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.exceptions import DataValidationError
from app.testing.look_ahead_validator import look_ahead_safe

logger = logging.getLogger(__name__)

# Durées en nanosecondes (pandas timestamp → int64)
_NANOSECONDS_PER_HOUR = int(3.6e12)
_WINDOWS_NS: dict[int, int] = {
    1: _NANOSECONDS_PER_HOUR,
    4: 4 * _NANOSECONDS_PER_HOUR,
    24: 24 * _NANOSECONDS_PER_HOUR,
}

# Monnaies pour lesquelles on produit des features booléennes
_CURRENCIES = ["USD", "EUR"]

# Parser de temps Forex Factory : "1:30pm", "10:00am", "All Day", "Tentative", ""
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(am|pm)", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers internes
# ═══════════════════════════════════════════════════════════════════════════════


def _to_ns(ts: pd.DatetimeIndex) -> np.ndarray:
    """Convertit un DatetimeIndex en int64 nanosecondes depuis epoch.

    Indépendant de la résolution native (us, ns, ms, s).
    pandas >= 2.0 utilise datetime64[us, UTC] par défaut pour les tz-aware.

    Args:
        ts: DatetimeIndex, éventuellement tz-aware.

    Returns:
        np.ndarray[int64] en nanosecondes.
    """
    unit = str(ts.dtype).split("[", 1)[1].split(",")[0] if "[" in str(ts.dtype) else "ns"
    if unit == "ns":
        return ts.asi8
    # Reconstruire en ns pour garantir l'unité
    return ts.as_unit("ns").asi8


def _parse_time_str(time_str: str) -> str:
    """Convertit une heure ForexFactory en HH:MM:SS 24h.

    Args:
        time_str: Chaîne brute (ex: "1:30pm", "All Day", "", "Tentative").

    Returns:
        Chaîne "HH:MM:SS" ou "00:00:00" si non parseable.
    """
    if not time_str or not time_str.strip():
        return "00:00:00"
    stripped = time_str.strip().lower()
    if stripped in ("all day", "tentative"):
        return "00:00:00"
    m = _TIME_RE.match(stripped)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}:00"
    logger.debug("Heure non parseable — fallback 00:00:00", extra={"time_str": time_str})
    return "00:00:00"


def _event_within_window(
    price_ns: np.ndarray,
    event_ns: np.ndarray,
    window_ns: int,
) -> np.ndarray:
    """Pour chaque barre, 1 si un événement est dans la fenêtre à venir.

    Une barre à t voit un événement à ts_e si : t <= ts_e <= t + window_ns.

    Args:
        price_ns: Timestamps des barres en nanosecondes int64, triés croissant.
        event_ns: Timestamps des événements (int64, triés).
        window_ns: Taille de la fenêtre forward en nanosecondes.

    Returns:
        np.ndarray[int8] de même longueur que price_ns.
    """
    result = np.zeros(len(price_ns), dtype=np.int8)
    if len(event_ns) == 0:
        return result
    for ts_e in event_ns:
        left = np.searchsorted(price_ns, ts_e - window_ns, side="left")
        right = np.searchsorted(price_ns, ts_e, side="right")
        if left < right:
            result[left:right] = 1
    return result


def _hours_since_last(
    price_ns: np.ndarray,
    event_ns: np.ndarray,
) -> np.ndarray:
    """Heures écoulées depuis le dernier événement ≤ chaque barre.

    Args:
        price_ns: Timestamps des barres (int64, triés).
        event_ns: Timestamps des événements (int64, triés).

    Returns:
        np.ndarray[float32] — NaN si aucun événement antérieur.
    """
    if len(event_ns) == 0:
        return np.full(len(price_ns), np.nan, dtype=np.float32)
    idx = np.searchsorted(event_ns, price_ns, side="right") - 1
    result = np.where(
        idx >= 0,
        (price_ns - event_ns[idx]).astype(np.float32) / _NANOSECONDS_PER_HOUR,
        np.nan,
    )
    return result.astype(np.float32)


def _hours_to_next(
    price_ns: np.ndarray,
    event_ns: np.ndarray,
) -> np.ndarray:
    """Heures avant le prochain événement > chaque barre.

    Args:
        price_ns: Timestamps des barres (int64, triés).
        event_ns: Timestamps des événements (int64, triés).

    Returns:
        np.ndarray[float32] — NaN si aucun événement futur.
    """
    if len(event_ns) == 0:
        return np.full(len(price_ns), np.nan, dtype=np.float32)
    idx = np.searchsorted(event_ns, price_ns, side="right")
    result = np.where(
        idx < len(event_ns),
        (event_ns[idx] - price_ns).astype(np.float32) / _NANOSECONDS_PER_HOUR,
        np.nan,
    )
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# API publique
# ═══════════════════════════════════════════════════════════════════════════════


@look_ahead_safe
def load_calendar(
    years: list[int],
    root: Path | None = None,
) -> pd.DataFrame:
    """Charge les calendriers économiques scrapés depuis data/raw/economic_calendar/.

    Combine les colonnes `date` + `time` (format scraper) en un timestamp UTC.

    Args:
        years: Années à charger (ex: [2023, 2024]).
        root: Répertoire racine contenant les CSV. Défaut: data/raw/economic_calendar.

    Returns:
        DataFrame avec colonnes : timestamp, currency, event, impact,
        actual, forecast, previous — trié par timestamp.

    Raises:
        DataValidationError: Si un fichier d'année est introuvable.
    """
    if root is None:
        root = Path("data/raw/economic_calendar")

    dfs: list[pd.DataFrame] = []
    for y in years:
        path = root / f"{y}.csv"
        if not path.exists():
            raise DataValidationError(
                f"Calendrier {y} introuvable dans {root}. "
                f"Lance `python scripts/scrape_forexfactory.py --year {y}` d'abord."
            )
        df_year = pd.read_csv(path)
        dfs.append(df_year)

    df = pd.concat(dfs, ignore_index=True)

    # Construire le timestamp UTC à partir de date + time
    time_parsed = df.get("time", pd.Series("")).apply(_parse_time_str)
    date_col = df.get("date", pd.Series(""))
    df["timestamp"] = pd.to_datetime(
        date_col.astype(str) + "T" + time_parsed,
        utc=True,
        errors="coerce",
    )

    # Filtrer les timestamps invalides (log warning)
    n_bad = df["timestamp"].isna().sum()
    if n_bad > 0:
        logger.warning(
            "%d événements avec timestamp invalide ignorés sur %d.",
            n_bad,
            len(df),
        )
        df = df[df["timestamp"].notna()]

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Garder uniquement les colonnes pertinentes
    cols_out = ["timestamp", "currency", "event", "impact", "actual", "forecast", "previous"]
    available = [c for c in cols_out if c in df.columns]

    logger.info(
        "Calendrier chargé : %d événements sur %d années.",
        len(df),
        len(years),
    )

    return df[available]


@look_ahead_safe
def compute_event_features(
    price_index: pd.DatetimeIndex,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule les features de calendrier économique pour chaque barre.

    Vectorisé via np.searchsorted — zéro boucle Python sur les barres.

    Args:
        price_index: Index temporel des barres (DatetimeIndex trié).
        calendar: DataFrame produit par load_calendar().

    Returns:
        DataFrame avec 9 colonnes de features, même index que price_index.
    """
    out = pd.DataFrame(index=price_index)

    # Conversion en nanosecondes int64 via _to_ns (robuste à la résolution native).
    price_ns = _to_ns(price_index)

    # ── Features booléennes : event dans les Xh à venir ─────────────────
    for currency in _CURRENCIES:
        mask = (calendar["currency"] == currency) & (calendar["impact"] == "High")
        event_ts = calendar.loc[mask, "timestamp"].dropna().sort_values()
        if len(event_ts) == 0:
            event_ns = np.array([], dtype=np.int64)
        else:
            event_ns = _to_ns(pd.DatetimeIndex(event_ts.to_numpy()))

        for hours, window_ns in _WINDOWS_NS.items():
            col = f"event_high_within_{hours}h_{currency}"
            out[col] = _event_within_window(price_ns, event_ns, window_ns)

    # ── Heures depuis dernier NFP / FOMC ────────────────────────────────
    event_col = calendar.get("event", pd.Series(dtype=str))

    for keyword, name in [("Non-Farm", "nfp"), ("FOMC", "fomc")]:
        col = f"hours_since_last_{name}"
        mask_kw = event_col.str.contains(keyword, case=False, na=False)
        event_ts = calendar.loc[mask_kw, "timestamp"].dropna().sort_values()
        if len(event_ts) == 0:
            event_ns = np.array([], dtype=np.int64)
        else:
            event_ns = _to_ns(pd.DatetimeIndex(event_ts.to_numpy()))
        out[col] = _hours_since_last(price_ns, event_ns)

    # ── Heures avant prochain événement High (toutes monnaies) ──────────
    mask_high = calendar["impact"] == "High"
    event_ts = calendar.loc[mask_high, "timestamp"].dropna().sort_values()
    if len(event_ts) == 0:
        event_ns = np.array([], dtype=np.int64)
    else:
        event_ns = _to_ns(pd.DatetimeIndex(event_ts.to_numpy()))
    out["hours_to_next_event_high"] = _hours_to_next(price_ns, event_ns)

    return out
