"""Pre-FOMC Drift — stratégie event-driven (Lucca & Moench, 2015).

Hypothèse théorique : aux 24 heures précédant l'annonce de la décision
de taux du FOMC, l'indice S&P 500 monte significativement plus que dans
n'importe quelle autre fenêtre de 24h. Sharpe historique ~1.5 (1994-2014).

Stratégie :
    - Entrée : long US500 à (FOMC_announcement - 24h)
    - Sortie : close à (FOMC_announcement - 1h)
    - Pas de TP/SL — hold à durée fixe.

Coûts modélisés :
    - Spread + slippage à l'entrée ET à la sortie
    - 1 nuit de swap (la fenêtre de 23h franchit minuit UTC dans la plupart des cas)

Calendrier source : data/raw/economic_calendar/<YEAR>.csv (Forex Factory scrape).
Les heures sont en ET (Eastern Time), converties en UTC ici.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.exceptions import DataValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)


def load_fomc_announcement_times(
    calendar_root: Path | str = "data/raw/economic_calendar",
    start_year: int = 2010,
    end_year: int = 2026,
) -> pd.DatetimeIndex:
    """Charge les timestamps UTC de tous les FOMC Statement scheduled.

    Filtre sur event == "FOMC Statement" (décision de taux). Les Forex
    Factory CSVs donnent les heures en ET ; on convertit en UTC via
    pandas tz handling pour gérer DST automatiquement.

    Args:
        calendar_root: Racine du calendrier scrapé.
        start_year, end_year: Bornes (inclusives) de filtrage.

    Returns:
        DatetimeIndex UTC trié, sans doublons.
    """
    root = Path(calendar_root)
    if not root.exists():
        raise DataValidationError(f"Calendar root introuvable : {root}")

    all_events: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        csv_path = root / f"{year}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        fomc = df[df["event"] == "FOMC Statement"].copy()
        if len(fomc) == 0:
            continue
        fomc["year"] = year
        all_events.append(fomc)

    if not all_events:
        raise DataValidationError(
            f"Aucun FOMC Statement trouvé entre {start_year} et {end_year}"
        )

    df = pd.concat(all_events, ignore_index=True)

    # Parse date + time → datetime ET → UTC
    timestamps: list[pd.Timestamp] = []
    for _, row in df.iterrows():
        date_str = str(row["date"]).strip()
        time_str = str(row["time"]).strip().lower()
        ts_et = _parse_et_datetime(date_str, time_str)
        if ts_et is not None:
            timestamps.append(ts_et.tz_convert("UTC"))

    idx = pd.DatetimeIndex(sorted(set(timestamps)))
    logger.info(
        "fomc_events_loaded",
        extra={"context": {
            "n_events": len(idx),
            "first": str(idx.min()),
            "last": str(idx.max()),
        }},
    )
    return idx


def _parse_et_datetime(date_str: str, time_str: str) -> pd.Timestamp | None:
    """Parse 'YYYY-MM-DD' + 'H:MMam/pm' en pd.Timestamp tz=US/Eastern."""
    import re

    m = re.match(r"(\d{1,2}):(\d{2})(am|pm)", time_str)
    if not m:
        logger.debug("time_unparseable", extra={"context": {"time": time_str}})
        return None

    hour = int(m.group(1))
    minute = int(m.group(2))
    ampm = m.group(3)
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    try:
        ts = pd.Timestamp(f"{date_str} {hour:02d}:{minute:02d}:00")
        return ts.tz_localize("US/Eastern", ambiguous="raise", nonexistent="raise")
    except Exception as exc:
        logger.warning(
            "ts_parse_failed",
            extra={"context": {"date": date_str, "time": time_str, "err": str(exc)}},
        )
        return None


def simulate_pre_fomc_trades(
    df: pd.DataFrame,
    fomc_times: pd.DatetimeIndex,
    spread_pips: float,
    slippage_pips: float,
    commission_pips: float,
    pip_size: float,
    swap_long_pips_per_night: float = 0.0,
    hours_before_entry: int = 24,
    hours_before_exit: int = 1,
) -> list[dict]:
    """Simule la stratégie Pre-FOMC drift sur un DataFrame OHLC indexé UTC.

    Pour chaque FOMC timestamp, trouve la barre d'entrée (au plus tôt à
    FOMC - hours_before_entry) et la barre de sortie (au plus tard à
    FOMC - hours_before_exit) via merge_asof avec direction.

    Args:
        df: OHLCV avec DatetimeIndex UTC, colonnes [Open, High, Low, Close].
        fomc_times: DatetimeIndex UTC des annonces FOMC.
        spread_pips: Spread broker en pips.
        slippage_pips: Slippage estimé en pips.
        commission_pips: Commission en pips.
        pip_size: Taille d'un pip dans l'unité du prix.
        swap_long_pips_per_night: Charge swap par nuit, position long.
        hours_before_entry: Nombre d'heures avant FOMC pour entrer (défaut 24).
        hours_before_exit: Nombre d'heures avant FOMC pour sortir (défaut 1).

    Returns:
        Liste de dicts trades avec clés : entry_time, exit_time, entry_price,
        exit_price, pips_brut, pips_net, nights_held, fomc_time, signal.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index doit être DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df.index doit être tz-aware (UTC)")

    cost_per_side = commission_pips + slippage_pips
    cost_total = 2 * cost_per_side + spread_pips  # spread une seule fois (round trip), slippage des 2 côtés
    # Convention : on traite spread comme "moitié à l'entrée, moitié à la sortie"
    # mais on l'applique une fois pour le round trip — consistent avec
    # half_cost convention des autres scripts.

    trades: list[dict] = []
    for fomc_ts in fomc_times:
        entry_target = fomc_ts - pd.Timedelta(hours=hours_before_entry)
        exit_target = fomc_ts - pd.Timedelta(hours=hours_before_exit)

        # Trouve la 1ère barre AU OU APRÈS entry_target
        entry_bars = df.index[df.index >= entry_target]
        if len(entry_bars) == 0:
            continue
        entry_ts = entry_bars[0]
        # Si la barre d'entrée est déjà après la sortie cible, skip
        if entry_ts >= exit_target:
            continue

        # Trouve la DERNIÈRE barre AU OU AVANT exit_target
        exit_bars = df.index[(df.index > entry_ts) & (df.index <= exit_target)]
        if len(exit_bars) == 0:
            continue
        exit_ts = exit_bars[-1]

        entry_price = float(df.at[entry_ts, "Close"])
        exit_price = float(df.at[exit_ts, "Close"])

        # PnL en pips, position LONG
        pips_brut = (exit_price - entry_price) / pip_size

        # Coûts
        pips_net = pips_brut - cost_total

        # Swap : nombre de minuits UTC traversés
        nights_held = max(0, (exit_ts.normalize() - entry_ts.normalize()).days)
        if nights_held > 0:
            pips_net += nights_held * swap_long_pips_per_night

        trades.append({
            "fomc_time": fomc_ts.isoformat(),
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pips_brut": pips_brut,
            "pips_net": pips_net,
            "nights_held": nights_held,
            "signal": 1,  # toujours long dans cette V1
        })

    return trades
