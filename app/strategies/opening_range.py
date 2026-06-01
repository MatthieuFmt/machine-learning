"""Opening Range Breakout (ORB) sur indices — stratégie intraday (Phase 1, option B).

Hypothèse PRÉ-ENREGISTRÉE (effet intraday documenté, ex. Zarattini & Aziz 2023
sur indices US) : le range de la 1ʳᵉ heure de la séance cash « cadre » la
journée ; une clôture au-delà de ce range amorce un mouvement directionnel qui
se poursuit jusqu'à la clôture. Stratégie INTRADAY stricte → flat la nuit →
ZÉRO swap (le tueur des holds multi-jours).

Règle (figée, sans TP — on laisse courir jusqu'au soir) :
    - Opening Range (OR) = High/Low de la barre H1 de l'heure d'ouverture
      (en HEURE LOCALE de la place : NYSE / Xetra), ce qui absorbe les
      changements d'heure été/hiver (DST).
    - Cassure CONFIRMÉE à la CLÔTURE d'une barre ultérieure de la séance :
        * Close > OR_high → LONG (signal +1)
        * Close < OR_low  → SHORT (signal −1)
      Première cassure de la journée seulement (1 trade/jour max).
    - Entrée à l'OPEN de la barre suivante (fill honnête, pas de look-ahead).
    - Stop = côté opposé de l'OR (long → OR_low ; short → OR_high).
      Intrabar conservateur : si une barre touche le stop, sortie au stop.
    - Sortie au plus tard à la CLÔTURE de la dernière barre de séance (EOD).
    - Coûts : spread + 2×(slippage+commission). Swap ≈ 0 (intraday).

⚠️ Limite : en données H1 le « range d'ouverture » est grossier (la littérature
utilise du 5 min). Un résultat négatif n'enterre pas l'ORB fin ; un positif
serait à reconfirmer en données plus fines.
"""
from __future__ import annotations

import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def simulate_orb_trades(
    df_h1: pd.DataFrame,
    asset_config: AssetConfig,
    *,
    session_tz: str,
    or_hour_local: int,
    last_hour_local: int,
) -> list[dict]:
    """Simule l'Opening Range Breakout sur un indice (long + short).

    Args:
        df_h1: OHLCV H1 indexé DatetimeIndex tz-aware UTC, colonnes
            [Open, High, Low, Close].
        asset_config: Coûts/paramètres broker (spread, slippage, swap, pip_size).
        session_tz: Fuseau de la place (ex. "America/New_York", "Europe/Berlin").
            Utilisé pour grouper par jour de séance et repérer l'heure locale.
        or_hour_local: Heure LOCALE de la barre servant d'opening range
            (ex. 9 → barre 09:00-10:00 locale).
        last_hour_local: Heure LOCALE de la dernière barre de séance ; la sortie
            EOD se fait à sa clôture. La cassure n'est cherchée que dans
            (or_hour_local, last_hour_local].

    Returns:
        Liste de dicts trades, clés : date, signal (+1/-1), entry_time,
        exit_time, entry_price, exit_price, or_high, or_low, or_range,
        stop_price, pips_brut, pips_net, nights_held, exit_reason ∈ {"stop","eod"}.
        ``pips_net`` et ``exit_time`` sont compatibles ``sharpe_daily_from_trades``.
    """
    if not isinstance(df_h1.index, pd.DatetimeIndex):
        raise TypeError("df_h1.index doit être DatetimeIndex")
    if df_h1.index.tz is None:
        raise ValueError("df_h1.index doit être tz-aware (UTC)")
    for col in ("Open", "High", "Low", "Close"):
        if col not in df_h1.columns:
            raise KeyError(f"df_h1 doit contenir la colonne '{col}'")
    if last_hour_local <= or_hour_local:
        raise ValueError(
            f"last_hour_local ({last_hour_local}) doit être > "
            f"or_hour_local ({or_hour_local})"
        )

    pip_size = asset_config.pip_size
    cost_total = (
        asset_config.spread_pips
        + 2 * (asset_config.slippage_pips + asset_config.commission_pips)
    )

    work = df_h1[["Open", "High", "Low", "Close"]].sort_index().copy()
    local = work.index.tz_convert(session_tz)
    work["lhour"] = local.hour
    work["ldate"] = local.date

    trades: list[dict] = []

    for _ldate, day in work.groupby("ldate", sort=True):
        rows = list(day.itertuples())  # ordre chronologique (UTC trié)
        n = len(rows)

        # ── Opening range (barre de l'heure d'ouverture locale) ──────
        or_high = or_low = None
        for r in rows:
            if r.lhour == or_hour_local:
                or_high, or_low = float(r.High), float(r.Low)
                break
        if or_high is None:
            continue
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        # ── Cassure confirmée à la close (1ʳᵉ de la séance) ──────────
        sig_i: int | None = None
        direction = 0
        for i, r in enumerate(rows):
            if not (or_hour_local < r.lhour <= last_hour_local):
                continue
            if r.Close > or_high:
                sig_i, direction = i, 1
                break
            if r.Close < or_low:
                sig_i, direction = i, -1
                break
        if sig_i is None:
            continue

        # ── Entrée à l'open de la barre suivante ─────────────────────
        entry_i = sig_i + 1
        if entry_i >= n:
            continue
        entry_row = rows[entry_i]
        if entry_row.lhour > last_hour_local:
            continue
        entry_price = float(entry_row.Open)
        entry_ts = entry_row.Index
        stop_price = or_low if direction == 1 else or_high

        # ── Scan sortie : stop intrabar (conservateur) sinon EOD ─────
        exit_price = exit_ts = exit_reason = None
        last_valid = entry_row
        for j in range(entry_i, n):
            rj = rows[j]
            if rj.lhour > last_hour_local:
                break
            last_valid = rj
            if direction == 1 and float(rj.Low) <= stop_price:
                exit_price, exit_ts, exit_reason = stop_price, rj.Index, "stop"
                break
            if direction == -1 and float(rj.High) >= stop_price:
                exit_price, exit_ts, exit_reason = stop_price, rj.Index, "stop"
                break
        if exit_price is None:
            exit_price, exit_ts, exit_reason = float(last_valid.Close), last_valid.Index, "eod"

        # ── PnL ──────────────────────────────────────────────────────
        if direction == 1:
            pips_brut = (exit_price - entry_price) / pip_size
        else:
            pips_brut = (entry_price - exit_price) / pip_size
        pips_net = pips_brut - cost_total
        nights_held = max(0, (exit_ts.normalize() - entry_ts.normalize()).days)
        if nights_held > 0:
            swap = (
                asset_config.swap_long_pips_per_night
                if direction == 1
                else asset_config.swap_short_pips_per_night
            )
            pips_net += nights_held * swap

        trades.append({
            "date": str(_ldate),
            "signal": direction,
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "or_high": or_high,
            "or_low": or_low,
            "or_range": or_range,
            "stop_price": float(stop_price),
            "pips_brut": float(pips_brut),
            "pips_net": float(pips_net),
            "nights_held": int(nights_held),
            "exit_reason": exit_reason,
        })

    logger.info(
        "orb_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_bars": len(work),
            "session_tz": session_tz,
            "or_hour_local": or_hour_local,
            "last_hour_local": last_hour_local,
        }},
    )
    return trades


def _parse_hhmm(s: str) -> int:
    """'09:30' → minutes depuis minuit (570)."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def simulate_orb_session(
    df_intraday: pd.DataFrame,
    asset_config: AssetConfig,
    *,
    session_tz: str,
    open_time: str = "09:30",
    or_minutes: int = 5,
    close_time: str = "16:00",
) -> list[dict]:
    """ORB FINE-résolution : range = les N premières MINUTES de séance.

    Version générique (M1/M5/M15…) de l'ORB documenté : l'opening range n'est
    plus une barre horaire mais une FENÊTRE de `or_minutes` minutes à partir de
    l'ouverture cash `open_time` (heure LOCALE de la place). C'est la résolution
    où l'effet ORB est censé exister (cf. littérature 5 min).

    Règle (identique à `simulate_orb_trades`, fenêtre fine) :
        - OR = High/Low des barres dans [open_time, open_time + or_minutes).
        - Cassure confirmée à la CLÔTURE d'une barre ultérieure de la séance
          (tmin ∈ [open_time+or_minutes, close_time]).
        - Entrée à l'OPEN de la barre suivante. Stop = côté opposé de l'OR.
        - Sortie au stop (intrabar conservateur) sinon à la close de la dernière
          barre ≤ close_time (EOD). Flat la nuit → zéro swap.

    Args:
        df_intraday: OHLCV intraday (M5 recommandé) indexé tz-aware UTC.
        asset_config: Coûts/paramètres broker.
        session_tz: Fuseau de la place (ex. "America/New_York").
        open_time: Heure d'ouverture cash locale "HH:MM" (ex. "09:30" NYSE).
        or_minutes: Durée de l'opening range en minutes (ex. 5, 15, 30).
        close_time: Heure de clôture locale "HH:MM" (sortie EOD).

    Returns:
        Liste de dicts trades (mêmes clés que `simulate_orb_trades`, +`or_minutes`).
    """
    if not isinstance(df_intraday.index, pd.DatetimeIndex):
        raise TypeError("df_intraday.index doit être DatetimeIndex")
    if df_intraday.index.tz is None:
        raise ValueError("df_intraday.index doit être tz-aware (UTC)")
    for col in ("Open", "High", "Low", "Close"):
        if col not in df_intraday.columns:
            raise KeyError(f"df_intraday doit contenir la colonne '{col}'")

    open_min = _parse_hhmm(open_time)
    or_end_min = open_min + or_minutes
    close_min = _parse_hhmm(close_time)
    if not (open_min < or_end_min <= close_min):
        raise ValueError(
            f"Incohérence horaire : open={open_time} +{or_minutes}min doit être "
            f"≤ close={close_time}"
        )

    pip_size = asset_config.pip_size
    cost_total = (
        asset_config.spread_pips
        + 2 * (asset_config.slippage_pips + asset_config.commission_pips)
    )

    work = df_intraday[["Open", "High", "Low", "Close"]].sort_index().copy()
    local = work.index.tz_convert(session_tz)
    work["tmin"] = local.hour * 60 + local.minute
    work["ldate"] = local.date

    trades: list[dict] = []

    for _ldate, day in work.groupby("ldate", sort=True):
        rows = list(day.itertuples())
        n = len(rows)

        # ── Opening range (fenêtre des or_minutes premières minutes) ──
        or_high = or_low = None
        for r in rows:
            if open_min <= r.tmin < or_end_min:
                hi, lo = float(r.High), float(r.Low)
                or_high = hi if or_high is None else max(or_high, hi)
                or_low = lo if or_low is None else min(or_low, lo)
        if or_high is None:
            continue
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        # ── Cassure confirmée à la close (1ʳᵉ de la séance) ──────────
        sig_i: int | None = None
        direction = 0
        for i, r in enumerate(rows):
            if not (or_end_min <= r.tmin <= close_min):
                continue
            if r.Close > or_high:
                sig_i, direction = i, 1
                break
            if r.Close < or_low:
                sig_i, direction = i, -1
                break
        if sig_i is None:
            continue

        # ── Entrée à l'open de la barre suivante ─────────────────────
        entry_i = sig_i + 1
        if entry_i >= n:
            continue
        entry_row = rows[entry_i]
        if entry_row.tmin > close_min:
            continue
        entry_price = float(entry_row.Open)
        entry_ts = entry_row.Index
        stop_price = or_low if direction == 1 else or_high

        # ── Scan sortie : stop intrabar sinon EOD ────────────────────
        exit_price = exit_ts = exit_reason = None
        last_valid = entry_row
        for j in range(entry_i, n):
            rj = rows[j]
            if rj.tmin > close_min:
                break
            last_valid = rj
            if direction == 1 and float(rj.Low) <= stop_price:
                exit_price, exit_ts, exit_reason = stop_price, rj.Index, "stop"
                break
            if direction == -1 and float(rj.High) >= stop_price:
                exit_price, exit_ts, exit_reason = stop_price, rj.Index, "stop"
                break
        if exit_price is None:
            exit_price, exit_ts, exit_reason = float(last_valid.Close), last_valid.Index, "eod"

        # ── PnL ──────────────────────────────────────────────────────
        if direction == 1:
            pips_brut = (exit_price - entry_price) / pip_size
        else:
            pips_brut = (entry_price - exit_price) / pip_size
        pips_net = pips_brut - cost_total
        nights_held = max(0, (exit_ts.normalize() - entry_ts.normalize()).days)
        if nights_held > 0:
            swap = (
                asset_config.swap_long_pips_per_night
                if direction == 1
                else asset_config.swap_short_pips_per_night
            )
            pips_net += nights_held * swap

        trades.append({
            "date": str(_ldate),
            "signal": direction,
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "or_high": or_high,
            "or_low": or_low,
            "or_range": or_range,
            "or_minutes": or_minutes,
            "stop_price": float(stop_price),
            "pips_brut": float(pips_brut),
            "pips_net": float(pips_net),
            "nights_held": int(nights_held),
            "exit_reason": exit_reason,
        })

    logger.info(
        "orb_session_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_bars": len(work),
            "session_tz": session_tz,
            "open_time": open_time,
            "or_minutes": or_minutes,
            "close_time": close_time,
        }},
    )
    return trades
