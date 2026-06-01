"""Gap Fade (overnight→intraday reversal) — stratégie intraday (Phase 1).

Hypothèse PRÉ-ENREGISTRÉE (sur-réaction nocturne ; cf. overnight-intraday
reversal, Della Corte & Kosowski) : quand un indice OUVRE avec un écart (gap)
vs la clôture de la veille, cet écart se REFERME souvent dans la journée. On
parie contre le gap, on entre à l'ouverture, on sort à la clôture → flat la
nuit → ZÉRO swap (le tueur des holds multi-jours).

Règle (figée) :
    - gap = open(séance) − close(séance précédente), en prix.
    - On ne trade QUE si |gap| > seuil = `min_gap_cost_mult` × coût aller-retour
      (sinon, même un comblement total ne couvre pas les frais — seuil
      PRINCIPIEL, pas tuné).
    - gap > 0 (ouverture au-dessus) → SHORT à l'open (signal −1).
    - gap < 0 (ouverture en-dessous) → LONG à l'open (signal +1).
    - Sortie à la clôture de séance (EOD). Pas de TP/SL en V1.
    - Coûts : spread + 2×(slippage+commission). Swap = 0 (intraday).

⚠️ Effet documenté mais réputé AFFAIBLI (marchés efficaces) — à mesurer chez
nous avec vrais coûts.
"""
from __future__ import annotations

import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def _parse_hhmm(s: str) -> int:
    """'09:30' → minutes depuis minuit (570)."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def simulate_gap_fade(
    df_intraday: pd.DataFrame,
    asset_config: AssetConfig,
    *,
    session_tz: str,
    open_time: str = "09:30",
    close_time: str = "16:00",
    min_gap_cost_mult: float = 1.0,
) -> list[dict]:
    """Simule le fade du gap d'ouverture (long + short), intraday.

    Args:
        df_intraday: OHLCV intraday (M5 recommandé) indexé tz-aware UTC.
        asset_config: Coûts/paramètres broker.
        session_tz: Fuseau de la place (ex. "America/New_York").
        open_time: Ouverture cash locale "HH:MM" (le prix d'ouverture de séance
            = Open de la 1ʳᵉ barre dont l'heure locale ≥ open_time).
        close_time: Clôture cash locale "HH:MM" (le prix de clôture = Close de la
            dernière barre dont l'heure locale ≤ close_time).
        min_gap_cost_mult: Seuil de gap minimal, en multiples du coût a/r
            (1.0 = on exige un gap > coût aller-retour).

    Returns:
        Liste de dicts trades (clés : date, signal, gap_pips, entry_time,
        exit_time, entry_price, exit_price, pips_brut, pips_net, nights_held,
        exit_reason). ``pips_net``/``exit_time`` compatibles
        ``sharpe_daily_from_trades``.
    """
    if not isinstance(df_intraday.index, pd.DatetimeIndex):
        raise TypeError("df_intraday.index doit être DatetimeIndex")
    if df_intraday.index.tz is None:
        raise ValueError("df_intraday.index doit être tz-aware (UTC)")
    for col in ("Open", "Close"):
        if col not in df_intraday.columns:
            raise KeyError(f"df_intraday doit contenir la colonne '{col}'")

    open_min = _parse_hhmm(open_time)
    close_min = _parse_hhmm(close_time)
    if close_min <= open_min:
        raise ValueError(f"close_time ({close_time}) doit être > open_time ({open_time})")

    pip_size = asset_config.pip_size
    cost_total = (
        asset_config.spread_pips
        + 2 * (asset_config.slippage_pips + asset_config.commission_pips)
    )
    floor_price = min_gap_cost_mult * cost_total * pip_size

    work = df_intraday[["Open", "Close"]].sort_index().copy()
    local = work.index.tz_convert(session_tz)
    work["tmin"] = local.hour * 60 + local.minute
    work["ldate"] = local.date

    # Résumé par séance : (date, open_price/ts, close_price/ts).
    sessions: list[dict] = []
    for _ldate, day in work.groupby("ldate", sort=True):
        sess = day[(day["tmin"] >= open_min) & (day["tmin"] <= close_min)].sort_index()
        if sess.empty:
            continue
        sessions.append({
            "date": _ldate,
            "open_price": float(sess.iloc[0]["Open"]),
            "open_ts": sess.index[0],
            "close_price": float(sess.iloc[-1]["Close"]),
            "close_ts": sess.index[-1],
        })

    trades: list[dict] = []
    for i in range(1, len(sessions)):
        cur, prev = sessions[i], sessions[i - 1]
        gap = cur["open_price"] - prev["close_price"]
        if abs(gap) <= floor_price:
            continue
        direction = -1 if gap > 0 else 1  # fade
        move = cur["close_price"] - cur["open_price"]
        pips_brut = direction * move / pip_size
        pips_net = pips_brut - cost_total
        trades.append({
            "date": str(cur["date"]),
            "signal": direction,
            "gap_pips": float(gap / pip_size),
            "entry_time": cur["open_ts"].isoformat(),
            "exit_time": cur["close_ts"].isoformat(),
            "entry_price": cur["open_price"],
            "exit_price": cur["close_price"],
            "pips_brut": float(pips_brut),
            "pips_net": float(pips_net),
            "nights_held": 0,
            "exit_reason": "eod",
        })

    logger.info(
        "gap_fade_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_sessions": len(sessions),
            "session_tz": session_tz,
            "min_gap_cost_mult": min_gap_cost_mult,
        }},
    )
    return trades
