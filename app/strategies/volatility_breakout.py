"""Volatility Breakout (Crabel NR4/NR7) — stratégie D1 event-driven (Phase H3).

Hypothèse théorique (Crabel, *Day Trading with Short Term Price Patterns*, 1990) :
une consolidation extrême du range (« Narrow Range » sur 4 ou 7 jours) précède
souvent une expansion directionnelle. Le breakout des extrêmes du jour NR
matérialise le début du mouvement.

Stratégie V1 NR7 (long ET short symétriques) :
    - À la close du jour J, si `range(J) == min(range(J-6..J))` (NR7),
      placer 2 ordres stop pour J+1 :
        * BUY STOP à High(J)
        * SELL STOP à Low(J)
    - Le 1er stop touché en J+1 ouvre la position (un seul trade/jour).
    - Si gap (Open J+1 > High J ou Open J+1 < Low J) → entry au Open.
    - Si les deux stops sont touchés intra-J+1 sans gap → ambigu, on skip
      (convention conservatrice : pas de trade).
    - TP = entry ± tp_mult × range_NR, SL = entry ∓ sl_mult × range_NR.
      Défauts Crabel : tp_mult=2.0, sl_mult=1.0 (R:R = 2:1, breakeven WR 33%).
    - Time-stop : Close(J+1) si ni TP ni SL touché.

Conventions :
    - Path-dependent intra-J+1 : SL prioritaire si TP et SL touchés
      (convention conservatrice — sur D1 on ne connaît pas l'ordre intra-bar).
    - Approximation swap D1 : nights_held=1 si time-stop, 0 si TP/SL.
      (La position dure ~24h pour time-stop, < 24h pour TP/SL touché — pessimiste
      sur le time-stop, neutre sur TP/SL.)
    - Coûts : cost_total = spread + 2 × (slippage + commission).
"""
from __future__ import annotations

import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_nr_days(df_d1: pd.DataFrame, lookback: int = 7) -> pd.DataFrame:
    """Détecte les jours Narrow Range (NR) sur le lookback donné.

    Un jour J est NR(lookback) ssi `range(J) == min(range(J-lookback+1..J))`.

    Args:
        df_d1: OHLCV D1 indexé tz-aware UTC, colonnes [Open, High, Low, Close].
        lookback: Nombre de jours rolling inclus le jour J (défaut 7 = NR7).

    Returns:
        DataFrame indexé comme df_d1, colonnes :
        - day_range : float = High - Low
        - is_nr : bool = True si NR(lookback)
        Les `lookback - 1` premières lignes ont is_nr=False (warmup).
    """
    if not isinstance(df_d1.index, pd.DatetimeIndex):
        raise TypeError("df_d1.index doit être DatetimeIndex")
    if df_d1.index.tz is None:
        raise ValueError("df_d1.index doit être tz-aware (UTC)")
    if lookback < 2:
        raise ValueError(f"lookback doit être >= 2, reçu {lookback}")

    day_range = df_d1["High"] - df_d1["Low"]
    rolling_min = day_range.rolling(window=lookback, min_periods=lookback).min()
    is_nr = (day_range == rolling_min) & rolling_min.notna()

    return pd.DataFrame(
        {"day_range": day_range, "is_nr": is_nr},
        index=df_d1.index,
    )


def simulate_nr_breakout_trades(
    df_d1: pd.DataFrame,
    asset_config: AssetConfig,
    lookback: int = 7,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
) -> list[dict]:
    """Simule la stratégie NR breakout sur D1 (long + short symétriques).

    Args:
        df_d1: OHLCV D1 indexé tz-aware UTC.
        asset_config: Coûts broker (spread, slippage, swap, pip_size).
        lookback: Lookback NR (défaut 7 = NR7).
        tp_mult: Multiplicateur TP (défaut 2.0).
        sl_mult: Multiplicateur SL (défaut 1.0).

    Returns:
        Liste de dicts trades, clés :
        setup_date (date du jour NR), entry_time/exit_time (timestamp J+1),
        signal (+1/-1), entry_price, exit_price, high_J, low_J, range_J,
        tp_price, sl_price, pips_brut, pips_net, nights_held,
        exit_reason ∈ {"tp", "sl", "time_stop"}.
    """
    if not isinstance(df_d1.index, pd.DatetimeIndex):
        raise TypeError("df_d1.index doit être DatetimeIndex")
    if df_d1.index.tz is None:
        raise ValueError("df_d1.index doit être tz-aware (UTC)")

    nr = compute_nr_days(df_d1, lookback=lookback)
    nr_setup_positions = nr.index[nr["is_nr"].values]

    cost_per_side = asset_config.commission_pips + asset_config.slippage_pips
    cost_total = 2 * cost_per_side + asset_config.spread_pips
    pip_size = asset_config.pip_size

    trades: list[dict] = []

    for setup_ts in nr_setup_positions:
        idx_j = df_d1.index.get_loc(setup_ts)
        if idx_j + 1 >= len(df_d1):
            continue  # Pas de J+1 disponible

        ts_j1 = df_d1.index[idx_j + 1]
        bar_j = df_d1.iloc[idx_j]
        bar_j1 = df_d1.iloc[idx_j + 1]

        high_j = float(bar_j["High"])
        low_j = float(bar_j["Low"])
        range_j = float(nr["day_range"].iloc[idx_j])

        open_j1 = float(bar_j1["Open"])
        high_j1 = float(bar_j1["High"])
        low_j1 = float(bar_j1["Low"])
        close_j1 = float(bar_j1["Close"])

        # ── Détermination signal + entry ────────────────────────────
        if open_j1 > high_j:
            signal = 1
            entry_price = open_j1
        elif open_j1 < low_j:
            signal = -1
            entry_price = open_j1
        else:
            up_hit = high_j1 >= high_j
            down_hit = low_j1 <= low_j
            if up_hit and down_hit:
                continue  # Ambigu : les 2 stops touchés sans gap → skip
            if up_hit:
                signal = 1
                entry_price = high_j
            elif down_hit:
                signal = -1
                entry_price = low_j
            else:
                continue  # Pas de breakout

        # ── TP / SL en prix absolus ──────────────────────────────────
        if signal == 1:
            tp_price = entry_price + tp_mult * range_j
            sl_price = entry_price - sl_mult * range_j
        else:
            tp_price = entry_price - tp_mult * range_j
            sl_price = entry_price + sl_mult * range_j

        # ── Path-dependent sur la barre J+1 ─────────────────────────
        if signal == 1:
            if low_j1 <= sl_price:
                exit_price, exit_reason, nights_held = sl_price, "sl", 0
            elif high_j1 >= tp_price:
                exit_price, exit_reason, nights_held = tp_price, "tp", 0
            else:
                exit_price, exit_reason, nights_held = close_j1, "time_stop", 1
        else:
            if high_j1 >= sl_price:
                exit_price, exit_reason, nights_held = sl_price, "sl", 0
            elif low_j1 <= tp_price:
                exit_price, exit_reason, nights_held = tp_price, "tp", 0
            else:
                exit_price, exit_reason, nights_held = close_j1, "time_stop", 1

        # ── PnL ──────────────────────────────────────────────────────
        if signal == 1:
            pips_brut = (exit_price - entry_price) / pip_size
        else:
            pips_brut = (entry_price - exit_price) / pip_size

        pips_net = pips_brut - cost_total
        if nights_held > 0:
            swap = (
                asset_config.swap_long_pips_per_night
                if signal == 1
                else asset_config.swap_short_pips_per_night
            )
            pips_net += nights_held * swap

        trades.append({
            "setup_date": str(setup_ts.date()),
            "entry_time": ts_j1.isoformat(),
            "exit_time": ts_j1.isoformat(),
            "signal": signal,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "high_J": high_j,
            "low_J": low_j,
            "range_J": range_j,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "pips_brut": float(pips_brut),
            "pips_net": float(pips_net),
            "nights_held": nights_held,
            "exit_reason": exit_reason,
        })

    logger.info(
        "nr_breakout_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_nr_setups": len(nr_setup_positions),
            "lookback": lookback,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
        }},
    )
    return trades
