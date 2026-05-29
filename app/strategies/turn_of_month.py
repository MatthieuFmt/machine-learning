"""Turn-of-Month (TOM) — stratégie de calendrier (Lakonishok & Smidt 1988).

Hypothèse PRÉ-ENREGISTRÉE (anomalie très documentée) : les indices actions
montent anormalement autour du changement de mois. Fenêtre canonique : on entre
LONG au close du dernier jour de bourse du mois et on sort au close du N-ième
jour de bourse du mois suivant (N=3 par défaut → fenêtre « −1 à +3 »).

Aucun paramètre à optimiser : la fenêtre est fixée par la littérature.

Coûts modélisés : spread+slippage round-trip + swap × nuits détenues.
Données : D1 OHLC, DatetimeIndex UTC.
"""
from __future__ import annotations

import pandas as pd


def simulate_turn_of_month_trades(
    df: pd.DataFrame,
    spread_pips: float,
    slippage_pips: float,
    commission_pips: float,
    pip_size: float,
    swap_long_pips_per_night: float = 0.0,
    hold_days: int = 3,
) -> list[dict]:
    """Simule la stratégie Turn-of-Month sur un DataFrame OHLC D1 indexé UTC.

    Pour chaque mois : entrée au close du DERNIER jour de bourse du mois,
    sortie au close du ``hold_days``-ième jour de bourse suivant.

    Args:
        df: OHLCV avec DatetimeIndex UTC, colonne 'Close' requise.
        spread_pips, slippage_pips, commission_pips: coûts en pips.
        pip_size: taille d'un pip dans l'unité du prix.
        swap_long_pips_per_night: charge swap par nuit (position long).
        hold_days: nombre de jours de bourse détenus après l'entrée (défaut 3).

    Returns:
        Liste de dicts : entry_time, exit_time, entry_price, exit_price,
        pips_brut, pips_net, nights_held, signal (toujours 1 = long).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index doit être DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df.index doit être tz-aware (UTC)")
    if hold_days < 1:
        raise ValueError(f"hold_days doit être >= 1, reçu {hold_days}")

    # Coût round-trip : spread une fois + slippage/commission des deux côtés.
    cost_total = spread_pips + 2 * (slippage_pips + commission_pips)

    idx = df.index
    closes = df["Close"].to_numpy()
    # tz_localize(None) avant to_period : les bornes de mois sont identiques en
    # UTC, et ça évite le UserWarning « drop timezone ».
    months = idx.tz_localize(None).to_period("M")
    # Position entière du DERNIER jour de bourse de chaque mois : la barre dont
    # le mois diffère de la barre suivante (la dernière barre n'a pas de suivante).
    is_last_of_month = months[:-1] != months[1:]
    last_positions = [i for i in range(len(idx) - 1) if is_last_of_month[i]]

    trades: list[dict] = []
    for p in last_positions:
        exit_pos = p + hold_days  # p+1 = 1er jour du mois suivant, +hold_days = N-ième
        if exit_pos >= len(idx):
            continue
        entry_price = float(closes[p])
        exit_price = float(closes[exit_pos])
        if entry_price <= 0 or exit_price <= 0:
            continue

        pips_brut = (exit_price - entry_price) / pip_size
        nights_held = max(0, (idx[exit_pos].normalize() - idx[p].normalize()).days)
        pips_net = pips_brut - cost_total + nights_held * swap_long_pips_per_night

        trades.append({
            "entry_time": idx[p].isoformat(),
            "exit_time": idx[exit_pos].isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pips_brut": pips_brut,
            "pips_net": pips_net,
            "nights_held": nights_held,
            "signal": 1,
        })

    return trades
