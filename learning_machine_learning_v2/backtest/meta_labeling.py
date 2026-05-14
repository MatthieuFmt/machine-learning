"""Méta-labeling — labellise les signaux Donchian selon l'issue du trade.

Pour chaque barre où Donchian émet un signal (±1) :
- Simule un trade stateful (mêmes règles que run_deterministic_backtest)
- Label = 1 si TP touché avant SL/timeout, 0 sinon
- Pour les barres sans signal : NaN

Réutilise le moteur stateful : 1 trade à la fois, TP prime sur SL si même barre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_meta_labels(
    df: pd.DataFrame,
    donchian_signals: pd.Series,
    tp_pips: float = 200.0,
    sl_pips: float = 100.0,
    window_hours: int = 120,
    pip_size: float = 1.0,
) -> pd.Series:
    """Pour chaque barre où donchian_signals ≠ 0, détermine si le trade est gagnant.

    Même moteur stateful que run_deterministic_backtest :
    - 1 trade à la fois (stateful)
    - TP prime sur SL si les deux sont touchés dans la même barre (conservateur)
    - Timeout : sortie au Close de la barre d'expiration (prix réel)

    Args:
        df: DataFrame OHLC, index=Time (datetime).
        donchian_signals: pd.Series −1, 0, 1, même index que df.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window_hours: Durée max du trade en heures.
        pip_size: Taille d'un pip (1.0 pour US30).

    Returns:
        pd.Series avec même index que df :
            1 = trade gagnant, 0 = trade perdant, NaN = pas de signal.
    """
    if "Time" in df.columns:
        df = df.set_index("Time")

    common_idx = df.index.intersection(donchian_signals.index)
    df = df.loc[common_idx]
    signals = donchian_signals.loc[common_idx]

    n = len(df)
    labels = pd.Series(np.nan, index=df.index, dtype="float64")

    if n == 0:
        return labels

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size

    # Calcul du window en nombre de barres
    if n >= 2:
        typical_td = (times[-1] - times[0]) / n
        typical_hours = typical_td.total_seconds() / 3600.0
        window_bars = (
            max(1, int(window_hours / typical_hours))
            if typical_hours > 0
            else window_hours
        )
    else:
        window_bars = window_hours

    i = 0
    while i < n:
        sig_val = int(signals.iloc[i])

        if sig_val == 0:
            i += 1
            continue

        signal = sig_val  # 1 ou -1
        entry_bar = i
        entry_price = closes[i]

        if signal == 1:
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        result: int = 0  # 0 = perdant par défaut

        for j in range(1, window_bars + 1):
            idx = i + j
            if idx >= n:
                # Fin des données → timeout → perdant
                result = 0
                i = n
                break

            curr_high = highs[idx]
            curr_low = lows[idx]

            if signal == 1:
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low <= sl_price
            else:
                tp_hit = curr_low <= tp_price
                sl_hit = curr_high >= sl_price

            if tp_hit and sl_hit:
                # Même barre : TP prime (conservateur, spec H03 §5.1)
                result = 1
                i = idx
                break
            elif sl_hit:
                result = 0
                i = idx
                break
            elif tp_hit:
                result = 1
                i = idx
                break
        else:
            # Timeout → perdant
            result = 0
            i += window_bars

        labels.iloc[entry_bar] = float(result)

    return labels
