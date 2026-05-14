"""Moteur de backtest déterministe stateful bar-by-bar pour H03.

Règles :
- Une seule position à la fois (stateful).
- Entrée au Close de la barre où signal ≠ 0 ET pas de position ouverte.
- Sortie au premier de : TP touché, SL touché, window_hours écoulé.
- Si TP et SL touchés dans la même barre : TP prime (conservateur, spec H03 §5.1).
- Commission + slippage déduits à l'entrée ET à la sortie.
- Timeout : sortie au Close de la barre d'expiration (prix réel).
- Zéro ML, zéro filtre de régime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_deterministic_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    tp_pips: float,
    sl_pips: float,
    window_hours: int,
    commission_pips: float,
    slippage_pips: float,
    pip_size: float = 1.0,
) -> dict[str, Any]:
    """Backtest bar-by-bar avec TP/SL fixes, moteur stateful.

    Args:
        df: DataFrame avec colonnes Open, High, Low, Close, index=Time (datetime).
        signals: pd.Series 1=LONG, -1=SHORT, 0=FLAT, même index que df.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window_hours: Durée max d'un trade en heures.
        commission_pips: Commission en pips (aller-retour par trade).
        slippage_pips: Slippage estimé en pips.
        pip_size: Taille d'un pip (ex: 0.0001 pour EURUSD, 1.0 pour XAUUSD).

    Returns:
        dict avec clés:
            sharpe: float — Sharpe ratio annualisé.
            wr: float — Win rate (0.0 à 1.0).
            total_trades: int.
            total_pnl_pips: float.
            trades: list[dict] — détail de chaque trade.
    """
    if "Time" in df.columns:
        df = df.set_index("Time")

    # Alignement
    common_idx = df.index.intersection(signals.index)
    df = df.loc[common_idx]
    signals = signals.loc[common_idx]

    n = len(df)
    if n == 0:
        return _empty_result()

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size
    cost_per_side = commission_pips + slippage_pips  # entrée + sortie
    cost_total = cost_per_side * 2  # entrée ET sortie

    # Calcul du window en nombre de barres
    if n >= 2:
        typical_td = (times[-1] - times[0]) / n
        typical_hours = typical_td.total_seconds() / 3600.0
        window_bars = max(1, int(window_hours / typical_hours)) if typical_hours > 0 else window_hours
    else:
        window_bars = window_hours

    trades: list[dict[str, Any]] = []
    i = 0

    while i < n:
        sig_val = int(signals.iloc[i])

        if sig_val == 0:
            i += 1
            continue

        signal = sig_val  # 1 ou -1
        entry_time = times[i]
        entry_price = closes[i]

        if signal == 1:
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        pips_net = -cost_total
        result_type = "loss_timeout"
        exit_time = entry_time
        exit_price = entry_price

        # Boucle forward jusqu'à window_bars
        for j in range(1, window_bars + 1):
            idx = i + j
            if idx >= n:
                # Fin des données
                exit_idx = n - 1
                exit_time = times[exit_idx]
                exit_price = closes[exit_idx]
                if signal == 1:
                    pips_net = (exit_price - entry_price) / pip_size - cost_total
                else:
                    pips_net = (entry_price - exit_price) / pip_size - cost_total
                result_type = "loss_timeout"
                i = n
                break

            curr_high = highs[idx]
            curr_low = lows[idx]

            if signal == 1:
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low <= sl_price

                if tp_hit and sl_hit:
                    # Même barre : TP prime (conservateur, spec H03)
                    exit_idx = idx
                    exit_time = times[idx]
                    exit_price = tp_price
                    pips_net = tp_pips - cost_total
                    result_type = "win"
                    i = idx
                    break
                elif sl_hit:
                    exit_idx = idx
                    exit_time = times[idx]
                    exit_price = sl_price
                    pips_net = -sl_pips - cost_total
                    result_type = "loss_sl"
                    i = idx
                    break
                elif tp_hit:
                    exit_idx = idx
                    exit_time = times[idx]
                    exit_price = tp_price
                    pips_net = tp_pips - cost_total
                    result_type = "win"
                    i = idx
                    break
            else:  # signal == -1
                tp_hit = curr_low <= tp_price
                sl_hit = curr_high >= sl_price

                if tp_hit and sl_hit:
                    exit_idx = idx
                    exit_time = times[idx]
                    exit_price = tp_price
                    pips_net = tp_pips - cost_total
                    result_type = "win"
                    i = idx
                    break
                elif sl_hit:
                    exit_idx = idx
                    exit_time = times[idx]
                    exit_price = sl_price
                    pips_net = -sl_pips - cost_total
                    result_type = "loss_sl"
                    i = idx
                    break
                elif tp_hit:
                    exit_idx = idx
                    exit_time = times[idx]
                    exit_price = tp_price
                    pips_net = tp_pips - cost_total
                    result_type = "win"
                    i = idx
                    break
        else:
            # Timeout : sortie au Close de la dernière barre
            exit_idx = min(i + window_bars, n - 1)
            exit_time = times[exit_idx]
            exit_price = closes[exit_idx]
            if signal == 1:
                pips_net = (exit_price - entry_price) / pip_size - cost_total
            else:
                pips_net = (entry_price - exit_price) / pip_size - cost_total
            result_type = "loss_timeout"
            i += window_bars

        trades.append({
            "entry_time": str(entry_time),
            "exit_time": str(exit_time),
            "signal": signal,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pips_net": float(pips_net),
            "result": result_type,
        })
        continue

    return _compute_metrics(trades)


def _compute_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcule les métriques à partir de la liste de trades."""
    if not trades:
        return _empty_result()

    pnls = np.array([t["pips_net"] for t in trades])
    total_trades = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    wr = float(len(wins) / total_trades) if total_trades > 0 else 0.0
    total_pnl = float(pnls.sum())

    # Sharpe annualisé — calculé sur les returns quotidiens de la courbe d'equity
    from learning_machine_learning_v2.backtest.metrics import sharpe_daily_from_trades
    sharpe = sharpe_daily_from_trades(trades)

    # Maximum drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(drawdown.min())

    # Profit factor
    if len(wins) > 0 and len(losses) > 0:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    elif len(wins) > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "sharpe": sharpe,
        "wr": wr,
        "total_trades": total_trades,
        "total_pnl_pips": total_pnl,
        "max_drawdown_pips": max_dd,
        "profit_factor": profit_factor,
        "mean_pnl_per_trade": float(pnls.mean()),
        "trades": trades,
    }


def _empty_result() -> dict[str, Any]:
    """Résultat vide quand aucun trade n'est généré."""
    return {
        "sharpe": 0.0,
        "wr": 0.0,
        "total_trades": 0,
        "total_pnl_pips": 0.0,
        "max_drawdown_pips": 0.0,
        "profit_factor": 0.0,
        "mean_pnl_per_trade": 0.0,
        "trades": [],
    }
