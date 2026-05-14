"""Métriques de backtest — fonctions pures, vectorisées NumPy.

Toutes les métriques sont calculées à partir d'un DataFrame de trades.
Aucune dépendance à l'instrument ou à la configuration — les paramètres
sont passés explicitement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _pips_to_return(
    pips: float | np.ndarray | pd.Series,
    pip_value_eur: float = 1.0,
    initial_capital: float = 10_000.0,
) -> float | np.ndarray:
    """Conversion pips → fraction de capital."""
    return pips * pip_value_eur / initial_capital


def sharpe_ratio(
    returns: np.ndarray | pd.Series,
    annual_factor: float = 252.0,
) -> float:
    """Sharpe ratio annualisé.

    Args:
        returns: Séquence de returns (quotidiens recommandés).
        annual_factor: Facteur d'annualisation (252 pour daily, 12 pour monthly).

    Returns:
        Sharpe ratio annualisé, 0.0 si volatilité nulle ou < 2 observations.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if not np.isfinite(std_val) or std_val == 0:
        return 0.0
    return (mean_val / std_val) * np.sqrt(annual_factor)


def sharpe_daily_from_trades(
    trades: list[dict[str, Any]],
    annual_factor: float = 252.0,
) -> float:
    """Sharpe annualisé à partir d'une liste de trades — calculé sur les returns
    quotidiens de la courbe d'equity (pas sur les PnL par trade).

    Méthode correcte : construit la courbe d'equity cumulée → resample quotidien
    (forward fill) → returns journaliers → annualisation √252.

    Args:
        trades: Liste de dicts avec 'pips_net' (float) et 'exit_time' (str).
        annual_factor: Facteur d'annualisation (252 = quotidien).

    Returns:
        Sharpe ratio annualisé. 0.0 si < 2 jours de returns.
    """
    if not trades or len(trades) < 2:
        return 0.0

    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    equity = np.cumsum(pnls)

    # Edge case: tous les trades perdants -> Sharpe negatif ou zero
    if len(pnls) > 0 and np.all(pnls <= 0):
        return 0.0

    exit_times = pd.to_datetime([t["exit_time"] for t in trades])
    equity_series = pd.Series(equity, index=exit_times).sort_index()

    equity_daily = equity_series.resample("D").last().ffill()
    if len(equity_daily) < 2:
        return 0.0

    daily_returns = equity_daily.pct_change().dropna()
    return sharpe_ratio(daily_returns, annual_factor=annual_factor)


def max_drawdown(pnl_series: pd.Series) -> float:
    """Drawdown maximum (valeur négative ou zéro).

    Args:
        pnl_series: Série de PnL cumulatif.

    Returns:
        Drawdown max (≤ 0).
    """
    if pnl_series.empty:
        return 0.0
    cummax = pnl_series.cummax()
    dd = (pnl_series - cummax).min()
    return float(dd)


def buy_and_hold_pips(df: pd.DataFrame, pip_size: float = 0.0001) -> float:
    """Profit en pips d'un buy & hold long.

    Args:
        df: DataFrame avec colonne 'Close'.
        pip_size: Taille d'un pip.

    Returns:
        Pips gagnés entre la première et la dernière Close.
    """
    if df is None or df.empty:
        return 0.0
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return 0.0
    return (closes.iloc[-1] - closes.iloc[0]) / pip_size


def compute_metrics(
    trades_df: pd.DataFrame,
    annee: int | None = None,
    df: pd.DataFrame | None = None,
    pip_value_eur: float = 1.0,
    initial_capital: float = 10_000.0,
    pip_size: float = 0.0001,
) -> dict:
    """Métriques agrégées sur un DataFrame de trades.

    Args:
        trades_df: DataFrame indexé par Time avec colonnes Pips_Nets, Pips_Bruts, Weight, result.
        annee: Année du backtest (pour le rapport).
        df: DataFrame d'entrée du backtest avec 'Close' (pour B&H benchmark).
        pip_value_eur: Valeur d'un pip en EUR.
        initial_capital: Capital de référence.
        pip_size: Taille d'un pip.

    Returns:
        Dict de métriques.
    """
    base: dict = {
        "annee": annee,
        "profit_net": 0.0,
        "dd": 0.0,
        "trades": 0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "sharpe_per_trade": 0.0,
        "total_return_pct": 0.0,
        "max_dd_pct": 0.0,
        "bh_pips": 0.0,
        "bh_return_pct": 0.0,
        "alpha_pips": 0.0,
        "alpha_return_pct": 0.0,
    }

    if trades_df.empty:
        return base

    n_trades = len(trades_df)
    win_rate = (trades_df["Pips_Bruts"] > 0).mean() * 100
    profit_net = trades_df["Pips_Nets"].sum()
    cum = trades_df["Pips_Nets"].cumsum()
    dd = max_drawdown(cum)

    # Sharpe quotidien
    daily_pips = trades_df["Pips_Nets"].resample("D").sum().dropna()
    daily_returns = _pips_to_return(daily_pips, pip_value_eur, initial_capital)
    sharpe = sharpe_ratio(daily_returns)

    # Sharpe per-trade (neutralise l'effet diversification intraday)
    trade_returns = _pips_to_return(
        trades_df["Pips_Nets"].values, pip_value_eur, initial_capital
    )
    sharpe_per_trade = sharpe_ratio(trade_returns, annual_factor=n_trades) if n_trades > 1 else 0.0

    total_return_pct = float(_pips_to_return(profit_net, pip_value_eur, initial_capital)) * 100
    max_dd_pct = float(_pips_to_return(dd, pip_value_eur, initial_capital)) * 100

    # Benchmark B&H
    bh_pips = buy_and_hold_pips(df, pip_size) if df is not None else 0.0
    bh_return_pct = float(_pips_to_return(bh_pips, pip_value_eur, initial_capital)) * 100
    alpha_pips = profit_net - bh_pips
    alpha_return_pct = total_return_pct - bh_return_pct

    return {
        "annee": annee,
        "profit_net": profit_net,
        "dd": dd,
        "trades": n_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "sharpe_per_trade": sharpe_per_trade,
        "total_return_pct": total_return_pct,
        "max_dd_pct": max_dd_pct,
        "bh_pips": bh_pips,
        "bh_return_pct": bh_return_pct,
        "alpha_pips": alpha_pips,
        "alpha_return_pct": alpha_return_pct,
    }
