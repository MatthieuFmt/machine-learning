"""Métriques de backtest — fonctions pures, vectorisées NumPy.

Toutes les métriques sont calculées à partir d'un DataFrame de trades.
Aucune dépendance à l'instrument ou à la configuration — les paramètres
sont passés explicitement.

Pivot v4 A1 : compute_metrics() supporte AssetConfig pour equity €,
DD borné [−100%, 0%], Sharpe sur retours quotidiens du capital.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.config.instruments import AssetConfig


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
    initial_capital_pips: float = 10_000.0,
    frequency_aware: bool = True,
) -> float:
    """Sharpe annualisé à partir d'une liste de trades, sur retours linéaires.

    Fix F2 : retours linéaires `daily_pnl / initial_capital`, pas `pct_change`
    sur une cumsum (qui mélange composé et linéaire). Le sizing en lots est
    calculé une fois sur un capital fixe → les retours doivent être linéaires
    pour rester cohérents.

    Fix E4 (2026-05-29) — annualisation routée par fréquence. Le resample daily
    avec `ffill` insère des jours à retour nul (jours sans trade) qui écrasent
    la volatilité et GONFLENT le Sharpe pour les stratégies basse-fréquence
    (ex. ~12 trades / 30 mois en D1). On route donc selon trades/an :
        ≥ 100 trades/an → daily  (√252)
        30-99 trades/an → weekly (√52)
        < 30 trades/an  → per-trade × √(trades_par_an)  (sans jours nuls fantômes)
    Mettre `frequency_aware=False` pour retrouver l'ancien comportement daily pur.

    Args:
        trades: Liste de dicts avec 'pips_net' (float) et 'exit_time' (str).
        annual_factor: Facteur d'annualisation utilisé si `frequency_aware=False`.
        initial_capital_pips: Capital de référence en pips. Défaut 10 000.
        frequency_aware: Si True (défaut), route l'annualisation par fréquence.

    Returns:
        Sharpe ratio annualisé. 0.0 si insuffisamment d'observations.
    """
    if not trades or len(trades) < 2:
        return 0.0

    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    if np.all(pnls <= 0):
        return 0.0

    exit_times = pd.to_datetime([t["exit_time"] for t in trades])

    if frequency_aware:
        span_seconds = (exit_times.max() - exit_times.min()).total_seconds()
        years = max(span_seconds / (365.25 * 86400), 1e-3)
        tpy = len(trades) / years

        if tpy < 30.0:
            # Per-trade : pas de jours nuls fantômes. Annualisation √(trades/an).
            per_trade = pnls / initial_capital_pips
            return sharpe_ratio(per_trade, annual_factor=tpy)

        resample_rule, factor = ("W-FRI", 52.0) if tpy < 100.0 else ("D", 252.0)
    else:
        resample_rule, factor = "D", annual_factor

    equity = np.cumsum(pnls)
    equity_series = pd.Series(equity, index=exit_times).sort_index()
    equity_resampled = equity_series.resample(resample_rule).last().ffill()
    if len(equity_resampled) < 2:
        return 0.0

    returns = equity_resampled.diff().dropna() / initial_capital_pips
    return sharpe_ratio(returns, annual_factor=factor)


def sharpe_annualized(
    equity: pd.Series,
    trades_df: pd.DataFrame,
    asset_cfg: AssetConfig | None = None,
    capital_eur: float = 10_000.0,
) -> tuple[float, Literal["daily", "weekly", "per_trade"]]:
    """Sharpe annualisé, route selon la fréquence des trades.

    Routing :
        ≥ 100 trades/an → daily resample (√252)
        30-99 trades/an → weekly resample (√52)
        < 30 trades/an  → per-trade × √trades_per_year

    Args:
        equity: Courbe d'equity (index DatetimeIndex recommandé).
        trades_df: DataFrame de trades avec Pips_Nets, position_size_lots.
        asset_cfg: Config actif (requis pour per-trade).
        capital_eur: Capital initial en €.

    Returns:
        (sharpe_annualized, method) — 0.0 si impossible à calculer.
    """
    if len(trades_df) < 2 or equity.empty:
        return 0.0, "daily"

    # Calculer la période en années
    if isinstance(trades_df.index, pd.DatetimeIndex):
        span_seconds = (trades_df.index[-1] - trades_df.index[0]).total_seconds()
    else:
        span_seconds = max(1, len(trades_df)) * 86400
    years = max(span_seconds / (365.25 * 86400), 1e-3)
    tpy = len(trades_df) / years

    if tpy >= 100:
        # Méthode daily : retours linéaires (fix F2).
        # daily.diff() / capital_eur → cohérent avec sizing à capital fixe.
        if isinstance(equity.index, pd.DatetimeIndex):
            daily = equity.resample("D").last().ffill()
        else:
            daily = equity
        returns = daily.diff().dropna() / capital_eur
        sr = sharpe_ratio(returns, annual_factor=252.0)
        return sr, "daily"

    if tpy >= 30:
        # Méthode weekly : retours linéaires (fix F2).
        if isinstance(equity.index, pd.DatetimeIndex):
            weekly = equity.resample("W-FRI").last().ffill()
        else:
            weekly = equity
        returns = weekly.diff().dropna() / capital_eur
        sr = sharpe_ratio(returns, annual_factor=52.0)
        return sr, "weekly"

    # Méthode per-trade : pour stratégies < 30 trades/an
    if asset_cfg is None:
        return 0.0, "per_trade"
    if "position_size_lots" not in trades_df.columns:
        per_trade_returns = (
            trades_df["Pips_Nets"] * asset_cfg.pip_value_eur / capital_eur
        )
    else:
        per_trade_returns = (
            trades_df["Pips_Nets"]
            * trades_df["position_size_lots"]
            * asset_cfg.pip_value_eur
            / capital_eur
        )
    if per_trade_returns.std() == 0 or len(per_trade_returns) < 2:
        return 0.0, "per_trade"
    sr_per_trade = float(per_trade_returns.mean() / per_trade_returns.std())
    sr_annualized = sr_per_trade * np.sqrt(tpy)
    return sr_annualized, "per_trade"


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
    asset_cfg: AssetConfig | None = None,
    capital_eur: float = 10_000.0,
) -> dict:
    """Métriques agrégées sur un DataFrame de trades.

    Deux modes :
    - Pivot v4 A1 (asset_cfg non-None) : equity en €, DD borné [−100%, 0%],
      Sharpe sur retours quotidiens du capital, PnL via position_size_lots.
    - Legacy (asset_cfg None) : comportement inchangé (pips → € via
      pip_value_eur / initial_capital).

    Args:
        trades_df: DataFrame indexé par Time avec colonnes Pips_Nets, Pips_Bruts,
            Weight, result. En mode A1, colonne 'position_size_lots' requise.
        annee: Année du backtest (pour le rapport).
        df: DataFrame d'entrée du backtest avec 'Close' (pour B&H benchmark).
        pip_value_eur: [Legacy] Valeur d'un pip en EUR.
        initial_capital: [Legacy] Capital de référence.
        pip_size: [Legacy] Taille d'un pip.
        asset_cfg: Pivot v4 A1 — config actif. Active le mode equity € si fourni.
        capital_eur: Pivot v4 A1 — capital initial en €.

    Returns:
        Dict de métriques. Inclut 'blowup_detected' en mode A1.
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
        if asset_cfg is not None:
            base["profit_net_eur"] = 0.0
            base["final_equity_eur"] = capital_eur
            base["blowup_detected"] = False
        return base

    n_trades = len(trades_df)
    win_rate = (trades_df["Pips_Bruts"] > 0).mean() * 100

    # ── Pivot v4 A1 : mode equity € ──────────────────────────────────
    if asset_cfg is not None:
        if "position_size_lots" not in trades_df.columns:
            raise ValueError(
                "trades_df doit contenir 'position_size_lots'. "
                "Re-run le simulator avec sizing au risque 2 %."
            )

        from app.backtest.sizing import expected_pnl_eur

        pnl_eur: np.ndarray = expected_pnl_eur(  # type: ignore[assignment]
            trades_df["Pips_Nets"].values,
            trades_df["position_size_lots"].values,
            asset_cfg,
        )
        pnl_eur_series = pd.Series(pnl_eur, index=trades_df.index)

        equity = capital_eur + pnl_eur_series.cumsum()
        # Protection blow-up : equity ne descend pas sous 0.01 €
        blowup_detected = bool((equity < 0.01).any())
        equity = equity.clip(lower=0.01)

        # Drawdown borné [−1, 0]
        cummax = equity.cummax()
        dd_series = (equity / cummax) - 1.0
        max_dd_pct = float(dd_series.min()) * 100  # négatif, borné à −100 %
        # Fix F12 : assertion défensive. Si on sort de [-100, 0], il y a un bug
        # de chemin de calcul (ex: trades_df['pips_net'] confondu avec EUR).
        if not (-100.01 <= max_dd_pct <= 0.01):
            raise AssertionError(
                f"max_dd_pct hors bornes [-100, 0] : {max_dd_pct:.2f}. "
                f"Vérifier que trades_df['Pips_Nets'] est en pips et "
                f"'position_size_lots' est cohérent avec asset_cfg."
            )

        # Sharpe routing par fréquence (pivot v4 A3)
        sharpe, sharpe_method = sharpe_annualized(
            equity, trades_df, asset_cfg, capital_eur
        )

        # Sharpe per-trade (annualisé par n_trades)
        if n_trades > 1:
            per_trade_returns = pnl_eur / equity.shift(1).fillna(capital_eur).values
            sharpe_per_trade = sharpe_ratio(
                per_trade_returns, annual_factor=float(max(n_trades, 1))
            )
        else:
            sharpe_per_trade = 0.0

        profit_net_eur = float(pnl_eur.sum())
        total_return_pct = (float(equity.iloc[-1]) / capital_eur - 1.0) * 100
        final_equity_eur = float(equity.iloc[-1])
        profit_net_pips = float(trades_df["Pips_Nets"].sum())
        dd_pips = max_drawdown(trades_df["Pips_Nets"].cumsum())

        # B&H benchmark (en €)
        bh_pips = buy_and_hold_pips(df, asset_cfg.pip_size) if df is not None else 0.0
        bh_return_pct = (bh_pips * asset_cfg.pip_value_eur / capital_eur) * 100 if df is not None else 0.0
        alpha_pips = profit_net_pips - bh_pips
        alpha_return_pct = total_return_pct - bh_return_pct

        return {
            "annee": annee,
            "profit_net": profit_net_pips,
            "profit_net_eur": profit_net_eur,
            "dd": dd_pips,
            "trades": n_trades,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "sharpe_method": sharpe_method,
            "sharpe_per_trade": sharpe_per_trade,
            "total_return_pct": total_return_pct,
            "max_dd_pct": max_dd_pct,
            "bh_pips": bh_pips,
            "bh_return_pct": bh_return_pct,
            "alpha_pips": alpha_pips,
            "alpha_return_pct": alpha_return_pct,
            "final_equity_eur": final_equity_eur,
            "blowup_detected": blowup_detected,
        }

    # ── Mode legacy (asset_cfg None) ─────────────────────────────────
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
        "sharpe_method": "daily",
        "sharpe_per_trade": sharpe_per_trade,
        "total_return_pct": total_return_pct,
        "max_dd_pct": max_dd_pct,
        "bh_pips": bh_pips,
        "bh_return_pct": bh_return_pct,
        "alpha_pips": alpha_pips,
        "alpha_return_pct": alpha_return_pct,
    }
