"""Grid search pour stratégies déterministes H03.

Pour un actif donné, teste toutes les combinaisons de param_grid sur train,
sélectionne le meilleur Sharpe train, évalue une fois sur val et test.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from app.backtest.deterministic import run_deterministic_backtest
from app.strategies.base import BaseStrategy


def grid_search_asset(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    strategy_class: type[BaseStrategy],
    param_grid: dict[str, list],
    tp_pips: float,
    sl_pips: float,
    window_hours: int,
    commission_pips: float,
    slippage_pips: float,
    pip_size: float,
) -> dict[str, Any]:
    """Grid search sur train, évaluation unique sur val et test.

    Args:
        df_train: Période ≤ 2022.
        df_val: Période 2023.
        df_test: Période ≥ 2024.
        strategy_class: Classe de stratégie (sous-classe de BaseStrategy).
        param_grid: Dictionnaire {param_name: [valeurs]}.
        tp_pips, sl_pips: Take-profit / stop-loss en pips.
        window_hours: Durée max d'un trade en heures.
        commission_pips: Commission en pips.
        slippage_pips: Slippage en pips.
        pip_size: Taille d'un pip.

    Returns:
        dict avec best_params, sharpe_train/val/test, wr_train/val/test, all_results.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_results: list[dict[str, Any]] = []
    best_sharpe = -float("inf")
    best_params: dict[str, Any] = {}
    best_train_result: dict[str, Any] = {}

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))

        try:
            strategy = strategy_class(**params)
            signals_train = strategy.generate_signals(df_train)
            result_train = run_deterministic_backtest(
                df_train, signals_train,
                tp_pips=tp_pips, sl_pips=sl_pips,
                window_hours=window_hours,
                commission_pips=commission_pips,
                slippage_pips=slippage_pips,
                pip_size=pip_size,
            )
        except Exception as exc:
            all_results.append({
                "params": params,
                "sharpe_train": None,
                "error": str(exc),
            })
            continue

        sharpe_t = result_train["sharpe"]

        all_results.append({
            "params": params,
            "sharpe_train": sharpe_t,
            "wr_train": result_train["wr"],
            "total_trades_train": result_train["total_trades"],
            "total_pnl_train": result_train["total_pnl_pips"],
        })

        if sharpe_t > best_sharpe:
            best_sharpe = sharpe_t
            best_params = dict(params)
            best_train_result = result_train

    if not best_params:
        return {
            "best_params": {},
            "sharpe_train": 0.0,
            "sharpe_val": 0.0,
            "sharpe_test": 0.0,
            "wr_train": 0.0,
            "wr_val": 0.0,
            "wr_test": 0.0,
            "all_results": all_results,
            "error": "No valid parameter combination found",
        }

    # Évaluation sur val (unique)
    best_strategy = strategy_class(**best_params)
    signals_val = best_strategy.generate_signals(df_val)
    result_val = run_deterministic_backtest(
        df_val, signals_val,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=window_hours,
        commission_pips=commission_pips,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
    )

    # Évaluation sur test (unique)
    signals_test = best_strategy.generate_signals(df_test)
    result_test = run_deterministic_backtest(
        df_test, signals_test,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=window_hours,
        commission_pips=commission_pips,
        slippage_pips=slippage_pips,
        pip_size=pip_size,
    )

    return {
        "best_params": best_params,
        "sharpe_train": best_sharpe,
        "sharpe_val": result_val["sharpe"],
        "sharpe_test": result_test["sharpe"],
        "wr_train": best_train_result.get("wr", 0.0),
        "wr_val": result_val["wr"],
        "wr_test": result_test["wr"],
        "total_trades_train": best_train_result.get("total_trades", 0),
        "total_trades_val": result_val["total_trades"],
        "total_trades_test": result_test["total_trades"],
        "total_pnl_train": best_train_result.get("total_pnl_pips", 0.0),
        "total_pnl_val": result_val["total_pnl_pips"],
        "total_pnl_test": result_test["total_pnl_pips"],
        "all_results": all_results,
    }
