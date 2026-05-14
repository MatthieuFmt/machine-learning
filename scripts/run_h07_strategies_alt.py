# ruff: noqa: E402  # sys.path.insert avant imports (nécessaire pour `app.*`)
"""Run H07 — Stratégies trend-following alternatives (Prompt 08).

Grid search de 4 stratégies (Dual MA, Keltner Channel, Chandelier Exit,
Parabolic SAR) sur US30 D1. Sélection du meilleur Sharpe train ≤ 2022,
évaluation val=2023, test ≥ 2024 avec coûts réalistes, puis validate_edge()
et corrélation rolling 60j vs Donchian.

Produit predictions/h07_strategies_alt.json.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ajouter la racine du projet au PYTHONPATH pour les imports `app.*`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from app.analysis.edge_validation import EdgeReport, validate_edge
from app.backtest.deterministic import run_deterministic_backtest
from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.strategies.chandelier import ChandelierExit
from app.strategies.donchian import DonchianBreakout
from app.strategies.dual_ma import DualMovingAverage
from app.strategies.keltner import KeltnerChannel
from app.strategies.parabolic import ParabolicSAR
from app.testing.snooping_guard import check_unlocked, read_oos

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes figées (Constitution §3)
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"

# Actif cible : US30 (baseline H06, seul actif avec edge historique)
PRIMARY_ASSET = "US30"

# Grilles de paramètres (docs/v3_roadmap.md § H07)
STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "dual_ma",
        "class": DualMovingAverage,
        "params_grid": {
            "fast": [5, 10, 20],
            "slow": [50, 100, 200],
        },
    },
    {
        "name": "keltner",
        "class": KeltnerChannel,
        "params_grid": {
            "period": [10, 20, 50],
            "mult": [1.5, 2.0, 2.5],
        },
    },
    {
        "name": "chandelier",
        "class": ChandelierExit,
        "params_grid": {
            "period": [11, 22, 44],
            "k_atr": [2.0, 3.0, 4.0],
        },
    },
    {
        "name": "parabolic",
        "class": ParabolicSAR,
        "params_grid": {
            "step": [0.01, 0.02, 0.03],
            "af_max": [0.1, 0.2, 0.3],
        },
    },
]

# Donchian de référence pour comparaison
DONCHIAN_PARAMS = {"N": 20, "M": 20}  # best v2

N_TRIALS_CUMUL = 7  # 6 hérités (v1+v2+H06) + H07

OUTPUT_PATH = Path("predictions/h07_strategies_alt.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _split_periods(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict : train ≤ 2022, val = 2023, test ≥ 2024."""
    train = df[df.index <= TRAIN_END]
    val = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test = df[df.index >= TEST_START]
    return train, val, test


def _build_equity_from_trades(
    trades: list[dict[str, Any]],
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit equity curve et trades DataFrame à partir de la sortie backtest."""
    if not trades:
        empty_idx = pd.DatetimeIndex([])
        return pd.Series(dtype=np.float64, index=empty_idx), pd.DataFrame({"pnl": []})

    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])

    equity_vals = np.cumsum(pnls)
    equity = pd.Series(equity_vals, index=exit_times).sort_index()

    trades_df = pd.DataFrame({"pnl": pnls}, index=exit_times)
    return equity, trades_df


def _compute_sharpe_from_equity(equity: pd.Series) -> float:
    """Sharpe annualisé depuis une courbe d'équité (Constitution § Règle 10)."""
    if len(equity) < 2:
        return 0.0
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2 or np.isclose(daily_returns.std(), 0.0):
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


def _compute_daily_returns(
    trades: list[dict[str, Any]],
    date_range: pd.DatetimeIndex,
) -> pd.Series:
    """Convertit les trades en série de retours quotidiens pour corrélation."""
    if not trades:
        return pd.Series(np.zeros(len(date_range)), index=date_range, dtype=np.float64)

    pnl_series = pd.Series(
        [t["pips_net"] for t in trades],
        index=pd.to_datetime([t["exit_time"] for t in trades]),
        dtype=np.float64,
    ).sort_index()

    # Rééchantillonner en daily : somme des PnL par jour
    daily_pnl = pnl_series.resample("D").sum().fillna(0.0)
    daily_pnl = daily_pnl.reindex(date_range, fill_value=0.0)

    # Convertir PnL en pourcentage de capital (10k€)
    capital = 10_000.0
    cumulative_pnl = daily_pnl.cumsum()
    equity_daily = capital + cumulative_pnl * 0.92  # pip_value_eur pour US30
    returns = equity_daily.pct_change().fillna(0.0)
    return returns


def _grid_search_strategy(
    strategy_def: dict[str, Any],
    df_train: pd.DataFrame,
    config: AssetConfig,
) -> dict[str, Any]:
    """Grid search sur train, retourne le meilleur paramétrage.

    Args:
        strategy_def: Dict avec 'name', 'class', 'params_grid'.
        df_train: DataFrame OHLCV train (≤ 2022).
        config: AssetConfig avec coûts et TP/SL.

    Returns:
        Dict avec best_params, best_sharpe_train, all_combinations.
    """
    strategy_cls = strategy_def["class"]
    params_grid: dict[str, list[Any]] = strategy_def["params_grid"]
    param_names = list(params_grid.keys())

    best_sharpe = -np.inf
    best_params: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    # Générer toutes les combinaisons via produit cartésien
    from itertools import product

    param_values = list(params_grid.values())
    for combo in product(*param_values):
        params = dict(zip(param_names, combo, strict=True))

        strat = strategy_cls(**params)
        signals = strat.generate_signals(df_train)

        result = run_deterministic_backtest(
            df=df_train,
            signals=signals,
            tp_pips=config.tp_points,
            sl_pips=config.sl_points,
            window_hours=config.window_hours,
            commission_pips=config.spread_pips,
            slippage_pips=config.slippage_pips,
            pip_size=config.pip_size,
        )

        trades_list: list[dict[str, Any]] = result.get("trades", [])
        equity, _ = _build_equity_from_trades(trades_list)
        sr = _compute_sharpe_from_equity(equity)

        all_results.append({
            **params,
            "sharpe_train": sr,
            "n_trades_train": result.get("total_trades", 0),
        })

        if sr > best_sharpe:
            best_sharpe = sr
            best_params = params

    return {
        "best_params": best_params,
        "best_sharpe_train": best_sharpe,
        "all_combinations": all_results,
    }


def _evaluate_strategy(
    strategy_def: dict[str, Any],
    params: dict[str, Any],
    df_period: pd.DataFrame,
    config: AssetConfig,
    period_label: str,
) -> dict[str, Any]:
    """Backtest d'une stratégie sur une période donnée.

    Returns:
        Dict avec sharpe, wr, n_trades, equity, trades_df, daily_returns.
    """
    strategy_cls = strategy_def["class"]
    strat = strategy_cls(**params)
    signals = strat.generate_signals(df_period)

    result = run_deterministic_backtest(
        df=df_period,
        signals=signals,
        tp_pips=config.tp_points,
        sl_pips=config.sl_points,
        window_hours=config.window_hours,
        commission_pips=config.spread_pips,
        slippage_pips=config.slippage_pips,
        pip_size=config.pip_size,
    )

    trades_list: list[dict[str, Any]] = result.get("trades", [])
    equity, trades_df = _build_equity_from_trades(trades_list)
    daily_returns = _compute_daily_returns(trades_list, df_period.index)

    return {
        "period": period_label,
        "sharpe": _compute_sharpe_from_equity(equity),
        "wr": float(result.get("wr", 0.0)),
        "n_trades": result.get("total_trades", 0),
        "total_pnl_pips": float(result.get("total_pnl_pips", 0.0)),
        "max_drawdown_pips": float(result.get("max_drawdown_pips", 0.0)),
        "equity": equity,
        "trades_df": trades_df,
        "daily_returns": daily_returns,
    }


def _compute_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 60,
) -> float:
    """Corrélation rolling des retours quotidiens entre deux stratégies.

    Retourne la corrélation moyenne sur la période de test.
    """
    common_idx = returns_a.index.intersection(returns_b.index)
    if len(common_idx) < window:
        return float("nan")

    ra = returns_a.loc[common_idx]
    rb = returns_b.loc[common_idx]

    rolling_corr = ra.rolling(window=window).corr(rb)
    return float(rolling_corr.dropna().mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Orchestrateur H07 : grid search 4 stratégies alternatives sur US30 D1."""
    set_global_seeds()
    check_unlocked()

    config = ASSET_CONFIGS.get(PRIMARY_ASSET)
    if config is None:
        logger.error("US30 absent de ASSET_CONFIGS. Abandon.")
        return

    logger.info("=== H07 : Stratégies alternatives sur %s D1 ===", PRIMARY_ASSET)
    logger.info("Coûts US30: spread=%.1f, slippage=%.1f, TP=%.0f, SL=%.0f",
                config.spread_pips, config.slippage_pips,
                config.tp_points, config.sl_points)

    # ── Charger les données ──────────────────────────────────────────────────
    try:
        df = load_asset(PRIMARY_ASSET, "D1")
    except Exception as e:
        logger.error("Chargement %s D1 échoué: %s", PRIMARY_ASSET, e)
        return

    df_train, df_val, df_test = _split_periods(df)
    logger.info("US30: train=%d barres, val=%d, test=%d",
                len(df_train), len(df_val), len(df_test))

    if len(df_train) < 100:
        logger.error("Train trop court (%d barres). Abandon.", len(df_train))
        return

    # ── Donchian baseline (référence pour corrélation) ───────────────────────
    donchian_test = _evaluate_strategy(
        {"name": "donchian", "class": DonchianBreakout},
        DONCHIAN_PARAMS,
        df_test,
        config,
        "test_2024+",
    )
    donchian_daily_returns = donchian_test["daily_returns"]
    logger.info("Donchian baseline test: sharpe=%.4f, wr=%.1f%%, n_trades=%d",
                donchian_test["sharpe"],
                donchian_test["wr"] * 100,
                donchian_test["n_trades"])

    # ── Boucle principale sur les 4 stratégies ───────────────────────────────
    results: dict[str, dict[str, Any]] = {
        "donchian_baseline": {
            "params": DONCHIAN_PARAMS,
            "test": {
                "sharpe": donchian_test["sharpe"],
                "wr": donchian_test["wr"],
                "n_trades": donchian_test["n_trades"],
                "total_pnl_pips": donchian_test["total_pnl_pips"],
                "max_drawdown_pips": donchian_test["max_drawdown_pips"],
            },
        },
    }
    strategies_go: list[str] = []

    for strat_def in STRATEGIES:
        strat_name: str = strat_def["name"]
        logger.info("── %s ──", strat_name)

        # Grid search sur train
        grid_result = _grid_search_strategy(strat_def, df_train, config)
        best_params = grid_result["best_params"]
        best_sr_train = grid_result["best_sharpe_train"]

        logger.info("%s: best_params=%s, sharpe_train=%.4f",
                    strat_name, best_params, best_sr_train)

        # Évaluer sur val 2023
        val_result: dict[str, Any] = {}
        if len(df_val) > 0:
            val_result = _evaluate_strategy(
                strat_def, best_params, df_val, config, "val_2023",
            )

        # Évaluer sur test ≥ 2024
        test_result: dict[str, Any] = {}
        edge_report: EdgeReport | None = None
        correlation: float = float("nan")

        if len(df_test) > 0:
            test_result = _evaluate_strategy(
                strat_def, best_params, df_test, config, "test_2024+",
            )

            # read_oos — obligatoire avant analyse du test set (Règle 9/14)
            sr_test = float(test_result.get("sharpe", 0.0))
            n_trades_test = int(test_result.get("n_trades", 0))
            read_oos(
                prompt="08",
                hypothesis="H07",
                sharpe=sr_test,
                n_trades=n_trades_test,
            )

            equity_test: pd.Series = test_result["equity"]
            trades_df_test: pd.DataFrame = test_result["trades_df"]

            edge_report = validate_edge(equity_test, trades_df_test, N_TRIALS_CUMUL)
            logger.info(
                "%s test: sharpe=%.4f, wr=%.1f%%, n_trades=%d, go=%s",
                strat_name,
                sr_test,
                float(test_result.get("wr", 0.0)) * 100,
                n_trades_test,
                edge_report.go,
            )

            # Corrélation rolling 60j avec Donchian
            strat_returns = test_result["daily_returns"]
            correlation = _compute_correlation(
                strat_returns, donchian_daily_returns, window=60,
            )
            logger.info("%s corrélation 60j vs Donchian: %.4f", strat_name, correlation)

            if edge_report.go:
                strategies_go.append(strat_name)

        # ── Agréger ──────────────────────────────────────────────────────
        results[strat_name] = {
            "status": "ok",
            "grid_search": {
                "best_params": best_params,
                "sharpe_train": best_sr_train,
                "n_combinations": len(grid_result["all_combinations"]),
                "all_combinations": grid_result["all_combinations"],
            },
            "val": {
                "sharpe": float(val_result.get("sharpe", 0.0)),
                "wr": float(val_result.get("wr", 0.0)),
                "n_trades": int(val_result.get("n_trades", 0)),
                "total_pnl_pips": float(val_result.get("total_pnl_pips", 0.0)),
                "max_drawdown_pips": float(val_result.get("max_drawdown_pips", 0.0)),
            } if val_result else None,
            "test": {
                "sharpe": float(test_result.get("sharpe", 0.0)),
                "wr": float(test_result.get("wr", 0.0)),
                "n_trades": int(test_result.get("n_trades", 0)),
                "total_pnl_pips": float(test_result.get("total_pnl_pips", 0.0)),
                "max_drawdown_pips": float(test_result.get("max_drawdown_pips", 0.0)),
            } if test_result else None,
            "correlation_vs_donchian_60d": correlation,
            "edge_report": {
                "go": edge_report.go,
                "reasons": edge_report.reasons,
                "metrics": edge_report.metrics,
            } if edge_report else None,
        }

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    n_tested = len(STRATEGIES)
    n_go = len(strategies_go)

    summary = {
        "hypothesis": "H07",
        "prompt": "08",
        "title": "Stratégies trend-following alternatives",
        "timestamp": datetime.now(UTC).isoformat(),
        "asset": PRIMARY_ASSET,
        "tf": "D1",
        "split": {
            "train": f"≤ {TRAIN_END}",
            "val": f"{VAL_START} → {VAL_END}",
            "test": f"≥ {TEST_START}",
        },
        "n_trials_cumul": N_TRIALS_CUMUL,
        "n_strategies_tested": n_tested,
        "n_go": n_go,
        "strategies_go": strategies_go,
        "donchian_params": DONCHIAN_PARAMS,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Résultats sauvegardés: %s", OUTPUT_PATH)
    logger.info("GO: %d stratégie(s) sur %d testées: %s",
                n_go, n_tested, strategies_go if strategies_go else "aucune")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
