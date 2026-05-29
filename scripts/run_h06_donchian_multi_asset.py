"""Run H06 — Donchian Breakout grid search multi-actif (Prompt 07).

Grid search Donchian N ∈ {20, 50, 100}, M ∈ {10, 20, 50} (9 combinaisons)
sur chaque actif D1 disponible. Sélection du meilleur Sharpe train ≤ 2022,
évaluation val=2023, test ≥ 2024 avec coûts réalistes, puis validate_edge().

Produit predictions/h06_donchian_multi_asset.json.
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
from app.data.registry import discover_assets
from app.strategies.donchian import DonchianBreakout
from app.testing.snooping_guard import check_unlocked, read_oos

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes figées (Constitution §3)
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"

N_VALUES = (20, 50, 100)
M_VALUES = (10, 20, 50)
N_TRIALS_CUMUL = 6  # 5 hérités v2 + H06

OUTPUT_PATH = Path("predictions/h06_donchian_multi_asset.json")


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
    """Construit equity curve et trades DataFrame à partir de la sortie backtest.

    Returns:
        (equity: pd.Series indexée datetime, trades_df: DataFrame avec colonne 'pnl')
    """
    if not trades:
        empty_idx = pd.DatetimeIndex([])
        return pd.Series(dtype=np.float64, index=empty_idx), pd.DataFrame({"pnl": []})

    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])

    equity_vals = np.cumsum(pnls)
    equity = pd.Series(equity_vals, index=exit_times).sort_index()

    trades_df = pd.DataFrame({"pnl": pnls}, index=exit_times)
    return equity, trades_df


def _compute_sharpe_train(backtest_result: dict[str, Any]) -> float:
    """Extrait le Sharpe annualisé depuis le résultat du backtest."""
    return float(backtest_result.get("sharpe", 0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Grid search par actif
# ═══════════════════════════════════════════════════════════════════════════════


def _grid_search_asset(
    df_train: pd.DataFrame,
    config: AssetConfig,
) -> dict[str, Any]:
    """Grid search Donchian(N, M) sur train, retourne le meilleur paramétrage.

    Args:
        df_train: DataFrame OHLCV train (≤ 2022).
        config: AssetConfig avec coûts et TP/SL.

    Returns:
        Dict avec best_N, best_M, best_sharpe_train, all_combinations.
    """
    best_sharpe = -np.inf
    best_n = 0  # noqa: N806 — Donchian convention
    best_m = 0  # noqa: N806 — Donchian convention
    all_results: list[dict[str, Any]] = []

    for n_val in N_VALUES:  # noqa: N806
        for m_val in M_VALUES:  # noqa: N806
            strat = DonchianBreakout(N=n_val, M=m_val)
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

            sr = _compute_sharpe_train(result)
            all_results.append({
                "N": n_val,
                "M": m_val,
                "sharpe_train": sr,
                "n_trades_train": result.get("total_trades", 0),
            })

            if sr > best_sharpe:
                best_sharpe = sr
                best_n = n_val
                best_m = m_val

    return {
        "best_N": best_n,
        "best_M": best_m,
        "best_sharpe_train": best_sharpe,
        "all_combinations": all_results,
    }


def _evaluate_on_period(
    df_period: pd.DataFrame,
    param_n: int,
    param_m: int,
    config: AssetConfig,
    period_label: str,
) -> dict[str, Any]:
    """Backtest Donchian(N, M) sur une période donnée.

    Returns:
        Dict avec sharpe, wr, n_trades, equity, trades_df.
    """
    strat = DonchianBreakout(N=param_n, M=param_m)
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

    return {
        "period": period_label,
        "sharpe": _compute_sharpe_train(result),
        "wr": float(result.get("wr", 0.0)),
        "n_trades": result.get("total_trades", 0),
        "total_pnl_pips": float(result.get("total_pnl_pips", 0.0)),
        "max_drawdown_pips": float(result.get("max_drawdown_pips", 0.0)),
        "equity": equity,
        "trades_df": trades_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Orchestrateur H06 : grid search multi-actif sur Donchian Breakout D1."""
    set_global_seeds()
    check_unlocked()

    # ── Découvrir les actifs D1 disponibles ────────────────────────────────
    available = discover_assets()
    d1_assets = sorted(
        asset for asset, tfs in available.items() if "D1" in tfs
    )
    logger.info("Actifs D1 disponibles: %s", d1_assets)

    # ── Filtrer sur la liste H06 ───────────────────────────────────────────
    h06_targets = set(ASSET_CONFIGS.keys())
    candidates = [a for a in d1_assets if a in h06_targets]
    missing = h06_targets - set(d1_assets)
    if missing:
        logger.warning("Actifs H06 sans données D1: %s", sorted(missing))

    if not candidates:
        logger.error("Aucun actif H06 avec données D1 trouvé. Abandon.")
        return

    logger.info("Actifs H06 à tester: %s", candidates)

    # ── Boucle principale par actif ────────────────────────────────────────
    results: dict[str, dict[str, Any]] = {}
    n_go = 0

    for asset in candidates:
        config = ASSET_CONFIGS[asset]
        logger.info("── %s ──", asset)

        try:
            df = load_asset(asset, "D1")
        except Exception as e:
            logger.error("Chargement %s D1 échoué: %s", asset, e)
            results[asset] = {"status": "error", "reason": str(e)}
            continue

        df_train, df_val, df_test = _split_periods(df)
        logger.info(
            "%s: train=%d barres, val=%d, test=%d",
            asset, len(df_train), len(df_val), len(df_test),
        )

        if len(df_train) < 100:
            logger.warning("%s: train trop court (%d barres), skip.", asset, len(df_train))
            results[asset] = {"status": "skip", "reason": "train < 100 barres"}
            continue

        # ── Grid search sur train ──────────────────────────────────────
        grid_result = _grid_search_asset(df_train, config)
        best_n = grid_result["best_N"]
        best_m = grid_result["best_M"]
        best_sr_train = grid_result["best_sharpe_train"]

        logger.info(
            "%s: best=(N=%d, M=%d), sharpe_train=%.4f",
            asset, best_n, best_m, best_sr_train,
        )

        # ── Évaluer sur val 2023 ───────────────────────────────────────
        val_result: dict[str, Any] = {}
        if len(df_val) > 0:
            val_result = _evaluate_on_period(df_val, best_n, best_m, config, "val_2023")

        # ── Évaluer sur test ≥ 2024 ────────────────────────────────────
        test_result: dict[str, Any] = {}
        edge_report: EdgeReport | None = None
        if len(df_test) > 0:
            test_result = _evaluate_on_period(df_test, best_n, best_m, config, "test_2024+")

            # ── validate_edge ──────────────────────────────────────────
            equity_test: pd.Series = test_result["equity"]
            trades_df_test: pd.DataFrame = test_result["trades_df"]

            # read_oos — obligatoire avant analyse du test set (Règle 9/14)
            sr_test = float(test_result.get("sharpe", 0.0))
            n_trades_test = int(test_result.get("n_trades", 0))
            read_oos(
                prompt="07",
                hypothesis="H06",
                sharpe=sr_test,
                n_trades=n_trades_test,
            )

            edge_report = validate_edge(equity_test, trades_df_test, N_TRIALS_CUMUL)
            logger.info(
                "%s test: sharpe=%.4f, wr=%.1f%%, n_trades=%d, go=%s",
                asset,
                sr_test,
                float(test_result.get("wr", 0.0)) * 100,
                n_trades_test,
                edge_report.go,
            )
            if edge_report.go:
                n_go += 1

        # ── Agréger les résultats ──────────────────────────────────────
        results[asset] = {
            "status": "ok",
            "config": {
                "spread_pips": config.spread_pips,
                "slippage_pips": config.slippage_pips,
                "tp_points": config.tp_points,
                "sl_points": config.sl_points,
                "window_hours": config.window_hours,
                "pip_size": config.pip_size,
            },
            "grid_search": {
                "best_N": best_n,
                "best_M": best_m,
                "sharpe_train": best_sr_train,
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
            "edge_report": {
                "go": edge_report.go,
                "reasons": edge_report.reasons,
                "metrics": edge_report.metrics,
            } if edge_report else None,
        }

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    n_tested = len([r for r in results.values() if r.get("status") == "ok"])
    n_skipped = len([r for r in results.values() if r.get("status") == "skip"])
    n_error = len([r for r in results.values() if r.get("status") == "error"])

    summary = {
        "hypothesis": "H06",
        "prompt": "07",
        "title": "Donchian Breakout multi-actif grid search",
        "timestamp": datetime.now(UTC).isoformat(),
        "split": {
            "train": f"≤ {TRAIN_END}",
            "val": f"{VAL_START} → {VAL_END}",
            "test": f"≥ {TEST_START}",
        },
        "grid": {"N_values": list(N_VALUES), "M_values": list(M_VALUES), "combinations": 9},
        "n_trials_cumul": N_TRIALS_CUMUL,
        "n_assets_tested": n_tested,
        "n_assets_skipped": n_skipped,
        "n_assets_error": n_error,
        "n_go": n_go,
        "missing_data": sorted(missing),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Résultats sauvegardés: %s", OUTPUT_PATH)
    logger.info("GO: %d actif(s) sur %d testés.", n_go, n_tested)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
