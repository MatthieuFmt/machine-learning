"""Pivot v4 A4 — Replay H06/H07 sur train+val avec simulateur corrigé.

⚠️ AUDIT INFORMATIF : ne touche jamais au test set ≥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ajouter la racine du projet au PYTHONPATH pour les imports `app.*`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from app.backtest.deterministic import run_deterministic_backtest
from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.data.registry import discover_assets
from app.strategies.chandelier import ChandelierExit
from app.strategies.donchian import DonchianBreakout
from app.strategies.dual_ma import DualMovingAverage
from app.strategies.keltner import KeltnerChannel
from app.strategies.parabolic import ParabolicSAR

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"

# ⚠️ INVARIANT CRITIQUE : aucune donnée > 2023-12-31 ne doit entrer dans ce script.
CUTOFF_DATE = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")


# ── Helpers ────────────────────────────────────────────────────────────────

def _backtest(
    strategy: Any,
    df: pd.DataFrame,
    cfg: AssetConfig,
) -> dict[str, Any]:
    """Wrapper qui génère les signaux puis exécute le backtest stateful."""
    signals = strategy.generate_signals(df)
    return run_deterministic_backtest(
        df,
        signals,
        tp_pips=cfg.tp_points,
        sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=cfg.slippage_pips,
        pip_size=cfg.pip_size,
    )


def _filter_to_train_val(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre strict aux données ≤ 2023-12-31. Lève si du test set est présent."""
    filtered = df.loc[df.index <= CUTOFF_DATE]
    if filtered.empty:
        raise ValueError("Aucune donnée train/val disponible")
    if filtered.index.max() > CUTOFF_DATE:
        raise AssertionError("Test set leak détecté ! Vérifier le filtrage.")
    return filtered


def _extract_metrics(backtest_result: dict[str, Any]) -> dict[str, Any]:
    """Extrait les métriques du résultat brut du backtest."""
    dd = backtest_result.get("max_drawdown_pips", 0.0)

    return {
        "sharpe": float(backtest_result.get("sharpe", 0.0)),
        "wr": float(backtest_result.get("wr", 0.0)),
        "trades": int(backtest_result.get("total_trades", 0)),
        "total_pnl_pips": float(backtest_result.get("total_pnl_pips", 0.0)),
        "max_dd_pips": float(dd),
        "profit_factor": float(backtest_result.get("profit_factor", 0.0)),
    }


# ── H06 Replay ─────────────────────────────────────────────────────────────

def replay_donchian_all_assets() -> dict[str, Any]:
    """Rejoue H06 sur train ≤ 2022 et val 2023, agnostique du test set."""
    set_global_seeds()
    discovered = discover_assets()

    # Intersection : actifs avec D1 dans discovered ET dans ASSET_CONFIGS
    candidates = [
        a for a, tfs in discovered.items()
        if "D1" in tfs and a in ASSET_CONFIGS
    ]
    results: dict[str, Any] = {}

    for asset in sorted(candidates):
        try:
            df = load_asset(asset, "D1")
        except Exception as exc:
            results[asset] = {"error": f"load_asset: {exc}"}
            continue

        try:
            df = _filter_to_train_val(df)
        except (ValueError, AssertionError) as exc:
            results[asset] = {"error": f"_filter: {exc}"}
            continue

        cfg = ASSET_CONFIGS[asset]

        df_train = df.loc[:TRAIN_END]
        df_val = df.loc[VAL_START:VAL_END]
        if df_train.empty or df_val.empty:
            results[asset] = {"error": "train ou val vide"}
            continue

        # Grid search Donchian sur train
        best_sharpe_train, best_params = -1e9, None
        for N in [20, 50, 100]:
            for M in [10, 20, 50]:
                strat = DonchianBreakout(N=N, M=M)
                bt = _backtest(strat, df_train, cfg)
                if bt["sharpe"] > best_sharpe_train:
                    best_sharpe_train = bt["sharpe"]
                    best_params = {"N": N, "M": M}

        if best_params is None:
            results[asset] = {"error": "grid search vide"}
            continue

        # Eval val 2023
        strat = DonchianBreakout(**best_params)
        bt_val = _backtest(strat, df_val, cfg)
        m_val = _extract_metrics(bt_val)

        # DD en % approximatif (pips vs capital 10k €)
        max_dd_pct = _dd_pips_to_pct(
            m_val["max_dd_pips"], cfg.pip_value_eur, 10_000.0
        )

        results[asset] = {
            "best_params": best_params,
            "sharpe_train_v4": float(best_sharpe_train),
            "sharpe_val_v4": float(m_val["sharpe"]),
            "wr_val": float(m_val["wr"]),
            "max_dd_pips_val": float(m_val["max_dd_pips"]),
            "max_dd_pct_val": max_dd_pct,
            "trades_val": int(m_val["trades"]),
            "pnl_val_pips": float(m_val["total_pnl_pips"]),
            "total_cost_pips": cfg.total_cost_pips,
        }
    return results


def _dd_pips_to_pct(dd_pips: float, pip_value_eur: float, capital_eur: float) -> float:
    """Convertit un drawdown en pips → pourcentage du capital."""
    if capital_eur <= 0:
        return 0.0
    return float((dd_pips * pip_value_eur) / capital_eur * 100.0)


# ── H07 Replay ─────────────────────────────────────────────────────────────

def replay_h07_us30() -> dict[str, Any]:
    """Rejoue les 4 strats alt sur US30 D1 train+val."""
    set_global_seeds()

    if "US30" not in ASSET_CONFIGS:
        return {"error": "US30 absent de ASSET_CONFIGS"}

    try:
        df = load_asset("US30", "D1")
    except Exception as exc:
        return {"error": f"load_asset US30: {exc}"}

    try:
        df = _filter_to_train_val(df)
    except (ValueError, AssertionError) as exc:
        return {"error": f"_filter US30: {exc}"}

    cfg = ASSET_CONFIGS["US30"]
    df_train = df.loc[:TRAIN_END]
    df_val = df.loc[VAL_START:VAL_END]

    if df_train.empty or df_val.empty:
        return {"error": "US30 train ou val vide"}

    # Donchian baseline avec les nouveaux coûts
    best_donchian_sharpe, best_donchian_params = -1e9, None
    for N in [20, 50, 100]:
        for M in [10, 20, 50]:
            strat = DonchianBreakout(N=N, M=M)
            bt = _backtest(strat, df_train, cfg)
            if bt["sharpe"] > best_donchian_sharpe:
                best_donchian_sharpe = bt["sharpe"]
                best_donchian_params = {"N": N, "M": M}

    donchian_val: dict[str, Any] = {"error": "no donchian baseline"}
    if best_donchian_params is not None:
        strat = DonchianBreakout(**best_donchian_params)
        bt_val = _backtest(strat, df_val, cfg)
        m = _extract_metrics(bt_val)
        donchian_val = {
            "best_params": best_donchian_params,
            "sharpe_train_v4": float(best_donchian_sharpe),
            "sharpe_val_v4": float(m["sharpe"]),
            "wr_val": float(m["wr"]),
            "trades_val": int(m["trades"]),
            "pnl_val_pips": float(m["total_pnl_pips"]),
            "total_cost_pips": cfg.total_cost_pips,
        }

    # Stratégies alternatives (H07)
    strats: dict[str, list[tuple[Any, dict]]] = {
        "dual_ma": [
            (DualMovingAverage(fast=f, slow=s), {"fast": f, "slow": s})
            for f in (5, 10, 20)
            for s in (50, 100, 200)
        ],
        "keltner": [
            (KeltnerChannel(period=p, mult=m), {"period": p, "mult": m})
            for p in (10, 20, 50)
            for m in (1.5, 2.0, 2.5)
        ],
        "chandelier": [
            (ChandelierExit(period=p, k_atr=k), {"period": p, "k_atr": k})
            for p in (11, 22, 44)
            for k in (2.0, 3.0, 4.0)
        ],
        "parabolic": [
            (ParabolicSAR(step=s, af_max=a), {"step": s, "af_max": a})
            for s in (0.01, 0.02, 0.03)
            for a in (0.1, 0.2, 0.3)
        ],
    }

    alt_results: dict[str, Any] = {"donchian_baseline": donchian_val}

    for name, candidate_list in strats.items():
        best_sharpe, best_params, best_strat = -1e9, None, None
        for strat_inst, params in candidate_list:
            bt = _backtest(strat_inst, df_train, cfg)
            if bt["sharpe"] > best_sharpe:
                best_sharpe = bt["sharpe"]
                best_params = dict(params)
                best_strat = strat_inst

        if best_strat is None or best_params is None:
            alt_results[name] = {"error": "no candidate"}
            continue

        bt_val = _backtest(best_strat, df_val, cfg)
        m_val = _extract_metrics(bt_val)

        alt_results[name] = {
            "best_params": best_params,
            "sharpe_train_v4": float(best_sharpe),
            "sharpe_val_v4": float(m_val["sharpe"]),
            "wr_val": float(m_val["wr"]),
            "trades_val": int(m_val["trades"]),
            "pnl_val_pips": float(m_val["total_pnl_pips"]),
            "total_cost_pips": cfg.total_cost_pips,
        }

    return alt_results


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    out: dict[str, Any] = {
        "h06_replay": replay_donchian_all_assets(),
        "h07_replay": replay_h07_us30(),
    }
    out_path = Path("predictions/pivot_a4_replay.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Replay terminé. Résultats dans {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
