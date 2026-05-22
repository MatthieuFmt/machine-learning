"""Pivot v4 C4 â€” Hyperparam tuning multi-actifs (nested CPCV train uniquement).

âš ï¸ Aucune lecture du test set â‰¥ 2024.
0 n_trial consommÃ©.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from app.config.features_selected import FEATURES_SELECTED
from app.config.instruments import ASSET_CONFIGS
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.superset import build_superset
from app.models.nested_tuning import nested_cpcv_tuning
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42
SHORTLIST_C5_SHARPE = 0.5
MAX_INNER_OUTER_GAP = 1.0

# â”€â”€ Grilles d'hyperparams (C4 â€” identiques A8 pour comparabilitÃ©) â”€â”€â”€â”€â”€â”€

RF_GRID: dict[str, list] = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "min_samples_leaf": [5, 10, 20],
}

HGBM_GRID: dict[str, list] = {
    "max_depth": [3, 6, None],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}

THRESHOLD_CANDIDATES: list[float] = [0.50, 0.55, 0.60]

# â”€â”€ Model factories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def rf_factory(params: dict, seed: int) -> RandomForestClassifier:
    """Construit un RandomForest avec les hyperparams donnÃ©s."""
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 4)),
        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def hgbm_factory(params: dict, seed: int) -> HistGradientBoostingClassifier:
    """Construit un HistGradientBoosting avec les hyperparams donnÃ©s."""
    return HistGradientBoostingClassifier(
        max_iter=int(params.get("max_iter", 200)),
        max_depth=params.get("max_depth", 5),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)) if params.get("max_leaf_nodes") is not None else 31,
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
        early_stopping=False,
    )


FACTORIES: dict[str, Any] = {
    "rf": rf_factory,
    "hgbm": hgbm_factory,
}

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _run_backtest(
    df_train: pd.DataFrame,
    asset: str,
    donchian: dict,
) -> pd.DataFrame:
    """ExÃ©cute le backtest Donchian dÃ©terministe et retourne le DataFrame trades."""
    from app.backtest.deterministic import run_deterministic_backtest

    cfg = ASSET_CONFIGS[asset]
    strat = DonchianBreakout(N=donchian["N"], M=donchian["M"])
    signals = strat.generate_signals(df_train)
    result = run_deterministic_backtest(
        df=df_train,
        signals=signals,
        tp_pips=cfg.tp_points,
        sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=cfg.slippage_pips,
        pip_size=cfg.pip_size,
        asset_config=cfg,
    )
    trades_list: list[dict] = result.get("trades", [])
    if not trades_list:
        return pd.DataFrame()
    trades = pd.DataFrame(trades_list)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.set_index("entry_time").sort_index()
    return trades


def _build_X_y_pnl(
    asset: str, tf: str, donchian: dict
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reconstruit X, y, pnl filtrÃ© par FEATURES_SELECTED."""
    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF]
    if df_train.empty:
        raise ValueError(f"No train data for {asset}/{tf}")

    trades = _run_backtest(df_train, asset, donchian)
    if trades.empty:
        raise ValueError(f"No trades for {asset}/{tf}")

    feat_train = build_superset(df_train, asset=asset)
    selected_features = list(FEATURES_SELECTED[(asset, tf)])
    common_idx = feat_train.index.intersection(trades.index)
    X = feat_train.loc[common_idx, selected_features].dropna(axis=0, how="any")
    y = (trades["pips_net"] > 0).astype(int).loc[X.index]
    pnl = trades["pips_net"].loc[X.index]

    return X, y, pnl


def _compute_inner_outer_gap(outer_fold_results: list[dict]) -> float:
    """Ã‰cart moyen entre le Sharpe inner (biaisÃ©) et outer (honnÃªte)."""
    gaps = []
    for r in outer_fold_results:
        inner = r.get("inner_best_score", np.nan)
        outer = r.get("outer_sharpe", np.nan)
        if not np.isnan(inner) and not np.isnan(outer):
            gaps.append(inner - outer)
    return float(np.mean(gaps)) if gaps else np.nan


# â”€â”€ Core â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _process_couple(asset: str, tf: str, donchian: dict, model_name: str) -> dict:
    """Tune un couple via nested CPCV, ou retourne defaults pour stacking."""
    set_global_seeds(SEED)

    try:
        X, y, pnl = _build_X_y_pnl(asset, tf, donchian)
    except (ValueError, KeyError) as exc:
        return {
            "asset": asset, "tf": tf,
            "status": "error",
            "error": str(exc),
        }

    if len(y) < 100:
        return {
            "asset": asset, "tf": tf,
            "status": "insufficient_trades",
            "n_trades": int(len(y)),
        }

    if model_name == "stacking":
        return {
            "asset": asset, "tf": tf,
            "status": "stacking_excluded_from_tuning",
            "model": "stacking",
            "params": {},
            "threshold": 0.50,
            "expected_sharpe_outer": 0.0,
            "outer_sharpe_std": 0.0,
            "sharpes_outer_per_fold": [],
            "expected_wr": 0.0,
            "expected_n_kept": 0.0,
            "inner_outer_gap": 0.0,
            "pass_c5": False,
            "n_trades_train": int(len(y)),
        }

    if model_name not in FACTORIES:
        return {
            "asset": asset, "tf": tf,
            "status": "error",
            "error": f"Unknown model: {model_name}",
        }

    factory = FACTORIES[model_name]
    grid = RF_GRID if model_name == "rf" else HGBM_GRID

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print(
        f"  Tuning {asset}/{tf} ({model_name}) on {len(y)} samples, "
        f"{len(grid)} axes ({n_combos} combos), "
        f"{len(THRESHOLD_CANDIDATES)} thresholds..."
    )

    result = nested_cpcv_tuning(
        model_factory=factory,
        param_grid=grid,
        threshold_grid=THRESHOLD_CANDIDATES,
        X=X,
        y=y,
        pnl=pnl,
        outer_k=5,
        inner_k=3,
        embargo_pct=0.01,
        seed=SEED,
    )

    inner_outer_gap = _compute_inner_outer_gap(result.outer_fold_results)
    sharpes_outer_per_fold = [r["outer_sharpe"] for r in result.outer_fold_results]

    pass_c5 = (
        result.sharpe_outer_mean >= SHORTLIST_C5_SHARPE
        and not np.isnan(inner_outer_gap)
        and inner_outer_gap < MAX_INNER_OUTER_GAP
    )

    return {
        "asset": asset,
        "tf": tf,
        "status": "ok",
        "model": model_name,
        "params": result.best_params,
        "threshold": result.best_threshold,
        "expected_sharpe_outer": float(result.sharpe_outer_mean),
        "outer_sharpe_std": float(result.sharpe_outer_std),
        "sharpes_outer_per_fold": [float(s) for s in sharpes_outer_per_fold],
        "expected_wr": float(result.wr_outer_mean),
        "expected_n_kept": float(result.n_kept_outer_mean),
        "inner_outer_gap": float(inner_outer_gap),
        "pass_c5": pass_c5,
        "n_trades_train": int(len(y)),
        "n_combos_evaluated": result.n_combos_evaluated,
        "outer_folds": result.outer_fold_results,
    }


# â”€â”€ Update hyperparams_tuned.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _update_hyperparams_tuned(path: Path, results: list[dict]) -> None:
    """Merge les nouveaux rÃ©sultats avec les 3 entrÃ©es A8 existantes."""
    from app.config.hyperparams_tuned import HYPERPARAMS_TUNED  # noqa: N811

    new_entries: dict[tuple[str, str], dict] = {}
    for r in results:
        if r["status"] in ("ok", "stacking_excluded_from_tuning"):
            new_entries[(r["asset"], r["tf"])] = {
                "model": r["model"],
                "params": r["params"],
                "threshold": r["threshold"],
                "expected_sharpe_outer": r["expected_sharpe_outer"],
                "expected_wr": r["expected_wr"],
            }

    # Merge: existing A8 entries first, then new C4 entries
    merged: dict[tuple[str, str], dict] = {}
    # A8 originals always come first
    for key in HYPERPARAMS_TUNED:
        merged[key] = HYPERPARAMS_TUNED[key]
    for key in new_entries:
        merged[key] = new_entries[key]

    lines = [
        '"""FROZEN aprÃ¨s pivot v4 A8 (3 entrÃ©es) + C4 (extension multi-actifs).',
        "",
        "NE PAS MODIFIER MANUELLEMENT. Seules les phases A8 / C4 peuvent y ajouter.",
        '"""',
        "from __future__ import annotations",
        "",
        "HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {",
    ]
    for (asset, tf), entry in merged.items():
        lines.append(f"    ({asset!r}, {tf!r}): {{")
        lines.append(f"        'model': {entry['model']!r},")
        lines.append(f"        'params': {entry['params']!r},")
        lines.append(f"        'threshold': {entry['threshold']!r},")
        lines.append(f"        'expected_sharpe_outer': {entry['expected_sharpe_outer']!r},")
        lines.append(f"        'expected_wr': {entry['expected_wr']!r},")
        lines.append("    },")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def main() -> int:
    sel_path = _PROJECT_ROOT / "predictions" / "c3_model_selection_multi_assets.json"
    selections = json.loads(sel_path.read_text(encoding="utf-8"))

    # RÃ©cupÃ©rer les Donchian (N, M) depuis C2
    rank_path = _PROJECT_ROOT / "predictions" / "c2_ranking_multi_assets.json"
    rankings = json.loads(rank_path.read_text(encoding="utf-8"))
    donchian_by_couple = {
        (r["asset"], r["tf"]): r["donchian"]
        for r in rankings
        if r["status"] == "ok"
    }

    shortlist = [r for r in selections if r["status"] == "ok" and r.get("pass_c4_threshold", False)]
    print(f"{len(shortlist)} couples en shortlist C4 (Sharpe CPCV â‰¥ 0.5).")

    results: list[dict] = []
    for r in shortlist:
        asset, tf = r["asset"], r["tf"]
        model_name = r["selected_model"]
        donchian = donchian_by_couple[(asset, tf)]
        print(f"\nâ†’ tuning {asset}/{tf} ({model_name}) ...")
        res = _process_couple(asset, tf, donchian, model_name)
        results.append(res)
        if res["status"] == "ok":
            print(
                f"  âœ“ params={res['params']}, threshold={res['threshold']}, "
                f"Sharpe outer={res['expected_sharpe_outer']:.2f}, "
                f"gap={res.get('inner_outer_gap', np.nan):.2f}, "
                f"pass_c5={res['pass_c5']}"
            )
        elif res["status"] == "stacking_excluded_from_tuning":
            print("  âš  stacking (defaults conservÃ©s)")
        else:
            print(f"  âœ— {res['status']}: {res.get('error', '')}")

    # Sauvegarde JSON
    out_path = _PROJECT_ROOT / "predictions" / "c4_hyperparam_tuning_multi_assets.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Mise Ã  jour hyperparams_tuned.py
    cfg_path = _PROJECT_ROOT / "app" / "config" / "hyperparams_tuned.py"
    _update_hyperparams_tuned(cfg_path, results)

    # RÃ©sumÃ©
    ok_results = [r for r in results if r["status"] in ("ok", "stacking_excluded_from_tuning")]
    final_shortlist = [r for r in results if r["status"] == "ok" and r.get("pass_c5", False)]
    print()
    print(f"Couples tunÃ©s : {len(ok_results)}")
    print(f"Shortlist finale C5 (Sharpe outer â‰¥ 0.5, gap < 1.0) : {len(final_shortlist)}")
    for r in final_shortlist:
        print(f"  {r['asset']}/{r['tf']} : {r['model']} Sharpe={r['expected_sharpe_outer']:.2f} gap={r['inner_outer_gap']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
