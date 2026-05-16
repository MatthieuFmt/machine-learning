"""Pivot v4 A8 — Tuning hyperparams + seuil via nested CPCV.

⚠️ Aucune lecture du test set ≥ 2024.
Coût estimé : ~30-60 min sur CPU 8-core.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from app.config.features_selected import FEATURES_SELECTED
from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.config.model_selected import MODEL_SELECTED
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.superset import build_superset
from app.models.nested_tuning import nested_cpcv_tuning
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42

# ── AssetConfig EURUSD (non présent dans ASSET_CONFIGS) ────────────────────
_EURUSD_CFG = AssetConfig(
    spread_pips=0.5,
    slippage_pips=1.0,
    commission_pips=0.0,
    pip_size=0.0001,
    pip_value_eur=1.0,
    tp_points=80,
    sl_points=40,
    window_hours=96,
    min_lot=0.01,
    max_lot=50.0,
)

# ── Mapping (asset, tf) → (AssetConfig, strat_cls, strat_kwargs) ───────────
STRAT_MAP: dict[tuple[str, str], tuple[AssetConfig, type, dict]] = {
    ("US30", "D1"): (ASSET_CONFIGS["US30"], DonchianBreakout, {"N": 20, "M": 20}),
    ("EURUSD", "H4"): (_EURUSD_CFG, DonchianBreakout, {"N": 20, "M": 20}),
    ("XAUUSD", "D1"): (ASSET_CONFIGS["XAUUSD"], DonchianBreakout, {"N": 100, "M": 20}),
}

# ── Model factories ────────────────────────────────────────────────────────


def rf_factory(params: dict, seed: int) -> RandomForestClassifier:
    """Construit un RandomForest avec les hyperparams donnés."""
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 4)),
        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def hgbm_factory(params: dict, seed: int) -> HistGradientBoostingClassifier:
    """Construit un HistGradientBoosting avec les hyperparams donnés."""
    return HistGradientBoostingClassifier(
        max_iter=int(params.get("max_iter", 200)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
        early_stopping=False,
    )


FACTORIES: dict[str, Any] = {
    "rf": rf_factory,
    "hgbm": hgbm_factory,
}

# ── Grids hyperparams (3 valeurs max par axe, 27 combos max) ───────────────

PARAM_GRIDS: dict[str, dict] = {
    "rf": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 4, 6],
        "min_samples_leaf": [5, 10, 20],
    },
    "hgbm": {
        "max_iter": [100, 200, 400],
        "max_depth": [4, 5, 7],
        "learning_rate": [0.02, 0.05, 0.10],
    },
    # stacking : pas tuné (trop lent)
}

THRESHOLD_GRID: list[float] = [0.50, 0.55, 0.60]


def _backtest_wrapper(
    df_train: pd.DataFrame,
    strat: DonchianBreakout,
    cfg: AssetConfig,
) -> pd.DataFrame:
    """Exécute le backtest déterministe et retourne le DataFrame trades."""
    from app.backtest.deterministic import run_deterministic_backtest

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
    )
    trades_list: list[dict] = result.get("trades", [])
    if not trades_list:
        return pd.DataFrame()

    trades = pd.DataFrame(trades_list)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.set_index("entry_time").sort_index()
    return trades


def tune_one_asset(asset: str, tf: str) -> dict:
    """Tune hyperparams + seuil via nested CPCV pour un actif."""
    key = (asset, tf)
    model_name = MODEL_SELECTED.get(key)
    if not model_name:
        return {"error": f"No model selected for {asset} {tf}"}

    if model_name == "stacking":
        return {
            "asset": asset,
            "tf": tf,
            "model": "stacking",
            "best_params": {},
            "best_threshold": 0.50,
            "note": "Stacking pas tuné (trop lent). Defaults A7 conservés.",
        }

    if model_name not in FACTORIES:
        return {"error": f"Unknown model: {model_name}"}

    factory = FACTORIES[model_name]
    grid = PARAM_GRIDS[model_name]

    if key not in STRAT_MAP:
        return {"error": f"No strategy mapped for {asset} {tf}"}

    cfg, strat_cls, strat_kwargs = STRAT_MAP[key]

    # ── Charger train ≤ 2022 ──────────────────────────────────────────────
    df = load_asset(asset, tf)
    df_train = df.loc[df.index <= CUTOFF].copy()
    if df_train.empty:
        return {"error": "No train data"}

    strat = strat_cls(**strat_kwargs)
    trades = _backtest_wrapper(df_train, strat, cfg)
    if trades.empty or len(trades) < 80:
        return {"error": f"Too few trades on train: {len(trades)}"}

    # ── Features ──────────────────────────────────────────────────────────
    selected = list(FEATURES_SELECTED[key])
    X_full = build_superset(df_train, asset=asset)
    X = X_full.loc[trades.index, selected].dropna(axis=0, how="any")
    y = (trades["pips_net"] > 0).astype(int).loc[X.index]
    pnl = trades["pips_net"].loc[X.index]

    if len(X) < 80:
        return {"error": f"Too few samples after align: {len(X)}"}

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print(
        f"  Tuning {asset} {tf} ({model_name}) on {len(X)} samples, "
        f"{len(grid)} axes ({n_combos} combos), "
        f"{len(THRESHOLD_GRID)} thresholds..."
    )

    r = nested_cpcv_tuning(
        model_factory=factory,
        param_grid=grid,
        threshold_grid=THRESHOLD_GRID,
        X=X,
        y=y,
        pnl=pnl,
        outer_k=5,
        inner_k=3,
        embargo_pct=0.01,
        seed=SEED,
    )

    return {
        "asset": asset,
        "tf": tf,
        "model": model_name,
        "best_params": r.best_params,
        "best_threshold": r.best_threshold,
        "expected_sharpe_outer": r.sharpe_outer_mean,
        "expected_wr": r.wr_outer_mean,
        "expected_n_kept": r.n_kept_outer_mean,
        "sharpe_outer_std": r.sharpe_outer_std,
        "n_combos_evaluated": r.n_combos_evaluated,
        "outer_folds": r.outer_fold_results,
    }


def main() -> int:
    set_global_seeds()

    out: dict[str, dict] = {}
    for asset, tf in FEATURES_SELECTED:
        key_str = f"{asset}_{tf}"
        print(f"\nTuning {key_str}...")
        out[key_str] = tune_one_asset(asset, tf)
        entry = out[key_str]
        if "error" in entry:
            print(f"  [WARN] Erreur: {entry['error']}")
        else:
            print(
                f"  [OK] Best: {entry.get('best_params')}, "
                f"threshold={entry.get('best_threshold')}, "
                f"Sharpe outer={entry.get('expected_sharpe_outer', 0):.3f}"
            )

    # ── Sauvegarde JSON ───────────────────────────────────────────────────
    out_path = Path("predictions/hyperparam_tuning_v4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Génère app/config/hyperparams_tuned.py ────────────────────────────
    lines: list[str] = []
    for asset, tf in FEATURES_SELECTED:
        key_str = f"{asset}_{tf}"
        entry = out.get(key_str, {})
        if "error" in entry:
            continue
        params_repr = repr(entry.get("best_params", {}))
        thr = entry.get("best_threshold", 0.50)
        sr = entry.get("expected_sharpe_outer", 0.0)
        wr = entry.get("expected_wr", 0.0)
        lines.append(
            f'    ("{asset}", "{tf}"): {{\n'
            f'        "model": "{entry["model"]}",\n'
            f'        "params": {params_repr},\n'
            f'        "threshold": {thr},\n'
            f'        "expected_sharpe_outer": {sr:.3f},\n'
            f'        "expected_wr": {wr:.3f},\n'
            f"    }},"
        )

    content = (
        '"""FROZEN après pivot v4 A8. NE PAS MODIFIER sans nouveau pivot."""\n'
        "from __future__ import annotations\n\n"
        "HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {\n"
        + "\n".join(lines)
        + "\n}\n"
    )
    config_path = Path("app/config/hyperparams_tuned.py")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")

    print(f"\nHyperparams sauvegardés dans {out_path} et {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
