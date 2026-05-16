"""Pivot v4 A7 — Sélection de modèle via CPCV train uniquement.

⚠️ Aucune lecture du test set ≥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.features_selected import FEATURES_SELECTED
from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.superset import build_superset
from app.models.candidates import CANDIDATES
from app.models.cpcv_evaluation import evaluate_model_cpcv
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# ── AssetConfig EURUSD locale (non présente dans ASSET_CONFIGS) ──────────
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

# ── Mapping (asset, tf) → (AssetConfig, strat_cls, strat_kwargs) ─────────
STRAT_MAP: dict[tuple[str, str], tuple[AssetConfig, type, dict]] = {
    ("US30", "D1"): (ASSET_CONFIGS["US30"], DonchianBreakout, {"N": 20, "M": 20}),
    ("EURUSD", "H4"): (_EURUSD_CFG, DonchianBreakout, {"N": 20, "M": 20}),
    ("XAUUSD", "D1"): (ASSET_CONFIGS["XAUUSD"], DonchianBreakout, {"N": 100, "M": 20}),
}


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


def evaluate_one_asset(asset: str, tf: str) -> dict:
    """Évalue les 3 candidats via CPCV sur train ≤ 2022 pour un actif."""
    key = (asset, tf)
    selected_features = FEATURES_SELECTED.get(key)
    if not selected_features:
        return {"error": f"No selected features for {asset} {tf}"}

    if key not in STRAT_MAP:
        return {"error": f"No strategy mapped for {asset} {tf}"}

    cfg, strat_cls, strat_kwargs = STRAT_MAP[key]

    # Charger train uniquement
    df = load_asset(asset, tf)
    df_train = df.loc[df.index <= CUTOFF].copy()
    if df_train.empty:
        return {"error": "no train data"}

    strat = strat_cls(**strat_kwargs)
    trades = _backtest_wrapper(df_train, strat, cfg)
    if trades.empty or len(trades) < 80:
        return {"error": f"Too few trades on train: {len(trades)}"}

    # Features
    X_full = build_superset(df_train, asset=asset)
    X = X_full.loc[trades.index, list(selected_features)].dropna(axis=0, how="any")
    y = (trades["pips_net"] > 0).astype(int).loc[X.index]
    pnl = trades["pips_net"].loc[X.index]

    if len(X) < 80:
        return {"error": f"Too few samples after align: {len(X)}"}

    results: dict[str, dict] = {}
    for name, builder in CANDIDATES.items():
        print(f"  CPCV {name}...")
        r = evaluate_model_cpcv(
            model_builder=builder,
            X=X,
            y=y,
            pnl=pnl,
            model_name=name,
            n_splits=5,
            embargo_pct=0.01,
            threshold=0.50,
            seed=42,
        )
        results[name] = {
            "sharpe_mean": r.sharpe_mean,
            "sharpe_std": r.sharpe_std,
            "stability": r.sharpe_ratio_stability,
            "wr_mean": r.wr_mean,
            "n_kept_mean": r.n_kept_mean,
            "fold_sharpes": r.fold_sharpes,
        }

    # Sélectionner : meilleur Sharpe moyen avec stabilité < 1.0
    candidates_stable = {
        n: res for n, res in results.items() if res["stability"] < 1.0
    }
    if not candidates_stable:
        # Fallback : meilleur Sharpe peu importe la stabilité
        best_name = max(results.items(), key=lambda kv: kv[1]["sharpe_mean"])[0]
    else:
        best_name = max(candidates_stable.items(), key=lambda kv: kv[1]["sharpe_mean"])[0]

    return {
        "asset": asset,
        "tf": tf,
        "n_trades_train": len(X),
        "n_features": len(selected_features),
        "winner_rate_train": float(y.mean()),
        "results_per_model": results,
        "best_model": best_name,
        "best_sharpe_mean": results[best_name]["sharpe_mean"],
    }


def main() -> int:
    set_global_seeds()
    out: dict[str, dict] = {}

    for (asset, tf) in FEATURES_SELECTED:
        key_str = f"{asset}_{tf}"
        print(f"Evaluating {key_str}...")
        out[key_str] = evaluate_one_asset(asset, tf)

    # ── Sauvegarde JSON ─────────────────────────────────────────────────
    out_path = Path("predictions/model_selection_v4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Génère app/config/model_selected.py ──────────────────────────────
    lines: list[str] = []
    for (asset, tf) in FEATURES_SELECTED:
        key_str = f"{asset}_{tf}"
        entry = out.get(key_str, {})
        if "best_model" in entry:
            lines.append(f'    ("{asset}", "{tf}"): "{entry["best_model"]}",')

    content = (
        '"""FROZEN après pivot v4 A7. NE PAS MODIFIER sans nouveau pivot."""\n'
        "from __future__ import annotations\n\n"
        "MODEL_SELECTED: dict[tuple[str, str], str] = {\n"
        + "\n".join(lines)
        + "\n}\n"
    )
    config_path = Path("app/config/model_selected.py")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")

    print(f"Modèles sélectionnés sauvegardés dans {out_path} et {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
