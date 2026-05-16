"""Pivot v4 A6 — Ranking robuste des features train uniquement.

⚠️ Aucune lecture du test set ≥ 2024.
Hard filter: toutes les données postérieures à 2022-12-31 sont EXCLUES.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.ranking import rank_features_bootstrap
from app.features.superset import build_superset
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

RANKING_CONFIGS: list[dict] = [
    {
        "asset": "US30",
        "tf": "D1",
        "strat_cls": DonchianBreakout,
        "strat_kwargs": {"N": 20, "M": 20},
        "cfg": ASSET_CONFIGS["US30"],
    },
    {
        "asset": "EURUSD",
        "tf": "H4",
        "strat_cls": DonchianBreakout,
        "strat_kwargs": {"N": 20, "M": 20},
        "cfg": _EURUSD_CFG,
    },
    {
        "asset": "XAUUSD",
        "tf": "D1",
        "strat_cls": DonchianBreakout,
        "strat_kwargs": {"N": 100, "M": 20},
        "cfg": ASSET_CONFIGS["XAUUSD"],
    },
]


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
    # entry_time is a string, convert back to datetime
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.set_index("entry_time").sort_index()
    return trades


def build_target_from_strat(
    df_train: pd.DataFrame,
    strat: DonchianBreakout,
    cfg: AssetConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Génère la cible binaire (winner) à partir des trades sur train."""
    trades = _backtest_wrapper(df_train, strat, cfg)
    if trades.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    # pips_net: on utilise la colonne si présente, sinon on calcule
    if "pips_net" in trades.columns:
        y = (trades["pips_net"] > 0).astype(int)
    else:
        y = pd.Series(0, index=trades.index, dtype=int)
    return trades, y


def rank_one_config(cfg_entry: dict) -> dict:
    """Exécute le ranking pour une config (asset, tf, strat)."""
    asset = cfg_entry["asset"]
    tf = cfg_entry["tf"]
    cfg = cfg_entry["cfg"]

    df = load_asset(asset, tf)
    df_train = df.loc[df.index <= CUTOFF]
    if df_train.empty:
        return {"asset": asset, "tf": tf, "error": "no train data"}

    strat = cfg_entry["strat_cls"](**cfg_entry["strat_kwargs"])
    trades, y = build_target_from_strat(df_train, strat, cfg)
    if trades.empty or len(y) < 50:
        return {"asset": asset, "tf": tf, "error": f"too few trades ({len(y)})"}

    X_full = build_superset(df_train, asset=asset)
    # Align X to trades entry timestamps
    X = X_full.loc[trades.index].dropna(axis=1, how="all")
    # Drop NaN rows
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y_aligned = y.loc[X.index]

    if len(X) < 80:
        return {
            "asset": asset,
            "tf": tf,
            "error": f"too few train samples after align ({len(X)})",
        }

    result = rank_features_bootstrap(X, y_aligned, n_bootstrap=5, top_k=15, seed=42)
    return {
        "asset": asset,
        "tf": tf,
        "n_train_trades": len(X),
        "winner_rate": float(y_aligned.mean()),
        "top_features": list(result.top_features),
        "stability_score": result.stability_score,
        "metrics_per_feature": result.metrics_per_feature.to_dict(orient="records"),
    }


def main() -> int:
    set_global_seeds()
    out: dict = {}
    for entry in RANKING_CONFIGS:
        key = f"{entry['asset']}_{entry['tf']}"
        print(f"Ranking {key}...")
        out[key] = rank_one_config(entry)

    # ── Sauvegarde JSON ─────────────────────────────────────────────────
    out_path = Path("predictions/feature_ranking_v4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Génération features_selected.py ──────────────────────────────────
    selected_lines: list[str] = []
    for entry in RANKING_CONFIGS:
        key = f"{entry['asset']}_{entry['tf']}"
        result_entry = out.get(key, {})
        if "top_features" in result_entry:
            tup = tuple(result_entry["top_features"])
            selected_lines.append(
                f'    ("{entry["asset"]}", "{entry["tf"]}"): {tup!r},'
            )

    config_content = (
        '"""FROZEN après pivot v4 A6. NE PAS MODIFIER sans nouveau pivot."""\n'
        "from __future__ import annotations\n\n"
        "FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {\n"
        + "\n".join(selected_lines)
        + "\n}\n"
    )
    config_dir = Path("app/config/features_selected.py")
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text(config_content, encoding="utf-8")

    print(f"Top features sauvegardés dans {out_path} et {config_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
