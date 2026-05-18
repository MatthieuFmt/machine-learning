"""Pivot v4 C2 — Ranking robuste multi-actifs train uniquement.

⚠️ Aucune lecture du test set ≥ 2024.
Hard filter: toutes les données postérieures à 2022-12-31 sont EXCLUES.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.instruments import ASSET_CONFIGS, AssetConfig  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.ranking import rank_features_bootstrap  # noqa: E402
from app.features.superset import build_superset  # noqa: E402
from app.strategies.donchian import DonchianBreakout  # noqa: E402

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42

DEFAULT_DONCHIAN: dict[str, tuple[int, int]] = {
    "D1": (20, 20),
    "H4": (20, 20),
    "H1": (50, 20),
}

FALLBACK_DONCHIAN = (10, 10)  # si trop peu de trades avec le défaut

# Top-N à figer par couple
TOP_K = 15
MIN_TRADES_TRAIN = 50  # seuil minimum pour ranking fiable


def _backtest_target(
    df_train: pd.DataFrame,
    strat: DonchianBreakout,
    cfg: AssetConfig,
) -> pd.DataFrame:
    """Génère les trades (target = winner) sur le train."""
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
    trades_list = result.get("trades", [])
    if not trades_list:
        return pd.DataFrame()
    trades = pd.DataFrame(trades_list)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.set_index("entry_time").sort_index()
    return trades


def _build_target_x_y(
    df_train: pd.DataFrame,
    trades: pd.DataFrame,
    feat_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aligne features et label binaire 'winner' (pnl > 0)."""
    if trades.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    y = (trades["pips_net"] > 0).astype(int)
    y.index = trades.index
    # Aligner sur l'index features (entry_time)
    common_idx = feat_train.index.intersection(y.index)
    X = feat_train.loc[common_idx]  # noqa: N806
    y = y.loc[common_idx]
    return X, y


def _process_couple(asset: str, tf: str, inventory_entry: dict) -> dict:
    """Ranking pour un couple (asset, tf)."""
    set_global_seeds(SEED)
    cfg = ASSET_CONFIGS.get(asset)
    if cfg is None:
        return {"asset": asset, "tf": tf, "status": "no_asset_config"}

    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF]
    if len(df_train) < 300:
        return {"asset": asset, "tf": tf, "status": "train_too_short", "n_bars_train": len(df_train)}

    feat_train = build_superset(df_train, asset=asset)

    # Tentative 1 : Donchian par défaut
    N, M = DEFAULT_DONCHIAN[tf]  # noqa: N806
    strat = DonchianBreakout(N=N, M=M)
    trades = _backtest_target(df_train, strat, cfg)
    X, y = _build_target_x_y(df_train, trades, feat_train)  # noqa: N806

    fallback_used = False
    if len(y) < MIN_TRADES_TRAIN:
        N, M = FALLBACK_DONCHIAN  # noqa: N806
        strat = DonchianBreakout(N=N, M=M)
        trades = _backtest_target(df_train, strat, cfg)
        X, y = _build_target_x_y(df_train, trades, feat_train)  # noqa: N806
        fallback_used = True

    if len(y) < MIN_TRADES_TRAIN:
        return {
            "asset": asset, "tf": tf,
            "status": "insufficient_trades",
            "n_trades": int(len(y)),
            "donchian": {"N": N, "M": M, "fallback_used": fallback_used},
        }

    wr_train = float(y.mean())

    # Bootstrap stability
    ranking = rank_features_bootstrap(
        X=X, y=y,
        n_bootstrap=5,
        seed=SEED,
        top_k=TOP_K,
    )
    top_features = list(ranking.top_features)
    stability_mean = float(np.mean([ranking.stability_score.get(f, 0.0) for f in top_features]))
    stability_top1 = float(ranking.stability_score.get(top_features[0], 0.0))

    return {
        "asset": asset, "tf": tf,
        "status": "ok",
        "n_trades_train": int(len(y)),
        "wr_train": wr_train,
        "donchian": {"N": N, "M": M, "fallback_used": fallback_used},
        "top_features": top_features,
        "stability_mean": stability_mean,
        "stability_top1": stability_top1,
        "stability_per_feature": {f: float(ranking.stability_score.get(f, 0.0)) for f in top_features},
    }


def main() -> int:
    inv_path = _PROJECT_ROOT / "predictions" / "c1_couples_inventory.json"
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    new_couples = [e for e in inventory if e["status"] == "new_phase_c"]
    print(f"{len(new_couples)} couples nouveaux à traiter.")

    results: list[dict] = []
    for entry in new_couples:
        asset, tf = entry["asset"], entry["tf"]
        print(f"  → ranking {asset}/{tf} ...")
        res = _process_couple(asset, tf, entry)
        results.append(res)
        if res["status"] == "ok":
            print(f"    ✓ {len(res['top_features'])} features, stability moy={res['stability_mean']:.2f}, n_trades={res['n_trades_train']}")
        else:
            print(f"    ✗ {res['status']}")

    # Sauvegarde JSON
    out_path = _PROJECT_ROOT / "predictions" / "c2_ranking_multi_assets.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Mise à jour features_selected.py
    cfg_path = _PROJECT_ROOT / "app" / "config" / "features_selected.py"
    _update_features_selected(cfg_path, results)

    # Bilan
    ok = [r for r in results if r["status"] == "ok"]
    shortlist = [r for r in ok if r["stability_mean"] >= 0.5]
    print()
    print(f"Couples ranked OK : {len(ok)} / {len(new_couples)}")
    print(f"Shortlist (stab moyenne ≥ 0.5) pour C3 : {len(shortlist)}")
    for r in shortlist:
        print(f"  {r['asset']}/{r['tf']} : stab={r['stability_mean']:.2f}, n_trades={r['n_trades_train']}")
    return 0


def _update_features_selected(path: Path, results: list[dict]) -> None:
    """Ajoute les nouvelles entrées tout en préservant les 3 originales."""
    from app.config.features_selected import FEATURES_SELECTED as EXISTING  # noqa: E402

    new_entries: dict[tuple[str, str], tuple[str, ...]] = {}
    for r in results:
        if r["status"] == "ok":
            new_entries[(r["asset"], r["tf"])] = tuple(r["top_features"])

    merged = {**EXISTING, **new_entries}

    lines = [
        '"""FROZEN après pivot v4 A6 (3 entrées) + C2 (extension multi-actifs).',
        '',
        'NE PAS MODIFIER MANUELLEMENT. Seules les phases A6 / C2 peuvent y ajouter.',
        '"""',
        "from __future__ import annotations",
        "",
        "FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {",
    ]
    for (asset, tf), feats in merged.items():
        feat_repr = "(" + ", ".join(f"'{f}'" for f in feats) + ")"
        lines.append(f"    ({asset!r}, {tf!r}): {feat_repr},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
