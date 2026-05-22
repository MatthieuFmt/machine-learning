"""Pivot v4 C3 â€” SÃ©lection de modÃ¨le multi-actifs (CPCV train uniquement).

âš ï¸ Aucune lecture du test set â‰¥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.features_selected import FEATURES_SELECTED
from app.config.instruments import ASSET_CONFIGS
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.superset import build_superset
from app.models.candidates import CANDIDATES
from app.models.cpcv_evaluation import evaluate_model_cpcv
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42
THRESHOLD = 0.50  # seuil mÃ©ta par dÃ©faut A7
SHORTLIST_THRESHOLD = 0.5  # stab moyenne C2 minimale
SHORTLIST_C4_SHARPE = 0.5  # Sharpe CPCV moyen minimum pour passer en C4
CANDIDATE_NAMES = ["rf", "hgbm", "stacking"]


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


def _build_x_y_pnl(
    asset: str, tf: str, donchian: dict
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reconstruit X, y, pnl comme en A7 mais en filtrant par FEATURES_SELECTED."""
    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF]
    if df_train.empty:
        raise ValueError(f"No train data for {asset}/{tf}")

    trades = _run_backtest(df_train, asset, donchian)
    if trades.empty:
        raise ValueError(f"No trades for {asset}/{tf}")

    feat_train = build_superset(df_train, asset=asset)

    # Filtrer aux 15 features figÃ©es en C2
    selected_features = list(FEATURES_SELECTED[(asset, tf)])
    common_idx = feat_train.index.intersection(trades.index)
    X = feat_train.loc[common_idx, selected_features].dropna(axis=0, how="any")
    y = (trades["pips_net"] > 0).astype(int).loc[X.index]
    pnl = trades["pips_net"].loc[X.index]

    return X, y, pnl


def _process_couple(asset: str, tf: str, donchian: dict) -> dict:
    """Ã‰value les 3 candidats via CPCV pour un couple (asset, tf)."""
    set_global_seeds(SEED)
    try:
        X, y, pnl = _build_x_y_pnl(asset, tf, donchian)
    except (ValueError, KeyError) as exc:
        return {
            "asset": asset,
            "tf": tf,
            "status": "error",
            "error": str(exc),
        }

    if len(y) < 50:
        return {
            "asset": asset,
            "tf": tf,
            "status": "insufficient_trades",
            "n_trades": int(len(y)),
        }

    per_candidate: dict[str, dict] = {}
    for cand_name in CANDIDATE_NAMES:
        builder = CANDIDATES[cand_name]
        cpcv_result = evaluate_model_cpcv(
            model_builder=builder,
            X=X,
            y=y,
            pnl=pnl,
            model_name=cand_name,
            n_splits=5,
            embargo_pct=0.01,
            threshold=THRESHOLD,
            seed=SEED,
        )
        per_candidate[cand_name] = {
            "sharpe_mean": float(cpcv_result.sharpe_mean),
            "sharpe_std": float(cpcv_result.sharpe_std),
            "sharpes_per_fold": [float(s) for s in cpcv_result.fold_sharpes],
            "stability_inter_fold": float(cpcv_result.sharpe_ratio_stability),
            "n_kept_mean": float(cpcv_result.n_kept_mean),
            "wr_meta_mean": float(cpcv_result.wr_mean),
        }

    # SÃ©lection : argmax Sharpe moyen
    best = max(per_candidate.items(), key=lambda kv: kv[1]["sharpe_mean"])
    best_name = best[0]
    best_metrics = best[1]

    return {
        "asset": asset,
        "tf": tf,
        "status": "ok",
        "n_trades_train": int(len(y)),
        "wr_train": float(y.mean()),
        "candidates": per_candidate,
        "selected_model": best_name,
        "selected_sharpe_mean": best_metrics["sharpe_mean"],
        "selected_stability": best_metrics["stability_inter_fold"],
        "pass_c4_threshold": best_metrics["sharpe_mean"] >= SHORTLIST_C4_SHARPE,
    }


def _update_model_selected(path: Path, results: list[dict]) -> None:
    """Ajoute les nouvelles entrÃ©es tout en prÃ©servant les 3 originales A7."""
    from app.config.model_selected import MODEL_SELECTED as existing

    new_entries: dict[tuple[str, str], str] = {}
    for r in results:
        if r["status"] == "ok":
            new_entries[(r["asset"], r["tf"])] = r["selected_model"]

    merged = {**existing, **new_entries}

    lines = [
        '"""FROZEN aprÃ¨s pivot v4 A7 (3 entrÃ©es) + C3 (extension multi-actifs).',
        "",
        "NE PAS MODIFIER MANUELLEMENT. Seules les phases A7 / C3 peuvent y ajouter.",
        '"""',
        "from __future__ import annotations",
        "",
        "MODEL_SELECTED: dict[tuple[str, str], str] = {",
    ]
    for (asset, tf), model_name in merged.items():
        lines.append(f"    ({asset!r}, {tf!r}): {model_name!r},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rank_path = _PROJECT_ROOT / "predictions" / "c2_ranking_multi_assets.json"
    rankings = json.loads(rank_path.read_text(encoding="utf-8"))
    shortlist = [
        r
        for r in rankings
        if r["status"] == "ok" and r["stability_mean"] >= SHORTLIST_THRESHOLD
    ]
    print(f"{len(shortlist)} couples en shortlist C2 (stab >= {SHORTLIST_THRESHOLD}).")

    results: list[dict] = []
    for r in shortlist:
        asset, tf = r["asset"], r["tf"]
        donchian = r["donchian"]
        print(f"  -> model selection {asset}/{tf} ...")
        res = _process_couple(asset, tf, donchian)
        results.append(res)
        if res["status"] == "ok":
            print(
                f"    [OK] {res['selected_model']} "
                f"Sharpe={res['selected_sharpe_mean']:.2f}, "
                f"pass_c4={res['pass_c4_threshold']}"
            )
        else:
            print(f"    [FAIL] {res['status']}: {res.get('error', res.get('n_trades', ''))}")

    # Sauvegarde JSON
    out_path = _PROJECT_ROOT / "predictions" / "c3_model_selection_multi_assets.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Mise Ã  jour model_selected.py
    cfg_path = _PROJECT_ROOT / "app" / "config" / "model_selected.py"
    _update_model_selected(cfg_path, results)

    # Bilan
    ok_results = [r for r in results if r["status"] == "ok"]
    c4_shortlist = [r for r in ok_results if r["pass_c4_threshold"]]
    print()
    print(f"Couples evalues OK : {len(ok_results)}")
    print(f"Shortlist C4 (Sharpe CPCV >= {SHORTLIST_C4_SHARPE}) : {len(c4_shortlist)}")
    for r in c4_shortlist:
        print(f"  {r['asset']}/{r['tf']} : {r['selected_model']} Sharpe={r['selected_sharpe_mean']:.2f}")

    # VÃ©rification critÃ¨re go/no-go C4
    if len(c4_shortlist) < 2:
        print()
        print(
            f"[WARN] Seulement {len(c4_shortlist)} couples(s) passent le filtre Sharpe >= 0.5. "
            "La Phase C risque de ne pas produire d'edge ML suffisant. "
            "Discuter avec l'utilisateur avant C4."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
