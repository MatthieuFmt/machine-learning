"""Script de run du pipeline walk-forward EURUSD v14.

Contrairement à run_pipeline_v1.py (split statique train ≤ 2023, val=2024, test=2025),
ce script utilise le walk-forward retraining : fenêtre glissante 36 mois,
réentraînement tous les 3 mois, purge 48h entre train et OOS.

L'objectif est de réduire le biais directionnel et de mieux s'adapter
aux changements de régime (v13 a montré que le modèle statique ne
se généralise pas à 2025).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ajouter le workspace au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_machine_learning.analysis.edge_validation import validate_edge
from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.pipelines.eurusd import EurUsdPipeline


def main() -> dict[str, Any]:
    print("=== Pipeline EURUSD v14 (Walk-Forward Retraining) ===")
    pipeline = EurUsdPipeline()

    print("1/3 Chargement donnees...")
    data = pipeline.load_data()
    print(f"   H1: {len(data['h1'])} barres")

    print("2/3 Feature engineering...")
    ml = pipeline.build_features(data)
    print(f"   ML-ready: {len(ml)} lignes x {len(ml.columns)} colonnes")
    print(f"   Colonnes: {list(ml.columns)}")

    print("3/3 Walk-forward retraining + backtest...")
    result = pipeline.run_walk_forward(
        ml_data=ml,
        data=data,
        train_months=36,
        step_months=3,
    )

    predictions_agg = result["predictions_agg"]
    trades_agg = result["trades_agg"]
    metrics_agg = result["metrics_agg"]
    n_folds = result["fold_count"]
    fold_info = result["fold_info"]

    print(f"\n   Folds générés : {n_folds}")
    print(f"   Prédictions OOS agrégées : {len(predictions_agg)} barres")
    print(f"   Période couverte : {predictions_agg.index.min().date()} -> {predictions_agg.index.max().date()}")

    # Afficher le détail par fold
    print(f"\n   {'Fold':>5s}  {'Train':>22s}  {'Test':>22s}  {'n_train':>8s}  {'n_test':>7s}")
    print("   " + "-" * 72)
    for fi in fold_info:
        train_range = f"{fi['train_start']}->{fi['train_end']}"
        test_range = f"{fi['test_start']}->{fi['test_end']}"
        print(f"   {fi['fold']:5d}  {train_range:>22s}  {test_range:>22s}  {fi['n_train']:8d}  {fi['n_test']:7d}")

    # Afficher les métriques agrégées par année
    results_dir = Path("predictions")
    results_dir.mkdir(exist_ok=True)

    for year, m in metrics_agg.items():
        print(f"\n--- {year} (Walk-Forward) ---")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")

        # Sauvegarder en JSON
        out_path = results_dir / f"metrics_wf_v14_{year}.json"
        metrics_serializable = {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in m.items()
        }
        out_path.write_text(
            json.dumps(metrics_serializable, indent=2, ensure_ascii=False)
        )
        print(f"   -> Sauvegarde: {out_path}")

    # ── v13: Edge Validation sur les résultats walk-forward ────────────
    print(f"\n=== v13: Edge Validation (Walk-Forward) ===")
    backtest_cfg = BacktestConfig()
    n_trials = 13  # v1->v13 = 13 itérations de recherche

    all_edges: dict[int, dict[str, Any]] = {}
    for year in sorted(trades_agg.keys()):
        t = trades_agg[year]
        if t is None or t.empty:
            print(f"   {year}: pas de trades — validation impossible")
            continue

        edge = validate_edge(t, backtest_cfg, n_trials_searched=n_trials)
        all_edges[int(year)] = edge

        print(f"\n   --- {year} ({edge['n_trades']} trades) ---")
        be = edge["breakeven"]
        print(f"   Breakeven WR: {be['wr_pct']}% | Observé: {be['observed_wr_pct']}% | Marge: {be['margin_pct']:+.1f}%")
        bs = edge["bootstrap_sharpe"]
        print(f"   Bootstrap Sharpe: obs={bs['observed']:.4f} | p(>0)={bs['p_value_gt_0']:.4f} | CI95=[{bs['ci_95_lower']:.4f}, {bs['ci_95_upper']:.4f}]")
        ds = edge["deflated_sharpe"]
        print(f"   Deflated SR: DSR={ds['dsr']:.4f} | PSR={ds['psr']:.4f} | E[max SR]={ds['e_max_sr']:.4f}")
        tt = edge["t_statistic"]
        print(f"   t-test: mean={tt['mean_pnl']:.4f} | std={tt['std_pnl']:.4f} | t={tt['t_stat']:.4f} | p={tt['p_value']:.4f}")

        p_val = bs["p_value_gt_0"]
        if p_val < 0.05:
            decision = "EDGE REEL — passer au walk-forward 2-etapes"
        elif p_val > 0.10:
            decision = "EDGE NON CONFIRME — restructurer le probleme"
        else:
            decision = "ZONE GRISE — investiguer davantage"
        print(f"   => DECISION: {decision}")

    # Sauvegarder les edge validations
    edge_out = results_dir / "edge_validation_wf_v14.json"
    edge_serializable = {
        str(year): {
            k: v for k, v in edge.items()
        }
        for year, edge in all_edges.items()
    }
    edge_out.write_text(
        json.dumps(edge_serializable, indent=2, ensure_ascii=False)
    )

    # Résumé
    print(f"\n=== Resume Walk-Forward ===")
    print(f"Folds: {n_folds}")
    print(f"Features actives: {len(result['X_cols'])}")
    print(f"Features: {result['X_cols']}")
    for year, t in trades_agg.items():
        print(f"Trades {year}: {len(t)}")

    return result


if __name__ == "__main__":
    main()
