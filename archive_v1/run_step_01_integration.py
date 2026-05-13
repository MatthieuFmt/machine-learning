"""Script d'intégration Step 01 — teste les 4 target_modes en rafale.

Pour chaque target_mode :
1. Charge les données et build les features avec la target appropriée
2. Entraîne le modèle (classifieur ou régresseur selon mode)
3. Évalue OOS (val_year + test_year)
4. Backtest
5. Affiche les métriques clés + distribution cible
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_machine_learning.config.instruments import TargetMode
from learning_machine_learning.config.registry import ConfigRegistry
from learning_machine_learning.pipelines.eurusd import EurUsdPipeline


def _fmt_dist(dist: dict[str, float]) -> str:
    return f"-1={dist.get('-1', 0):.1f}%  0={dist.get('0', 0):.1f}%  1={dist.get('1', 0):.1f}%"


def run_mode(mode: TargetMode) -> dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"  MODE: {mode}")
    print(f"{'=' * 70}")

    pipeline = EurUsdPipeline()
    # Patcher le target_mode (InstrumentConfig est frozen → replace)
    pipeline.instrument = replace(pipeline.instrument, target_mode=mode)

    print("  [1] Chargement données...")
    data = pipeline.load_data()
    print(f"      H1: {len(data['h1'])} barres")

    print("  [2] Feature engineering...")
    ml = pipeline.build_features(data)
    print(f"      ML-ready: {len(ml)} lignes x {len(ml.columns)} colonnes")

    from learning_machine_learning.features.triple_barrier import label_distribution
    dist = label_distribution(ml["Target"])
    print(f"      Distribution cible: {_fmt_dist(dist)}")

    print("  [3] Entraînement + prédictions...")
    model, X_cols = pipeline.train_model(ml)
    predictions = pipeline.evaluate_model(model, ml, X_cols)

    for year, preds_df in predictions.items():
        if "Predicted_Return" in preds_df.columns:
            pr = preds_df["Predicted_Return"]
            print(f"      {year}: {len(preds_df)} prédictions, "
                  f"mean_pred={float(pr.mean()):.6f}, std_pred={float(pr.std()):.6f}")
        else:
            print(f"      {year}: {len(preds_df)} prédictions "
                  f"(classes={preds_df['Prediction_Modele'].value_counts().to_dict()})")

    print("  [4] Backtest...")
    trades, metrics = pipeline.run_backtest(predictions, ml, data.get("h1"))

    for year, m in metrics.items():
        print(f"\n      --- {year} ---")
        print(f"      Trades: {m.get('trades', 'N/A')}")
        print(f"      Sharpe: {m.get('sharpe', float('nan')):.4f}")
        print(f"      Win Rate: {m.get('win_rate', float('nan')):.1f}%")
        print(f"      Profit Net: {m.get('profit_net', float('nan')):.1f} pips")
        print(f"      Buy & Hold: {m.get('buy_and_hold_pips', float('nan')):.1f} pips")
        print(f"      Alpha: {m.get('alpha_pips', float('nan')):.1f} pips")
        print(f"      PnL €: {m.get('pnl_eur', float('nan')):.2f}")

    return {
        "mode": mode,
        "distribution": dist,
        "metrics": {str(y): m for y, m in metrics.items()},
        "n_trades": {str(y): len(t) for y, t in trades.items()},
        "X_cols": X_cols,
    }


def main() -> None:
    modes: list[TargetMode] = [
        "forward_return",
        "directional_clean",
        "cost_aware_v2",
    ]

    all_results: dict[str, Any] = {}

    for mode in modes:
        try:
            result = run_mode(mode)
            all_results[mode] = result
        except Exception as e:
            print(f"\n  ❌ Mode {mode} a échoué: {e}")
            import traceback
            traceback.print_exc()
            all_results[mode] = {"error": str(e)}

    # Résumé comparatif
    print(f"\n{'=' * 70}")
    print("  RÉSUMÉ COMPARATIF")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<22s} {'Val Sharpe':>12s} {'Test Sharpe':>12s} {'Distribution':>30s}")
    print("  " + "-" * 80)

    for mode, r in all_results.items():
        if "error" in r:
            print(f"  {mode:<22s} {'❌ ' + r['error'][:40]:>12s}")
            continue

        dist = r.get("distribution", {})
        metrics = r.get("metrics", {})

        val_sharpe = metrics.get("2024", {}).get("sharpe", float("nan"))
        test_sharpe = metrics.get("2025", {}).get("sharpe", float("nan"))

        print(
            f"  {mode:<22s} {val_sharpe:>12.4f} {test_sharpe:>12.4f} "
            f"{_fmt_dist(dist):>30s}"
        )

    # Sauvegarder les résultats
    out_path = Path("predictions/step_01_integration.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
