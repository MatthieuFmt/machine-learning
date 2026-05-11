"""Script de run du pipeline EURUSD v1 — génère predictions et rapports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ajouter le workspace au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_machine_learning.pipelines.eurusd import EurUsdPipeline

def main() -> None:
    print("=== Pipeline EURUSD v1 (features_dropped fix) ===")
    pipeline = EurUsdPipeline()
    
    print("1/4 Chargement donnees...")
    data = pipeline.load_data()
    print(f"   H1: {len(data['h1'])} barres")
    
    print("2/4 Feature engineering...")
    ml = pipeline.build_features(data)
    print(f"   ML-ready: {len(ml)} lignes x {len(ml.columns)} colonnes")
    print(f"   Colonnes: {list(ml.columns)}")
    
    print("3/4 Entrainement + predictions...")
    model, X_cols = pipeline.train_model(ml)
    predictions = pipeline.evaluate_model(model, ml, X_cols)
    
    print("4/4 Backtest...")
    trades, metrics = pipeline.run_backtest(predictions, ml, data.get("h1"))
    
    # Sauvegarder les metriques
    results_dir = Path("predictions")
    results_dir.mkdir(exist_ok=True)
    
    for year, m in metrics.items():
        print(f"\n--- {year} ---")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        
        # Sauvegarder en JSON
        out_path = results_dir / f"metrics_v1_{year}.json"
        metrics_serializable = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in m.items()}
        out_path.write_text(json.dumps(metrics_serializable, indent=2, ensure_ascii=False))
        print(f"   -> Sauvegarde: {out_path}")
    
    # Sauvegarder le nombre de colonnes pour le log
    print(f"\n=== Resume ===")
    print(f"Features actives: {len(X_cols)}")
    print(f"Features: {X_cols}")
    for year, t in trades.items():
        print(f"Trades {year}: {len(t)}")
    
    return metrics, trades, X_cols

if __name__ == "__main__":
    main()
