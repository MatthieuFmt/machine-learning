"""Script de run du pipeline EURUSD v1 — génère predictions et rapports."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier

# Ajouter le workspace au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_machine_learning.pipelines.eurusd import EurUsdPipeline
from learning_machine_learning.model.meta_labeling import (
    build_meta_labels,
    train_meta_model,
    apply_meta_filter,
)

def evaluate_meta_thresholds(
    pipeline: EurUsdPipeline,
    meta_model: RandomForestClassifier,
    predictions_val: pd.DataFrame,
    ml_data: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    X_cols: list[str],
    val_year: int,
    thresholds: list[float],
) -> pd.DataFrame:
    """Sweep des seuils méta sur val_year.

    Pour chaque seuil, applique le méta-filtre sur les prédictions val_year,
    exécute le backtest, et collecte les métriques. Le test_year n'est jamais
    exposé — anti-leakage garanti.

    Args:
        pipeline: Pipeline EURUSD instance.
        meta_model: RandomForestClassifier binaire entraîné.
        predictions_val: DataFrame de prédictions OOS pour val_year.
        ml_data: DataFrame ML-ready complet.
        ohlcv_h1: DataFrame OHLC H1.
        X_cols: Colonnes de features primaires.
        val_year: Année de validation (ex: 2024).
        thresholds: Liste de seuils à tester.

    Returns:
        DataFrame avec colonnes [threshold, sharpe, profit_net, win_rate,
        trades, dd, alpha_pips].
    """
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        filtered = apply_meta_filter(
            meta_model,
            df_predictions=predictions_val.copy(),
            ml_data=ml_data,
            X_cols=X_cols,
            threshold=t,
        )
        _, metrics_dict = pipeline.run_backtest(
            {val_year: filtered}, ml_data, ohlcv_h1,
        )
        if val_year in metrics_dict:
            m = metrics_dict[val_year]
            rows.append({
                "threshold": t,
                "sharpe": float(m.get("sharpe", float("nan"))),
                "profit_net": float(m.get("profit_net", float("nan"))),
                "win_rate": float(m.get("win_rate", float("nan"))),
                "trades": int(m.get("trades", 0)),
                "dd": float(m.get("dd", float("nan"))),
                "alpha_pips": float(m.get("alpha_pips", float("nan"))),
            })

    return pd.DataFrame(rows)


def main() -> None:
    print("=== Pipeline EURUSD v9 (méta-labeling + sweep seuil) ===")
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
    
    # v9: Méta-labeling avec optimisation du seuil sur val_year
    print("3.5/4 Méta-labeling + sweep seuil...")
    val_year = pipeline.model_cfg.val_year  # 2024
    test_year = pipeline.model_cfg.test_year  # 2025

    # Étape 1: Backtest primaire sur val_year pour générer les labels méta
    val_predictions = {val_year: predictions[val_year]}
    val_trades_dict, _ = pipeline.run_backtest(val_predictions, ml, data.get("h1"))
    val_trades = val_trades_dict.get(val_year)

    if val_trades is not None and not val_trades.empty:
        # Construire les méta-labels à partir des trades val_year
        X_meta, y_meta = build_meta_labels(
            trades_df=val_trades,
            predictions_df=predictions[val_year],
            ml_data=ml,
            X_cols=X_cols,
        )

        if not X_meta.empty:
            # Entraîner le méta-modèle
            meta_model = train_meta_model(X_meta, y_meta)

            # Étape 2: Sweep des seuils sur val_year
            thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
            print(f"   Sweep seuils {thresholds} sur val_year ({val_year})...")
            sweep_df = evaluate_meta_thresholds(
                pipeline, meta_model,
                predictions_val=predictions[val_year],
                ml_data=ml,
                ohlcv_h1=data.get("h1"),
                X_cols=X_cols,
                val_year=val_year,
                thresholds=thresholds,
            )

            # Afficher le tableau de sweep
            print("   " + "-" * 70)
            print(f"   {'Seuil':>6s}  {'Sharpe':>8s}  {'Net(pips)':>10s}  {'WR%':>6s}  {'Trades':>6s}  {'DD%':>6s}")
            print("   " + "-" * 70)
            for _, row in sweep_df.iterrows():
                print(
                    f"   {row['threshold']:6.2f}  {row['sharpe']:8.4f}  "
                    f"{row['profit_net']:10.1f}  {row['win_rate']:6.1f}  "
                    f"{row['trades']:6.0f}  {row['dd']:6.1f}"
                )

            # Sélection du meilleur seuil (max Sharpe)
            best_idx = sweep_df["sharpe"].idxmax()
            best_threshold = float(sweep_df.loc[best_idx, "threshold"])
            best_sharpe = float(sweep_df.loc[best_idx, "sharpe"])
            print(f"   => Meilleur seuil: {best_threshold:.2f} (Sharpe val={best_sharpe:.4f})")

            # Appliquer le meilleur seuil sur test_year (2025) — jamais vu par le sweep
            predictions[test_year] = apply_meta_filter(
                meta_model,
                df_predictions=predictions[test_year],
                ml_data=ml,
                X_cols=X_cols,
                threshold=best_threshold,
            )

            # Appliquer aussi sur val_year pour rapport complet (métriques potentiellement optimistes)
            predictions[val_year] = apply_meta_filter(
                meta_model,
                df_predictions=predictions[val_year].copy(),
                ml_data=ml,
                X_cols=X_cols,
                threshold=best_threshold,
            )
            print(f"   Méta-filtre appliqué sur val_year={val_year} et test_year={test_year} (seuil={best_threshold:.2f})")
        else:
            print("   ⚠️ X_meta vide, méta-labeling désactivé")
    else:
        print(f"   ⚠️ Aucun trade val_year ({val_year}), méta-labeling désactivé")
    
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
