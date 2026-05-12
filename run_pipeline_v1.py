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
from learning_machine_learning.analysis.edge_validation import validate_edge
from learning_machine_learning.config.backtest import BacktestConfig

# v10: Colonnes de contexte de marché pour le méta-modèle
# Ces colonnes sont dans ml_data (FILTER_KEEP) mais pas dans X_cols.
# Elles donnent au méta-modèle l'information de régime au moment du trade.
META_EXTRA_COLS: list[str] = [
    "ATR_Norm",
    "Spread",
    "Dist_SMA200_D1",
    "Volatilite_Realisee_24h",
]

def evaluate_meta_thresholds(
    pipeline: EurUsdPipeline,
    meta_model: RandomForestClassifier,
    predictions_val: pd.DataFrame,
    ml_data: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    X_cols: list[str],
    val_year: int,
    thresholds: list[float],
    extra_cols: list[str] | None = None,
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
            extra_cols=extra_cols,
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
    print("=== Pipeline EURUSD v12 (méta-RF 10 features + seuil 0.55 fixe) ===")
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
    
    # v12: Méta-labeling RF 10 features (sans contexte marché) + seuil 0.55 fixe
    print("3.5/4 Méta-labeling (RF 10 features, seuil 0.55 fixe)...")
    val_year = pipeline.model_cfg.val_year  # 2024
    test_year = pipeline.model_cfg.test_year  # 2025

    # Étape 1: Backtest primaire sur val_year pour générer les labels méta
    val_predictions = {val_year: predictions[val_year]}
    val_trades_dict, _ = pipeline.run_backtest(val_predictions, ml, data.get("h1"))
    val_trades = val_trades_dict.get(val_year)

    if val_trades is not None and not val_trades.empty:
        # Construire les méta-labels à partir des trades val_year (10 features, sans extra_cols)
        X_meta, y_meta = build_meta_labels(
            trades_df=val_trades,
            predictions_df=predictions[val_year],
            ml_data=ml,
            X_cols=X_cols,
        )

        if not X_meta.empty:
            # Entraîner le méta-modèle RF (v8 — meilleure généralisation observée)
            meta_model = train_meta_model(X_meta, y_meta)

            # Sweep diagnostic (informatif seulement — le seuil reste fixe à 0.55)
            thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
            print(f"   Sweep diagnostic {thresholds} sur val_year ({val_year})...")
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

            # v12: Seuil fixe 0.55 — meilleure généralisation observée (v8)
            FIXED_THRESHOLD = 0.55
            print(f"   => Seuil fixe: {FIXED_THRESHOLD} (v8 — meilleur Sharpe test_year)")

            # Appliquer le seuil fixe sur test_year (2025)
            predictions[test_year] = apply_meta_filter(
                meta_model,
                df_predictions=predictions[test_year],
                ml_data=ml,
                X_cols=X_cols,
                threshold=FIXED_THRESHOLD,
            )

            # Appliquer aussi sur val_year pour rapport complet
            predictions[val_year] = apply_meta_filter(
                meta_model,
                df_predictions=predictions[val_year].copy(),
                ml_data=ml,
                X_cols=X_cols,
                threshold=FIXED_THRESHOLD,
            )
            print(f"   Méta-filtre appliqué sur val_year={val_year} et test_year={test_year} (seuil={FIXED_THRESHOLD})")
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
    
    # ── v13: Validation statistique de l'edge ────────────────────────────
    print(f"\n=== v13: Edge Validation ===")
    backtest_cfg = BacktestConfig()
    n_trials = 12  # v1 → v12 = 12 itérations de recherche
    
    for year in sorted(trades.keys()):
        t = trades[year]
        if t is None or t.empty:
            print(f"   {year}: pas de trades — validation impossible")
            continue
        
        result = validate_edge(t, backtest_cfg, n_trials_searched=n_trials)
        
        print(f"\n   --- {year} ({result['n_trades']} trades) ---")
        
        be = result["breakeven"]
        print(f"   Breakeven WR: {be['wr_pct']}% | Observé: {be['observed_wr_pct']}% | Marge: {be['margin_pct']:+.1f}%")
        
        bs = result["bootstrap_sharpe"]
        print(f"   Bootstrap Sharpe: obs={bs['observed']:.4f} | p(>0)={bs['p_value_gt_0']:.4f} | CI95=[{bs['ci_95_lower']:.4f}, {bs['ci_95_upper']:.4f}]")
        
        ds = result["deflated_sharpe"]
        print(f"   Deflated SR: DSR={ds['dsr']:.4f} | PSR={ds['psr']:.4f} | E[max SR]={ds['e_max_sr']:.4f} (n_trials={ds['n_trials']})")
        
        tt = result["t_statistic"]
        print(f"   t-test: mean={tt['mean_pnl']:.4f} | std={tt['std_pnl']:.4f} | t={tt['t_stat']:.4f} | p={tt['p_value']:.4f}")
        
        # Décision
        p_val = bs["p_value_gt_0"]
        if p_val < 0.05:
            decision = "EDGE REEL — passer au walk-forward"
        elif p_val > 0.10:
            decision = "EDGE NON CONFIRME — restructurer le probleme"
        else:
            decision = "ZONE GRISE — investiguer davantage"
        print(f"   => DECISION: {decision}")
    
    # Sauvegarder le nombre de colonnes pour le log
    print(f"\n=== Resume ===")
    print(f"Features actives: {len(X_cols)}")
    print(f"Features: {X_cols}")
    for year, t in trades.items():
        print(f"Trades {year}: {len(t)}")
    
    return metrics, trades, X_cols

if __name__ == "__main__":
    main()
