"""Script de run du pipeline BTCUSD — validation cross-actif.

Backtest simple : RF → prédictions → simulateur → métriques.
Pas de méta-labeling (le méta-modèle EURUSD n'est pas transférable à BTC).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ajouter le workspace au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_machine_learning.pipelines.btcusd import BtcUsdPipeline
from learning_machine_learning.config.backtest import BacktestConfig


BTC_BACKTEST = BacktestConfig(
    tp_pips=30.0,          # effectif = 30 × 5.0 = 150 pips BTC
    sl_pips=10.0,          # effectif = 10 × 5.0 = 50 pips BTC
    commission_pips=10.0,  # ≈ $10 round-trip BTC spot
    slippage_pips=5.0,     # ≈ $5 slippage
    use_momentum_filter=True,
    use_vol_filter=True,
    use_session_filter=True,
    use_calendar_filter=True,
)


def main() -> None:
    print("=== Pipeline BTCUSD H1 — Validation Cross-Actif ===")

    # Instancier le pipeline avec la config backtest BTC spécifique
    pipeline = BtcUsdPipeline(backtest_cfg=BTC_BACKTEST)

    # 1. Chargement des données
    print("1/4 Chargement données...")
    data = pipeline.load_data()
    print(f"   H1: {len(data['h1'])} barres")

    # 2. Feature engineering
    print("2/4 Feature engineering...")
    ml = pipeline.build_features(data)
    print(f"   ML-ready: {len(ml)} lignes × {len(ml.columns)} colonnes")

    # 3. Entraînement + prédictions
    print("3/4 Entraînement + prédictions...")
    model, X_cols = pipeline.train_model(ml)
    predictions = pipeline.evaluate_model(model, ml, X_cols)

    # 4. Backtest
    print("4/4 Backtest...")
    trades, metrics = pipeline.run_backtest(predictions, ml, data.get("h1"))

    # ── Affichage et sauvegarde ──
    results_dir = Path("predictions")
    results_dir.mkdir(exist_ok=True)

    for year, m in metrics.items():
        print(f"\n--- BTCUSD {year} ---")
        sharpe = m.get("sharpe", float("nan"))
        profit = m.get("profit_net", float("nan"))
        wr = m.get("win_rate", float("nan"))
        n_trades = m.get("trades", 0)
        dd = m.get("dd", float("nan"))
        alpha = m.get("alpha_pips", float("nan"))

        print(f"   Sharpe:      {sharpe:.4f}")
        print(f"   Profit net:  {profit:.1f} pips")
        print(f"   Win rate:    {wr:.1f}%")
        print(f"   Trades:      {n_trades}")
        print(f"   DD:          {dd:.1f}%")
        print(f"   Alpha pips:  {alpha:.4f}")

        # Sauvegarder en JSON
        out_path = results_dir / f"btcusd_metrics_{year}.json"
        metrics_serializable = {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in m.items()
        }
        out_path.write_text(json.dumps(metrics_serializable, indent=2, ensure_ascii=False))
        print(f"   → Sauvegarde: {out_path}")

    print(f"\n=== Résumé ===")
    print(f"Features actives: {len(X_cols)}")
    for year, t in trades.items():
        print(f"Trades BTCUSD {year}: {len(t)}")

    # Comparaison EURUSD
    print(f"\n=== Comparaison Cross-Actif ===")
    for year in [2024, 2025]:
        eurusd_path = results_dir / f"metrics_v1_{year}.json"
        btcusd_path = results_dir / f"btcusd_metrics_{year}.json"

        eurusd_sharpe = float("nan")
        btcusd_sharpe = float("nan")

        if eurusd_path.exists():
            eurusd_data = json.loads(eurusd_path.read_text())
            eurusd_sharpe = float(eurusd_data.get("sharpe", float("nan")))

        if btcusd_path.exists():
            btcusd_data = json.loads(btcusd_path.read_text())
            btcusd_sharpe = float(btcusd_data.get("sharpe", float("nan")))

        print(f"   {year}: EURUSD Sharpe = {eurusd_sharpe:.4f} | BTCUSD Sharpe = {btcusd_sharpe:.4f}")

    return metrics, trades, X_cols


if __name__ == "__main__":
    main()
