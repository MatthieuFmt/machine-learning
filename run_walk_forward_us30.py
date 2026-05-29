"""H05 — Walk-Forward Paper Trading US30 D1.

Orchestrateur comparant deux configurations en walk-forward réaliste :
- Config A : Donchian(20,20) pur (baseline robuste)
- Config B : Donchian(20,20) + RF méta-labeling (réentraîné tous les 6 mois)

Protocole :
- Zéro look-ahead
- Slippage aléatoire reproductible (seed=42)
- Réentraînement strict aux dates spécifiées
- Un seul run, pas d'ajustement post-hoc
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ajouter le répertoire racine au PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.meta_labeling import compute_meta_labels
from app.backtest.walk_forward import run_walk_forward
from app.pipelines.us30 import Us30Pipeline
from app.strategies.donchian import DonchianBreakout

# ── Configuration figée ex ante (H03 + H04 + H05) ──
TP_PIPS = 200.0
SL_PIPS = 100.0
WINDOW_HOURS = 120
COMMISSION_PIPS = 3.0
SLIPPAGE_PIPS = 5.0
SLIPPAGE_RANDOM = 2.0
SPREAD_PIPS = 2.0
PIP_SIZE = 1.0  # US30

RF_META_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

THRESHOLDS: list[float] = [0.45, 0.50, 0.55, 0.60]

RETRAIN_DATES: list[str] = [
    "2023-07-01",
    "2024-01-01",
    "2024-07-01",
    "2025-01-01",
]
INITIAL_TRAIN_END = "2022-12-31"


def main() -> int:
    """Exécute le walk-forward complet et affiche la comparaison."""
    # ── 1. Chargement des données ──
    pipeline = Us30Pipeline()
    data = pipeline.load_data()
    d1 = data["us30_d1"].sort_index()
    print(f"US30 D1 charge : {len(d1)} barres, "
          f"{d1.index.min().date()} -> {d1.index.max().date()}")

    # ── 2. Signaux Donchian(20,20) ──
    donchian = DonchianBreakout(N=20, M=20)
    all_signals = donchian.generate_signals(d1)
    n_long = int((all_signals == 1).sum())
    n_short = int((all_signals == -1).sum())
    print(f"Signaux Donchian(20,20) : {n_long} LONG, {n_short} SHORT, "
          f"{n_long + n_short} total")

    # ── 3. Features ──
    ml_data = pipeline.build_features(data)
    # ml_data contient OHLC, Target, RSI_14, ADX_14, Dist_SMA50, Dist_SMA200,
    #   ATR_Norm, Log_Return_5d, (+-Volume_Ratio)

    # Ajouter Donchian_Position
    ml_data["Donchian_Position"] = (
        all_signals.reindex(ml_data.index).fillna(0).astype(int)
    )

    feature_cols = [
        "RSI_14", "ADX_14", "Dist_SMA50", "Dist_SMA200",
        "ATR_Norm", "Log_Return_5d", "Donchian_Position",
    ]
    if "Volume_Ratio" in ml_data.columns:
        feature_cols.append("Volume_Ratio")

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"ml_data : {len(ml_data)} lignes après dropna")

    # ── 4. Méta-labels sur tout le dataset ──
    meta_labels_all = compute_meta_labels(
        df=d1,
        donchian_signals=all_signals,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        pip_size=PIP_SIZE,
    )

    # Aligner sur ml_data (qui a perdu des lignes avec dropna dans build_features)
    common_idx = ml_data.index.intersection(meta_labels_all.index)
    ml_data = ml_data.loc[common_idx]
    meta_labels_all = meta_labels_all.loc[common_idx]
    all_signals = all_signals.loc[common_idx]

    n_win = int((meta_labels_all == 1).sum())
    n_loss = int((meta_labels_all == 0).sum())
    print(f"Méta-labels : {n_win} gagnants, {n_loss} perdants "
          f"(sur {len(common_idx)} barres alignées)")

    # ── 5. Walk-forward Config A : Donchian pur ──
    print("\n--- Config A : Donchian(20,20) pur ---")
    wf_baseline = run_walk_forward(
        df=d1,
        donchian_signals=all_signals,
        features=ml_data[feature_cols],
        meta_labels=meta_labels_all,
        retrain_dates=RETRAIN_DATES,
        initial_train_end=INITIAL_TRAIN_END,
        use_rf=False,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        commission_pips=COMMISSION_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        slippage_random=SLIPPAGE_RANDOM,
        spread_pips=SPREAD_PIPS,
        pip_size=PIP_SIZE,
    )

    print(f"  Sharpe WF: {wf_baseline['sharpe']:.3f}  "
          f"WR: {wf_baseline['wr']:.1%}  "
          f"Trades: {wf_baseline['trades']}  "
          f"PnL: {wf_baseline['pnl_pips']:+.1f} pips")

    # ── 6. Walk-forward Config B : Donchian + RF méta-labeling ──
    print("\n--- Config B : Donchian(20,20) + RF méta-labeling ---")
    wf_meta = run_walk_forward(
        df=d1,
        donchian_signals=all_signals,
        features=ml_data[feature_cols],
        meta_labels=meta_labels_all,
        retrain_dates=RETRAIN_DATES,
        initial_train_end=INITIAL_TRAIN_END,
        use_rf=True,
        rf_params=RF_META_PARAMS,
        thresholds=THRESHOLDS,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        commission_pips=COMMISSION_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        slippage_random=SLIPPAGE_RANDOM,
        spread_pips=SPREAD_PIPS,
        pip_size=PIP_SIZE,
    )

    print(f"  Sharpe WF: {wf_meta['sharpe']:.3f}  "
          f"WR: {wf_meta['wr']:.1%}  "
          f"Trades: {wf_meta['trades']}  "
          f"PnL: {wf_meta['pnl_pips']:+.1f} pips")

    # ── 7. Rapport comparatif ──
    print()
    print("=" * 60)
    print("=== WALK-FORWARD US30 D1 ===")
    print(f"{'':<22} {'Sharpe WF':>10}   {'WR':>7}   {'Trades':>6}   {'PnL (pips)':>12}")
    print(f"{'Donchian pur':<22} {wf_baseline['sharpe']:>10.3f}   "
          f"{wf_baseline['wr']:>6.1%}   {wf_baseline['trades']:>6}   "
          f"{wf_baseline['pnl_pips']:>+11.1f}")
    print(f"{'Donchian + RF':<22} {wf_meta['sharpe']:>10.3f}   "
          f"{wf_meta['wr']:>6.1%}   {wf_meta['trades']:>6}   "
          f"{wf_meta['pnl_pips']:>+11.1f}")

    # Détail par segment (5 segments: 2023H1, 2023H2, 2024S1, 2024S2, 2025S1)
    print()
    print("Par segment :")
    seg_details_a = wf_baseline["segment_details"]
    seg_details_b = wf_meta["segment_details"]
    seg_labels_raw = ["2023H1", "2023H2", "2024S1", "2024S2", "2025S1"]
    for i in range(len(seg_details_a)):
        seg_a = seg_details_a[i]
        seg_b = seg_details_b[i]
        label = seg_labels_raw[i] if i < len(seg_labels_raw) else seg_a["segment"]
        print(f"  {label}:  Donchian pur={seg_a['sharpe']:.3f} / "
              f"RF={seg_b['sharpe']:.3f}")

    # ── 8. Décision GO / NO-GO ──
    sharpe_a = wf_baseline["sharpe"]
    sharpe_b = wf_meta["sharpe"]

    if sharpe_a <= 0 and sharpe_b <= 0:
        verdict = "NO-GO — Retour à la planche à dessin."
        deploy = "Aucune"
    elif sharpe_b > sharpe_a + 0.5:
        verdict = f"GO — Config B (RF) déploie Sharpe WF={sharpe_b:.3f} > {sharpe_a:.3f} (écart > 0.5)"
        deploy = "Config B (Donchian + RF méta-labeling)"
    elif sharpe_b > sharpe_a:
        verdict = (f"GO - Config B meilleure mais ecart < 0.5 -> "
                   f"deploiement Config A (plus simple, plus robuste)")
        deploy = "Config A (Donchian pur)"
    elif sharpe_a > 0:
        verdict = f"GO — Config A (Donchian pur) déploie Sharpe WF={sharpe_a:.3f}"
        deploy = "Config A (Donchian pur)"
    else:
        verdict = "NO-GO — Aucune configuration positive"
        deploy = "Aucune"

    print()
    print(f">>> Déploiement: {deploy}")
    print(f">>> Verdict: {verdict}")
    print("=" * 60)

    # ── 9. Sauvegarde ──
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Nettoyer pour JSON (pd.Series -> list, etc.)
    def _safe_equity(series: pd.Series) -> list[dict[str, Any]]:
        if series.empty:
            return []
        return [
            {"time": str(idx), "equity": float(val)}
            for idx, val in series.items()
        ]

    report: dict[str, Any] = {
        "hypothesis": "v2_05",
        "instrument": "USA30IDXUSD",
        "primary_tf": "D1",
        "strategy": "Donchian(20,20) ± RF méta-labeling",
        "walk_forward_config": {
            "retrain_dates": RETRAIN_DATES,
            "initial_train_end": INITIAL_TRAIN_END,
            "tp_pips": TP_PIPS,
            "sl_pips": SL_PIPS,
            "window_hours": WINDOW_HOURS,
            "commission_pips": COMMISSION_PIPS,
            "slippage_pips": SLIPPAGE_PIPS,
            "slippage_random": SLIPPAGE_RANDOM,
            "spread_pips": SPREAD_PIPS,
            "pip_size": PIP_SIZE,
        },
        "rf_params": {k: v for k, v in RF_META_PARAMS.items() if k != "n_jobs"},
        "thresholds": THRESHOLDS,
        "config_a_baseline": {
            "sharpe": sharpe_a,
            "wr": wf_baseline["wr"],
            "trades": wf_baseline["trades"],
            "pnl_pips": wf_baseline["pnl_pips"],
            "segment_details": wf_baseline["segment_details"],
        },
        "config_b_meta_rf": {
            "sharpe": sharpe_b,
            "wr": wf_meta["wr"],
            "trades": wf_meta["trades"],
            "pnl_pips": wf_meta["pnl_pips"],
            "segment_details": wf_meta["segment_details"],
        },
        "decision": {
            "deployment": deploy,
            "verdict": verdict,
            "best_config": "A" if sharpe_a >= sharpe_b else "B",
        },
        "n_trials_cumulatif_v2": 5,
    }

    output_path = predictions_dir / "walk_forward_us30_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nRapport sauvegardé : {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
