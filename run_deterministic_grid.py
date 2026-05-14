"""Orchestrateur H03 — Backtest systématique multi-actif, stratégies déterministes.

Règles:
- Split unique : train ≤ 2022, val = 2023, test ≥ 2024.
- Sélection meilleur paramétrage sur train uniquement.
- Arrêt au premier GO (Sharpe val > 0 ET Sharpe test > 0, Règle 4).
- Sauvegarde rapport complet dans predictions/deterministic_grid_results.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from learning_machine_learning_v2.strategies import ALL_STRATEGIES
from learning_machine_learning_v2.backtest.grid_search import grid_search_asset

# ── Configuration des actifs (spec H03 §1.4) ────────────────────────────
ASSETS: list[dict[str, Any]] = [
    {
        "name": "XAUUSD",
        "tf": "H4",
        "csv": "cleaned-data/XAUUSD_H4_cleaned.csv",
        "tp_pips": 300,
        "sl_pips": 150,
        "window_hours": 96,
        "commission_pips": 25,
        "slippage_pips": 10,
        "pip_size": 1.0,
    },
    {
        "name": "USA30IDXUSD",
        "tf": "D1",
        "csv": "cleaned-data/USA30IDXUSD_D1_cleaned.csv",
        "tp_pips": 200,
        "sl_pips": 100,
        "window_hours": 120,
        "commission_pips": 3,
        "slippage_pips": 5,
        "pip_size": 1.0,
    },
    {
        "name": "EURUSD",
        "tf": "H1",
        "csv": "cleaned-data/EURUSD_H1_cleaned.csv",
        "tp_pips": 30,
        "sl_pips": 10,
        "window_hours": 24,
        "commission_pips": 1.5,
        "slippage_pips": 1,
        "pip_size": 0.0001,
    },
    {
        "name": "BTCUSD",
        "tf": "H1",
        "csv": "cleaned-data/BTCUSD_H1_cleaned.csv",
        "tp_pips": 30,
        "sl_pips": 10,
        "window_hours": 24,
        "commission_pips": 10,
        "slippage_pips": 5,
        "pip_size": 1.0,
    },
]

TRAIN_END = "2023-01-01"
VAL_END = "2024-01-01"


def load_and_split(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge un CSV et le split en train/val/test."""
    df = pd.read_csv(
        csv_path,
        parse_dates=["Time"],
        index_col="Time",
    ).sort_index()

    train = df[df.index < TRAIN_END].copy()
    val = df[(df.index >= TRAIN_END) & (df.index < VAL_END)].copy()
    test = df[df.index >= VAL_END].copy()

    return train, val, test


def main() -> None:
    print("=" * 80)
    print("V2 H03 -- Backtest deterministe multi-actif")
    print(f"Demarrage : {datetime.now().isoformat()}")
    print(f"Split : train < {TRAIN_END}, val = [{TRAIN_END}, {VAL_END}[, test >= {VAL_END}")
    print(f"{len(ASSETS)} actifs x {len(ALL_STRATEGIES)} strategies = "
          f"{len(ASSETS) * sum(len(p[1].keys()) for p in ALL_STRATEGIES)} combinaisons max")
    print("=" * 80)

    all_reports: list[dict[str, Any]] = []
    go_found = False
    go_asset = ""
    go_strategy = ""

    for asset in ASSETS:
        if go_found:
            print(f"\n[STOP] Regle 4 : GO trouve sur {go_asset}/{go_strategy}, arret.")
            break

        csv_path = asset["csv"]
        if not Path(csv_path).exists():
            print(f"\n[SKIP] {asset['name']} {asset['tf']} : CSV absent ({csv_path}), skip.")
            continue

        print(f"\n{'-' * 80}")
        print(f"[ACTIF] {asset['name']} {asset['tf']}")
        print(f"   TP={asset['tp_pips']} SL={asset['sl_pips']} "
              f"Window={asset['window_hours']}h "
              f"Comm={asset['commission_pips']} Slip={asset['slippage_pips']} "
              f"PipSize={asset['pip_size']}")

        try:
            train, val, test = load_and_split(csv_path)
        except Exception as exc:
            print(f"   [ERREUR] Chargement CSV : {exc}")
            continue

        print(f"   Train: {len(train)} barres ({train.index.min()} -> {train.index.max()})")
        print(f"   Val:   {len(val)} barres ({val.index.min()} -> {val.index.max()})")
        print(f"   Test:  {len(test)} barres ({test.index.min()} -> {test.index.max()})")

        for strategy_class, param_grid in ALL_STRATEGIES:
            if go_found:
                break

            strat_name = strategy_class.__name__
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)

            print(f"\n   [STRAT] {strat_name} ({n_combos} combinaisons)...")

            try:
                result = grid_search_asset(
                    df_train=train,
                    df_val=val,
                    df_test=test,
                    strategy_class=strategy_class,
                    param_grid=param_grid,
                    tp_pips=asset["tp_pips"],
                    sl_pips=asset["sl_pips"],
                    window_hours=asset["window_hours"],
                    commission_pips=asset["commission_pips"],
                    slippage_pips=asset["slippage_pips"],
                    pip_size=asset["pip_size"],
                )
            except Exception as exc:
                print(f"      [ERREUR] Grid search : {exc}")
                import traceback
                traceback.print_exc()
                continue

            sharpe_train = result.get("sharpe_train", 0.0)
            sharpe_val = result.get("sharpe_val", 0.0)
            sharpe_test = result.get("sharpe_test", 0.0)
            wr_train = result.get("wr_train", 0.0)
            wr_val = result.get("wr_val", 0.0)
            wr_test = result.get("wr_test", 0.0)

            print(f"      Best params: {result.get('best_params', {})}")
            print(f"      Sharpe: train={sharpe_train:+.3f}  "
                  f"val={sharpe_val:+.3f}  test={sharpe_test:+.3f}")
            print(f"      WR:     train={wr_train:.1%}  "
                  f"val={wr_val:.1%}  test={wr_test:.1%}")

            report_entry = {
                "asset": asset["name"],
                "tf": asset["tf"],
                "strategy": strat_name,
                **result,
            }
            all_reports.append(report_entry)

            # Règle 4 : GO si Sharpe val > 0 ET Sharpe test > 0
            if sharpe_val > 0 and sharpe_test > 0:
                print(f"\n   >>> [GO] {strat_name} sur {asset['name']} {asset['tf']} !")
                print(f"   >>> Sharpe val={sharpe_val:+.3f}, Sharpe test={sharpe_test:+.3f}")
                go_found = True
                go_asset = f"{asset['name']} {asset['tf']}"
                go_strategy = strat_name

    # ── Sauvegarde du rapport ─────────────────────────────────────────────
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "deterministic_grid_results.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_assets_tested": len(ASSETS),
        "n_strategies": len(ALL_STRATEGIES),
        "split": {"train_end": TRAIN_END, "val_end": VAL_END},
        "go_found": go_found,
        "go_asset": go_asset if go_found else None,
        "go_strategy": go_strategy if go_found else None,
        "results": all_reports,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 80}")
    if go_found:
        print(f"[GO] {go_strategy} sur {go_asset}")
    else:
        print(f"[NO-GO] Aucune strategie n'a passe Sharpe val > 0 ET Sharpe test > 0")
    print(f"Rapport sauvegardé : {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
