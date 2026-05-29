#!/usr/bin/env python3
"""Screening d'edge HONNÊTE multi-actifs — point d'entrée unique de la Phase 1.

Pour CHAQUE couple (actif, timeframe) découvert dans data/raw/ :
  1. construit un jeu de stratégies trend/momentum simples (peu de paramètres) ;
  2. sélectionne le meilleur candidat sur l'IN-SAMPLE (entrée < --oos-start) ;
  3. lit l'OUT-OF-SAMPLE UNE SEULE fois et applique les 5 critères de la
     constitution (Sharpe ≥ 1, DSR > 0 & p < 0.05, MaxDD < 15 %, WR > 30 %,
     ≥ 30 trades/an), avec n_trials dérivé du registre anti-snooping.

Le backtest est honnête : entrée à l'ouverture de la barre suivante (pas de
look-ahead), coûts XTB réels (spread+slippage+commission) et swap overnight.

USAGE (en local, après avoir peuplé data/raw/<ASSET>/<*>_<TF>.csv) :
    python scripts/screen_edge.py                       # tous les actifs, D1+H4
    python scripts/screen_edge.py --assets XAUUSD,BTCUSD,ETHUSD --timeframes D1
    python scripts/screen_edge.py --oos-start 2024-01-01 --out predictions/edge.csv

⚠️ Chaque exécution qui découvre de NOUVELLES hypothèses incrémente n_trials
(registre TEST_SET_LOCK.json) — c'est voulu (anti-data-snooping).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permet `python scripts/screen_edge.py` depuis la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.data.registry import discover_assets  # noqa: E402
from app.research.edge_harness import EdgeResult, screen_candidates  # noqa: E402
from app.strategies.donchian import DonchianBreakout  # noqa: E402
from app.strategies.dual_ma import DualMovingAverage  # noqa: E402
from app.strategies.sma_crossover import SmaCrossover  # noqa: E402
from app.strategies.ts_momentum import TsMomentum  # noqa: E402


def build_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Jeu de stratégies trend/momentum simples (peu de paramètres = robustes).

    Direction guidée par les preuves du projet : le trend-following marche mieux
    sur actifs tendanciels (crypto, or) ; on garde des périodes standard.
    """
    specs: dict[str, pd.Series] = {}
    for n in (20, 40, 55):
        specs[f"Donchian{n}"] = DonchianBreakout(N=n, M=n // 2).generate_signals(df)
    for fast, slow in ((10, 50), (20, 100), (50, 200)):
        specs[f"DualMA{fast}-{slow}"] = DualMovingAverage(fast=fast, slow=slow).generate_signals(df)
    for t in (20, 60, 120):
        specs[f"TsMom{t}"] = TsMomentum(T=t).generate_signals(df)
    for fast, slow in ((5, 20), (10, 50)):
        specs[f"SmaX{fast}-{slow}"] = SmaCrossover(fast=fast, slow=slow).generate_signals(df)
    return specs


def vol_tp_sl_grid(df: pd.DataFrame, cfg, oos_start: pd.Timestamp) -> list[tuple[float, float]]:
    """Grille TP/SL calée sur la volatilité IN-SAMPLE de l'actif (en pips).

    Unité = médiane du range journalier (High-Low) en pips sur l'in-sample.
    Rend les niveaux comparables d'un actif à l'autre (20 pips EURUSD ≠ 20 pts US30).
    """
    is_df = df[df.index < oos_start]
    if len(is_df) < 30:
        is_df = df
    unit = float(((is_df["High"] - is_df["Low"]) / cfg.pip_size).median())
    unit = max(unit, 1.0)
    return [
        (round(2.0 * unit, 1), round(1.0 * unit, 1)),
        (round(3.0 * unit, 1), round(1.5 * unit, 1)),
        (round(1.5 * unit, 1), round(1.0 * unit, 1)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Screening d'edge honnête multi-actifs.")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--assets", default="", help="Liste CSV (défaut : tous les découverts).")
    parser.add_argument("--timeframes", default="D1,H4")
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--out", default="predictions/edge_screen_results.csv", type=Path)
    args = parser.parse_args()

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    wanted_tfs = {tf.strip() for tf in args.timeframes.split(",") if tf.strip()}
    wanted_assets = {a.strip() for a in args.assets.split(",") if a.strip()}

    available = discover_assets(args.data_root)
    if not available:
        print(f"❌ Aucune donnée dans {args.data_root}/. "
              f"Peuple data/raw/<ASSET>/<*>_<TF>.csv puis relance.")
        return 1

    results: list[EdgeResult] = []
    for asset, tfs in available.items():
        if wanted_assets and asset not in wanted_assets:
            continue
        if asset not in ASSET_CONFIGS:
            print(f"⏭️  {asset} : pas de config coûts XTB (ASSET_CONFIGS) — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        for tf in tfs:
            if tf not in wanted_tfs:
                continue
            try:
                df = load_asset(asset, tf, data_root=args.data_root)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  {asset}/{tf} : chargement échoué ({exc}) — ignoré.")
                continue
            if len(df) < 200:
                print(f"⏭️  {asset}/{tf} : trop peu de barres ({len(df)}) — ignoré.")
                continue
            try:
                candidates = build_candidates(df)
                grid = vol_tp_sl_grid(df, cfg, oos_start)
                res = screen_candidates(
                    df, candidates, cfg,
                    asset=asset, timeframe=tf,
                    tp_sl_grid=grid, oos_start=oos_start, capital=args.capital,
                )
                results.append(res)
                print("  " + res.summary())
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  {asset}/{tf} : screening échoué ({exc}) — ignoré.")

    if not results:
        print("❌ Aucun résultat. Vérifie les actifs/timeframes et les données.")
        return 1

    # Tri : GO d'abord, puis par DSR décroissant.
    def _sort_key(r: EdgeResult) -> tuple[int, float]:
        dsr = r.oos_dsr if r.oos_dsr == r.oos_dsr else -1e9  # NaN -> très bas
        return (1 if r.go else 0, dsr)

    results.sort(key=_sort_key, reverse=True)

    rows = [{
        "asset": r.asset, "timeframe": r.timeframe, "candidat": r.label,
        "tp_pips": r.tp_pips, "sl_pips": r.sl_pips,
        "is_sharpe": round(r.is_sharpe, 3), "is_trades": r.is_trades,
        "oos_sharpe": round(r.oos_sharpe, 3), "oos_trades": r.oos_trades,
        "oos_wr": round(r.oos_win_rate, 3), "oos_dsr": round(r.oos_dsr, 3),
        "oos_p": round(r.oos_p_value, 4), "oos_max_dd": round(r.oos_max_dd_pct, 3),
        "n_trials": r.n_trials, "go": r.go, "reasons": " ; ".join(r.reasons),
    } for r in results]
    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    gos = [r for r in results if r.go]
    print("\n" + "=" * 70)
    print(f"Résultats : {len(results)} couples évalués, {len(gos)} GO.")
    print(f"CSV complet : {args.out}")
    if gos:
        print("\n🎯 Candidats GO (à valider en démo XTB avant tout argent réel) :")
        for r in gos:
            print("  " + r.summary())
    else:
        print("\nAucun edge n'a passé les 5 critères. C'est le résultat le plus "
              "probable et honnête — on itère sur de nouvelles familles de stratégies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
