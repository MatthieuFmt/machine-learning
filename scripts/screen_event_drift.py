#!/usr/bin/env python3
"""Effet de PRÉ-ANNONCE généralisé — l'edge pre-FOMC se généralise-t-il ?

Notre seul signal réel (pre-FOMC) est un cas d'« annonce premium » (Ai & Bansal
2018) : les actions monteraient avant les grandes annonces macro. On teste si la
même fenêtre (long de annonce−24h à annonce−1h) marche pour d'autres événements
programmés : emploi US (NFP) et inflation (CPI), avec FOMC en témoin.

Hypothèses PRÉ-ENREGISTRÉES (événements documentés), même discipline :
coûts XTB + swap, Sharpe annualisé routé par fréquence, DSR (Sharpe annualisé
honnête), test « bat-on une fenêtre 23h au hasard ? » (sépare l'effet du beta),
n_trials = nb de couples (événement × actif) testés dans ce batch.

USAGE :
    python scripts/screen_event_drift.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.strategies.pre_fomc_drift import (  # noqa: E402
    load_fomc_announcement_times,
    simulate_pre_fomc_trades,
)

# (label, nom exact dans le calendrier Forex Factory)
EVENTS = [
    ("FOMC (témoin)", "FOMC Statement"),
    ("NFP (emploi)", "Non-Farm Employment Change"),
    ("CPI (inflation)", "CPI m/m"),
]


def _equity_and_df(trades: list[dict], pip_value_eur: float, capital: float):
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times, pnl_eur = exit_times[order], pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    return equity, pd.DataFrame({"pnl": pnl_eur}, index=exit_times)


def _beats_random(df: pd.DataFrame, gross: np.ndarray, pip_size: float,
                  hours: int = 23, seed: int = 0) -> tuple[float, float, float]:
    c = df["Close"].to_numpy()
    allw = (c[hours:] - c[:-hours]) / pip_size
    mu_all, mu_ev, n = float(allw.mean()), float(gross.mean()), len(gross)
    t = (mu_ev - mu_all) / (gross.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    boot = np.array([allw[rng.integers(0, len(allw), n)].mean() for _ in range(5000)])
    return mu_all, float(t), float((boot < mu_ev).mean() * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description="Effet de pré-annonce généralisé.")
    parser.add_argument("--assets", default="US500,US30")
    parser.add_argument("--tf", default="H1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    # Nb total de couples testés (pour la pénalité DSR de data-snooping).
    n_trials = len(assets) * len(EVENTS)

    # Pré-charge les données (une fois par actif).
    data = {a: load_asset(a, args.tf, data_root=args.data_root)
            for a in assets if a in ASSET_CONFIGS}

    for label, event_name in EVENTS:
        try:
            times = load_fomc_announcement_times(
                args.data_root / "economic_calendar", 2010, 2026, event_name=event_name)
        except Exception as exc:  # noqa: BLE001
            print(f"⏭️  {label} : {exc}\n")
            continue
        print(f"━━━ {label} — {len(times)} annonces ({times.min().date()}→{times.max().date()}) ━━━")
        for asset in assets:
            if asset not in data:
                continue
            cfg = ASSET_CONFIGS[asset]
            df = data[asset]
            trades = simulate_pre_fomc_trades(
                df, times, cfg.total_cost_pips, 0.0, 0.0, cfg.pip_size,
                cfg.swap_long_pips_per_night, 24, 1)
            if len(trades) < 5:
                print(f"  {asset}: {len(trades)} trades — insuffisant.")
                continue
            equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
            ann = sharpe_daily_from_trades(trades)
            rep = validate_edge(equity, tdf, n_trials=n_trials, annualized_sharpe=ann)
            gross = np.array([t["pips_brut"] for t in trades])
            mu_all, t_stat, pct = _beats_random(df, gross, cfg.pip_size)
            print(f"  {asset}: Sharpe {ann:.2f}  DSR {rep.metrics['dsr']:.2f} "
                  f"(p={rep.metrics['p_value']:.2f})  WR {float((np.array([t['pips_net'] for t in trades])>0).mean()):.0%}  "
                  f"net€ {tdf['pnl'].sum():.0f}  | vs hasard +{gross.mean()-mu_all:.0f} pips "
                  f"(t={t_stat:.2f}, pct={pct:.0f}%)  | GO={rep.go}")
        print()

    print(f"(n_trials={n_trials} = {len(assets)} actifs × {len(EVENTS)} événements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
