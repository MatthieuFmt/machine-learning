#!/usr/bin/env python3
"""Test HONNÊTE du Turn-of-Month (Lakonishok & Smidt 1988) — effet calendrier.

Hypothèse PRÉ-ENREGISTRÉE : les indices actions montent autour du changement de
mois. Stratégie sans paramètre libre : long au close du dernier jour de bourse
du mois, sortie au close du 3e jour de bourse suivant (fenêtre canonique −1/+3).

Même discipline que screen_pre_fomc.py :
  - coûts XTB round-trip + swap × nuits ;
  - Sharpe annualisé routé par fréquence ; DSR canonique par-période (fix 2026-06-09) ;
  - hypothèse pré-enregistrée → n_trials = cumul du registre anti-snooping ;
  - test « bat-on une fenêtre de même durée au hasard ? » (sépare l'effet du
    simple beta haussier) ; découpage pré/post pour la stabilité.

USAGE :
    python scripts/screen_turn_of_month.py
    python scripts/screen_turn_of_month.py --assets US500,US30,GER30,XAUUSD --hold-days 3
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
from app.research.edge_harness import record_and_resolve_n_trials  # noqa: E402
from app.strategies.turn_of_month import simulate_turn_of_month_trades  # noqa: E402


def _equity_and_df(trades: list[dict], pip_value_eur: float, capital: float):
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times, pnl_eur = exit_times[order], pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    trades_df = pd.DataFrame({"pnl": pnl_eur}, index=exit_times)
    return equity, trades_df


def _beta_benchmark(df: pd.DataFrame, tom_gross: np.ndarray, pip_size: float,
                    hold_days: int, seed: int = 0) -> tuple[float, float, float]:
    """Compare la moyenne TOM (pips bruts) à des fenêtres de même durée au hasard.

    Returns: (moy_hasard, t_stat, percentile) — percentile = % de tirages
    bootstrap dont la moyenne est < moyenne TOM.
    """
    c = df["Close"].to_numpy()
    allw = (c[hold_days:] - c[:-hold_days]) / pip_size
    mu_all = float(allw.mean())
    mu_tom, n = float(tom_gross.mean()), len(tom_gross)
    t = (mu_tom - mu_all) / (tom_gross.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    boot = np.array([allw[rng.integers(0, len(allw), n)].mean() for _ in range(5000)])
    pct = float((boot < mu_tom).mean() * 100)
    return mu_all, float(t), pct


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête Turn-of-Month.")
    parser.add_argument("--assets", default="US500,US30,GER30,XAUUSD")
    parser.add_argument("--tf", default="D1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--hold-days", default=3, type=int)
    parser.add_argument("--split-year", default=2018, type=int)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"⏭️  {asset} : pas de config coûts XTB — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue

        trades = simulate_turn_of_month_trades(
            df=df,
            spread_pips=cfg.total_cost_pips,
            slippage_pips=0.0,
            commission_pips=0.0,
            pip_size=cfg.pip_size,
            swap_long_pips_per_night=cfg.swap_long_pips_per_night,
            hold_days=args.hold_days,
        )
        if len(trades) < 2:
            print(f"{asset}/{args.tf} : {len(trades)} trade(s) — insuffisant.\n")
            continue

        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        ann_sharpe = sharpe_daily_from_trades(trades)
        n_trials = record_and_resolve_n_trials(
            prompt="screen_turn_of_month",
            hypothesis=f"{asset}/{args.tf}:tom_hold{args.hold_days}",
            sharpe=ann_sharpe,
            n_trades=len(trades),
        )
        report = validate_edge(equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe)

        gross = np.array([t["pips_brut"] for t in trades])
        mu_all, t_stat, pct = _beta_benchmark(df, gross, cfg.pip_size, args.hold_days)

        span_years = max((df.index[-1] - df.index[0]).days / 365.25, 1)
        wr = float((np.array([t["pips_net"] for t in trades]) > 0).mean())

        split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
        pre = [t["pips_net"] for t in trades if pd.Timestamp(t["entry_time"]) < split]
        post = [t["pips_net"] for t in trades if pd.Timestamp(t["entry_time"]) >= split]

        print(f"══ {asset}/{args.tf} ══ ({len(trades)} trades, {len(trades)/span_years:.1f}/an)")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
              f"MaxDD : {report.metrics['max_dd']:.1%}   [n_trials={n_trials}]")
        print(f"  Preuves primaires : t/trade = {report.metrics['t_stat']:.2f} "
              f"(p={report.metrics['p_t']:.3f})   p_bootstrap = "
              f"{report.metrics['p_bootstrap']:.3f}")
        print(f"  WR : {wr:.0%}   pips moy/trade : {gross.mean():.1f} brut / "
              f"{np.mean([t['pips_net'] for t in trades]):.1f} net   "
              f"gain net € : {tdf['pnl'].sum():.0f}")
        print(f"  GO : {report.go}"
              + (f"   (NO-GO : {' ; '.join(report.reasons)})" if report.reasons else ""))
        print(f"  vs hasard : TOM {gross.mean():.1f} pips vs {mu_all:.1f} au hasard  "
              f"→ surplus {gross.mean()-mu_all:+.1f}  | t={t_stat:.2f}  | pct={pct:.0f}%")
        print(f"  stabilité (net moy/trade) : "
              f"{np.mean(pre) if pre else float('nan'):.1f} ({len(pre)} tr avant {args.split_year})  →  "
              f"{np.mean(post) if post else float('nan'):.1f} ({len(post)} tr depuis)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
