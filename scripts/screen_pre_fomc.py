#!/usr/bin/env python3
"""Test HONNÊTE du Pre-FOMC Drift (Lucca & Moench 2015) — source d'edge différente.

Hypothèse PRÉ-ENREGISTRÉE (publiée, donc pas data-minée par nous) : l'indice US
monte dans les ~24 h précédant la décision de taux du FOMC. Stratégie sans
paramètre libre : long à FOMC−24h, sortie à FOMC−1h, toujours long.

Discipline appliquée :
  - Coûts XTB réels (spread+slippage round-trip) + 1 nuit de swap (cfg).
  - Sharpe annualisé routé par fréquence (sharpe_daily_from_trades).
  - DSR avec le Sharpe PAR-PÉRIODE (fix 2026-05-29) — n_trials petit car
    hypothèse pré-enregistrée (≠ recherche technique massive).
  - Comme la stratégie n'a AUCUN paramètre à ajuster, on l'évalue sur tout
    l'échantillon (≈ 8 trades/an × ~16 ans ≈ 128 trades → DSR calculable),
    PLUS un découpage temporel pré/post-2015 pour détecter une décroissance
    de l'anomalie après sa publication.

USAGE :
    python scripts/screen_pre_fomc.py
    python scripts/screen_pre_fomc.py --assets US500,US30 --tf H1
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


def _equity_and_df(trades: list[dict], pip_value_eur: float, capital: float):
    """(equity €, trades_df['pnl']) triés par heure de sortie — pour validate_edge."""
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times, pnl_eur = exit_times[order], pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    trades_df = pd.DataFrame({"pnl": pnl_eur}, index=exit_times)
    return equity, trades_df


def _summarize(trades: list[dict], label: str) -> dict:
    """Statistiques descriptives d'un lot de trades (long pre-FOMC)."""
    if not trades:
        return {"label": label, "n": 0}
    pips = np.array([t["pips_net"] for t in trades], dtype=float)
    wr = float((pips > 0).mean())
    return {
        "label": label,
        "n": len(trades),
        "avg_pips": float(pips.mean()),
        "median_pips": float(np.median(pips)),
        "wr": wr,
        "sum_pips": float(pips.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête Pre-FOMC drift.")
    parser.add_argument("--assets", default="US500,US30")
    parser.add_argument("--tf", default="H1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--split-year", default=2015, type=int,
                        help="Année de césure pour le test de décroissance post-publication.")
    args = parser.parse_args()

    fomc = load_fomc_announcement_times(args.data_root / "economic_calendar",
                                        start_year=2010, end_year=2026)
    print(f"FOMC chargés : {len(fomc)} annonces, {fomc.min().date()} → {fomc.max().date()}\n")

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

        trades = simulate_pre_fomc_trades(
            df=df,
            fomc_times=fomc,
            spread_pips=cfg.total_cost_pips,   # coût round-trip XTB complet
            slippage_pips=0.0,
            commission_pips=0.0,
            pip_size=cfg.pip_size,
            swap_long_pips_per_night=cfg.swap_long_pips_per_night,
            hours_before_entry=24,
            hours_before_exit=1,
        )
        if len(trades) < 2:
            print(f"{asset}/{args.tf} : {len(trades)} trade(s) — insuffisant.\n")
            continue

        # Découpage temporel pré/post-publication (descriptif, PAS de sélection).
        split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
        pre = [t for t in trades if pd.Timestamp(t["entry_time"]) < split]
        post = [t for t in trades if pd.Timestamp(t["entry_time"]) >= split]

        # Verdict honnête sur tout l'échantillon (hypothèse pré-enregistrée).
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        ann_sharpe = sharpe_daily_from_trades(trades)
        # n_trials=2 : on teste 2 actifs (US500, US30) sur cette hypothèse unique.
        report = validate_edge(equity, tdf, n_trials=len(assets), annualized_sharpe=ann_sharpe)

        full = _summarize(trades, "TOUT")
        pre_s = _summarize(pre, f"<{args.split_year}")
        post_s = _summarize(post, f">={args.split_year}")

        print(f"══ {asset}/{args.tf} ══ ({full['n']} trades, "
              f"{full['n'] / max((fomc.max() - fomc.min()).days / 365.25, 1):.1f}/an)")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
              f"MaxDD : {report.metrics['max_dd']:.1%}")
        print(f"  WR : {full['wr']:.0%}   pips moy/trade : {full['avg_pips']:.1f}   "
              f"gain net € : {tdf['pnl'].sum():.0f}")
        print(f"  GO : {report.go}")
        if report.reasons:
            print(f"    raisons NO-GO : {' ; '.join(report.reasons)}")
        # Décroissance post-publication
        print(f"  Décroissance (pips moy/trade) : "
              f"{pre_s.get('avg_pips', float('nan')):.1f} ({pre_s['n']} tr) avant {args.split_year}  →  "
              f"{post_s.get('avg_pips', float('nan')):.1f} ({post_s['n']} tr) depuis")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
