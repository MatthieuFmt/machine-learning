#!/usr/bin/env python3
"""Test HONNÊTE du Gap Fade (overnight→intraday reversal) — Phase 1.

On parie que le gap d'ouverture se referme dans la journée. Entrée à l'open,
sortie à la clôture (flat la nuit → zéro swap). On ne trade que si le gap dépasse
le coût aller-retour.

⚠️ NÉCESSITE des données intraday (M5) locales pour une séance précise :
    python scripts/download_orb_data.py --asset US500 --tf M5 --start 2015 --end 2026

Discipline : signal connu à l'open (gap = open − close veille), exécution
open→close, coûts XTB réels, Sharpe annualisé routé par fréquence →
validate_edge + DSR. Paramètres FIGÉS, n_trials = nombre d'indices.

⚠️ Effet documenté mais réputé AFFAIBLI — on mesure ce qu'il en reste.

GO ssi : Sharpe ≥ 1 · DSR > 0 (p<0.05) · MaxDD < 15 % · WR > 30 % · ≥ 30 tr/an.

USAGE :
    python scripts/screen_gap_fade.py
    python scripts/screen_gap_fade.py --assets US500 --tf M5 --min-gap-mult 1.0
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
from app.strategies.gap_fade import simulate_gap_fade  # noqa: E402

# Séances cash (figées) : (fuseau, ouverture locale, clôture locale).
SESSIONS: dict[str, tuple[str, str, str]] = {
    "US500": ("America/New_York", "09:30", "16:00"),
    "US30": ("America/New_York", "09:30", "16:00"),
    "GER30": ("Europe/Berlin", "09:00", "17:30"),
}


def _equity_and_df(
    trades: list[dict], pip_value_eur: float, capital: float
) -> tuple[pd.Series, pd.DataFrame]:
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times, pnl_eur = exit_times[order], pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    trades_df = pd.DataFrame({"pnl": pnl_eur}, index=exit_times)
    return equity, trades_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête Gap Fade.")
    parser.add_argument("--assets", default="US500")
    parser.add_argument("--tf", default="M5")
    parser.add_argument("--min-gap-mult", default=1.0, type=float,
                        help="Seuil de gap minimal en multiples du coût a/r.")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--split-year", default=2020, type=int)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    n_trials = len(assets)

    print("=" * 72)
    print(f"GAP FADE (honnête) — {args.tf}, intraday flat la nuit")
    print(f"seuil gap ≥ {args.min_gap_mult:g}× coût a/r   Indices : {', '.join(assets)}   "
          f"n_trials={n_trials}")
    print("=" * 72)

    any_go = False
    for asset in assets:
        if asset not in ASSET_CONFIGS or asset not in SESSIONS:
            print(f"\n⏭️  {asset} : config/séance manquante — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        tz, open_t, close_t = SESSIONS[asset]
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            print(f"     → télécharge d'abord : python scripts/download_orb_data.py "
                  f"--asset {asset} --tf {args.tf}")
            continue

        trades = simulate_gap_fade(
            df, cfg, session_tz=tz, open_time=open_t, close_time=close_t,
            min_gap_cost_mult=args.min_gap_mult,
        )
        if len(trades) < 2:
            print(f"\n══ {asset}/{args.tf} ══ {len(trades)} trade(s) — insuffisant.")
            continue

        years = max((df.index.max() - df.index.min()).days / 365.25, 1e-3)
        tpy = len(trades) / years
        ann_sharpe = sharpe_daily_from_trades(trades)
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        report = validate_edge(
            equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe
        )
        any_go = any_go or report.go

        n_long = sum(1 for t in trades if t["signal"] == 1)
        avg_gap = float(np.mean([abs(t["gap_pips"]) for t in trades]))

        print(f"\n══ {asset}/{args.tf} ══ ({len(trades)} trades, {tpy:.0f}/an, "
              f"{df.index.min().date()}→{df.index.max().date()})")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})")
        print(f"  MaxDD : {report.metrics['max_dd']:.1%}   WR : {report.metrics['wr']:.0%}   "
              f"trades/an : {report.metrics['trades_per_year']:.1f}")
        print(f"  PnL net : {tdf['pnl'].sum():+.0f} €   long/short : {n_long}/{len(trades) - n_long}   "
              f"gap moyen : {avg_gap:.0f} pts")
        print(f"  ==> {'✅ GO' if report.go else '❌ NO-GO'}")
        if report.reasons:
            print(f"      raisons : {' ; '.join(report.reasons)}")

        split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
        pre = [t for t in trades if pd.Timestamp(t["entry_time"]) < split]
        post = [t for t in trades if pd.Timestamp(t["entry_time"]) >= split]
        sr_pre = sharpe_daily_from_trades(pre) if len(pre) >= 2 else float("nan")
        sr_post = sharpe_daily_from_trades(post) if len(post) >= 2 else float("nan")
        print(f"  Stabilité : Sharpe avant {args.split_year} = {sr_pre:.2f} ({len(pre)} tr)"
              f"  →  depuis = {sr_post:.2f} ({len(post)} tr)")

    print("\n" + "=" * 72)
    print(f"VERDICT GLOBAL : {'✅ au moins un indice GO' if any_go else '❌ aucun indice GO'}")
    return 0 if any_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
