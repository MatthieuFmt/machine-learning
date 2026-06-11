#!/usr/bin/env python3
"""Test HONNÊTE de l'ORB en données FINES (M5) — Phase 1, option B (vraie résolution).

L'ORB en H1 était trop grossier (NO-GO plat, US500 Sharpe +0.24). Ici on teste
l'effet à sa résolution documentée : opening range = les `--or-minutes` premières
minutes de séance, sur données M5.

⚠️ NÉCESSITE des données M5 locales (le cloud ne télécharge pas) :
    python scripts/download_orb_data.py --asset US500 --tf M5 --start 2015 --end 2026

Discipline identique aux autres screens : entrée à l'open de la barre suivante,
stop = côté opposé de l'OR, sortie EOD (flat la nuit → 0 swap), coûts réels,
Sharpe annualisé routé par fréquence → validate_edge + DSR. Paramètres FIGÉS.

⚠️ n_trials : chaque (--or-minutes × indice) testé = +1 essai, compté
   AUTOMATIQUEMENT dans le registre anti-snooping (TEST_SET_LOCK.json).
   Pré-enregistrer UNE config (défaut : OR=5 min, US500) ; explorer plusieurs
   = data-snooping (la pénalité DSR grossit d'elle-même).

USAGE :
    python scripts/screen_orb_fine.py
    python scripts/screen_orb_fine.py --assets US500 --tf M5 --or-minutes 5
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
from app.strategies.opening_range import simulate_orb_session  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Test honnête ORB fin (M5).")
    parser.add_argument("--assets", default="US500")
    parser.add_argument("--tf", default="M5")
    parser.add_argument("--or-minutes", default=5, type=int)
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--split-year", default=2020, type=int)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    print("=" * 72)
    print(f"ORB FIN (honnête) — {args.tf}, opening range = {args.or_minutes} min")
    print(f"Indices : {', '.join(assets)}   n_trials : registre anti-snooping")
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

        trades = simulate_orb_session(
            df, cfg, session_tz=tz, open_time=open_t,
            or_minutes=args.or_minutes, close_time=close_t,
        )
        if len(trades) < 2:
            print(f"\n══ {asset}/{args.tf} ══ {len(trades)} trade(s) — insuffisant.")
            continue

        years = max((df.index.max() - df.index.min()).days / 365.25, 1e-3)
        tpy = len(trades) / years
        ann_sharpe = sharpe_daily_from_trades(trades)
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        n_trials = record_and_resolve_n_trials(
            prompt="screen_orb_fine",
            hypothesis=f"{asset}/{args.tf}:ORB{args.or_minutes}min",
            sharpe=ann_sharpe,
            n_trades=len(trades),
        )
        report = validate_edge(
            equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe
        )
        any_go = any_go or report.go

        n_stop = sum(1 for t in trades if t["exit_reason"] == "stop")
        n_eod = sum(1 for t in trades if t["exit_reason"] == "eod")
        n_long = sum(1 for t in trades if t["signal"] == 1)
        max_nights = max(t["nights_held"] for t in trades)

        print(f"\n══ {asset}/{args.tf} ══ ({len(trades)} trades, {tpy:.0f}/an, OR={args.or_minutes}min, "
              f"{df.index.min().date()}→{df.index.max().date()})")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
              f"[n_trials={n_trials}]")
        print(f"  Preuves primaires : t/trade = {report.metrics['t_stat']:.2f} "
              f"(p={report.metrics['p_t']:.3f})   p_bootstrap = "
              f"{report.metrics['p_bootstrap']:.3f}")
        print(f"  MaxDD : {report.metrics['max_dd']:.1%}   WR : {report.metrics['wr']:.0%}   "
              f"trades/an : {report.metrics['trades_per_year']:.1f}")
        print(f"  PnL net : {tdf['pnl'].sum():+.0f} €   long/short : {n_long}/{len(trades) - n_long}   "
              f"sorties : {n_stop} stop / {n_eod} EOD   max nuits : {max_nights}")
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
