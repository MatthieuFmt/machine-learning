#!/usr/bin/env python3
"""Test HONNÊTE de l'Opening Range Breakout (ORB) sur indices — Phase 1, option B.

Hypothèse PRÉ-ENREGISTRÉE (effet intraday documenté sur indices) : le range de
la 1ʳᵉ heure de séance « cadre » la journée ; une clôture au-delà amorce un
mouvement qui se poursuit jusqu'au soir. Stratégie INTRADAY → flat la nuit →
ZÉRO swap.

Discipline (cf. CLAUDE.md §5) :
  - Range d'ouverture en HEURE LOCALE (NYSE/Xetra) → robuste aux changements
    d'heure (DST).
  - Cassure confirmée à la close, entrée à l'open de la barre suivante (fill
    honnête). Stop = côté opposé de l'OR. Sortie EOD (pas de TP fixe).
  - Coûts XTB réels par trade ; swap ≈ 0 (intraday).
  - Sharpe annualisé routé par fréquence → ``validate_edge(annualized_sharpe=)``.
  - Paramètres FIGÉS (aucun tuning). ``n_trials`` = cumul du registre
    anti-snooping (chaque indice testé y est journalisé).

⚠️ Données H1 = range d'ouverture grossier (la littérature utilise du 5 min).
   Négatif ≠ enterre l'ORB fin ; positif = à reconfirmer en données fines.

GO ssi (constitution §2) : Sharpe ≥ 1 · DSR > 0 (p<0.05) · MaxDD < 15 %
                            · WR > 30 % · ≥ 30 trades/an.

USAGE :
    python scripts/screen_orb.py
    python scripts/screen_orb.py --assets US500,US30,GER30
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
from app.strategies.opening_range import simulate_orb_trades  # noqa: E402

# Séances (figées) : (fuseau de la place, heure locale du range, dernière heure locale).
# US RTH 9:30-16:00 ET → OR = barre 9h (couvre l'ouverture), exit close 15h-16h.
# DAX Xetra 9:00-17:30 CET → OR = barre 9h, exit close 16h-17h.
SESSIONS: dict[str, tuple[str, int, int]] = {
    "US500": ("America/New_York", 9, 15),
    "US30": ("America/New_York", 9, 15),
    "GER30": ("Europe/Berlin", 9, 16),
}


def _equity_and_df(
    trades: list[dict], pip_value_eur: float, capital: float
) -> tuple[pd.Series, pd.DataFrame]:
    """(equity €, trades_df['pnl']) triés par heure de sortie — pour validate_edge."""
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times, pnl_eur = exit_times[order], pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    trades_df = pd.DataFrame({"pnl": pnl_eur}, index=exit_times)
    return equity, trades_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête ORB indices.")
    parser.add_argument("--assets", default="US500,US30,GER30")
    parser.add_argument("--tf", default="H1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--split-year", default=2020, type=int)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    print("=" * 72)
    print(f"OPENING RANGE BREAKOUT (honnête) — {args.tf}, intraday flat la nuit")
    print("OR = 1ʳᵉ heure de séance (locale), stop = côté opposé, sortie EOD")
    print(f"Indices : {', '.join(assets)}   n_trials : registre anti-snooping")
    print("=" * 72)

    any_go = False
    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"\n⏭️  {asset} : pas de config coûts XTB — ignoré.")
            continue
        if asset not in SESSIONS:
            print(f"\n⏭️  {asset} : pas de séance définie (SESSIONS) — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        tz, or_hour, last_hour = SESSIONS[asset]
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue

        trades = simulate_orb_trades(
            df, cfg, session_tz=tz, or_hour_local=or_hour, last_hour_local=last_hour,
        )
        if len(trades) < 2:
            print(f"\n══ {asset}/{args.tf} ══ {len(trades)} trade(s) — insuffisant.")
            continue

        years = max((df.index.max() - df.index.min()).days / 365.25, 1e-3)
        tpy = len(trades) / years
        ann_sharpe = sharpe_daily_from_trades(trades)
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        n_trials = record_and_resolve_n_trials(
            prompt="screen_orb",
            hypothesis=f"{asset}/{args.tf}:orb_h1_session",
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

        print(f"\n══ {asset}/{args.tf} ══ ({len(trades)} trades, {tpy:.0f}/an, tz {tz}, "
              f"{df.index.min().date()}→{df.index.max().date()})")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
              f"[n_trials={n_trials}]")
        print(f"  Preuves primaires : t/trade = {report.metrics['t_stat']:.2f} "
              f"(p={report.metrics['p_t']:.3f})   p_bootstrap = "
              f"{report.metrics['p_bootstrap']:.3f}")
        print(f"  MaxDD : {report.metrics['max_dd']:.1%}   WR : {report.metrics['wr']:.0%}   "
              f"trades/an : {report.metrics['trades_per_year']:.1f}")
        print(f"  PnL net : {tdf['pnl'].sum():+.0f} €   "
              f"long/short : {n_long}/{len(trades) - n_long}   "
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
