#!/usr/bin/env python3
"""Test HONNÊTE de l'Asian Range Breakout (kill zone Londres) — Phase 1, option A.

Hypothèse PRÉ-ENREGISTRÉE (effet de session documenté) : le range de la nuit
asiatique (00h-06h UTC) « contient » le prix ; sa cassure à l'ouverture de
Londres (close 07h UTC) amorce un mouvement directionnel jusqu'au soir.
Stratégie INTRADAY stricte : entrée 08h, sortie au plus tard 22h → **flat la
nuit → ZÉRO swap** (le tueur de nos 7 familles multi-jours).

Discipline appliquée (cf. CLAUDE.md §5) :
  - Fill HONNÊTE : signal à la close 07h, entrée à l'Open 08h (barre suivante).
  - Intrabar conservateur : SL prioritaire sur TP dans la même barre.
  - Coûts XTB réels par trade (spread + 2×slippage), swap ≈ 0 (intraday).
  - Sharpe annualisé routé par fréquence (``sharpe_daily_from_trades``), passé
    à ``validate_edge(annualized_sharpe=...)`` pour le critère ET le DSR.
  - Paramètres FIGÉS (TP=1.5×range, SL=0.5×range, fenêtres horaires standard) :
    aucun tuning. ``n_trials`` = cumul du registre anti-snooping.

⚠️ Cette stratégie avait été regardée en Phase H2 (forex, barre molle 0.7, OOS
   ≥2024 aujourd'hui brûlé). Ici : re-test propre, sur tout l'historique.

GO ssi (constitution §2) : Sharpe ≥ 1 · DSR > 0 (p<0.05) · MaxDD < 15 %
                            · WR > 30 % · ≥ 30 trades/an.

USAGE :
    python scripts/screen_asian_range.py
    python scripts/screen_asian_range.py --assets EURUSD,GBPUSD,USDJPY --tf H1
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
from app.strategies.asian_range import simulate_asian_range_trades  # noqa: E402

# Paramètres FIGÉS (pré-enregistrés) — ne PAS tuner.
TP_MULT = 1.5
SL_MULT = 0.5
TIME_STOP_HOUR_UTC = 22


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
    parser = argparse.ArgumentParser(description="Test honnête Asian Range Breakout.")
    parser.add_argument("--assets", default="EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--tf", default="H1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--split-year", default=2020, type=int,
                        help="Année de césure pour le test de stabilité pré/post.")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    print("=" * 72)
    print(f"ASIAN RANGE BREAKOUT (honnête) — {args.tf}, intraday flat la nuit")
    print(f"TP={TP_MULT}×range  SL={SL_MULT}×range  time-stop {TIME_STOP_HOUR_UTC}h UTC")
    print(f"Paires : {', '.join(assets)}   n_trials : registre anti-snooping")
    print("=" * 72)

    any_go = False
    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"\n⏭️  {asset} : pas de config coûts XTB — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue

        trades = simulate_asian_range_trades(
            df, cfg, tp_mult=TP_MULT, sl_mult=SL_MULT,
            time_stop_hour_utc=TIME_STOP_HOUR_UTC,
        )
        if len(trades) < 2:
            print(f"\n══ {asset}/{args.tf} ══ {len(trades)} trade(s) — insuffisant.")
            continue

        years = max((df.index.max() - df.index.min()).days / 365.25, 1e-3)
        tpy = len(trades) / years
        ann_sharpe = sharpe_daily_from_trades(trades)
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        n_trials = record_and_resolve_n_trials(
            prompt="screen_asian_range",
            hypothesis=f"{asset}/{args.tf}:asian_range_tp{TP_MULT}_sl{SL_MULT}",
            sharpe=ann_sharpe,
            n_trades=len(trades),
        )
        report = validate_edge(
            equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe
        )
        any_go = any_go or report.go

        n_tp = sum(1 for t in trades if t["exit_reason"] == "tp")
        n_sl = sum(1 for t in trades if t["exit_reason"] == "sl")
        n_ts = sum(1 for t in trades if t["exit_reason"] == "time_stop")
        max_nights = max(t["nights_held"] for t in trades)

        print(f"\n══ {asset}/{args.tf} ══ ({len(trades)} trades, {tpy:.0f}/an, "
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
              f"sorties : {n_tp} TP / {n_sl} SL / {n_ts} time-stop   "
              f"max nuits détenues : {max_nights}")
        print(f"  ==> {'✅ GO' if report.go else '❌ NO-GO'}")
        if report.reasons:
            print(f"      raisons : {' ; '.join(report.reasons)}")

        # Stabilité pré/post (descriptif).
        split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
        pre = [t for t in trades if pd.Timestamp(t["entry_time"]) < split]
        post = [t for t in trades if pd.Timestamp(t["entry_time"]) >= split]
        sr_pre = sharpe_daily_from_trades(pre) if len(pre) >= 2 else float("nan")
        sr_post = sharpe_daily_from_trades(post) if len(post) >= 2 else float("nan")
        print(f"  Stabilité : Sharpe avant {args.split_year} = {sr_pre:.2f} ({len(pre)} tr)"
              f"  →  depuis = {sr_post:.2f} ({len(post)} tr)")

    print("\n" + "=" * 72)
    print(f"VERDICT GLOBAL : {'✅ au moins une paire GO' if any_go else '❌ aucune paire GO'}")
    return 0 if any_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
