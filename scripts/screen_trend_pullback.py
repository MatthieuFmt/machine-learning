#!/usr/bin/env python3
"""Test HONNÊTE de la stratégie MANUELLE TradingView (trend-pullback D1/H4).

Mesure ce que la stratégie manuelle (strategie-forex/) aurait donné, au lieu de
la croire sur parole. Mêmes règles que les 2 indicateurs Pine, version
mécanique anti-fuite (cf. app/strategies/trend_pullback.py) :
  - D1 : EMA20>EMA50, Close>EMA200, ADX>25 (régime de la VEILLE, jamais du jour) ;
  - H4 : repli zone EMA20-50 + RSI recroise 50 + bougie alignée ;
  - entrée Open suivant, SL=1.5×ATR, TP=2R, SL prioritaire, swap par nuit.

⚠️ Contexte honnête : la famille « pullback de tendance D1/H4 forex » est NO-GO
historiquement sur ce repo (10 familles testées). Ce screen sert à donner un
CHIFFRE au mainteneur (espérance, WR) — résultat négatif attendu et acceptable.

Discipline :
  - paramètres FIGÉS (ceux des indicateurs Pine : 20/50/200, ADX 25, ATR 1.5, R:R 2) ;
  - coûts XTB ×{marge} (défaut 1.5 tant que les coûts démo ne sont pas relevés) ;
  - n_trials = cumul du registre anti-snooping ;
  - DSR canonique + t-test par trade + bootstrap (validate_edge).

USAGE :
    python scripts/screen_trend_pullback.py
    python scripts/screen_trend_pullback.py --assets EURUSD,GBPUSD,USDJPY,XAUUSD
    python scripts/screen_trend_pullback.py --cost-margin 1.0   # coûts nominaux
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
from app.strategies.trend_pullback import simulate_trend_pullback_trades  # noqa: E402

# Paramètres FIGÉS = ceux des indicateurs Pine. Ne PAS tuner.
ATR_MULT_SL = 1.5
RR = 2.0


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
    parser = argparse.ArgumentParser(
        description="Test honnête de la stratégie manuelle trend-pullback D1/H4."
    )
    parser.add_argument("--assets", default="EURUSD,GBPUSD,USDJPY,XAUUSD",
                        help="Watchlist de la stratégie manuelle (HTML).")
    parser.add_argument("--tf", default="H4", help="Timeframe d'exécution.")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--cost-margin", default=1.5, type=float,
                        help="Marge de sécurité sur les coûts estimés (≥1).")
    parser.add_argument("--split-year", default=2020, type=int)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    print("=" * 72)
    print(f"STRATÉGIE MANUELLE trend-pullback — D1 filtre + {args.tf} entrée")
    print(f"SL={ATR_MULT_SL}×ATR  TP={RR}R  coûts ×{args.cost_margin:g}  "
          f"n_trials : registre anti-snooping")
    print("=" * 72)

    any_go = False
    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"\n⏭️  {asset} : pas de config coûts XTB — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        try:
            df_h4 = load_asset(asset, args.tf, data_root=args.data_root)
            df_d1 = load_asset(asset, "D1", data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  {asset} : chargement échoué ({exc}).")
            continue

        trades = simulate_trend_pullback_trades(
            df_h4, df_d1, cfg,
            atr_mult_sl=ATR_MULT_SL, rr=RR, cost_multiplier=args.cost_margin,
        )
        if len(trades) < 2:
            print(f"\n══ {asset}/{args.tf} ══ {len(trades)} trade(s) — insuffisant.")
            continue

        years = max((df_h4.index.max() - df_h4.index.min()).days / 365.25, 1e-3)
        tpy = len(trades) / years
        ann_sharpe = sharpe_daily_from_trades(trades)
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        n_trials = record_and_resolve_n_trials(
            prompt="screen_trend_pullback",
            hypothesis=f"{asset}/{args.tf}:trend_pullback_cm{args.cost_margin:g}",
            sharpe=ann_sharpe,
            n_trades=len(trades),
        )
        report = validate_edge(
            equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe
        )
        any_go = any_go or report.go

        pips = np.array([t["pips_net"] for t in trades])
        n_tp = sum(1 for t in trades if t["exit_reason"] == "tp")
        n_sl = sum(1 for t in trades if t["exit_reason"] == "sl")
        mean_nights = float(np.mean([t["nights_held"] for t in trades]))
        n_long = sum(1 for t in trades if t["signal"] == 1)

        print(f"\n══ {asset}/{args.tf} ══ ({len(trades)} trades, {tpy:.1f}/an, "
              f"{df_h4.index.min().date()}→{df_h4.index.max().date()})")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
              f"[n_trials={n_trials}]")
        print(f"  Preuves primaires : t/trade = {report.metrics['t_stat']:.2f} "
              f"(p={report.metrics['p_t']:.3f})   p_bootstrap = "
              f"{report.metrics['p_bootstrap']:.3f}")
        print(f"  MaxDD : {report.metrics['max_dd']:.1%}   WR : {report.metrics['wr']:.0%}   "
              f"espérance : {pips.mean():+.1f} pips/trade net")
        print(f"  PnL net : {tdf['pnl'].sum():+.0f} €   long/short : "
              f"{n_long}/{len(trades) - n_long}   sorties : {n_tp} TP / {n_sl} SL   "
              f"durée moy : {mean_nights:.1f} nuits")
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
    print(f"VERDICT GLOBAL : {'✅ au moins un actif GO' if any_go else '❌ aucun actif GO'}")
    print("Rappel : NO-GO attendu (famille morte 10×) — l'objectif est le CHIFFRE.")
    return 0 if any_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
