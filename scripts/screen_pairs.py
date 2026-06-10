#!/usr/bin/env python3
"""Test HONNÊTE du Pairs Trading Or / Argent (XAUUSD vs XAGUSD) — Phase 1.

Hypothèse PRÉ-ENREGISTRÉE (relative value, anomalie documentée) : deux métaux
précieux liés (« ratio or/argent ») reviennent l'un vers l'autre quand leur
écart se tend. On parie sur la convergence quand le z-score du spread dépasse
un seuil. Marché-neutre → protégé des krachs directionnels.

Discipline appliquée (cf. CLAUDE.md §5) :
  - Fill HONNÊTE : signal sur Close[i], exécution à l'Open[i+1]
    (``simulate_pairs_honest``). Pas de look-ahead.
  - Sizing dollar-neutral (jambes équilibrées en €) → pas de jambe argent qui
    écrase l'or. PnL en rendement de prix (indépendant des conventions pip).
  - Coûts XTB réels par jambe + swap signé par nuit sur les DEUX jambes
    (sur CFD on paie le financement des deux côtés — pas de compensation).
  - Sharpe annualisé routé par fréquence (``sharpe_daily_from_trades``), puis
    passé à ``validate_edge(annualized_sharpe=...)`` pour le critère ET le DSR.
  - Paramètres FIGÉS (z=2.0 / 0.5, time-stop 30 H4 ≈ 5 j, lookbacks 60) : aucun
    tuning → ``n_trials`` petit. UNE seule paire pré-choisie pour une raison
    économique (or/argent). Ne PAS pêcher d'autres paires après coup.

GO ssi (constitution §2) : Sharpe ≥ 1 · DSR > 0 (p<0.05) · MaxDD < 15 %
                            · WR > 30 % · ≥ 30 trades/an.

USAGE :
    python scripts/screen_pairs.py
    python scripts/screen_pairs.py --asset-a XAUUSD --asset-b XAGUSD --tf H4
    python scripts/screen_pairs.py --split-year 2020   # stabilité pré/post
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
from app.strategies.pairs_trading import simulate_pairs_honest  # noqa: E402

# Paramètres FIGÉS (pré-enregistrés) — ne PAS tuner.
Z_ENTRY = 2.0
Z_EXIT = 0.5
TIME_STOP_BARS = 30  # 30 barres H4 ≈ 5 jours
BETA_LOOKBACK = 60   # ~10 jours H4
ZSCORE_LOOKBACK = 60


def _equity_and_df(trades: list[dict], capital: float) -> tuple[pd.Series, pd.DataFrame]:
    """(equity €, trades_df['pnl']) triés par heure de sortie — pour validate_edge."""
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl = np.array([t["pnl_eur_net"] for t in trades], dtype=float)
    order = np.argsort(exit_times.values)
    exit_times, pnl = exit_times[order], pnl[order]
    equity = pd.Series(capital + np.cumsum(pnl), index=exit_times)
    trades_df = pd.DataFrame({"pnl": pnl}, index=exit_times)
    return equity, trades_df


def _cointegration(close_a: pd.Series, close_b: pd.Series) -> str:
    """Test Engle-Granger (info, PAS un critère). Fallback si statsmodels absent."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return "statsmodels absent (pip install statsmodels) — test sauté"
    common = close_a.index.intersection(close_b.index)
    t_stat, p_value, _ = coint(close_a.loc[common].values, close_b.loc[common].values)
    verdict = "cointégrés" if p_value < 0.10 else "NON cointégrés"
    return f"t={t_stat:+.2f}  p={p_value:.4f}  → {verdict} (p<0.10 ?)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête Pairs Trading Or/Argent.")
    parser.add_argument("--asset-a", default="XAUUSD")
    parser.add_argument("--asset-b", default="XAGUSD")
    parser.add_argument("--tf", default="H4")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float,
                        help="Notionnel € par jambe ET capital de référence.")
    parser.add_argument("--n-trials", default=None, type=int,
                        help="Essais DSR (défaut : cumul du registre anti-snooping).")
    parser.add_argument("--swap-scale", default=1.0, type=float,
                        help="Multiplicateur swap (1=réel, 0=sans swap pour test sensibilité).")
    parser.add_argument("--split-year", default=2020, type=int,
                        help="Année de césure pour le test de stabilité pré/post.")
    args = parser.parse_args()

    a, b = args.asset_a, args.asset_b
    for asset in (a, b):
        if asset not in ASSET_CONFIGS:
            print(f"❌ {asset} : pas de config coûts XTB.")
            return 1
    cfg_a, cfg_b = ASSET_CONFIGS[a], ASSET_CONFIGS[b]

    try:
        df_a = load_asset(a, args.tf, data_root=args.data_root)
        df_b = load_asset(b, args.tf, data_root=args.data_root)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Chargement échoué : {exc}")
        return 1

    print("=" * 72)
    print(f"PAIRS TRADING (honnête) {a} vs {b} {args.tf}")
    print(f"z_entry={Z_ENTRY}  z_exit={Z_EXIT}  time_stop={TIME_STOP_BARS} barres (~5 j)"
          f"  lookback={BETA_LOOKBACK}")
    print(f"Notionnel/jambe : {args.capital:.0f} €  (gross 2× / net ≈ 0)")
    print(f"Coûts a/r : {a}={cfg_a.total_cost_pips:g} pips  {b}={cfg_b.total_cost_pips:g} pips")
    print(f"Swap/nuit (long/short) : {a}={cfg_a.swap_long_pips_per_night:g}/"
          f"{cfg_a.swap_short_pips_per_night:g}  {b}={cfg_b.swap_long_pips_per_night:g}/"
          f"{cfg_b.swap_short_pips_per_night:g}  (pips)")
    print("=" * 72)

    common = df_a.index.intersection(df_b.index)
    print(f"\nIndex commun : {len(common)} barres "
          f"({common.min().date()} → {common.max().date()})")
    print(f"Cointégration (info) : {_cointegration(df_a['Close'], df_b['Close'])}")

    trades = simulate_pairs_honest(
        df_a, df_b, cfg_a, cfg_b,
        z_entry=Z_ENTRY, z_exit=Z_EXIT, time_stop_bars=TIME_STOP_BARS,
        beta_lookback=BETA_LOOKBACK, zscore_lookback=ZSCORE_LOOKBACK,
        notional_per_leg_eur=args.capital, swap_scale=args.swap_scale,
    )
    if len(trades) < 2:
        print(f"\n{len(trades)} trade(s) — insuffisant pour conclure.")
        return 1

    years = max((common.max() - common.min()).days / 365.25, 1e-3)
    tpy = len(trades) / years
    ann_sharpe = sharpe_daily_from_trades(trades)
    equity, tdf = _equity_and_df(trades, args.capital)
    n_trials = args.n_trials if args.n_trials is not None else record_and_resolve_n_trials(
        prompt="screen_pairs",
        hypothesis=f"{a}-{b}/{args.tf}:z{Z_ENTRY}_{Z_EXIT}_swap{args.swap_scale:g}",
        sharpe=ann_sharpe,
        n_trades=len(trades),
    )
    report = validate_edge(
        equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe
    )

    net = float(tdf["pnl"].sum())
    n_mr = sum(1 for t in trades if t["exit_reason"] == "mean_reversion")
    n_ts = sum(1 for t in trades if t["exit_reason"] == "time_stop")
    mean_nights = float(np.mean([t["nights_held"] for t in trades]))
    swap_total = float(np.sum([t["swap_eur"] for t in trades]))
    cost_total = float(np.sum([t["cost_eur"] for t in trades]))
    gross_total = float(np.sum([t["pnl_eur_brut"] for t in trades]))
    net_no_swap = net - swap_total  # = brut − coûts

    print(f"\n── SENSIBILITÉ aux coûts (swap_scale={args.swap_scale:g}) ──")
    print(f"  Brut (avant frais) : {gross_total:+.0f} €   "
          f"− spread {cost_total:.0f} €   − swap {swap_total:+.0f} €")
    print(f"  Net SANS swap : {net_no_swap:+.0f} €   |   Net réel : {net:+.0f} €")

    print(f"\n── RÉSULTAT (tout l'historique, {len(trades)} trades, {tpy:.0f}/an) ──")
    print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
          f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
          f"n_trials={n_trials}")
    print(f"  Preuves primaires : t/trade = {report.metrics['t_stat']:.2f} "
          f"(p={report.metrics['p_t']:.3f})   p_bootstrap = "
          f"{report.metrics['p_bootstrap']:.3f}")
    print(f"  MaxDD : {report.metrics['max_dd']:.1%}   WR : {report.metrics['wr']:.0%}   "
          f"trades/an : {report.metrics['trades_per_year']:.1f}")
    print(f"  PnL net : {net:+.0f} €   (coûts cumulés {cost_total:.0f} € ; "
          f"swap cumulé {swap_total:+.0f} €)")
    print(f"  Sorties : {n_mr} retour-moyenne / {n_ts} time-stop   "
          f"durée moy : {mean_nights:.1f} nuits")
    print(f"\n  ==> {'✅ GO' if report.go else '❌ NO-GO'}")
    if report.reasons:
        print(f"      raisons : {' ; '.join(report.reasons)}")

    # Stabilité pré/post (descriptif, PAS de sélection).
    split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
    pre = [t for t in trades if pd.Timestamp(t["entry_time"]) < split]
    post = [t for t in trades if pd.Timestamp(t["entry_time"]) >= split]
    sr_pre = sharpe_daily_from_trades(pre) if len(pre) >= 2 else float("nan")
    sr_post = sharpe_daily_from_trades(post) if len(post) >= 2 else float("nan")
    print(f"\n  Stabilité : Sharpe avant {args.split_year} = {sr_pre:.2f} ({len(pre)} tr)"
          f"  →  depuis = {sr_post:.2f} ({len(post)} tr)")

    return 0 if report.go else 2


if __name__ == "__main__":
    raise SystemExit(main())
