#!/usr/bin/env python3
"""Pre-ECB Drift — le SEUL signal survivant se généralise-t-il à l'Europe ?

POURQUOI CETTE HYPOTHÈSE (et pourquoi elle n'est PAS du data-mining)
--------------------------------------------------------------------
Le pre-FOMC drift (Lucca & Moench 2015) est le seul signal du projet qui survit
à la pile statistique corrigée (t = 2,80 ; p = 0,003). L'explication théorique
retenue dans la littérature est l'« announcement premium » (Ai & Bansal 2018) :
les actions rémunèrent le risque d'*attente* d'une annonce macro majeure. Ce
mécanisme n'a rien de spécifiquement américain.

→ Prédiction testable AVANT de regarder les données : le même effet doit
  apparaître sur un indice EUROPÉEN avant les décisions de la BCE.

C'est une hypothèse *dérivée d'un mécanisme*, pré-enregistrée ici par écrit, avec
la fenêtre FIGÉE à l'identique du pre-FOMC (annonce−24 h → annonce−1 h). Aucun
paramètre n'est ajusté. Deux issues, toutes deux informatives :
  - ça marche  → ~8 trades/an de PLUS, faiblement corrélés au FOMC (banques
    centrales différentes) → un panier FOMC+BCE approche les ≥30 trades/an de la
    constitution et améliore le Sharpe combiné ;
  - ça ne marche pas → le pre-FOMC est probablement un artefact spécifique aux
    US, ce qui AFFAIBLIT le seul survivant. C'est un vrai test, pas une loterie.

⚠️ Rappel du 2026-07-31 : les coûts estimés se sont révélés faux dans le sens
qui arrangeait (spread US500 ×15, swap long JPY jusqu'au signe). GER30 n'a PAS
encore été relevé sur l'app XTB → `--cost-margin` est à 1.5 par défaut ici.

USAGE :
    python scripts/screen_pre_ecb.py
    python scripts/screen_pre_ecb.py --assets GER30 --tf H1
    python scripts/screen_pre_ecb.py --list-events     # si le nom BCE diffère
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.research.edge_harness import record_and_resolve_n_trials  # noqa: E402
from app.strategies.pre_fomc_drift import (  # noqa: E402
    load_fomc_announcement_times,
    simulate_pre_fomc_trades,
)

# Noms possibles de la décision de taux BCE dans les CSV Forex Factory.
# Essayés dans l'ordre ; le premier qui existe est retenu.
ECB_EVENT_CANDIDATES: tuple[str, ...] = (
    "Main Refinancing Rate",
    "Minimum Bid Rate",          # ancien libellé Forex Factory (< 2016)
    "ECB Press Conference",
    "Monetary Policy Statement",
    "Rate Statement",
)

# Motifs pour l'auto-inspection quand aucun candidat ne matche.
ECB_HINTS: tuple[str, ...] = ("ecb", "refinanc", "bid rate", "monetary policy")


def discover_event_names(
    calendar_root: Path, start_year: int, end_year: int, hints: tuple[str, ...]
) -> list[tuple[str, int]]:
    """Liste les noms d'événements du calendrier contenant l'un des `hints`.

    Sert de filet de sécurité : si le libellé BCE de tes CSV diffère de nos
    candidats, ce diagnostic dit exactement quoi passer à `--event`.
    """
    counter: Counter[str] = Counter()
    for year in range(start_year, end_year + 1):
        path = calendar_root / f"{year}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["event"])
        except Exception:  # noqa: BLE001 — CSV malformé : on saute l'année
            continue
        for name in df["event"].dropna().astype(str):
            low = name.lower()
            if any(h in low for h in hints):
                counter[name] += 1
    return sorted(counter.items(), key=lambda kv: -kv[1])


def resolve_ecb_event(
    calendar_root: Path, start_year: int, end_year: int, forced: str | None
) -> tuple[str, pd.DatetimeIndex] | None:
    """Retourne (nom_événement, timestamps UTC) pour la décision BCE."""
    candidates = (forced,) if forced else ECB_EVENT_CANDIDATES
    for name in candidates:
        if name is None:
            continue
        try:
            times = load_fomc_announcement_times(
                calendar_root, start_year=start_year, end_year=end_year, event_name=name
            )
        except Exception:  # noqa: BLE001 — nom absent du calendrier : suivant
            continue
        if len(times) >= 2:
            return name, times
    return None


def _equity_and_df(
    trades: list[dict], pip_value_eur: float, capital: float
) -> tuple[pd.Series, pd.DataFrame]:
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times, pnl_eur = exit_times[order], pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    return equity, pd.DataFrame({"pnl": pnl_eur}, index=exit_times)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête du Pre-ECB drift.")
    parser.add_argument("--assets", default="GER30",
                        help="Indices européens disponibles chez XTB (défaut : GER30/DAX).")
    parser.add_argument("--tf", default="H1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--event", default=None,
                        help="Forcer le libellé BCE du calendrier (voir --list-events).")
    parser.add_argument("--cost-margin", default=1.5, type=float,
                        help="Marge de sécurité sur les coûts (GER30 non encore relevé chez XTB).")
    parser.add_argument("--split-year", default=2015, type=int,
                        help="Césure pour le test de décroissance (comme pre-FOMC).")
    parser.add_argument("--list-events", action="store_true",
                        help="Diagnostic : lister les événements BCE du calendrier puis sortir.")
    args = parser.parse_args()

    cal_root = args.data_root / "economic_calendar"
    if not cal_root.exists():
        print(f"❌ Calendrier introuvable : {cal_root}")
        print("   Ce screen a besoin des CSV Forex Factory (machine locale).")
        return 1

    if args.list_events:
        found = discover_event_names(cal_root, 2010, 2026, ECB_HINTS)
        if not found:
            print("Aucun événement BCE repéré dans le calendrier.")
            return 2
        print("Événements liés à la BCE trouvés (nom → occurrences) :")
        for name, n in found:
            print(f"  {n:5d} × {name}")
        print("\n→ relance avec : --event \"<le nom voulu>\"")
        return 0

    resolved = resolve_ecb_event(cal_root, 2010, 2026, args.event)
    if resolved is None:
        print("❌ Décision BCE introuvable dans le calendrier.")
        print(f"   Libellés essayés : {', '.join(ECB_EVENT_CANDIDATES)}")
        print("   → lance `python scripts/screen_pre_ecb.py --list-events` "
              "et donne-moi la sortie.")
        return 2

    event_name, ecb_times = resolved
    print("=" * 76)
    print("PRE-ECB DRIFT — le pre-FOMC se généralise-t-il à la zone euro ?")
    print(f"Événement retenu : « {event_name} »  —  {len(ecb_times)} annonces, "
          f"{ecb_times.min().date()} → {ecb_times.max().date()}")
    print("Fenêtre FIGÉE (identique au pre-FOMC) : annonce−24 h → annonce−1 h")
    print(f"Marge de sécurité sur les coûts : ×{args.cost_margin}")
    print("=" * 76)

    any_go = False
    for asset in [a.strip() for a in args.assets.split(",") if a.strip()]:
        if asset not in ASSET_CONFIGS:
            print(f"\n⏭️  {asset} : pas de config coûts XTB — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        # Marge de sécurité : les coûts estimés se sont révélés optimistes (2026-07-31).
        cfg = replace(
            cfg,
            spread_pips=cfg.spread_pips * args.cost_margin,
            slippage_pips=cfg.slippage_pips * args.cost_margin,
        )
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue

        trades = simulate_pre_fomc_trades(
            df=df,
            fomc_times=ecb_times,
            spread_pips=cfg.total_cost_pips,
            slippage_pips=0.0,
            commission_pips=0.0,
            pip_size=cfg.pip_size,
            swap_long_pips_per_night=cfg.swap_long_pips_per_night,
            hours_before_entry=24,
            hours_before_exit=1,
        )
        if len(trades) < 2:
            print(f"\n══ {asset}/{args.tf} ══ {len(trades)} trade(s) — insuffisant.")
            continue

        ann_sharpe = sharpe_daily_from_trades(trades)
        equity, tdf = _equity_and_df(trades, cfg.pip_value_eur, args.capital)
        n_trials = record_and_resolve_n_trials(
            prompt="screen_pre_ecb",
            hypothesis=f"{asset}/{args.tf}:pre_ecb_drift",
            sharpe=ann_sharpe,
            n_trades=len(trades),
        )
        report = validate_edge(equity, tdf, n_trials=n_trials, annualized_sharpe=ann_sharpe)
        any_go = any_go or report.go

        pips = np.array([t["pips_net"] for t in trades], dtype=float)
        years = max((ecb_times.max() - ecb_times.min()).days / 365.25, 1e-3)

        print(f"\n══ {asset}/{args.tf} ══ ({len(trades)} trades, {len(trades) / years:.1f}/an)")
        print(f"  Sharpe annualisé : {ann_sharpe:.2f}   "
              f"DSR : {report.metrics['dsr']:.2f} (p={report.metrics['p_value']:.3f})   "
              f"[n_trials={n_trials}]")
        print(f"  PREUVE PRIMAIRE : t/trade = {report.metrics['t_stat']:.2f} "
              f"(p={report.metrics['p_t']:.3f})   p_bootstrap = "
              f"{report.metrics['p_bootstrap']:.3f}")
        print(f"  WR : {(pips > 0).mean():.0%}   pips moy/trade : {pips.mean():.1f}   "
              f"MaxDD : {report.metrics['max_dd']:.1%}   PnL net : {tdf['pnl'].sum():+.0f} €")
        print(f"  ==> {'✅ GO' if report.go else '❌ NO-GO'}")
        if report.reasons:
            print(f"      raisons : {' ; '.join(report.reasons)}")

        split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
        pre = [t for t in trades if pd.Timestamp(t["entry_time"]) < split]
        post = [t for t in trades if pd.Timestamp(t["entry_time"]) >= split]
        f = lambda ts: np.mean([t["pips_net"] for t in ts]) if ts else float("nan")  # noqa: E731
        print(f"  Décroissance : {f(pre):.1f} pips/trade ({len(pre)} tr) avant "
              f"{args.split_year}  →  {f(post):.1f} ({len(post)} tr) depuis")

    print("\n" + "=" * 76)
    print(f"VERDICT : {'✅ au moins un indice GO' if any_go else '❌ aucun indice GO'}")
    print("Rappel : le t-test est la preuve primaire (indépendante du data-snooping).")
    return 0 if any_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
