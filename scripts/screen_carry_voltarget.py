#!/usr/bin/env python3
"""Carry JPY + VOLATILITY TARGETING — Phase 1 (amélioration de sizing).

Compare le panier carry BRUT (exposition constante 1×) au même panier avec
*volatility targeting* : la position vise une volatilité constante (plus grosse
quand c'est calme, ~zéro quand c'est agité). La recherche montre que ce sizing
améliore le Sharpe du carry et réduit ses krachs (0,76 → 0,84 ; jusqu'à ~1,07
combiné à d'autres signaux).

Discipline :
  - Poids vol-target calculés SANS look-ahead (vol réalisée jusqu'à t-1, cf.
    ``volatility_target_weights``).
  - Paramètres FIGÉS : fenêtre 60 j, cible 10 %/an, levier max 3× (standard,
    non tunés ; Sharpe ~invariant à la cible/au plafond).
  - DSR canonique par-période (fix 2026-06-09, via screen_carry._metrics) ;
    n_trials = cumul du registre anti-snooping.

GO ssi : Sharpe ≥ 1 · DSR > 0 (p<0.05) · MaxDD < 15 %.

USAGE :
    python scripts/screen_carry_voltarget.py
    python scripts/screen_carry_voltarget.py --assets AUDJPY,GBPJPY,EURJPY --lookback 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.sizing import volatility_target_weights  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.research.edge_harness import record_and_resolve_n_trials  # noqa: E402
from scripts.screen_carry import _daily_carry_returns, _metrics  # noqa: E402


def _equity(ret: pd.Series, capital: float) -> pd.Series:
    return capital + (ret * capital).cumsum()


def _print_block(label: str, m: dict) -> bool:
    go = bool(m["sharpe"] >= 1.0 and m["dsr"] > 0 and m["p"] < 0.05 and m["maxdd"] < 0.15)
    print(f"  {label:<16} Sharpe {m['sharpe']:+.2f}   DSR {m['dsr']:+.2f} (p={m['p']:.3f})   "
          f"t-jour {m['t']:+.2f} (p={m['p_t']:.3f})   "
          f"MaxDD {m['maxdd']:.0%}   Rendt/an {m['ann_return_pct']:+.1f}%   "
          f"{'✅ GO' if go else '❌'}")
    return go


def main() -> int:
    parser = argparse.ArgumentParser(description="Carry JPY + volatility targeting.")
    parser.add_argument("--assets", default="AUDJPY,GBPJPY,EURJPY")
    parser.add_argument("--tf", default="D1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--lookback", default=60, type=int)
    parser.add_argument("--target-vol", default=0.10, type=float)
    parser.add_argument("--max-leverage", default=3.0, type=float)
    parser.add_argument("--n-trials", default=None, type=int,
                        help="Essais DSR (défaut : cumul du registre anti-snooping).")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    rets: list[pd.Series] = []
    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"⏭️  {asset} : pas de config — ignoré.")
            continue
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue
        ret, _, _, _ = _daily_carry_returns(df, ASSET_CONFIGS[asset], args.capital)
        rets.append(ret.rename(asset))

    if len(rets) < 2:
        print("Pas assez d'actifs chargés.")
        return 1

    mat = pd.concat(rets, axis=1, sort=True).dropna()
    basket = mat.mean(axis=1)

    print("=" * 72)
    print(f"CARRY {', '.join(assets)} — BRUT vs VOLATILITY TARGETING ({args.tf})")
    print(f"vol cible {args.target_vol:.0%}/an · fenêtre {args.lookback} j · "
          f"levier max {args.max_leverage:g}× · n_trials : registre anti-snooping")
    print(f"{len(basket)} jours ({basket.index.min().date()} → {basket.index.max().date()})")
    print("=" * 72)

    def _resolve(label: str, ret: pd.Series) -> int:
        if args.n_trials is not None:
            return int(args.n_trials)
        std = float(ret.std(ddof=1))
        sr_ann = float(ret.mean()) / std * (252.0 ** 0.5) if std > 0 else 0.0
        return record_and_resolve_n_trials(
            prompt="screen_carry_voltarget",
            hypothesis=f"basket[{','.join(assets)}]/{args.tf}:{label}",
            sharpe=sr_ann,
            n_trades=len(ret),
        )

    # ── Baseline (exposition constante 1×) ───────────────────────────
    m_base = _metrics(basket, _equity(basket, args.capital), _resolve("carry_1x", basket))

    # ── Volatility targeting ─────────────────────────────────────────
    weights = volatility_target_weights(
        basket, target_vol_annual=args.target_vol,
        lookback=args.lookback, max_leverage=args.max_leverage,
    )
    scaled = (weights * basket).dropna()
    m_vt = _metrics(
        scaled, _equity(scaled, args.capital),
        _resolve(f"carry_voltarget{args.target_vol:g}", scaled),
    )

    print("\n── Comparaison ──")
    _print_block("Carry BRUT", m_base)
    go_vt = _print_block("Carry VOL-TARGET", m_vt)

    w_valid = weights.dropna()
    print(f"\n  Poids vol-target : moyen {w_valid.mean():.2f}×  "
          f"médian {w_valid.median():.2f}×  "
          f"temps < 0,5× (dé-risqué) : {(w_valid < 0.5).mean():.0%}  "
          f"temps plafonné : {(w_valid >= args.max_leverage - 1e-9).mean():.0%}")

    print(f"\n  ==> {'✅ GO (vol-target)' if go_vt else '❌ NO-GO (vol-target)'}")
    return 0 if go_vt else 2


if __name__ == "__main__":
    raise SystemExit(main())
