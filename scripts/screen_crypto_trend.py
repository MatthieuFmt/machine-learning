#!/usr/bin/env python3
"""Tendance crypto (TSMOM) BTC/ETH + volatility targeting — Phase 1.

Suit la tendance (signe du rendement des `lookback` derniers jours) sur BTC/ETH
en D1, long/short, avec coûts + swap réels et vol-scaling. La crypto est le seul
marché « multi-jours » où les moves peuvent dépasser le swap.

Discipline :
  - Signal SANS look-ahead (position du jour t = signe du momentum jusqu'à t-1).
  - Coûts XTB réels (ETH corrigé) + swap signé/nuit + coût a/r à chaque
    retournement.
  - Vol-scaling sans look-ahead (vol jusqu'à t-1).
  - DSR avec Sharpe annualisé honnête (retours quotidiens → √252).
  - Paramètres FIGÉS : momentum 100 j, vol cible 20 %/an, fenêtre vol 60 j,
    levier max 3×. n_trials = nb d'actifs.

GO ssi : Sharpe ≥ 1 · DSR > 0 (p<0.05) · MaxDD < 15 %.

USAGE :
    python scripts/screen_crypto_trend.py
    python scripts/screen_crypto_trend.py --assets BTCUSD,ETHUSD --lookback 100
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
from app.strategies.crypto_trend import tsmom_daily_returns  # noqa: E402
from scripts.screen_carry import _metrics  # noqa: E402


def _equity(ret: pd.Series, capital: float) -> pd.Series:
    return capital + (ret * capital).cumsum()


def _print_block(label: str, m: dict) -> bool:
    go = bool(m["sharpe"] >= 1.0 and m["dsr"] > 0 and m["p"] < 0.05 and m["maxdd"] < 0.15)
    print(f"  {label:<22} Sharpe {m['sharpe']:+.2f}   DSR {m['dsr']:+.2f} (p={m['p']:.3f})   "
          f"MaxDD {m['maxdd']:.0%}   Rendt/an {m['ann_return_pct']:+.1f}%   "
          f"{'✅ GO' if go else '❌'}")
    return go


def main() -> int:
    parser = argparse.ArgumentParser(description="Tendance crypto TSMOM + vol-target.")
    parser.add_argument("--assets", default="BTCUSD,ETHUSD")
    parser.add_argument("--tf", default="D1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--lookback", default=100, type=int, help="Momentum (jours).")
    parser.add_argument("--target-vol", default=0.20, type=float)
    parser.add_argument("--vol-lookback", default=60, type=int)
    parser.add_argument("--max-leverage", default=3.0, type=float)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    n_trials = len(assets)

    print("=" * 78)
    print(f"TENDANCE CRYPTO (TSMOM {args.lookback} j) + VOL-TARGET — {args.tf}")
    print(f"vol cible {args.target_vol:.0%}/an · fenêtre vol {args.vol_lookback} j · "
          f"levier max {args.max_leverage:g}× · n_trials={n_trials}")
    print("=" * 78)

    nets: list[pd.Series] = []
    any_go = False
    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"\n⏭️  {asset} : pas de config — ignoré.")
            continue
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue

        net, gross, position = tsmom_daily_returns(
            df, ASSET_CONFIGS[asset], lookback=args.lookback
        )
        net = net.dropna()
        nets.append(net.rename(asset))

        m_base = _metrics(net, _equity(net, args.capital), n_trials)
        w = volatility_target_weights(
            net, target_vol_annual=args.target_vol,
            lookback=args.vol_lookback, max_leverage=args.max_leverage,
        )
        scaled = (w * net).dropna()
        m_vt = _metrics(scaled, _equity(scaled, args.capital), n_trials)

        pct_long = float((position > 0).mean())
        print(f"\n══ {asset}/{args.tf} ══ ({len(net)} jours, "
              f"{net.index.min().date()}→{net.index.max().date()}, "
              f"long {pct_long:.0%} du temps)")
        any_go |= _print_block("Trend BRUT", m_base)
        any_go |= _print_block("Trend VOL-TARGET", m_vt)

    # Panier équipondéré (diversification BTC+ETH), vol-targeté.
    if len(nets) >= 2:
        mat = pd.concat(nets, axis=1, sort=True).dropna()
        basket = mat.mean(axis=1)
        w = volatility_target_weights(
            basket, target_vol_annual=args.target_vol,
            lookback=args.vol_lookback, max_leverage=args.max_leverage,
        )
        scaled = (w * basket).dropna()
        m_b = _metrics(scaled, _equity(scaled, args.capital), n_trials)
        print(f"\n══ PANIER {'+'.join(assets)} (vol-targeté) ══ ({len(basket)} jours)")
        any_go |= _print_block("Panier VOL-TARGET", m_b)

    print(f"\n{'=' * 78}")
    print(f"VERDICT GLOBAL : {'✅ au moins une variante GO' if any_go else '❌ aucune variante GO'}")
    return 0 if any_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
