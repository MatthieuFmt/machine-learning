#!/usr/bin/env python3
"""Test HONNÊTE du Carry (paires JPY) — source d'edge structurelle, pas technique.

Hypothèse PRÉ-ENREGISTRÉE (anomalie FX la plus documentée) : être long une
devise à taux élevé contre une à taux faible (AUDJPY, GBPJPY, EURJPY) rapporte
le différentiel de taux (encaissé via le swap), au prix d'un risque de krach
(« carry unwind »).

Évaluation adaptée à une position CONTINUE (pas des trades discrets) :
  - rendement quotidien = variation spot du jour + swap × nuits détenues ;
  - Sharpe annualisé sur retours quotidiens (×√252, correct car retours daily) ;
  - DSR avec Sharpe PAR-PÉRIODE (fix 2026-05-29), n_obs = nb de jours ;
  - décomposition spot-seul vs swap-seul vs total (le swap aide-t-il vraiment ?) ;
  - MaxDD (le talon d'Achille du carry) + stabilité pré/post.

⚠️ Les swaps sont PROVISOIRES (estimés, cf. instruments.py) — résultat sensible
à ces valeurs ; à valider en démo XTB.

USAGE :
    python scripts/screen_carry.py
    python scripts/screen_carry.py --assets AUDJPY,GBPJPY,EURJPY --tf D1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.edge_validation import deflated_sharpe, max_drawdown, sharpe_ratio  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.data.loader import load_asset  # noqa: E402


def _daily_carry_returns(df: pd.DataFrame, cfg, capital: float):
    """Retours quotidiens FRACTIONNELS d'une position LONG continue 1× (sans levier).

    Notionnel = capital → rendement = variation spot (%) + carry (% du notionnel).
    Le carry quotidien = swap_pips × pip_size / prix (conversion pips → fraction).

    Returns: (ret_total, ret_spot, ret_swap, equity) — Series alignées.
    """
    close = df["Close"]
    nights = close.index.to_series().diff().dt.days.fillna(1).clip(lower=0)
    spot_ret = close.pct_change()
    # swap en pips → unités de prix (×pip_size) → fraction du notionnel (/prix)
    carry_ret = (cfg.swap_long_pips_per_night * cfg.pip_size / close) * nights
    spot_ret = spot_ret.fillna(0.0)
    carry_ret = carry_ret.fillna(0.0)
    total_ret = spot_ret + carry_ret
    # Coût d'ouverture/fermeture (une seule fois, négligeable sur des années).
    total_ret.iloc[0] -= cfg.total_cost_pips * cfg.pip_size / float(close.iloc[0])
    equity = capital * (1.0 + total_ret.cumsum())
    return total_ret, spot_ret, carry_ret, equity


def _metrics(ret: pd.Series, equity: pd.Series, n_trials: int) -> dict:
    r = ret.dropna()
    sr_ann = sharpe_ratio(r, freq=252)
    # DSR : Sharpe ANNUALISÉ honnête (retours déjà quotidiens → √252 correct).
    dsr, p = deflated_sharpe(sr_ann, n_trials=n_trials, n_obs=len(r),
                             skew=float(r.skew()), kurtosis=float(r.kurtosis()) + 3.0)
    return {
        "sharpe": sr_ann, "dsr": dsr, "p": p,
        "maxdd": max_drawdown(equity),
        "wr_days": float((r > 0).mean()),
        "ann_return_pct": float(r.mean() * 252 * 100),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test honnête Carry JPY.")
    parser.add_argument("--assets", default="AUDJPY,GBPJPY,EURJPY")
    parser.add_argument("--tf", default="D1")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--capital", default=10_000.0, type=float)
    parser.add_argument("--split-year", default=2018, type=int)
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    n_trials = len(assets)
    basket_rets: list[pd.Series] = []

    for asset in assets:
        if asset not in ASSET_CONFIGS:
            print(f"⏭️  {asset} : pas de config — ignoré.")
            continue
        cfg = ASSET_CONFIGS[asset]
        try:
            df = load_asset(asset, args.tf, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  {asset}/{args.tf} : chargement échoué ({exc}).")
            continue

        ret, ret_spot, ret_swap, equity = _daily_carry_returns(df, cfg, args.capital)
        basket_rets.append(ret.rename(asset))
        m = _metrics(ret, equity, n_trials)

        # Sharpe spot-seul vs total (le swap aide-t-il ?)
        sr_spot = sharpe_ratio(ret_spot.dropna(), freq=252)
        carry_pct = float(ret_swap.mean() * 252 * 100)

        split = pd.Timestamp(f"{args.split_year}-01-01", tz="UTC")
        pre, post = ret[ret.index < split], ret[ret.index >= split]
        sr_pre = sharpe_ratio(pre.dropna(), freq=252)
        sr_post = sharpe_ratio(post.dropna(), freq=252)

        print(f"══ {asset}/{args.tf} ══ ({len(ret.dropna())} jours, "
              f"{df.index.min().date()}→{df.index.max().date()})")
        print(f"  Sharpe TOTAL : {m['sharpe']:.2f}   (spot seul : {sr_spot:.2f})   "
              f"DSR : {m['dsr']:.2f} (p={m['p']:.3f})")
        print(f"  Rendt/an : {m['ann_return_pct']:.1f}%   dont carry (swap) : {carry_pct:+.1f}%/an   "
              f"MaxDD : {m['maxdd']:.0%}")
        print(f"  Jours gagnants : {m['wr_days']:.0%}   "
              f"Sharpe avant {args.split_year} : {sr_pre:.2f}  →  depuis : {sr_post:.2f}")
        go = (m['sharpe'] >= 1.0 and m['dsr'] > 0 and m['p'] < 0.05 and m['maxdd'] < 0.15)
        print(f"  GO : {go}")
        print()

    # Panier équipondéré (la vraie façon de trader le carry : diversifié)
    if len(basket_rets) >= 2:
        mat = pd.concat(basket_rets, axis=1, sort=True).dropna()
        bret = mat.mean(axis=1)
        beq = args.capital + (bret * args.capital).cumsum()
        m = _metrics(bret, beq, n_trials)
        print(f"══ PANIER équipondéré ({', '.join(assets)}) ══ ({len(bret)} jours)")
        print(f"  Sharpe : {m['sharpe']:.2f}   DSR : {m['dsr']:.2f} (p={m['p']:.3f})   "
              f"Rendt/an : {m['ann_return_pct']:.1f}%   MaxDD : {m['maxdd']:.0%}")
        go = (m['sharpe'] >= 1.0 and m['dsr'] > 0 and m['p'] < 0.05 and m['maxdd'] < 0.15)
        print(f"  GO : {go}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
