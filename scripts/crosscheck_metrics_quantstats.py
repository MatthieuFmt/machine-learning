#!/usr/bin/env python3
"""Recoupe NOS métriques avec QuantStats — implémentation indépendante.

POURQUOI
--------
Les trois faux résultats de l'histoire du projet venaient d'erreurs de calcul,
pas de mauvaises idées. Et ce qui les a démasqués, ce n'est jamais un test
interne : c'est toujours un **recoupement extérieur** (calcul à la main, captures
de l'app XTB, littérature académique).

Ce script applique le même principe au code : il calcule Sharpe, Sortino,
max-drawdown et volatilité **deux fois** — une fois avec `app/analysis` et
`app/backtest`, une fois avec [QuantStats](https://github.com/ranaroussi/quantstats),
une bibliothèque tierce largement utilisée. Si les deux divergent au-delà de la
tolérance, l'un des deux ment et il faut trouver lequel.

Attention aux conventions : QuantStats annualise par défaut sur 252 périodes et
calcule le Sharpe sur des retours simples. On aligne explicitement les
conventions avant de comparer — une divergence de convention n'est pas un bug,
mais elle doit être visible plutôt que silencieuse.

USAGE :
    python scripts/crosscheck_metrics_quantstats.py
    python scripts/crosscheck_metrics_quantstats.py --n-days 2000 --seed 7
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.edge_validation import max_drawdown, sharpe_ratio, sortino_ratio  # noqa: E402

TOLERANCE = 0.02  # écart relatif acceptable (2 %) — au-delà, on enquête


def _build_series(n_days: int, seed: int, mu: float, sigma: float) -> pd.Series:
    """Retours quotidiens synthétiques (loi de Student, queues épaisses réalistes)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df=4, size=n_days)
    raw = raw / raw.std(ddof=1)
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B", tz="UTC")
    return pd.Series(mu + sigma * raw, index=idx)


def _compare(label: str, ours: float, theirs: float, rows: list[tuple]) -> None:
    denom = max(abs(ours), abs(theirs), 1e-12)
    rel = abs(ours - theirs) / denom
    rows.append((label, ours, theirs, rel, rel <= TOLERANCE))


def main() -> int:
    p = argparse.ArgumentParser(description="Recoupe nos métriques avec QuantStats.")
    p.add_argument("--n-days", default=2520, type=int, help="≈ 10 ans ouvrés")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--mu", default=0.0004, type=float, help="dérive quotidienne")
    p.add_argument("--sigma", default=0.011, type=float, help="vol quotidienne")
    args = p.parse_args()

    try:
        import quantstats as qs
    except ImportError:
        print("❌ QuantStats absent.  →  pip install quantstats")
        return 1

    warnings.filterwarnings("ignore")
    returns = _build_series(args.n_days, args.seed, args.mu, args.sigma)
    equity = (1.0 + returns).cumprod()

    rows: list[tuple] = []
    _compare("Sharpe annualisé", sharpe_ratio(returns, freq=252),
             float(qs.stats.sharpe(returns, periods=252, annualize=True)), rows)
    _compare("Sortino annualisé", sortino_ratio(returns, freq=252),
             float(qs.stats.sortino(returns, periods=252, annualize=True)), rows)
    # Nos max_drawdown/QuantStats renvoient une fraction négative ou positive selon
    # la convention : on compare les valeurs absolues.
    _compare("Max drawdown (abs)", abs(max_drawdown(equity)),
             abs(float(qs.stats.max_drawdown(equity))), rows)
    _compare("Volatilité annualisée", float(returns.std(ddof=1) * np.sqrt(252)),
             float(qs.stats.volatility(returns, periods=252, annualize=True)), rows)

    print("=" * 78)
    print("RECOUPEMENT : app/analysis  vs  QuantStats (implémentation indépendante)")
    print(f"Série synthétique : {args.n_days} jours ouvrés, Student(4), "
          f"mu={args.mu}, sigma={args.sigma}, seed={args.seed}")
    print("=" * 78)
    print(f"\n{'Métrique':<26}{'NOUS':>12}{'QuantStats':>14}{'écart rel.':>13}   verdict")
    print("-" * 78)
    all_ok = True
    for label, ours, theirs, rel, ok in rows:
        all_ok &= ok
        print(f"{label:<26}{ours:>12.4f}{theirs:>14.4f}{rel:>12.2%}   "
              f"{'✅' if ok else '🚨 DIVERGENCE'}")
    print("-" * 78)

    if all_ok:
        print(f"\n✅ Les deux implémentations concordent (tolérance {TOLERANCE:.0%}).")
        print("   Nos métriques de base sont recoupées par une bibliothèque tierce.")
    else:
        print("\n🚨 DIVERGENCE : l'une des deux implémentations est fausse.")
        print("   Vérifier d'abord les CONVENTIONS (facteur d'annualisation, ddof,")
        print("   retours simples vs log, taux sans risque) avant de conclure au bug.")

    print("\n⚠️ Portée : ce recoupement couvre les métriques de PERFORMANCE.")
    print("   Il ne teste PAS le DSR (QuantStats ne l'implémente pas) ni le")
    print("   simulateur de trades. Le DSR reste couvert par tests/unit/test_dsr_sanity.py")
    print("   (valeurs recalculées à la main + régression ORB).")
    print("=" * 78)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
