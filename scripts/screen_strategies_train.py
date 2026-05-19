"""Screening exhaustif des 8 stratégies non testées sur train ≤ 2022.

8 stratégies × 9 actifs × 4 ratios TP/SL ATR = 288 backtests, tous sur
train (zéro n_trial consommé). Objectif : trouver UNE configuration qui
montre un edge minimal (Sharpe ≥ 0.5 sur train) pour la candidater
ensuite en OOS unique.

Stratégies testées :
- BollingerBands (mean-reversion sur bandes)
- KeltnerChannel (breakout du canal Keltner)
- DualMovingAverage (position permanente long/short selon trend MA)
- RsiContrarian (achat survente, vente surachat)
- TsMomentum (signe du rendement sur T barres)
- SmaCrossover (croisement de SMA)
- ParabolicSAR (trend-following stateful)
- MeanReversionRSIBB (RSI extrême + Bollinger)

Critères de pré-sélection train (pour candidater OOS) :
- Sharpe ≥ 0.5
- WR ≥ 35%
- n_trades ≥ 30 (sur 12 ans = 2.5/an minimum)
- max_dd_pips raisonnable (pas > 10× le SL moyen)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.deterministic import run_deterministic_backtest  # noqa: E402
from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.indicators import atr  # noqa: E402
from app.strategies.bollinger import BollingerBands  # noqa: E402
from app.strategies.dual_ma import DualMovingAverage  # noqa: E402
from app.strategies.keltner import KeltnerChannel  # noqa: E402
from app.strategies.mean_reversion import MeanReversionRSIBB  # noqa: E402
from app.strategies.parabolic import ParabolicSAR  # noqa: E402
from app.strategies.rsi_contrarian import RsiContrarian  # noqa: E402
from app.strategies.sma_crossover import SmaCrossover  # noqa: E402
from app.strategies.ts_momentum import TsMomentum  # noqa: E402

from scripts.run_validation_finale import TRAIN_CUTOFF  # noqa: E402

# 8 stratégies avec params défaut raisonnables
STRATEGIES: list[tuple[str, callable]] = [
    ("BollingerBands_20_2", lambda: BollingerBands(N=20, K=2.0)),
    ("BollingerBands_20_2_5", lambda: BollingerBands(N=20, K=2.5)),
    ("KeltnerChannel_20_2", lambda: KeltnerChannel(period=20, mult=2.0)),
    ("DualMovingAverage_10_50", lambda: DualMovingAverage(fast=10, slow=50)),
    ("DualMovingAverage_20_100", lambda: DualMovingAverage(fast=20, slow=100)),
    ("RsiContrarian_14_30_70", lambda: RsiContrarian(N=14, oversold=30, overbought=70)),
    ("RsiContrarian_2_10_90", lambda: RsiContrarian(N=2, oversold=10, overbought=90)),
    ("TsMomentum_20", lambda: TsMomentum(T=20)),
    ("TsMomentum_60", lambda: TsMomentum(T=60)),
    ("SmaCrossover_5_20", lambda: SmaCrossover(fast=5, slow=20)),
    ("SmaCrossover_20_50", lambda: SmaCrossover(fast=20, slow=50)),
    ("ParabolicSAR_default", lambda: ParabolicSAR(step=0.02, af_max=0.2)),
    ("MeanReversionRSIBB_14_30_20_2",
     lambda: MeanReversionRSIBB(rsi_period=14, rsi_long=30, rsi_short=70, bb_period=20, bb_mult=2.0)),
]

ASSETS: list[str] = [
    "GBPUSD", "EURUSD", "USDCHF", "ETHUSD", "BTCUSD", "US30", "US500", "GER30", "XAUUSD",
]
TF = "D1"
TP_SL_RATIOS = [0.5, 0.7, 1.0, 1.5]  # SL = ratio × ATR
TP_OVER_SL = 2.0


def _analyze_trades(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
    if not trades:
        return {"sharpe": 0.0, "wr": 0.0, "n_trades": 0, "mean_pnl": 0.0, "max_dd_pips": 0.0}
    pnls = np.array([t["pips_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    return {
        "sharpe": float(sharpe_daily_from_trades(trades, initial_capital_pips=capital_pips)),
        "wr": float((pnls > 0).mean()),
        "n_trades": int(len(trades)),
        "mean_pnl": float(pnls.mean()),
        "max_dd_pips": float((equity - peak).min()),
    }


def screen_one(strat_name: str, strat_factory: callable, asset: str) -> list[dict[str, Any]]:
    """Pour un (stratégie, asset) : balaye 4 ratios TP/SL et retourne 4 lignes."""
    try:
        df = load_asset(asset, TF)
    except Exception as exc:
        return [{"strat": strat_name, "asset": asset, "error": str(exc)}]
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    if df_train.empty:
        return [{"strat": strat_name, "asset": asset, "error": "train vide"}]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    atr14 = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_pips = atr14 / cfg.pip_size
    atr_mean = float(atr_pips.dropna().mean())
    if atr_mean <= 0:
        return [{"strat": strat_name, "asset": asset, "error": "ATR=0"}]

    # Générer les signaux (1 seule fois — c'est la stratégie qui change, pas le TP/SL)
    try:
        strat = strat_factory()
        signals = strat.generate_signals(df_train)
    except Exception as exc:
        return [{"strat": strat_name, "asset": asset, "error": f"signal: {exc}"}]
    n_signals = int((signals != 0).sum())
    if n_signals < 10:
        return [{"strat": strat_name, "asset": asset, "error": f"only {n_signals} signaux"}]

    results = []
    for ratio in TP_SL_RATIOS:
        sl_pips = max(round(ratio * atr_mean), 1)
        tp_pips = max(round(sl_pips * TP_OVER_SL), 1)
        bt = run_deterministic_backtest(
            df=df_train, signals=signals,
            tp_pips=tp_pips, sl_pips=sl_pips,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size,
        )
        m = _analyze_trades(bt.get("trades", []))
        m.update({
            "strat": strat_name, "asset": asset, "tf": TF,
            "sl_atr_ratio": ratio, "sl_pips": sl_pips, "tp_pips": tp_pips,
            "atr_mean_pips": atr_mean, "n_signals_raw": n_signals,
        })
        results.append(m)
    return results


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"SCREENING {len(STRATEGIES)} stratégies × {len(ASSETS)} actifs × "
          f"{len(TP_SL_RATIOS)} ratios = {len(STRATEGIES)*len(ASSETS)*len(TP_SL_RATIOS)} backtests")
    print(f"Train ≤ {TRAIN_CUTOFF.date()} uniquement — 0 n_trial consommé")
    print("=" * 70)

    all_results: list[dict[str, Any]] = []
    for strat_name, strat_factory in STRATEGIES:
        print(f"\n── {strat_name} ──")
        for asset in ASSETS:
            rows = screen_one(strat_name, strat_factory, asset)
            for r in rows:
                if "error" in r:
                    continue
                all_results.append(r)
            errors = [r["error"] for r in rows if "error" in r]
            if errors:
                print(f"  {asset}: ❌ {errors[0]}")
            else:
                best = max(rows, key=lambda x: x.get("sharpe", -np.inf))
                print(f"  {asset}: meilleur Sharpe={best['sharpe']:+.2f} "
                      f"(WR={best['wr']:.0%}, n={best['n_trades']}, "
                      f"SL={best['sl_atr_ratio']}×ATR)")

    # Tri par Sharpe descendant
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Critères de pré-sélection
    candidates = [
        r for r in all_results
        if r["sharpe"] >= 0.5 and r["wr"] >= 0.35
        and r["n_trades"] >= 30
    ]

    print("\n" + "=" * 90)
    print(f"TOP 20 RÉSULTATS (sur {len(all_results)} backtests valides)")
    print("=" * 90)
    print(f"{'Strat':<32} {'Asset':<8} {'SL/ATR':>7} {'Sharpe':>7} {'WR':>5} {'n':>5} {'mean_pnl':>10}")
    for r in all_results[:20]:
        print(f"{r['strat']:<32} {r['asset']:<8} {r['sl_atr_ratio']:>7.1f} "
              f"{r['sharpe']:>+7.2f} {r['wr']:>5.0%} {r['n_trades']:>5} "
              f"{r['mean_pnl']:>+10.2f}")

    print("\n" + "=" * 70)
    print(f"CANDIDATS pour OOS unique : {len(candidates)} "
          f"(Sharpe≥0.5, WR≥35%, n≥30 sur train)")
    print("=" * 70)
    for r in candidates[:30]:
        print(f"  ✅ {r['strat']:<30} {r['asset']:<8} SL={r['sl_atr_ratio']}×ATR "
              f"→ Sharpe {r['sharpe']:+.2f}, WR {r['wr']:.1%}, n={r['n_trades']}")

    if not candidates:
        print("\n🔴 AUCUN couple (stratégie, actif, ratio) ne passe le critère minimal sur train.")
        print("   → Aucune stratégie technique simple ne montre d'edge sur ce dataset.")
        print("   → Implications : revoir l'approche (timeframe, type de feature, type de marché).")
    else:
        print(f"\n💡 {len(candidates)} candidates. Tester en OOS unique (1 n_trial par couple).")
        print("   Recommandation : prendre le TOP 3-5 distincts par stratégie pour éviter overfitting.")

    out_json = Path("predictions/screen_strategies_train.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"all_results": all_results, "candidates": candidates},
                   indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
