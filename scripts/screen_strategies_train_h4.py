"""Screening intraday H4 â€” 13 stratÃ©gies Ã— 9 actifs Ã— 4 ratios = 468 backtests.

Pivot aprÃ¨s Ã©chec du D1 : tester si les patterns intraday capturent du
signal. H4 = compromis volume (â‰¥ 25k bars / actif) / temps de calcul.

DiffÃ©rences clÃ©s vs D1 :
- ATR H4 plus petit (la stratÃ©gie de TP/SL ATR-based s'adapte automatiquement).
- Plus de trades (~6Ã— plus de signaux qu'en D1).
- Frictions proportionnellement plus impactantes (Ã  surveiller via mean_pnl).
- Filtre sessions possible mais non testÃ© ici (params dÃ©faut).

Train â‰¤ 2022 uniquement. ZÃ©ro n_trial consommÃ©.
Si signal dÃ©tectÃ© sur H4 â†’ screener H1 ensuite (script sÃ©parÃ©).
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

# Params adaptÃ©s pour H4 â€” fenÃªtres plus longues pour compenser le bruit intraday.
# Sur H4, une "journÃ©e" = 6 bars, donc N=20 H4 â‰ˆ 3 jours D1.
STRATEGIES: list[tuple[str, callable]] = [
    ("BollingerBands_50_2", lambda: BollingerBands(N=50, K=2.0)),
    ("BollingerBands_30_2_5", lambda: BollingerBands(N=30, K=2.5)),
    ("KeltnerChannel_30_2", lambda: KeltnerChannel(period=30, mult=2.0)),
    ("DualMovingAverage_30_120", lambda: DualMovingAverage(fast=30, slow=120)),
    ("DualMovingAverage_60_200", lambda: DualMovingAverage(fast=60, slow=200)),
    ("RsiContrarian_14_30_70", lambda: RsiContrarian(N=14, oversold=30, overbought=70)),
    ("RsiContrarian_2_10_90", lambda: RsiContrarian(N=2, oversold=10, overbought=90)),
    ("RsiContrarian_28_20_80", lambda: RsiContrarian(N=28, oversold=20, overbought=80)),
    ("TsMomentum_60", lambda: TsMomentum(T=60)),
    ("TsMomentum_120", lambda: TsMomentum(T=120)),
    ("SmaCrossover_10_50", lambda: SmaCrossover(fast=10, slow=50)),
    ("ParabolicSAR_default", lambda: ParabolicSAR(step=0.02, af_max=0.2)),
    ("MeanReversionRSIBB_14_30_30_2",
     lambda: MeanReversionRSIBB(rsi_period=14, rsi_long=30, rsi_short=70, bb_period=30, bb_mult=2.0)),
]

ASSETS: list[str] = [
    "GBPUSD", "EURUSD", "USDCHF", "ETHUSD", "BTCUSD", "US30", "US500", "GER30", "XAUUSD",
]
TF = "H4"
TP_SL_RATIOS = [0.5, 0.7, 1.0, 1.5]
TP_OVER_SL = 2.0


def _analyze(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
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
    try:
        df = load_asset(asset, TF)
    except Exception as exc:
        return [{"strat": strat_name, "asset": asset, "error": str(exc)[:80]}]
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

    try:
        strat = strat_factory()
        signals = strat.generate_signals(df_train)
    except Exception as exc:
        return [{"strat": strat_name, "asset": asset, "error": f"signal: {exc}"}]
    n_signals = int((signals != 0).sum())
    if n_signals < 30:
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
            slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
        )
        m = _analyze(bt.get("trades", []))
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
    print(f"SCREENING H4 â€” {len(STRATEGIES)} strats Ã— {len(ASSETS)} actifs Ã— "
          f"{len(TP_SL_RATIOS)} ratios = {len(STRATEGIES)*len(ASSETS)*len(TP_SL_RATIOS)} backtests")
    print(f"Train â‰¤ {TRAIN_CUTOFF.date()} uniquement â€” 0 n_trial consommÃ©")
    print("=" * 70)

    all_results: list[dict[str, Any]] = []
    skipped: list[str] = []

    for strat_name, strat_factory in STRATEGIES:
        print(f"\nâ”€â”€ {strat_name} â”€â”€")
        for asset in ASSETS:
            rows = screen_one(strat_name, strat_factory, asset)
            ok_rows = [r for r in rows if "error" not in r]
            error_rows = [r for r in rows if "error" in r]
            for r in ok_rows:
                all_results.append(r)
            if error_rows:
                err = error_rows[0]["error"]
                skipped.append(f"{strat_name}/{asset}: {err}")
                print(f"  {asset}: âŒ {err}")
            elif ok_rows:
                best = max(ok_rows, key=lambda x: x["sharpe"])
                print(f"  {asset}: meilleur Sharpe={best['sharpe']:+.2f} "
                      f"(WR={best['wr']:.0%}, n={best['n_trades']}, "
                      f"SL={best['sl_atr_ratio']}Ã—ATR)")

    all_results.sort(key=lambda x: x["sharpe"], reverse=True)
    candidates = [
        r for r in all_results
        if r["sharpe"] >= 0.5 and r["wr"] >= 0.35
        and r["n_trades"] >= 30
    ]

    print("\n" + "=" * 90)
    print(f"TOP 20 RÃ‰SULTATS (sur {len(all_results)} backtests valides, {len(skipped)} skipped)")
    print("=" * 90)
    print(f"{'Strat':<35} {'Asset':<8} {'SL/ATR':>7} {'Sharpe':>7} {'WR':>5} {'n':>6} {'mean_pnl':>10}")
    for r in all_results[:20]:
        print(f"{r['strat']:<35} {r['asset']:<8} {r['sl_atr_ratio']:>7.1f} "
              f"{r['sharpe']:>+7.2f} {r['wr']:>5.0%} {r['n_trades']:>6} "
              f"{r['mean_pnl']:>+10.2f}")

    print("\n" + "=" * 70)
    print(f"CANDIDATS pour OOS unique : {len(candidates)} "
          f"(Sharpeâ‰¥0.5, WRâ‰¥35%, nâ‰¥30 sur train)")
    print("=" * 70)
    for r in candidates[:30]:
        print(f"  âœ… {r['strat']:<30} {r['asset']:<8} SL={r['sl_atr_ratio']}Ã—ATR "
              f"â†’ Sharpe {r['sharpe']:+.2f}, WR {r['wr']:.1%}, n={r['n_trades']}, "
              f"mean_pnl={r['mean_pnl']:+.1f}")

    if not candidates:
        print("\nðŸ”´ AUCUN couple ne passe le critÃ¨re minimal H4.")
        print("   â†’ Les patterns intraday H4 n'offrent pas plus d'edge que D1.")
    else:
        print(f"\nðŸ’¡ {len(candidates)} candidates H4. ")
        print("   Si l'un est solide â†’ screener H1 sur les meilleurs strats/actifs.")

    out_json = Path("predictions/screen_strategies_train_h4.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({
            "all_results": all_results,
            "candidates": candidates,
            "skipped": skipped,
        }, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardÃ© : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
