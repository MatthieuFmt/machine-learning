"""Test OOS unique du candidat H4 issu du screening.

Candidat principal :
- TsMomentum(T=60) sur ETHUSD H4, SL=1.5Ã—ATR_train, TP=2Ã—SL.
  Train : Sharpe +0.59, WR 37%, n=1330, mean_pnl +341.

Candidats bonus (Sharpe train 0.37-0.38, sous le seuil 0.5 mais cohÃ©rents) :
- SmaCrossover_10_50 sur GBPUSD H4, SL=1.0Ã—ATR_train.
- SmaCrossover_10_50 sur EURUSD H4, SL=1.0Ã—ATR_train.

Total : 3 lectures OOS = 3 n_trials (de 50 Ã  53).

CritÃ¨res GO :
- Sharpe test â‰¥ 0.5
- WR test â‰¥ 35%
- Î” Sharpe (test-train) â‰¥ -0.5 (pas d'effondrement)
- n_trades test â‰¥ 50 (Ã©chantillon plus large attendu en H4)
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
from app.strategies.sma_crossover import SmaCrossover  # noqa: E402
from app.strategies.ts_momentum import TsMomentum  # noqa: E402
from app.testing.snooping_guard import read_oos  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "TsMomentum_60_ETHUSD_H4_1.5xATR",
        "asset": "ETHUSD", "tf": "H4",
        "strat_factory": lambda: TsMomentum(T=60),
        "sl_atr_ratio": 1.5, "tp_over_sl": 2.0,
        "train_sharpe": 0.59, "train_wr": 0.37, "train_n": 1330,
        "tier": "primary",  # seul candidat sur critÃ¨re â‰¥0.5
    },
    {
        "name": "SmaCrossover_10_50_GBPUSD_H4_1.0xATR",
        "asset": "GBPUSD", "tf": "H4",
        "strat_factory": lambda: SmaCrossover(fast=10, slow=50),
        "sl_atr_ratio": 1.0, "tp_over_sl": 2.0,
        "train_sharpe": 0.38, "train_wr": 0.38, "train_n": 489,
        "tier": "bonus",  # sous le seuil mais cohÃ©rent multi-asset
    },
    {
        "name": "SmaCrossover_10_50_EURUSD_H4_1.0xATR",
        "asset": "EURUSD", "tf": "H4",
        "strat_factory": lambda: SmaCrossover(fast=10, slow=50),
        "sl_atr_ratio": 1.0, "tp_over_sl": 2.0,
        "train_sharpe": 0.38, "train_wr": 0.38, "train_n": 510,
        "tier": "bonus",
    },
]


def _analyze(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
    if not trades:
        return {"sharpe": 0.0, "wr": 0.0, "n_trades": 0, "mean_pnl": 0.0, "max_dd_pips": 0.0, "total_pnl": 0.0}
    pnls = np.array([t["pips_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    return {
        "sharpe": float(sharpe_daily_from_trades(trades, initial_capital_pips=capital_pips)),
        "wr": float((pnls > 0).mean()),
        "n_trades": int(len(trades)),
        "mean_pnl": float(pnls.mean()),
        "max_dd_pips": float((equity - peak).min()),
        "total_pnl": float(pnls.sum()),
    }


def test_one(spec: dict[str, Any]) -> dict[str, Any]:
    asset, tf = spec["asset"], spec["tf"]
    print(f"\n{'='*70}")
    print(f"[OOS test {spec['tier']}] {spec['name']}")
    print(f"{'='*70}")

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # ATR figÃ© sur train
    atr14_train = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_pips_train = float((atr14_train / cfg.pip_size).dropna().mean())
    sl_pips = max(round(spec["sl_atr_ratio"] * atr_pips_train), 1)
    tp_pips = max(round(sl_pips * spec["tp_over_sl"]), 1)
    print(f"  ATR train H4 moyen : {atr_pips_train:.1f} pips")
    print(f"  SL = {sl_pips} pips, TP = {tp_pips} pips (figÃ©s sur train)")
    print(f"  Test : {df_test.index.min().date()} â†’ {df_test.index.max().date()} ({len(df_test)} bars)")

    strat = spec["strat_factory"]()

    # Train (rÃ©fÃ©rence)
    signals_train = strat.generate_signals(df_train)
    bt_train = run_deterministic_backtest(
        df=df_train, signals=signals_train,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_train = _analyze(bt_train.get("trades", []))
    print(f"  Train  : Sharpe={m_train['sharpe']:+.2f}, WR={m_train['wr']:.1%}, n={m_train['n_trades']}, "
          f"total_pnl={m_train['total_pnl']:+.0f} pips")

    # TEST (OOS)
    signals_test = strat.generate_signals(df_test)
    bt_test = run_deterministic_backtest(
        df=df_test, signals=signals_test,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_test = _analyze(bt_test.get("trades", []))
    print(f"  TEST   : Sharpe={m_test['sharpe']:+.2f}, WR={m_test['wr']:.1%}, n={m_test['n_trades']}, "
          f"total_pnl={m_test['total_pnl']:+.0f} pips, max_dd={m_test['max_dd_pips']:.0f} pips")

    read_oos(
        prompt="test_oos_h4_candidate",
        hypothesis=spec["name"],
        sharpe=m_test["sharpe"],
        n_trades=m_test["n_trades"],
    )

    delta = m_test["sharpe"] - m_train["sharpe"]
    go_sharpe = m_test["sharpe"] >= 0.5
    go_wr = m_test["wr"] >= 0.35
    go_stability = delta >= -0.5
    go_volume = m_test["n_trades"] >= 50
    go = go_sharpe and go_wr and go_stability and go_volume

    print(f"\n  Verdict :")
    print(f"    Sharpe â‰¥ 0.5    : {'âœ…' if go_sharpe else 'âŒ'} ({m_test['sharpe']:+.2f})")
    print(f"    WR â‰¥ 35%        : {'âœ…' if go_wr else 'âŒ'} ({m_test['wr']:.1%})")
    print(f"    Î” Sharpe â‰¥ -0.5 : {'âœ…' if go_stability else 'âŒ'} ({delta:+.2f})")
    print(f"    n â‰¥ 50          : {'âœ…' if go_volume else 'âŒ'} ({m_test['n_trades']})")
    print(f"  â†’ {'ðŸŽ¯ GO' if go else 'âŒ NO-GO'}")

    return {
        "candidate": spec["name"], "tier": spec["tier"],
        "asset": asset, "tf": tf,
        "sl_pips": sl_pips, "tp_pips": tp_pips,
        "atr_train_pips": atr_pips_train,
        "metrics_train": m_train, "metrics_test": m_test,
        "delta_sharpe": delta, "go": go,
        "go_sharpe": go_sharpe, "go_wr": go_wr,
        "go_stability": go_stability, "go_volume": go_volume,
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"TEST OOS H4 â€” {len(CANDIDATES)} candidats (1 primary + 2 bonus)")
    print(f"Lectures OOS : {len(CANDIDATES)} (+{len(CANDIDATES)} n_trials)")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        try:
            r = test_one(spec)
            results.append(r)
        except Exception as exc:
            print(f"  âŒ {spec['name']} : {exc}")
            results.append({"candidate": spec["name"], "error": str(exc)})

    print("\n" + "=" * 90)
    print("RÃ‰CAP")
    print("=" * 90)
    print(f"{'Candidat':<45} {'Tier':>8} {'Tr Sh':>7} {'Tst Sh':>7} {'Tst WR':>7} {'Tst n':>6} {'GO?':>5}")
    n_go = 0
    for r in results:
        if r.get("error"):
            print(f"{r['candidate']:<45} ERROR : {r['error']}")
            continue
        mt = r["metrics_test"]; mtr = r["metrics_train"]
        verdict = "ðŸŽ¯ GO" if r["go"] else "âŒ"
        if r["go"]:
            n_go += 1
        print(f"{r['candidate']:<45} {r['tier']:>8} "
              f"{mtr['sharpe']:>+7.2f} {mt['sharpe']:>+7.2f} "
              f"{mt['wr']:>7.1%} {mt['n_trades']:>6} {verdict:>5}")
    print(f"\n{n_go}/{len(CANDIDATES)} candidats passent les 4 critÃ¨res.")

    out_json = Path("predictions/test_oos_h4_candidate.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardÃ© : {out_json}")
    return 0 if n_go > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
