"""Test OOS unique des 2 candidates issues du screening (Sharpe train â‰¥ 0.5).

Candidates :
1. RsiContrarian(14, 30/70) sur US500 D1, SL=1.0Ã—ATR_train, TP=2Ã—SL.
   Train : Sharpe +0.54, WR 50%, n=135.
2. RsiContrarian(2, 10/90) sur ETHUSD D1, SL=1.5Ã—ATR_train, TP=2Ã—SL.
   Train : Sharpe +0.51, WR 52%, n=197.

ParamÃ¨tres FIGÃ‰S sur train (ATR moyen calculÃ© sur train â‰¤ 2022 uniquement).
Pas de re-calibration sur test. 2 lectures OOS uniques (2 n_trials).

CritÃ¨res de succÃ¨s par couple (pour GO sur ce couple) :
- Sharpe test â‰¥ 0.5
- WR test â‰¥ 35%
- Sharpe test - Sharpe train â‰¥ -0.5 (pas d'effondrement OOS)
- n_trades test â‰¥ 10 (sinon variance Ã©norme)
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
from app.strategies.rsi_contrarian import RsiContrarian  # noqa: E402
from app.testing.snooping_guard import read_oos  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "RsiContrarian_14_30_70_US500_1.0xATR",
        "asset": "US500",
        "params": {"N": 14, "oversold": 30, "overbought": 70},
        "sl_atr_ratio": 1.0,
        "tp_over_sl": 2.0,
        "train_sharpe": 0.54,
        "train_wr": 0.50,
        "train_n": 135,
    },
    {
        "name": "RsiContrarian_2_10_90_ETHUSD_1.5xATR",
        "asset": "ETHUSD",
        "params": {"N": 2, "oversold": 10, "overbought": 90},
        "sl_atr_ratio": 1.5,
        "tp_over_sl": 2.0,
        "train_sharpe": 0.51,
        "train_wr": 0.52,
        "train_n": 197,
    },
]


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
        "total_pnl": float(pnls.sum()),
    }


def test_one_candidate(spec: dict[str, Any]) -> dict[str, Any]:
    asset = spec["asset"]
    print(f"\n{'='*70}")
    print(f"[OOS test] {spec['name']}")
    print(f"{'='*70}")

    df = load_asset(asset, "D1")
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # ATR figÃ© sur TRAIN uniquement (pas de leak du test)
    atr14_train = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_pips_train = float((atr14_train / cfg.pip_size).dropna().mean())
    sl_pips = max(round(spec["sl_atr_ratio"] * atr_pips_train), 1)
    tp_pips = max(round(sl_pips * spec["tp_over_sl"]), 1)
    print(f"  ATR train moyen : {atr_pips_train:.1f} pips")
    print(f"  SL = {sl_pips} pips, TP = {tp_pips} pips (figÃ©s sur train)")
    print(f"  Test : {df_test.index.min().date()} â†’ {df_test.index.max().date()} "
          f"({len(df_test)} barres)")

    # StratÃ©gie figÃ©e
    strat = RsiContrarian(**spec["params"])

    # Backtest TRAIN (rÃ©fÃ©rence, pas un n_trial)
    signals_train = strat.generate_signals(df_train)
    bt_train = run_deterministic_backtest(
        df=df_train, signals=signals_train,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_train = _analyze(bt_train.get("trades", []))
    print(f"  Train  : Sharpe={m_train['sharpe']:+.2f}, WR={m_train['wr']:.1%}, "
          f"n={m_train['n_trades']}, total_pnl={m_train['total_pnl']:+.0f} pips")

    # Backtest TEST (OOS unique â€” 1 n_trial)
    signals_test = strat.generate_signals(df_test)
    bt_test = run_deterministic_backtest(
        df=df_test, signals=signals_test,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_test = _analyze(bt_test.get("trades", []))
    print(f"  TEST   : Sharpe={m_test['sharpe']:+.2f}, WR={m_test['wr']:.1%}, "
          f"n={m_test['n_trades']}, total_pnl={m_test['total_pnl']:+.0f} pips, "
          f"max_dd={m_test['max_dd_pips']:.0f} pips")

    # Snooping log
    read_oos(
        prompt="test_oos_rsi_candidates",
        hypothesis=spec["name"],
        sharpe=m_test["sharpe"],
        n_trades=m_test["n_trades"],
    )

    # Verdict
    delta_sharpe = m_test["sharpe"] - m_train["sharpe"]
    go_sharpe = m_test["sharpe"] >= 0.5
    go_wr = m_test["wr"] >= 0.35
    go_stability = delta_sharpe >= -0.5
    go_volume = m_test["n_trades"] >= 10
    go = go_sharpe and go_wr and go_stability and go_volume

    print(f"\n  Verdict couple :")
    print(f"    Sharpe test â‰¥ 0.5      : {'âœ…' if go_sharpe else 'âŒ'} ({m_test['sharpe']:+.2f})")
    print(f"    WR test â‰¥ 35%           : {'âœ…' if go_wr else 'âŒ'} ({m_test['wr']:.1%})")
    print(f"    Î” Sharpe â‰¥ -0.5         : {'âœ…' if go_stability else 'âŒ'} ({delta_sharpe:+.2f})")
    print(f"    n_trades test â‰¥ 10      : {'âœ…' if go_volume else 'âŒ'} ({m_test['n_trades']})")
    print(f"  â†’ {'ðŸŽ¯ GO' if go else 'âŒ NO-GO'}")

    return {
        "candidate": spec["name"],
        "asset": asset, "params": spec["params"],
        "sl_pips": sl_pips, "tp_pips": tp_pips,
        "atr_train_pips": atr_pips_train,
        "metrics_train": m_train,
        "metrics_test": m_test,
        "delta_sharpe": delta_sharpe,
        "go_sharpe": go_sharpe, "go_wr": go_wr,
        "go_stability": go_stability, "go_volume": go_volume,
        "go": go,
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print("TEST OOS â€” 2 candidates issues du screening (Sharpe train â‰¥ 0.5)")
    print("Lectures OOS : 2 (les 2 candidates â†’ +2 n_trials)")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        try:
            r = test_one_candidate(spec)
            results.append(r)
        except Exception as exc:
            print(f"  âŒ {spec['name']} : {exc}")
            results.append({"candidate": spec["name"], "error": str(exc)})

    # RÃ©cap
    print("\n" + "=" * 70)
    print("RÃ‰CAP")
    print("=" * 70)
    print(f"{'Candidate':<45} {'Train Sh':>8} {'Test Sh':>8} {'Test WR':>8} {'Test n':>7} {'GO?':>5}")
    n_go = 0
    for r in results:
        if r.get("error"):
            print(f"{r['candidate']:<45} ERROR : {r['error']}")
            continue
        mt = r["metrics_test"]
        mtr = r["metrics_train"]
        verdict = "ðŸŽ¯ GO" if r["go"] else "âŒ"
        if r["go"]:
            n_go += 1
        print(f"{r['candidate']:<45} {mtr['sharpe']:>+8.2f} {mt['sharpe']:>+8.2f} "
              f"{mt['wr']:>8.1%} {mt['n_trades']:>7} {verdict:>5}")

    print(f"\n{n_go}/{len(CANDIDATES)} couples passent les 4 critÃ¨res.")

    out_json = Path("predictions/test_oos_rsi_candidates.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardÃ© : {out_json}")

    return 0 if n_go > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
