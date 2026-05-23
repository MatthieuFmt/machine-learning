"""Test OOS unique du candidat G-light H1 — Phase G2.

Candidat unique (1/1404 backtests G-light) :
- SmaCrossover(fast=24, slow=120) sur ETHUSD H1, SL=1.5×ATR_train, TP=2×SL.
- Train (≤ 2022-12-31) : Sharpe +0.75, WR 41.6%, n=303, mean_pnl +408.8 pips.

Particularités :
- ETHUSD H1 inclut swap_long=-80, swap_short=-10 pips/nuit (F6 calibration).
- SL/TP figés sur ATR train (1.5× ATR moyen H1 sur train).
- Test : TEST_START → fin de data (2024-01-01 → 2026-05-22).

Critères GO :
- Sharpe test ≥ 0.5
- WR test ≥ 35%
- Δ Sharpe (test - train) ≥ -0.5 (pas d'effondrement)
- n_trades test ≥ 30 (au moins 1.5 ans × 1.2 trades/semaine)

Total : 1 lecture OOS = +1 n_trial (compteur global avancé via read_oos).
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
from app.testing.snooping_guard import read_oos  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

CANDIDATE = {
    "name": "SmaCrossover_24_120_ETHUSD_H1_1.5xATR",
    "asset": "ETHUSD",
    "tf": "H1",
    "strat_factory": lambda: SmaCrossover(fast=24, slow=120),
    "sl_atr_ratio": 1.5,
    "tp_over_sl": 2.0,
    "train_sharpe": 0.75,
    "train_wr": 0.416,
    "train_n": 303,
    "train_mean_pnl": 408.8,
}


def _analyze(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
    if not trades:
        return {
            "sharpe": 0.0, "wr": 0.0, "n_trades": 0,
            "mean_pnl": 0.0, "max_dd_pips": 0.0, "total_pnl": 0.0,
        }
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


def main() -> int:
    set_global_seeds()
    spec = CANDIDATE
    asset, tf = spec["asset"], spec["tf"]

    print("=" * 70)
    print(f"G2 OOS — {spec['name']}")
    print("=" * 70)
    print(f"Train ref : Sharpe={spec['train_sharpe']}, WR={spec['train_wr']:.1%}, "
          f"n={spec['train_n']}, mean_pnl={spec['train_mean_pnl']:+.1f}")
    print(f"+1 n_trial OOS")
    print("=" * 70)

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # ATR figé sur train uniquement (pas de fuite)
    atr14_train = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_pips_train = float((atr14_train / cfg.pip_size).dropna().mean())
    sl_pips = max(round(spec["sl_atr_ratio"] * atr_pips_train), 1)
    tp_pips = max(round(sl_pips * spec["tp_over_sl"]), 1)

    print(f"\n  ATR train H1 moyen : {atr_pips_train:.1f} pips")
    print(f"  SL = {sl_pips} pips, TP = {tp_pips} pips (figés sur train)")
    print(f"  Swap appliqué : long={cfg.swap_long_pips_per_night}, short={cfg.swap_short_pips_per_night} pips/nuit")
    print(f"  Train period : {df_train.index.min().date()} → {df_train.index.max().date()} ({len(df_train)} bars)")
    print(f"  Test period  : {df_test.index.min().date()} → {df_test.index.max().date()} ({len(df_test)} bars)")

    strat = spec["strat_factory"]()

    # ── Re-run train (référence cohérente avec G-light) ─────────────────
    signals_train = strat.generate_signals(df_train)
    bt_train = run_deterministic_backtest(
        df=df_train, signals=signals_train,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_train = _analyze(bt_train.get("trades", []))
    print(f"\n  Train re-run : Sharpe={m_train['sharpe']:+.2f}, WR={m_train['wr']:.1%}, "
          f"n={m_train['n_trades']}, total_pnl={m_train['total_pnl']:+.0f} pips, "
          f"max_dd={m_train['max_dd_pips']:.0f}")

    # ── TEST OOS ─────────────────────────────────────────────────────────
    signals_test = strat.generate_signals(df_test)
    bt_test = run_deterministic_backtest(
        df=df_test, signals=signals_test,
        tp_pips=tp_pips, sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_test = _analyze(bt_test.get("trades", []))
    print(f"\n  TEST OOS     : Sharpe={m_test['sharpe']:+.2f}, WR={m_test['wr']:.1%}, "
          f"n={m_test['n_trades']}, total_pnl={m_test['total_pnl']:+.0f} pips, "
          f"max_dd={m_test['max_dd_pips']:.0f}")

    # Snooping guard : enregistre cet OOS read
    read_oos(
        prompt="test_oos_g2_ethusd_smacross_h1",
        hypothesis=spec["name"],
        sharpe=m_test["sharpe"],
        n_trades=m_test["n_trades"],
    )

    # ── Verdict ──────────────────────────────────────────────────────────
    delta = m_test["sharpe"] - m_train["sharpe"]
    go_sharpe = m_test["sharpe"] >= 0.5
    go_wr = m_test["wr"] >= 0.35
    go_stability = delta >= -0.5
    go_volume = m_test["n_trades"] >= 30
    go = go_sharpe and go_wr and go_stability and go_volume

    print(f"\n  Verdict G2 :")
    print(f"    Sharpe >= 0.5    : {'OK' if go_sharpe else 'FAIL'} ({m_test['sharpe']:+.2f})")
    print(f"    WR >= 35%        : {'OK' if go_wr else 'FAIL'} ({m_test['wr']:.1%})")
    print(f"    Delta Sh >= -0.5 : {'OK' if go_stability else 'FAIL'} ({delta:+.2f})")
    print(f"    n >= 30          : {'OK' if go_volume else 'FAIL'} ({m_test['n_trades']})")
    print(f"\n  ==> {'GO — bascule sur validate_edge() + documentation' if go else 'NO-GO — pivot Phase H'}")

    out = {
        "candidate": spec["name"],
        "asset": asset,
        "tf": tf,
        "strat_params": {"fast": 24, "slow": 120},
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "atr_train_pips": atr_pips_train,
        "swap_long": cfg.swap_long_pips_per_night,
        "swap_short": cfg.swap_short_pips_per_night,
        "metrics_train": m_train,
        "metrics_test": m_test,
        "delta_sharpe": delta,
        "go": go,
        "go_breakdown": {
            "sharpe": go_sharpe,
            "wr": go_wr,
            "stability": go_stability,
            "volume": go_volume,
        },
        "train_ref": {
            "sharpe": spec["train_sharpe"],
            "wr": spec["train_wr"],
            "n": spec["train_n"],
            "mean_pnl": spec["train_mean_pnl"],
        },
    }

    out_json = Path("predictions/test_oos_g2_ethusd_smacross_h1.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_json}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
