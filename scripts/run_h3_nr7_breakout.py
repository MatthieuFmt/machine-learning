"""Phase H3 — Backtest NR7 Volatility Breakout (Crabel) sur US30/US500 D1.

Stratégie déterministe (Étape 1 cascade) : long ou short selon le sens du
breakout du jour Narrow Range 7. TP = 2×range_NR, SL = 1×range_NR
(R:R = 2:1, breakeven WR théorique = 33%).

Train ≤ 2022-12-31, OOS ≥ 2024-01-01.

Critères GO par actif :
    - Sharpe OOS ≥ 0.7
    - Mean PnL OOS > 0
    - p-value bootstrap < 0.10
    - n_trades OOS ≥ 30
    - WR OOS > 35% (sécurité vs breakeven théorique 33%)

Si ≥ 1 actif GO, on bascule sur Étape 2 cascade (ML méta-labeling).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.strategies.volatility_breakout import simulate_nr_breakout_trades  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

logger = get_logger(__name__)

ASSETS = ("US30", "US500")
TF = "D1"

# Hyperparams figés (Crabel) — pas de tuning
LOOKBACK = 7
TP_MULT = 2.0
SL_MULT = 1.0


def _bootstrap_pvalue(pnls: np.ndarray, n_iter: int = 10_000, seed: int = 42) -> float:
    if len(pnls) < 5:
        return 1.0
    rng = np.random.default_rng(seed)
    centered = pnls - pnls.mean()
    observed = pnls.mean()
    if observed <= 0:
        return 1.0
    n = len(pnls)
    boots = np.array([
        rng.choice(centered, size=n, replace=True).mean() for _ in range(n_iter)
    ])
    return float((boots >= observed).mean())


def _analyze(trades: list[dict], label: str) -> dict[str, Any]:
    if not trades:
        return {
            "label": label, "n_trades": 0,
            "sharpe": 0.0, "mean_pnl": 0.0, "median_pnl": 0.0,
            "wr": 0.0, "total_pnl": 0.0, "max_dd_pips": 0.0,
            "p_value_bootstrap": 1.0,
            "n_long": 0, "n_short": 0,
            "tp_rate": 0.0, "sl_rate": 0.0, "time_stop_rate": 0.0,
        }
    pnls = np.array([t["pips_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    signals = np.array([t["signal"] for t in trades])
    reasons = [t["exit_reason"] for t in trades]
    return {
        "label": label,
        "n_trades": len(trades),
        "sharpe": float(sharpe_daily_from_trades(trades)),
        "mean_pnl": float(pnls.mean()),
        "median_pnl": float(np.median(pnls)),
        "wr": float((pnls > 0).mean()),
        "total_pnl": float(pnls.sum()),
        "max_dd_pips": float((equity - peak).min()),
        "p_value_bootstrap": _bootstrap_pvalue(pnls),
        "n_long": int((signals == 1).sum()),
        "n_short": int((signals == -1).sum()),
        "tp_rate": float(reasons.count("tp") / len(reasons)),
        "sl_rate": float(reasons.count("sl") / len(reasons)),
        "time_stop_rate": float(reasons.count("time_stop") / len(reasons)),
    }


def _print_metrics(m: dict[str, Any]) -> None:
    print(f"  {m['label']}: n={m['n_trades']} (L={m['n_long']}/S={m['n_short']}), "
          f"Sharpe={m['sharpe']:+.2f}, WR={m['wr']:.1%}, "
          f"mean={m['mean_pnl']:+.1f} pips, median={m['median_pnl']:+.1f}, "
          f"total={m['total_pnl']:+.0f}, max_dd={m['max_dd_pips']:.0f}, "
          f"p={m['p_value_bootstrap']:.3f}")
    print(f"    exit: TP={m['tp_rate']:.0%}, SL={m['sl_rate']:.0%}, "
          f"time_stop={m['time_stop_rate']:.0%}")


def _run_one_asset(asset: str) -> dict[str, Any]:
    print("\n" + "─" * 70)
    print(f"  {asset} {TF}")
    print("─" * 70)

    cfg = ASSET_CONFIGS[asset]
    print(f"  Cfg : spread={cfg.spread_pips}, slippage={cfg.slippage_pips}, "
          f"commission={cfg.commission_pips} pips, pip_size={cfg.pip_size}")
    print(f"        swap long={cfg.swap_long_pips_per_night}, "
          f"short={cfg.swap_short_pips_per_night} pips/nuit")

    df = load_asset(asset, TF)
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    print(f"  Data : {len(df)} bars total "
          f"({df.index.min().date()} → {df.index.max().date()})")
    print(f"         Train : {len(df_train)} bars (≤ {TRAIN_CUTOFF.date()})")
    print(f"         Test  : {len(df_test)} bars (≥ {TEST_START.date()})")

    # ── Simulation Train ─────────────────────────────────────────────
    print("\n  ── TRAIN ──")
    trades_train = simulate_nr_breakout_trades(
        df_train, cfg, lookback=LOOKBACK, tp_mult=TP_MULT, sl_mult=SL_MULT,
    )
    m_train = _analyze(trades_train, "Train")
    _print_metrics(m_train)

    # ── Simulation OOS ───────────────────────────────────────────────
    print("\n  ── TEST (OOS) ──")
    trades_test = simulate_nr_breakout_trades(
        df_test, cfg, lookback=LOOKBACK, tp_mult=TP_MULT, sl_mult=SL_MULT,
    )
    m_test = _analyze(trades_test, "Test")
    _print_metrics(m_test)

    # ── Verdict ─────────────────────────────────────────────────────
    go_sharpe = m_test["sharpe"] >= 0.7
    go_mean = m_test["mean_pnl"] > 0
    go_pvalue = m_test["p_value_bootstrap"] < 0.10
    go_ntrades = m_test["n_trades"] >= 30
    go_wr = m_test["wr"] > 0.35
    go = go_sharpe and go_mean and go_pvalue and go_ntrades and go_wr

    print(f"\n  VERDICT {asset} :")
    print(f"    Sharpe OOS ≥ 0.7   : {'OK' if go_sharpe else 'FAIL'} ({m_test['sharpe']:+.2f})")
    print(f"    Mean PnL OOS > 0   : {'OK' if go_mean else 'FAIL'} ({m_test['mean_pnl']:+.1f})")
    print(f"    p-value bootstrap  : {'OK' if go_pvalue else 'FAIL'} ({m_test['p_value_bootstrap']:.3f})")
    print(f"    n_trades OOS ≥ 30  : {'OK' if go_ntrades else 'FAIL'} ({m_test['n_trades']})")
    print(f"    WR OOS > 35%       : {'OK' if go_wr else 'FAIL'} ({m_test['wr']:.1%})")
    print(f"    ==> {'GO Étape 2 (ML méta-labeling)' if go else 'NO-GO Étape 1'}")

    return {
        "strategy": "nr7_volatility_breakout",
        "asset": asset,
        "tf": TF,
        "lookback": LOOKBACK,
        "tp_mult": TP_MULT,
        "sl_mult": SL_MULT,
        "train_cutoff": str(TRAIN_CUTOFF),
        "test_start": str(TEST_START),
        "asset_config": {
            "spread_pips": cfg.spread_pips,
            "slippage_pips": cfg.slippage_pips,
            "commission_pips": cfg.commission_pips,
            "pip_size": cfg.pip_size,
            "swap_long_pips_per_night": cfg.swap_long_pips_per_night,
            "swap_short_pips_per_night": cfg.swap_short_pips_per_night,
        },
        "metrics_train": m_train,
        "metrics_test": m_test,
        "go": go,
        "go_breakdown": {
            "sharpe": go_sharpe,
            "mean_positive": go_mean,
            "pvalue": go_pvalue,
            "n_trades": go_ntrades,
            "wr": go_wr,
        },
        "trades_train_sample": trades_train[:5],
        "trades_test_sample": trades_test[:5],
        "n_trades_train": len(trades_train),
        "n_trades_test": len(trades_test),
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print("PHASE H3 — NR7 Volatility Breakout (Crabel, Étape 1 cascade)")
    print(f"Actifs : {ASSETS} sur {TF}")
    print(f"NR{LOOKBACK} : range(J) == min(range(J-{LOOKBACK-1}..J))")
    print(f"Stops à High(J)/Low(J), TP = {TP_MULT}×range, SL = {SL_MULT}×range")
    print(f"Long ET short symétriques")
    print("=" * 70)

    all_results: dict[str, dict[str, Any]] = {}
    for asset in ASSETS:
        try:
            all_results[asset] = _run_one_asset(asset)
        except Exception as exc:
            logger.exception("nr7_run_failed", extra={"context": {"asset": asset}})
            print(f"\n  ⚠️ {asset} ÉCHEC : {exc}")
            all_results[asset] = {"asset": asset, "error": str(exc), "go": False}

    # ── Sauvegarde JSON par actif ───────────────────────────────────
    out_dir = Path("predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    for asset, res in all_results.items():
        out_path = out_dir / f"h3_nr7_breakout_{asset.lower()}.json"
        out_path.write_text(
            json.dumps(res, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  JSON sauvegardé : {out_path}")

    # ── Synthèse ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SYNTHÈSE PHASE H3")
    print("=" * 70)
    print(f"{'Actif':<8} {'Sharpe OOS':>10} {'Mean OOS':>10} {'WR OOS':>8} {'n_OOS':>6} {'p-val':>6} {'GO?':>4}")
    print("─" * 60)
    for asset, res in all_results.items():
        if "error" in res:
            print(f"{asset:<8} ERROR : {res['error']}")
            continue
        m = res["metrics_test"]
        go_mark = "✅" if res["go"] else "❌"
        print(f"{asset:<8} {m['sharpe']:>+10.2f} {m['mean_pnl']:>+10.1f} "
              f"{m['wr']:>7.1%} {m['n_trades']:>6d} {m['p_value_bootstrap']:>6.3f} {go_mark:>4}")

    n_go = sum(1 for r in all_results.values() if r.get("go", False))
    print(f"\n  {n_go}/{len(ASSETS)} actifs GO Étape 1")
    if n_go > 0:
        print(f"  ==> Bascule Étape 2 (ML méta-labeling) sur les actifs GO")
    else:
        print(f"  ==> 0 actif GO — bilan Phase H, considérer pivot stratégies différentes")

    return 0 if n_go > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
