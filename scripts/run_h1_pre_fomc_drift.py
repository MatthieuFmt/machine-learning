"""Phase H1 — Backtest Pre-FOMC drift sur US500 H1 (Niveau 1 cascade).

Stratégie déterministe sans ML :
    - Long US500 à FOMC - 24h
    - Close à FOMC - 1h
    - Tous les FOMC Statement scheduled depuis 2010

Pas de tuning. Une hypothèse théorique (Lucca-Moench 2015) testée une fois.

Train ≤ 2022-12-31 : ~96 FOMC events (déjà 0 n_trial car hypothèse a priori)
Test  ≥ 2024-01-01 : ~16 FOMC events

Critères GO (Étape 1 cascade, avant ajout ML/LLM) :
    - Sharpe OOS ≥ 0.7 (Lucca-Moench publié à ~1.0+ OOS)
    - Mean return OOS > 0
    - p-value bootstrap < 0.10
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

from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.strategies.pre_fomc_drift import (  # noqa: E402
    load_fomc_announcement_times,
    simulate_pre_fomc_trades,
)

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

logger = get_logger(__name__)

ASSET = "US500"
TF = "H1"


def _bootstrap_pvalue(pnls: np.ndarray, n_iter: int = 10_000, seed: int = 42) -> float:
    """P-value bootstrap unilatéral : H0 = mean(pnl) ≤ 0.

    Returns la proba sous H0 d'observer une mean au moins aussi grande
    en ré-échantillonnant avec recentrage.
    """
    if len(pnls) < 5:
        return 1.0
    rng = np.random.default_rng(seed)
    centered = pnls - pnls.mean()  # H0 : mean = 0
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
        }
    pnls = np.array([t["pips_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
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
    }


def _print_metrics(m: dict[str, Any]) -> None:
    print(f"  {m['label']}: n={m['n_trades']}, "
          f"Sharpe={m['sharpe']:+.2f}, WR={m['wr']:.1%}, "
          f"mean={m['mean_pnl']:+.1f} pips, median={m['median_pnl']:+.1f} pips, "
          f"total={m['total_pnl']:+.0f} pips, max_dd={m['max_dd_pips']:.0f} pips, "
          f"p={m['p_value_bootstrap']:.3f}")


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"PHASE H1 — Pre-FOMC drift sur {ASSET} {TF}")
    print("Stratégie : long à FOMC-24h, close à FOMC-1h")
    print("=" * 70)

    cfg = ASSET_CONFIGS[ASSET]
    print(f"\nCfg {ASSET} :")
    print(f"  spread={cfg.spread_pips}, slippage={cfg.slippage_pips}, "
          f"commission={cfg.commission_pips} pips")
    print(f"  pip_size={cfg.pip_size}, pip_value_eur={cfg.pip_value_eur}")
    print(f"  swap long={cfg.swap_long_pips_per_night}, "
          f"short={cfg.swap_short_pips_per_night} pips/nuit")

    # ── FOMC events ──────────────────────────────────────────────────────
    fomc_all = load_fomc_announcement_times(start_year=2010, end_year=2026)
    print(f"\n{len(fomc_all)} FOMC Statement chargés "
          f"({fomc_all.min().date()} → {fomc_all.max().date()})")

    fomc_train = fomc_all[fomc_all <= TRAIN_CUTOFF]
    fomc_test = fomc_all[fomc_all >= TEST_START]
    print(f"  Train (≤ {TRAIN_CUTOFF.date()}) : {len(fomc_train)} events")
    print(f"  Test  (≥ {TEST_START.date()})  : {len(fomc_test)} events")

    # ── US500 H1 ─────────────────────────────────────────────────────────
    df = load_asset(ASSET, TF)
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    print(f"\n{ASSET} {TF} : {len(df)} bars total "
          f"({df.index.min().date()} → {df.index.max().date()})")

    # ── Simulation Train ─────────────────────────────────────────────────
    print("\n── TRAIN ──")
    trades_train = simulate_pre_fomc_trades(
        df_train,
        fomc_times=fomc_train,
        spread_pips=cfg.spread_pips,
        slippage_pips=cfg.slippage_pips,
        commission_pips=cfg.commission_pips,
        pip_size=cfg.pip_size,
        swap_long_pips_per_night=cfg.swap_long_pips_per_night,
    )
    m_train = _analyze(trades_train, "Train")
    _print_metrics(m_train)

    # ── Simulation Test (OOS) ────────────────────────────────────────────
    print("\n── TEST (OOS) ──")
    trades_test = simulate_pre_fomc_trades(
        df_test,
        fomc_times=fomc_test,
        spread_pips=cfg.spread_pips,
        slippage_pips=cfg.slippage_pips,
        commission_pips=cfg.commission_pips,
        pip_size=cfg.pip_size,
        swap_long_pips_per_night=cfg.swap_long_pips_per_night,
    )
    m_test = _analyze(trades_test, "Test")
    _print_metrics(m_test)

    # ── Verdict Étape 1 cascade ──────────────────────────────────────────
    go_sharpe = m_test["sharpe"] >= 0.7
    go_mean_pos = m_test["mean_pnl"] > 0
    go_pvalue = m_test["p_value_bootstrap"] < 0.10
    go = go_sharpe and go_mean_pos and go_pvalue

    print("\n" + "=" * 70)
    print("VERDICT Étape 1 cascade (déterministe sans ML)")
    print("=" * 70)
    print(f"  Sharpe OOS ≥ 0.7      : {'OK' if go_sharpe else 'FAIL'} ({m_test['sharpe']:+.2f})")
    print(f"  Mean PnL OOS > 0      : {'OK' if go_mean_pos else 'FAIL'} ({m_test['mean_pnl']:+.1f})")
    print(f"  p-value bootstrap<.10 : {'OK' if go_pvalue else 'FAIL'} (p={m_test['p_value_bootstrap']:.3f})")
    print(f"\n  ==> {'GO Étape 2 (ML méta-labeling)' if go else 'NO-GO — soit autre stratégie, soit ajustement timing'}")

    out = {
        "strategy": "pre_fomc_drift",
        "asset": ASSET,
        "tf": TF,
        "hours_before_entry": 24,
        "hours_before_exit": 1,
        "train_cutoff": str(TRAIN_CUTOFF),
        "test_start": str(TEST_START),
        "n_fomc_train": len(fomc_train),
        "n_fomc_test": len(fomc_test),
        "metrics_train": m_train,
        "metrics_test": m_test,
        "go": go,
        "go_breakdown": {
            "sharpe": go_sharpe,
            "mean_positive": go_mean_pos,
            "pvalue": go_pvalue,
        },
        "trades_train_sample": trades_train[:5],
        "trades_test_sample": trades_test[:5],
        "trades_train_all": trades_train,
        "trades_test_all": trades_test,
    }
    out_json = Path("predictions/h1_pre_fomc_drift_us500.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_json}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
