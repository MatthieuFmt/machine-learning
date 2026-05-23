"""Phase H4 — Backtest Pairs Trading EURUSD-GBPUSD H4.

Stratégie déterministe (Étape 1 cascade) : mean-reversion sur spread cointégré.

1. Test cointegration Engle-Granger sur train (statsmodels.tsa.stattools.coint).
   Si p ≥ 0.10 → NO-GO formel (paire non cointégrée).
2. Sinon, simulate train + OOS avec z_entry=2.0, z_exit=0.5, time_stop_bars=30.
3. Verdict GO Étape 1 par critères Phase H cascade.

⚠️ Requires statsmodels (pip install statsmodels).
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

from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.strategies.pairs_trading import simulate_pairs_trades  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

logger = get_logger(__name__)

ASSET_A = "EURUSD"
ASSET_B = "GBPUSD"
TF = "H4"

Z_ENTRY = 2.0
Z_EXIT = 0.5
TIME_STOP_BARS = 30  # 30 H4 ≈ 5 jours
BETA_LOOKBACK = 60
ZSCORE_LOOKBACK = 60


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


def _sharpe_per_trade(pnls_eur: np.ndarray, trades_per_year: float = 100.0) -> float:
    """Sharpe per-trade annualisé. trades_per_year approximation pour pairs H4."""
    if len(pnls_eur) < 2:
        return 0.0
    std = pnls_eur.std(ddof=1)
    if std == 0:
        return 0.0
    return float((pnls_eur.mean() / std) * np.sqrt(trades_per_year))


def _analyze(trades: list[dict], label: str, trades_per_year_hint: float = 100.0) -> dict[str, Any]:
    if not trades:
        return {
            "label": label, "n_trades": 0,
            "sharpe": 0.0, "mean_pnl_eur": 0.0, "median_pnl_eur": 0.0,
            "wr": 0.0, "total_pnl_eur": 0.0, "max_dd_eur": 0.0,
            "p_value_bootstrap": 1.0,
            "n_long": 0, "n_short": 0,
            "mean_reversion_rate": 0.0, "time_stop_rate": 0.0,
            "mean_bars_held": 0.0,
        }
    pnls = np.array([t["pnl_eur_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    signals = np.array([t["signal"] for t in trades])
    reasons = [t["exit_reason"] for t in trades]
    bars = [t["bars_held"] for t in trades]
    return {
        "label": label,
        "n_trades": len(trades),
        "sharpe": _sharpe_per_trade(pnls, trades_per_year_hint),
        "mean_pnl_eur": float(pnls.mean()),
        "median_pnl_eur": float(np.median(pnls)),
        "wr": float((pnls > 0).mean()),
        "total_pnl_eur": float(pnls.sum()),
        "max_dd_eur": float((equity - peak).min()),
        "p_value_bootstrap": _bootstrap_pvalue(pnls),
        "n_long": int((signals == 1).sum()),
        "n_short": int((signals == -1).sum()),
        "mean_reversion_rate": float(reasons.count("mean_reversion") / len(reasons)),
        "time_stop_rate": float(reasons.count("time_stop") / len(reasons)),
        "mean_bars_held": float(np.mean(bars)),
    }


def _print(m: dict[str, Any]) -> None:
    print(f"  {m['label']}: n={m['n_trades']} (L={m['n_long']}/S={m['n_short']}), "
          f"Sharpe={m['sharpe']:+.2f}, WR={m['wr']:.1%}, "
          f"mean=€{m['mean_pnl_eur']:+.1f}, total=€{m['total_pnl_eur']:+.0f}, "
          f"max_dd=€{m['max_dd_eur']:.0f}, p={m['p_value_bootstrap']:.3f}")
    print(f"    exits: mean_rev={m['mean_reversion_rate']:.0%}, "
          f"time_stop={m['time_stop_rate']:.0%}, "
          f"mean_bars={m['mean_bars_held']:.1f}")


def _test_cointegration(close_a: pd.Series, close_b: pd.Series) -> dict[str, Any]:
    """Test Engle-Granger via statsmodels."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        print("⚠️ statsmodels manquant. Installer : pip install statsmodels")
        return {"available": False, "p_value": np.nan, "t_stat": np.nan}

    t_stat, p_value, crit = coint(close_a.values, close_b.values)
    return {
        "available": True,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "critical_values_1pct_5pct_10pct": [float(c) for c in crit],
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"PHASE H4 — Pairs Trading {ASSET_A}-{ASSET_B} {TF}")
    print(f"z_entry={Z_ENTRY}, z_exit={Z_EXIT}, time_stop={TIME_STOP_BARS} bars (~5 jours)")
    print(f"β rolling lookback={BETA_LOOKBACK}, z-score lookback={ZSCORE_LOOKBACK}")
    print("=" * 70)

    cfg_a = ASSET_CONFIGS[ASSET_A]
    cfg_b = ASSET_CONFIGS[ASSET_B]
    print(f"\n{ASSET_A} : spread={cfg_a.spread_pips}, slippage={cfg_a.slippage_pips}, "
          f"pip_value_eur={cfg_a.pip_value_eur}")
    print(f"{ASSET_B} : spread={cfg_b.spread_pips}, slippage={cfg_b.slippage_pips}, "
          f"pip_value_eur={cfg_b.pip_value_eur}")

    df_a = load_asset(ASSET_A, TF)
    df_b = load_asset(ASSET_B, TF)
    print(f"\n{ASSET_A} : {len(df_a)} bars ({df_a.index.min().date()} → {df_a.index.max().date()})")
    print(f"{ASSET_B} : {len(df_b)} bars ({df_b.index.min().date()} → {df_b.index.max().date()})")

    common_idx = df_a.index.intersection(df_b.index)
    print(f"Index commun : {len(common_idx)} bars")
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    df_a_train = df_a.loc[:TRAIN_CUTOFF]
    df_b_train = df_b.loc[:TRAIN_CUTOFF]
    df_a_test = df_a.loc[TEST_START:]
    df_b_test = df_b.loc[TEST_START:]

    # ── Test cointegration train ─────────────────────────────────────
    print(f"\n── Test cointegration Engle-Granger (Train ≤ {TRAIN_CUTOFF.date()}) ──")
    coint_train = _test_cointegration(df_a_train["Close"], df_b_train["Close"])
    if not coint_train["available"]:
        print("  ❌ Impossible de tester (statsmodels manquant)")
        return 1
    print(f"  t-stat = {coint_train['t_stat']:+.3f}")
    print(f"  p-value = {coint_train['p_value']:.4f}")
    print(f"  critical 1%/5%/10% : {coint_train['critical_values_1pct_5pct_10pct']}")
    coint_ok_train = coint_train["p_value"] < 0.10
    print(f"  ==> {'Cointégrés (p < 0.10)' if coint_ok_train else '⚠️ Non cointégrés (p ≥ 0.10)'}")

    # Test cointegration sur full sample (info)
    coint_full = _test_cointegration(df_a["Close"], df_b["Close"])
    print(f"\n  Full sample : p = {coint_full['p_value']:.4f} "
          f"({'cointégrés' if coint_full['p_value'] < 0.10 else 'non cointégrés'})")

    # ── Simulation Train ─────────────────────────────────────────────
    print(f"\n── TRAIN ──")
    trades_train = simulate_pairs_trades(
        df_a_train, df_b_train, cfg_a, cfg_b,
        z_entry=Z_ENTRY, z_exit=Z_EXIT,
        time_stop_bars=TIME_STOP_BARS,
        beta_lookback=BETA_LOOKBACK, zscore_lookback=ZSCORE_LOOKBACK,
    )
    m_train = _analyze(trades_train, "Train")
    _print(m_train)

    # ── Simulation OOS ───────────────────────────────────────────────
    print(f"\n── TEST (OOS) ──")
    trades_test = simulate_pairs_trades(
        df_a_test, df_b_test, cfg_a, cfg_b,
        z_entry=Z_ENTRY, z_exit=Z_EXIT,
        time_stop_bars=TIME_STOP_BARS,
        beta_lookback=BETA_LOOKBACK, zscore_lookback=ZSCORE_LOOKBACK,
    )
    m_test = _analyze(trades_test, "Test")
    _print(m_test)

    # ── Verdict ──────────────────────────────────────────────────────
    go_coint = coint_ok_train
    go_sharpe = m_test["sharpe"] >= 0.7
    go_mean = m_test["mean_pnl_eur"] > 0
    go_pvalue = m_test["p_value_bootstrap"] < 0.10
    go_ntrades = m_test["n_trades"] >= 30
    go = go_coint and go_sharpe and go_mean and go_pvalue and go_ntrades

    print(f"\n" + "=" * 70)
    print(f"VERDICT {ASSET_A}-{ASSET_B} {TF}")
    print("=" * 70)
    print(f"  Cointégration p<0.10 : {'OK' if go_coint else 'FAIL'} (p={coint_train['p_value']:.4f})")
    print(f"  Sharpe OOS ≥ 0.7     : {'OK' if go_sharpe else 'FAIL'} ({m_test['sharpe']:+.2f})")
    print(f"  Mean OOS > 0         : {'OK' if go_mean else 'FAIL'} (€{m_test['mean_pnl_eur']:+.1f})")
    print(f"  p-value bootstrap    : {'OK' if go_pvalue else 'FAIL'} ({m_test['p_value_bootstrap']:.3f})")
    print(f"  n_trades OOS ≥ 30    : {'OK' if go_ntrades else 'FAIL'} ({m_test['n_trades']})")
    print(f"  ==> {'GO Étape 2 (ML méta-labeling)' if go else 'NO-GO Étape 1'}")

    # ── Sauvegarde ────────────────────────────────────────────────────
    out = {
        "strategy": "pairs_trading_meanrev",
        "asset_a": ASSET_A, "asset_b": ASSET_B, "tf": TF,
        "z_entry": Z_ENTRY, "z_exit": Z_EXIT,
        "time_stop_bars": TIME_STOP_BARS,
        "beta_lookback": BETA_LOOKBACK,
        "zscore_lookback": ZSCORE_LOOKBACK,
        "train_cutoff": str(TRAIN_CUTOFF),
        "test_start": str(TEST_START),
        "cointegration_train": coint_train,
        "cointegration_full_sample": coint_full,
        "metrics_train": m_train,
        "metrics_test": m_test,
        "go": go,
        "go_breakdown": {
            "cointegration": go_coint,
            "sharpe": go_sharpe,
            "mean_positive": go_mean,
            "pvalue": go_pvalue,
            "n_trades": go_ntrades,
        },
        "trades_train_sample": trades_train[:5],
        "trades_test_sample": trades_test[:5],
        "n_trades_train": len(trades_train),
        "n_trades_test": len(trades_test),
    }
    out_path = Path("predictions/h4_pairs_eurusd_gbpusd.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_path}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
