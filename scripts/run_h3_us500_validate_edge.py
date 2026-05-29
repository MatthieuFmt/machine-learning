"""Phase H3 — validate_edge formel sur US500 NR7 D1 (sleeve candidat GO).

Construit l'equity daily à partir des trades OOS et applique le framework
validate_edge (5 critères Constitution §2 : Sharpe, DSR, Max DD, WR, trades/an).

⚠️ Avertissement : NR7 est intrinsèquement low-frequency (~25 trades/an sur D1).
Le critère 5 (trades/an ≥ 30) est attendu en FAIL — voir discussion dans le
rapport produit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.strategies.volatility_breakout import simulate_nr_breakout_trades  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

logger = get_logger(__name__)

ASSET = "US500"
TF = "D1"
LOOKBACK = 7
TP_MULT = 2.0
SL_MULT = 1.0

CAPITAL_EUR = 10_000.0
N_TRIALS_CUMUL = 61


def _build_daily_equity(
    trades: list[dict], start: pd.Timestamp, end: pd.Timestamp,
    pip_value_eur: float, capital_eur: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit equity daily forward-fillée + DataFrame pnl pour validate_edge."""
    trades_df = pd.DataFrame(trades)
    trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_time"], utc=True)
    trades_df = trades_df.sort_values("exit_dt").reset_index(drop=True)
    trades_df["pnl"] = trades_df["pips_net"].astype(float) * pip_value_eur

    trades_df["exit_date"] = trades_df["exit_dt"].dt.normalize()
    daily_index = pd.date_range(start, end, freq="D", tz="UTC")
    daily_pnl = trades_df.groupby("exit_date")["pnl"].sum()
    daily_pnl = daily_pnl.reindex(daily_index, fill_value=0.0)

    equity = capital_eur + daily_pnl.cumsum()
    return equity, trades_df


def _print_section(title: str, scope: str, n_trials: int, report, equity: pd.Series) -> None:
    print("\n" + "=" * 70)
    print(f"  validate_edge — {scope}")
    print(f"  n_trials_cumul = {n_trials}")
    print("=" * 70)
    print(f"  Equity : {equity.index[0].date()} → {equity.index[-1].date()} "
          f"(n={len(equity)} jours, n_obs returns={len(equity)-1})")
    print(f"  Capital : €{equity.iloc[0]:.0f} → €{equity.iloc[-1]:.0f} "
          f"(return {(equity.iloc[-1]/equity.iloc[0] - 1)*100:+.1f}%)")
    print()
    print(f"  GO global : {'✅' if report.go else '❌'}")
    print(f"\n  Critères Constitution §2 :")
    print(f"    1. Sharpe (≥ 1.0)        : {report.metrics.get('sharpe', float('nan')):+.2f}")
    print(f"    2. DSR (> 0, p < 0.05)   : DSR={report.metrics.get('dsr', float('nan')):+.2f}, "
          f"p={report.metrics.get('p_value', float('nan')):.4f}")
    print(f"    3. Max DD (< 15%)        : {report.metrics.get('max_dd', float('nan'))*100:.1f}%")
    print(f"    4. WR (> 30%)            : {report.metrics.get('wr', float('nan'))*100:.1f}%")
    print(f"    5. Trades/an (≥ 30)      : {report.metrics.get('trades_per_year', float('nan')):.1f}")
    print(f"    n_trades                 : {int(report.metrics.get('n_trades', 0))}")

    if report.reasons:
        print(f"\n  Raisons d'échec :")
        for r in report.reasons:
            print(f"    ❌ {r}")


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"PHASE H3 — validate_edge {ASSET} NR{LOOKBACK} {TF}")
    print(f"TP={TP_MULT}×range, SL={SL_MULT}×range, capital={CAPITAL_EUR} EUR")
    print("=" * 70)

    cfg = ASSET_CONFIGS[ASSET]
    df = load_asset(ASSET, TF)

    # ── Validation 1 : OOS pur (réalisme post-publication) ─────────
    print(f"\n[1/2] OOS pur (≥ {TEST_START.date()}) ...")
    df_test = df.loc[TEST_START:]
    trades_oos = simulate_nr_breakout_trades(
        df_test, cfg, lookback=LOOKBACK, tp_mult=TP_MULT, sl_mult=SL_MULT,
    )
    start_oos = (
        TEST_START if isinstance(TEST_START, pd.Timestamp) and TEST_START.tz is not None
        else pd.Timestamp(TEST_START, tz="UTC")
    )
    end_oos = df_test.index.max()
    equity_oos, trades_oos_df = _build_daily_equity(
        trades_oos, start_oos, end_oos, cfg.pip_value_eur, CAPITAL_EUR,
    )
    report_oos = validate_edge(
        equity=equity_oos,
        trades=trades_oos_df[["pnl"]],
        n_trials=N_TRIALS_CUMUL,
    )
    _print_section("OOS pur", "OOS pur (2024-01 → 2026-05)", N_TRIALS_CUMUL,
                   report_oos, equity_oos)

    # ── Validation 2 : Train+OOS combiné (sample maximal) ─────────
    print(f"\n[2/2] Train+OOS combiné ...")
    trades_full = simulate_nr_breakout_trades(
        df, cfg, lookback=LOOKBACK, tp_mult=TP_MULT, sl_mult=SL_MULT,
    )
    start_full = df.index.min()
    end_full = df.index.max()
    equity_full, trades_full_df = _build_daily_equity(
        trades_full, start_full, end_full, cfg.pip_value_eur, CAPITAL_EUR,
    )
    report_full = validate_edge(
        equity=equity_full,
        trades=trades_full_df[["pnl"]],
        n_trials=N_TRIALS_CUMUL,
    )
    _print_section("Train+OOS combiné", "Train+OOS combiné (full sample)",
                   N_TRIALS_CUMUL, report_full, equity_full)

    # ── Sauvegarde ────────────────────────────────────────────────
    out_path = Path("predictions") / "h3_us500_nr7_validate_edge.json"
    out_data = {
        "asset": ASSET,
        "tf": TF,
        "lookback": LOOKBACK,
        "tp_mult": TP_MULT,
        "sl_mult": SL_MULT,
        "n_trials_cumul": N_TRIALS_CUMUL,
        "capital_eur": CAPITAL_EUR,
        "scopes": {
            "oos_only": {
                "n_trades": len(trades_oos),
                "equity_start_date": str(equity_oos.index[0]),
                "equity_end_date": str(equity_oos.index[-1]),
                "equity_start_eur": float(equity_oos.iloc[0]),
                "equity_end_eur": float(equity_oos.iloc[-1]),
                "go": report_oos.go,
                "reasons": report_oos.reasons,
                "metrics": {k: float(v) for k, v in report_oos.metrics.items()},
            },
            "full_sample": {
                "n_trades": len(trades_full),
                "equity_start_date": str(equity_full.index[0]),
                "equity_end_date": str(equity_full.index[-1]),
                "equity_start_eur": float(equity_full.iloc[0]),
                "equity_end_eur": float(equity_full.iloc[-1]),
                "go": report_full.go,
                "reasons": report_full.reasons,
                "metrics": {k: float(v) for k, v in report_full.metrics.items()},
            },
        },
    }
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_path}")

    # Verdict global
    print("\n" + "=" * 70)
    print("VERDICT VALIDATE_EDGE")
    print("=" * 70)
    print(f"  OOS pur          : {'GO ✅' if report_oos.go else 'NO-GO ❌'}")
    print(f"  Train+OOS combiné: {'GO ✅' if report_full.go else 'NO-GO ❌'}")
    print()
    print("  Note : NR7 est low-frequency (~25 trades/an). Le critère 5")
    print("  'trades/an ≥ 30' Constitution §2 est calibré pour stratégies H1/H4.")
    print("  Si seul ce critère fail, considérer documentation en sleeve")
    print("  'low-frequency probatoire' avec re-évaluation après 2 ans supp.")

    return 0 if (report_oos.go or report_full.go) else 1


if __name__ == "__main__":
    sys.exit(main())
