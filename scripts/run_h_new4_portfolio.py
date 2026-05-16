"""Pivot v4 B4 — H_new4 : Portfolio des sleeves GO (single-sleeve fallback).

Avec 1 seul sleeve GO (EURUSD H4), le portfolio = le sleeve unique.
Le multi-sleeve est conditionnel a >= 2 sleeves GO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from app.core.seeds import set_global_seeds
from app.testing.snooping_guard import read_oos

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Sleeves GO confirmes dans JOURNAL.md ─────────────────────────────
# H_new1 US30 D1 : NO-GO (Sharpe 0.82, 12 trades)
# H_new2 walk-forward rolling : NO-GO
# H_new3 EURUSD H4 : GO (Sharpe +1.73, 25.2 trades/an)
SLEEVES_GO: dict[str, str] = {
    # "h_new1_us30_d1": "predictions/h_new1_meta_us30.json",  # NO-GO
    "h_new3_eurusd_h4": "predictions/h_new3_eurusd_h4.json",
}


def main() -> int:
    """Single-sleeve fallback: portfolio = unique sleeve GO."""
    set_global_seeds()

    n_sleeves_go = len(SLEEVES_GO)
    print(f"Sleeves GO disponibles : {n_sleeves_go}")

    # ── Single-sleeve fallback ───────────────────────────────────────
    best_name, best_file = next(iter(SLEEVES_GO.items()))
    data = json.loads(Path(best_file).read_text(encoding="utf-8"))
    metrics = data.get("metrics_walk_forward_oos", {})
    sr = float(metrics.get("sharpe", 0.0))
    dd = float(metrics.get("max_dd_pct", 0.0))
    trades = int(metrics.get("trades", 0))
    wr = float(metrics.get("win_rate", 0.0))

    # Duration estimate for trades/year
    segments = data.get("segments", [])
    if segments:
        start = pd.Timestamp(segments[0]["start"])
        end = pd.Timestamp(segments[-1]["end"])
        years = max((end - start).days / 365.25, 0.1)
        trades_per_year = trades / years
    else:
        trades_per_year = 0.0

    mode = "single_sleeve_fallback"
    if n_sleeves_go < 2:
        print("< 2 sleeves GO → single-sleeve fallback.")
    else:
        mode = "multi_sleeve_not_implemented"

    print(f"  Sleeve : {best_name}")
    print(f"  Sharpe : {sr:.2f}")
    print(f"  DD     : {dd:.1f}%")
    print(f"  Trades : {trades} ({trades_per_year:.1f}/an)")
    print(f"  WR     : {wr:.1f}%")

    # read_oos() — 1 lecture test set, n_trials 27→28
    read_oos(
        prompt="pivot_v4_B4",
        hypothesis="H_new4_portfolio_single_sleeve",
        sharpe=sr,
        n_trades=trades,
    )

    out = {
        "mode": mode,
        "rationale": (
            "Only 1 sleeve GO (EURUSD H4). Portfolio = unique sleeve."
            if n_sleeves_go < 2
            else "Multi-sleeve path not executed (B4 fallback)."
        ),
        "n_sleeves_go": n_sleeves_go,
        "n_trials_cumul": 28,
        "best_sleeve": best_name,
        "sharpe": sr,
        "sharpe_method": metrics.get("sharpe_method", "per_trade"),
        "max_dd_pct": dd,
        "trades": trades,
        "trades_per_year": round(trades_per_year, 1),
        "win_rate_pct": wr,
        "final_equity_eur": metrics.get("final_equity_eur", 0.0),
        "total_return_pct": metrics.get("total_return_pct", 0.0),
        "verdict": "NO-GO portfolio (single-sleeve). Production = EURUSD H4 seule.",
    }
    Path("predictions/h_new4_portfolio.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
