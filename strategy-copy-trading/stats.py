"""Honest 'luck vs edge' tests on the replayed Trading Family VIP signals.

For each (scenario, management policy) we take the per-trade R series and ask:
is the mean R distinguishable from zero, or is it consistent with luck?

Tests (all on per-trade R, so no spurious √252 annualisation — cf. CLAUDE.md §5):
  • one-sample t-test  (H0: mean R = 0)         -> t, one-sided p for mean>0
  • bootstrap of mean R (10k resamples)         -> 95% CI + P(mean>0)
  • per-trade Sharpe + PSR + DSR (n_trials=1)   -> reuses app/analysis/edge_validation
    (n_trials=1 = the MOST generous case: no data-snooping deflation at all;
     if there is no edge even here, the verdict is robust.)

Run:  python strategy-copy-trading/stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import replay  # noqa: E402
from app.analysis.edge_validation import deflated_sharpe, probabilistic_sharpe  # noqa: E402

RNG = np.random.default_rng(12345)


def _bootstrap_mean(rs: np.ndarray, n: int = 10000) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(rs), size=(n, len(rs)))
    means = rs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means > 0).mean())


def analyse(rs: list[float]) -> dict:
    a = np.asarray(rs, dtype=float)
    n = len(a)
    mean = a.mean()
    std = a.std(ddof=1)
    sr = mean / std if std > 0 else 0.0           # per-trade Sharpe
    t = sr * np.sqrt(n) if std > 0 else 0.0        # = mean / (std/√n)
    p_one = 1.0 - sps.t.cdf(t, df=n - 1)           # H1: mean > 0
    lo, hi, p_boot = _bootstrap_mean(a)
    skew = float(sps.skew(a, bias=False))
    kurt = float(sps.kurtosis(a, fisher=True, bias=False) + 3.0)
    dsr_z, dsr_p = deflated_sharpe(sr, n_trials=1, n_obs=n, skew=skew, kurtosis=kurt)
    psr = probabilistic_sharpe(sr, n_obs=n, skew=skew, kurtosis=kurt, sr_benchmark=0.0)
    return {
        "n": n, "mean_R": mean, "std_R": std, "sr_trade": sr,
        "t": t, "p_meanR>0": p_one,
        "boot_CI95": (lo, hi), "P(meanR>0)": p_boot,
        "PSR(SR>0)": psr, "DSR_z": dsr_z, "DSR_p": dsr_p,
    }


def main() -> None:
    import csv

    sig_rows = [
        r for r in csv.DictReader((OUT / "signals.csv").open(encoding="utf-8"))
        if r["testable"] == "True" and r["sanity_ok"] == "True"
        and r["direction"] in ("BUY", "SELL")
    ]
    for r in sig_rows:
        r["tps"] = [r[c] for c in ("tp1", "tp2", "tp3", "tp4") if r[c] not in ("", None)]

    scenarios = [
        replay.Scenario("ideal_nocost", "ideal", costs=False),
        replay.Scenario("ideal_costs", "ideal", costs=True),
        replay.Scenario("realistic_nocost", "realistic", costs=False),
        replay.Scenario("realistic_costs", "realistic", costs=True),
    ]

    lines = ["=== LUCK vs EDGE — per-trade tests (n_trials=1, most generous) ===", ""]
    for scen in scenarios:
        trades = [t for r in sig_rows if (t := replay.replay_signal(r, scen)) is not None]
        for pol, attr in (("scaleout", "r_scaleout"), ("hold", "r_hold")):
            rs = [getattr(t, attr) for t in trades]
            s = analyse(rs)
            lines.append(f"[{scen.name} / {pol}]  n={s['n']}")
            lines.append(
                f"   mean_R={s['mean_R']:+.4f}  sr/trade={s['sr_trade']:+.4f}  "
                f"t={s['t']:+.2f}  p(mean>0)={s['p_meanR>0']:.3f}"
            )
            lines.append(
                f"   bootstrap mean_R 95% CI=[{s['boot_CI95'][0]:+.3f}, {s['boot_CI95'][1]:+.3f}]  "
                f"P(mean>0)={s['P(meanR>0)']:.3f}"
            )
            lines.append(
                f"   PSR(SR>0)={s['PSR(SR>0)']:.3f}  DSR_z={s['DSR_z']:+.2f}  DSR_p={s['DSR_p']:.3f}"
            )
            lines.append("")

    txt = "\n".join(lines)
    (OUT / "stats_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
