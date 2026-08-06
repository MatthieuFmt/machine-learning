"""Honest replay of the 'Trading Family - VIP' alerts on local H1 price data.

For every parsed, testable signal we:
  1. Enter at the OPEN of the first H1 bar AFTER the alert timestamp
     (builds in a realistic execution delay, and is strictly anti-look-ahead).
  2. Apply XTB costs from app.config.instruments.ASSET_CONFIGS:
        per-side cost = (spread/2 + slippage) * pip_size   on entry AND exit
        swap          = nights_held * swap_{long,short} * pip_size
  3. Resolve the outcome bar-by-bar on H1 highs/lows, with the conservative
     tie-break: if SL and a TP are both touchable in the same bar -> SL wins.
  4. Score three management policies:
        tp1      : exit fully at first of {SL, TP1}
        scaleout : 50% at TP1 then move SL to break-even, scale rest at TP2/TP3
                   (matches the channel's stated "je sors 50% et SL au point d'entrée")
        hold     : hold full size to the last TP or the SL

Everything is expressed in R-multiples (R = |entry - SL|), so assets aggregate.

Outputs (strategy-copy-trading/out/):
  trades.csv          one row per replayed signal, all policies
  replay_summary.txt  per-policy + per-asset performance

Run:  python strategy-copy-trading/replay.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
DATA = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT))
from app.config.instruments import ASSET_CONFIGS  # noqa: E402

# How long we allow a trade to stay open before marking-to-market (timeout).
MAX_HOLD_BARS = 24 * 45  # ~45 calendar days of H1 bars

# Scale-out fractions for the 'scaleout' policy (must sum to <= 1; remainder
# rides to the last available TP).
SCALE_FRACS = [0.5, 0.25, 0.25]


# ───────────────────────── price data ──────────────────────────────────────

_CACHE: dict[str, pd.DataFrame | None] = {}


def load_h1(ticker: str) -> pd.DataFrame | None:
    if ticker in _CACHE:
        return _CACHE[ticker]
    path = DATA / ticker / f"{ticker}_H1.csv"
    if not path.exists():
        _CACHE[ticker] = None
        return None
    # Some files (crypto) carry a 7th column; read the first 6 by POSITION so a
    # column-count mismatch can't shift Time/OHLC (this silently corrupted the
    # crypto Time column when read by header).
    df = pd.read_csv(
        path, sep="\t", header=None, skiprows=1, usecols=[0, 1, 2, 3, 4, 5],
        names=["Time", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.set_index("Time").sort_index()
    _CACHE[ticker] = df
    return df


# ───────────────────────── trade result ────────────────────────────────────


@dataclass
class Trade:
    msg_id: int
    ticker: str
    direction: str
    dt_utc: str
    entry_signal: float
    entry_fill: float
    sl: float
    tps: list[float]
    risk: float
    # diagnostics
    entry_bar_ts: str = ""
    fill_gap_R: float = 0.0       # |fill - signal entry| in R (slippage of delay)
    bars_held: int = 0
    nights: int = 0
    mfe_R: float = 0.0            # max favorable excursion (R)
    mae_R: float = 0.0            # max adverse excursion (R)
    # per-policy results (R, net of costs)
    r_tp1: float = 0.0
    out_tp1: str = ""
    r_scaleout: float = 0.0
    out_scaleout: str = ""
    r_hold: float = 0.0
    out_hold: str = ""


def _per_side_cost(cfg) -> float:
    """Adverse price move charged per side (entry, exit)."""
    return (cfg.spread_pips / 2.0 + cfg.slippage_pips) * cfg.pip_size


def _nights(ts0: pd.Timestamp, ts1: pd.Timestamp) -> int:
    return max(0, (ts1.normalize() - ts0.normalize()).days)


@dataclass(frozen=True)
class Scenario:
    name: str
    entry_mode: str   # 'realistic' (next-bar open + delay) | 'ideal' (stated price)
    costs: bool       # apply spread/slippage/swap


def replay_signal(sig: dict, scen: Scenario) -> Trade | None:
    ticker = sig["ticker"]
    df = load_h1(ticker)
    if df is None:
        return None
    cfg = ASSET_CONFIGS.get(ticker)
    if cfg is None:
        return None

    alert = pd.Timestamp(sig["dt_utc"])
    if alert.tzinfo is None:
        alert = alert.tz_localize("UTC")

    direction = sig["direction"]
    side = 1 if direction == "BUY" else -1
    cost = _per_side_cost(cfg) if scen.costs else 0.0
    entry_signal = float(sig["entry"])

    if scen.entry_mode == "ideal":
        # benefit of the doubt: fill at their exact stated price, no delay; scan
        # from the bar covering the alert (slightly favourable -> upper bound).
        past = df.index[df.index <= alert]
        if len(past) == 0:
            return None
        entry_ts = past[-1]
        raw_fill = entry_signal
    else:  # realistic: first bar strictly after the alert -> enter at its open
        future = df.index[df.index > alert]
        if len(future) == 0:
            return None
        entry_ts = future[0]
        if (entry_ts - alert).total_seconds() > 6 * 3600:
            return None  # market closed >6h around the alert -> unreliable
        raw_fill = float(df.loc[entry_ts, "Open"])

    entry_fill = raw_fill + side * cost
    sl = float(sig["sl"])
    tps = [float(t) for t in sig["tps"]]
    risk = abs(entry_fill - sl)
    if risk <= 0:
        return None

    tr = Trade(
        msg_id=int(sig["msg_id"]), ticker=ticker, direction=direction,
        dt_utc=sig["dt_utc"], entry_signal=entry_signal, entry_fill=entry_fill,
        sl=sl, tps=tps, risk=risk, entry_bar_ts=str(entry_ts),
        fill_gap_R=side * (entry_fill - entry_signal) / risk,
    )

    window = df.loc[entry_ts:].iloc[:MAX_HOLD_BARS]
    swap_per_night = cfg.swap_long_pips_per_night if side == 1 else cfg.swap_short_pips_per_night
    swap_price_per_night = swap_per_night * cfg.pip_size  # signed (+credit)

    def to_R(level: float, ts: pd.Timestamp) -> tuple[float, int]:
        """PnL in R for exiting at a RAW price `level`. The per-side exit cost
        (half-spread + slippage) is applied adversely here; the entry cost is
        already baked into entry_fill. Swap is signed (+ = credit)."""
        n = _nights(entry_ts, ts)
        gross = side * (level - entry_fill) - cost
        gross += n * swap_price_per_night
        return gross / risk, n

    # ── walk bars; record first-touch events and excursions ────────────────
    sl_now = sl
    moved_be = False
    realized_R = 0.0
    remaining = 1.0
    scale_idx = 0
    scale_done = False
    tp1_done = False
    hold_done = False
    r_tp1 = out_tp1 = None
    r_hold = out_hold = None

    last_ts = window.index[-1]
    last_close = float(window.iloc[-1]["Close"])

    for ts, bar in window.iterrows():
        hi, lo = float(bar["High"]), float(bar["Low"])
        # excursions (in R, gross of costs, on raw price)
        fav = side * (hi - entry_fill) if side == 1 else side * (lo - entry_fill)
        adv = side * (lo - entry_fill) if side == 1 else side * (hi - entry_fill)
        tr.mfe_R = max(tr.mfe_R, fav / risk)
        tr.mae_R = min(tr.mae_R, adv / risk)

        hit_sl = (lo <= sl_now) if side == 1 else (hi >= sl_now)
        # next unrealized TP for scale; TP1 for tp1 policy; last TP for hold
        def touched(level: float) -> bool:
            return (hi >= level) if side == 1 else (lo <= level)

        # ---- policy: tp1-only ----
        if r_tp1 is None:
            sl_hit_tp1 = (lo <= sl) if side == 1 else (hi >= sl)
            tp1_hit = touched(tps[0]) if tps else False
            if sl_hit_tp1:                       # tie -> SL wins (checked first)
                r, _ = to_R(sl, ts); r_tp1, out_tp1 = r, "SL"
            elif tp1_hit:
                r, _ = to_R(tps[0], ts); r_tp1, out_tp1 = r, "TP1"

        # ---- policy: hold to last TP ----
        if r_hold is None:
            last_tp = tps[-1]
            sl_hit_hold = (lo <= sl) if side == 1 else (hi >= sl)
            tp_last_hit = touched(last_tp)
            if sl_hit_hold:                      # tie -> SL wins (checked first)
                r, _ = to_R(sl, ts); r_hold, out_hold = r, "SL"
            elif tp_last_hit:
                r, _ = to_R(last_tp, ts); r_hold, out_hold = r, f"TP{len(tps)}"

        # ---- policy: scale-out + breakeven ----
        if not scale_done:
            sl_hit_sc = (lo <= sl_now) if side == 1 else (hi >= sl_now)
            # check TP ladder first only if SL not also hit (tie -> SL)
            if sl_hit_sc:
                r, n = to_R(sl_now, ts)
                realized_R += remaining * r
                tr.nights = max(tr.nights, n)
                scale_done = True
                tr.out_scaleout = "BE" if moved_be else "SL"
            else:
                # realize as many TP rungs as are touched in this bar
                progressed = False
                while scale_idx < len(tps) and touched(tps[scale_idx]):
                    frac = SCALE_FRACS[scale_idx] if scale_idx < len(SCALE_FRACS) else 0.0
                    if scale_idx == len(tps) - 1:   # last TP closes everything
                        frac = remaining
                    frac = min(frac, remaining)
                    r, n = to_R(tps[scale_idx], ts)
                    realized_R += frac * r
                    remaining -= frac
                    tr.nights = max(tr.nights, n)
                    if scale_idx == 0:
                        moved_be = True
                        sl_now = entry_fill        # break-even after TP1
                    scale_idx += 1
                    progressed = True
                    if remaining <= 1e-9:
                        scale_done = True
                        tr.out_scaleout = f"TP{scale_idx}"
                        break
                _ = progressed

        tr.bars_held += 1
        if r_tp1 is not None and r_hold is not None and scale_done:
            break

    # timeouts: mark to market at last close
    if r_tp1 is None:
        r, _ = to_R(last_close, last_ts); r_tp1, out_tp1 = r, "TIMEOUT"
    if r_hold is None:
        r, _ = to_R(last_close, last_ts); r_hold, out_hold = r, "TIMEOUT"
    if not scale_done:
        r, n = to_R(last_close, last_ts)
        realized_R += remaining * r
        tr.nights = max(tr.nights, n)
        tr.out_scaleout = "TIMEOUT"

    tr.r_tp1, tr.out_tp1 = round(r_tp1, 4), out_tp1
    tr.r_hold, tr.out_hold = round(r_hold, 4), out_hold
    tr.r_scaleout = round(realized_R, 4)
    return tr


# ───────────────────────── aggregation ─────────────────────────────────────


def _stats(rs: list[float]) -> dict:
    import statistics as st

    n = len(rs)
    if n == 0:
        return {}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    eq = 0.0
    peak = 0.0
    maxdd = 0.0
    curve = []
    for r in rs:
        eq += r
        curve.append(eq)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq - peak)
    return {
        "n": n,
        "win_rate": round(len(wins) / n, 4),
        "mean_R": round(st.mean(rs), 4),
        "median_R": round(st.median(rs), 4),
        "total_R": round(eq, 3),
        "std_R": round(st.pstdev(rs), 4) if n > 1 else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "max_dd_R": round(maxdd, 3),
        "best_R": round(max(rs), 3),
        "worst_R": round(min(rs), 3),
    }


def _scenario_table(results: dict[str, list[Trade]]) -> None:
    lines = ["", "=== SCENARIO COMPARISON (mean_R / total_R / profit_factor per policy) ===",
             f"{'scenario':18s} {'policy':9s} {'n':>4} {'winR':>6} {'meanR':>8} {'totR':>8} {'PF':>6}"]
    for name, trades in results.items():
        for pol, attr in (("tp1", "r_tp1"), ("scaleout", "r_scaleout"), ("hold", "r_hold")):
            rs = [getattr(t, attr) for t in trades]
            s = _stats(rs)
            if not s:
                continue
            lines.append(
                f"{name:18s} {pol:9s} {s['n']:>4} {s['win_rate']:>6} "
                f"{s['mean_R']:>8} {s['total_R']:>8} {s['profit_factor']:>6}"
            )
        lines.append("")
    txt = "\n".join(lines)
    (OUT / "scenario_comparison.txt").write_text(txt, encoding="utf-8")
    print(txt)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sig_rows = [
        r for r in csv.DictReader((OUT / "signals.csv").open(encoding="utf-8"))
        if r["testable"] == "True" and r["sanity_ok"] == "True" and r["direction"] in ("BUY", "SELL")
    ]
    # rebuild tps list from columns
    for r in sig_rows:
        r["tps"] = [r[c] for c in ("tp1", "tp2", "tp3", "tp4") if r[c] not in ("", None)]

    scenarios = [
        Scenario("ideal_nocost", "ideal", costs=False),        # upper bound on their skill
        Scenario("ideal_costs", "ideal", costs=True),          # exact price BUT XTB costs (fast bot)
        Scenario("realistic_nocost", "realistic", costs=False),  # isolate delay effect
        Scenario("realistic_costs", "realistic", costs=True),  # what a copier really gets
    ]
    results: dict[str, list[Trade]] = {}
    for scen in scenarios:
        tr = [t for r in sig_rows if (t := replay_signal(r, scen)) is not None]
        results[scen.name] = tr

    # cross-scenario comparison
    _scenario_table(results)

    # primary scenario for detailed outputs
    trades = results["realistic_costs"]
    skipped = len(sig_rows) - len(trades)

    # write trades.csv
    with (OUT / "trades.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "msg_id", "ticker", "direction", "dt_utc", "entry_signal", "entry_fill",
            "sl", "risk", "entry_bar_ts", "fill_gap_R", "bars_held", "nights",
            "mfe_R", "mae_R", "r_tp1", "out_tp1", "r_scaleout", "out_scaleout",
            "r_hold", "out_hold",
        ])
        for t in trades:
            w.writerow([
                t.msg_id, t.ticker, t.direction, t.dt_utc, round(t.entry_signal, 5),
                round(t.entry_fill, 5), t.sl, round(t.risk, 5), t.entry_bar_ts,
                round(t.fill_gap_R, 3), t.bars_held, t.nights, round(t.mfe_R, 3),
                round(t.mae_R, 3), t.r_tp1, t.out_tp1, t.r_scaleout, t.out_scaleout,
                t.r_hold, t.out_hold,
            ])

    _summary(trades, skipped)


def _summary(trades: list[Trade], skipped: int) -> None:
    from collections import Counter

    lines = ["=== REPLAY SUMMARY — Trading Family VIP (H1, XTB costs) ===",
             f"replayed trades: {len(trades)}   (skipped/no-data/closed-market: {skipped})", ""]
    for pol, attr in (("tp1", "r_tp1"), ("scaleout", "r_scaleout"), ("hold", "r_hold")):
        rs = [getattr(t, attr) for t in trades]
        s = _stats(rs)
        lines.append(f"--- policy: {pol} ---")
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in s.items()))
        outs = Counter(getattr(t, "out_" + pol) for t in trades)
        lines.append(f"  outcomes: {dict(outs)}")
        lines.append("")

    # per-asset (scaleout policy)
    lines.append("--- per-asset (scaleout policy): n, win_rate, mean_R, total_R ---")
    by: dict[str, list[float]] = {}
    for t in trades:
        by.setdefault(t.ticker, []).append(t.r_scaleout)
    for tic, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        s = _stats(rs)
        lines.append(f"  {tic:8s} n={s['n']:<4} wr={s['win_rate']:<6} meanR={s['mean_R']:<8} totR={s['total_R']}")

    # diagnostics: fill gap (delay slippage) & nights
    fg = [t.fill_gap_R for t in trades]
    import statistics as st
    lines += ["",
              f"avg fill_gap_R (delay slippage, +adverse): {round(st.mean(fg),3)}  "
              f"median {round(st.median(fg),3)}  worst {round(max(fg),3)}",
              f"avg nights held: {round(st.mean([t.nights for t in trades]),2)}  "
              f"max {max(t.nights for t in trades)}"]
    txt = "\n".join(lines)
    (OUT / "replay_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
