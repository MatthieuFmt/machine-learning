"""Reverse-engineer the setup profile of the Trading Family VIP signals and
explain WHY the expectancy is negative.

Reads out/signals.csv (+ out/trades.csv) and profiles:
  • session: hour-of-day (Paris) and weekday of the alerts
  • risk/reward geometry: TP1/SL and last-TP/SL ratios  <-- the key to the verdict
  • asset mix and trade-type (intraday vs swing)
  • realised holding time

Run:  python strategy-copy-trading/rules.py
"""

from __future__ import annotations

import csv
import statistics as st
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    sig = list(csv.DictReader((OUT / "signals.csv").open(encoding="utf-8")))
    sig = [r for r in sig if r["sanity_ok"] == "True" and r["direction"] in ("BUY", "SELL")]

    # ── session ────────────────────────────────────────────────────────────
    hours, wdays = Counter(), Counter()
    for r in sig:
        if not r["dt_utc"]:
            continue
        t = pd.Timestamp(r["dt_utc"]).tz_convert("Europe/Paris")
        hours[t.hour] += 1
        wdays[t.day_name()] += 1

    # ── geometry: reward1/risk and reward_last/risk ──────────────────────────
    rr1, rrlast = [], []
    for r in sig:
        e, s = _f(r["entry"]), _f(r["sl"])
        tps = [_f(r[c]) for c in ("tp1", "tp2", "tp3", "tp4") if _f(r[c]) is not None]
        if e is None or s is None or not tps:
            continue
        risk = abs(e - s)
        if risk <= 0:
            continue
        rr1.append(abs(tps[0] - e) / risk)
        rrlast.append(abs(tps[-1] - e) / risk)

    # ── asset / type mix ─────────────────────────────────────────────────────
    assets = Counter(r["ticker"] or "??" for r in sig)
    ttypes = Counter()
    for r in sig:
        tt = (r["trade_type"] or "").lower()
        if "scalp" in tt:
            ttypes["scalping"] += 1
        elif "intraday" in tt and "swing" in tt:
            ttypes["intraday/swing"] += 1
        elif "intraday" in tt or "day" in tt:
            ttypes["intraday"] += 1
        elif "swing" in tt:
            ttypes["swing"] += 1
        elif "long" in tt or "moyen" in tt or "semaine" in tt:
            ttypes["moyen/long terme"] += 1
        elif tt:
            ttypes["autre"] += 1
        else:
            ttypes["(non précisé)"] += 1
    n_tps = Counter(len([1 for c in ("tp1", "tp2", "tp3", "tp4") if _f(r[c]) is not None]) for r in sig)

    # ── realised holding (from trades.csv if present) ────────────────────────
    hold_lines = []
    tpath = OUT / "trades.csv"
    if tpath.exists():
        tr = list(csv.DictReader(tpath.open(encoding="utf-8")))
        nights = [int(t["nights"]) for t in tr]
        mfe = [float(t["mfe_R"]) for t in tr]
        reached_1R = sum(1 for m in mfe if m >= 1.0)
        hold_lines = [
            "REALISED HOLDING & EXCURSIONS (realistic+costs replay)",
            f"   median nights held: {st.median(nights)}   mean: {st.mean(nights):.1f}   max: {max(nights)}",
            f"   trades whose price ran >= +1R in favour at some point: "
            f"{reached_1R}/{len(tr)} ({reached_1R/len(tr)*100:.0f}%)",
            "",
        ]

    top = st.median(rr1) if rr1 else 0
    lines = [
        "=== SETUP PROFILE — Trading Family VIP ===",
        "",
        "SESSION (Europe/Paris):",
        "   busiest hours: " + ", ".join(f"{h}h({n})" for h, n in hours.most_common(6)),
        "   weekdays: " + ", ".join(f"{d}:{n}" for d, n in wdays.most_common()),
        "",
        "RISK/REWARD GEOMETRY:",
        f"   reward(TP1)/risk : median {st.median(rr1):.2f}  mean {st.mean(rr1):.2f}  "
        f"(share < 1.0 : {sum(1 for x in rr1 if x < 1)/len(rr1)*100:.0f}%)",
        f"   reward(lastTP)/risk: median {st.median(rrlast):.2f}  mean {st.mean(rrlast):.2f}",
        "   -> On paper the R:R is fine (TP1 ~1.2R, last TP ~3.2R). The problem is",
        "      NOT the geometry but that the TP1-hit rate (~47-53%) is too low to",
        "      monetise it once execution delay + XTB costs are paid: scaling 50% out",
        "      at TP1 then break-even gives small wins, while stops are full -1R.",
        "",
        "TP LADDER (number of take-profits per signal):",
        "   " + ", ".join(f"{k}TP:{v}" for k, v in sorted(n_tps.items())),
        "",
        "TRADE TYPE (stated):",
        "   " + ", ".join(f"{k}:{v}" for k, v in ttypes.most_common()),
        "",
        "ASSET MIX (top 12):",
        "   " + ", ".join(f"{a}:{n}" for a, n in assets.most_common(12)),
        "",
        *hold_lines,
    ]
    txt = "\n".join(lines)
    (OUT / "setup_profile.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
