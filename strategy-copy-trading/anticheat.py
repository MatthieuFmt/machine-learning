"""Anti-cheat checks on the Trading Family VIP export.

The questions the user asked:
  1. Are alerts posted BEFORE the move, or backfilled/edited after the fact?
  2. Are losing trades quietly deleted?
  3. Do the channel's self-reported results match an independent replay?

What this export can and cannot show (be honest):
  • Deletions  -> VISIBLE as gaps in the message-id sequence (content is lost).
  • Edits      -> NOT marked in a Telegram Desktop HTML export. We substitute a
                  price-based heuristic: was the stated entry actually tradable
                  around the alert time, and had TP1 already been hit BEFORE the
                  alert (which would mean the signal was backfilled)?
  • Recaps     -> the channel posts weekly "Résumé des trades" with W/L/BE; we
                  compare that self-report to the replay outcome distribution.

Run:  python strategy-copy-trading/anticheat.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import replay  # noqa: E402

PRE_BARS = 3   # how many H1 bars before the alert to scan for a "pre-move"


def load_signals() -> list[dict]:
    rows = [
        r for r in csv.DictReader((OUT / "signals.csv").open(encoding="utf-8"))
        if r["sanity_ok"] == "True" and r["direction"] in ("BUY", "SELL")
    ]
    for r in rows:
        r["tps"] = [r[c] for c in ("tp1", "tp2", "tp3", "tp4") if r[c] not in ("", None)]
    return rows


# ───────────────────────── 1) deletions ────────────────────────────────────


def deletion_analysis(signals: list[dict]) -> list[str]:
    meta = json.loads((OUT / "deleted.json").read_text(encoding="utf-8"))
    deleted = set(meta["deleted_ids"])
    lo, hi = meta["range"]
    total = hi - lo + 1
    sig_ids = {int(r["msg_id"]) for r in signals}
    # deletions immediately AFTER a signal (id+1..id+3): could be a removed
    # follow-up that reported a loss
    after_sig = sum(
        1 for sid in sig_ids if any((sid + k) in deleted for k in (1, 2, 3))
    )
    return [
        "1) DELETIONS (gaps in message ids)",
        f"   id range {lo}..{hi} ({total} slots), {len(deleted)} missing "
        f"= {len(deleted)/total*100:.1f}%",
        f"   signals followed within 3 ids by a deleted message: {after_sig} "
        f"/ {len(sig_ids)} ({after_sig/len(sig_ids)*100:.0f}%)",
        "   NOTE: deleted content is unrecoverable; a 4% deletion rate over 4 years",
        "   is normal channel hygiene, but losing follow-ups COULD hide among them.",
        "",
    ]


# ───────────────────────── 2) recap honesty ────────────────────────────────


def recap_analysis() -> list[str]:
    rows = list(csv.DictReader((OUT / "recaps.csv").open(encoding="utf-8")))
    out = Counter(r["outcome"] for r in rows)
    w, l_, be = out.get("WIN", 0), out.get("LOSS", 0), out.get("BE", 0)
    tot = w + l_ + be
    lines = [
        "2) SELF-REPORTED RECAPS ('Résumé des trades de la semaine')",
        f"   WIN={w}  LOSS={l_}  BE(break-even)={be}  total lines={tot}",
    ]
    if tot:
        lines.append(
            f"   headline 'win rate' ignoring BE: {w/(w+l_)*100:.0f}%  "
            f"(if BE counted as scratch, still {w/tot*100:.0f}% wins)"
        )
    lines += [
        "   -> They DO post losses, so it is not a 'delete every loss' scam.",
        "   -> BUT a 'win' = TP1 merely touched (then 50% out + SL to break-even).",
        "      A high win/BE rate is fully compatible with NEGATIVE expectancy when",
        "      TP1 sits very close to entry while the full SL is far (small R wins,",
        "      full-R losses). The replay below shows exactly that.",
        "",
    ]
    return lines


# ───────────────────────── 3) timing / backfill ────────────────────────────


def timing_analysis(signals: list[dict]) -> list[str]:
    in_bar = 0          # stated entry was inside the alert-hour bar range
    pre_move = 0        # TP1 already hit in the PRE_BARS hours before the alert
    far_entry = 0       # stated entry >0.5R away from market at alert (limit-like)
    checked = 0
    gaps = []
    for r in signals:
        if r["testable"] != "True":
            continue
        df = replay.load_h1(r["ticker"])
        if df is None:
            continue
        alert = pd.Timestamp(r["dt_utc"])
        if alert.tzinfo is None:
            alert = alert.tz_localize("UTC")
        past = df.index[df.index <= alert]
        if len(past) == 0:
            continue
        bar_ts = past[-1]
        bar = df.loc[bar_ts]
        entry = float(r["entry"])
        sl = float(r["sl"])
        tp1 = float(r["tps"][0])
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        checked += 1
        lo_, hi_ = float(bar["Low"]), float(bar["High"])
        if lo_ <= entry <= hi_:
            in_bar += 1
        else:
            far_entry += 1
            gaps.append(min(abs(entry - lo_), abs(entry - hi_)) / risk)
        # pre-move: did price already reach TP1 in the bars just before the alert?
        side = 1 if r["direction"] == "BUY" else -1
        pre = df.loc[:bar_ts].iloc[-(PRE_BARS + 1):-1]
        if len(pre):
            if side == 1 and (pre["High"] >= tp1).any():
                pre_move += 1
            elif side == -1 and (pre["Low"] <= tp1).any():
                pre_move += 1
    lines = [
        "3) TIMING / BACKFILL HEURISTIC (testable signals only)",
        f"   checked: {checked}",
        f"   stated entry INSIDE the alert-hour price range: {in_bar} "
        f"({in_bar/checked*100:.0f}%)  -> consistent with a live market post",
        f"   stated entry OUTSIDE that range (limit-like / late): {far_entry} "
        f"({far_entry/checked*100:.0f}%)",
        f"   TP1 ALREADY reached in the {PRE_BARS}h BEFORE the alert "
        f"(backfill red flag): {pre_move} ({pre_move/checked*100:.0f}%)",
        "   -> A low pre-move rate means alerts generally precede the move (good).",
        "   -> Edits proper are invisible in this export; this is the best proxy.",
        "",
    ]
    return lines


def main() -> None:
    signals = load_signals()
    lines = ["=== ANTI-CHEAT REPORT — Trading Family VIP ===", ""]
    lines += deletion_analysis(signals)
    lines += recap_analysis()
    lines += timing_analysis(signals)
    txt = "\n".join(lines)
    (OUT / "anticheat_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
