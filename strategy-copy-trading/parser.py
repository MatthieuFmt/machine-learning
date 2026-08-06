"""Parse the Telegram HTML export of the 'Trading Family - VIP' channel
into a structured dataset of trading signals + follow-up messages.

The export is a Telegram Desktop HTML dump (messages*.html). Photos are NOT
included in the export, but the trading signals of interest are TEXT messages
of the form:

    ✔️Achat Nasdaq 11373
    ❌SL 11240
    ✅TP1 11481
    ✅TP2 11644

This module is intentionally self-contained (a brand-new strategy study), but
it normalises asset tickers to the project's local data convention so the
replay can reuse app/ infrastructure (costs, metrics, DSR).

Outputs (written to strategy-copy-trading/out/):
    signals.csv     one row per detected entry signal
    followups.csv   one row per follow-up message (TP/SL hit, breakeven, close)
    deleted.json    list of deleted message ids (gaps in the id sequence)
    parse_report.txt human-readable summary

Run:  python strategy-copy-trading/parser.py
"""

from __future__ import annotations

import html as _html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# ───────────────────────── HTML splitting ──────────────────────────────────

# Each message is a <div class="message default clearfix[ joined]" id="messageN">…</div>
_MSG_RE = re.compile(
    r'<div class="message default clearfix(?P<joined>[^"]*)" id="message(?P<id>\d+)">'
    r'(?P<body>.*?)'
    r'(?=<div class="message (?:default|service)|</div>\s*</div>\s*</div>\s*</body>)',
    re.S,
)
_DATE_RE = re.compile(r'class="pull_right date details" title="(?P<dt>[^"]+)"')
_TEXT_RE = re.compile(r'<div class="text">(?P<t>.*?)</div>', re.S)
_REPLY_RE = re.compile(r'In reply to <a href="#go_to_message(?P<rid>\d+)"')
_PHOTO_RE = re.compile(r"media_photo")
_FROM_RE = re.compile(r'<div class="from_name">\s*(?P<n>.*?)\s*</div>', re.S)


def _clean_text(raw: str) -> str:
    """Strip HTML tags, turn <br> into newlines, unescape entities."""
    t = re.sub(r"<br\s*/?>", "\n", raw)
    t = re.sub(r"<[^>]+>", "", t)
    return _html.unescape(t).strip()


def _parse_dt(title: str) -> datetime | None:
    """'12.08.2022 17:36:12 UTC+01:00' -> tz-aware UTC datetime.

    NB: the export stamps a FIXED offset (the export machine's offset at export
    time). Summer signals may therefore be off by up to 1h vs true UTC. We keep
    the literal offset here and let the replay cross-check entry price against
    market price to detect any systematic shift.
    """
    m = re.match(
        r"(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2}) UTC([+-]\d{2}):(\d{2})",
        title,
    )
    if not m:
        return None
    d, mo, y, h, mi, s, oh, om = m.groups()
    local = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
    off_min = int(oh) * 60 + (int(om) if int(oh) >= 0 else -int(om))
    return (local - _delta(off_min)).replace(tzinfo=timezone.utc)


def _delta(minutes: int):
    from datetime import timedelta

    return timedelta(minutes=minutes)


# ───────────────────────── signal extraction ───────────────────────────────

# Asset name -> local-data ticker (None = no local data / not on XTB)
ASSET_MAP: dict[str, str | None] = {
    # indices
    "nasdaq": "US100",
    "nasdaq100": "US100",
    "nas100": "US100",
    "us100": "US100",
    "ustech": "US100",
    "dow": "US30",
    "dowjones": "US30",
    "us30": "US30",
    "sp500": "US500",
    "sandp500": "US500",
    "us500": "US500",
    "spx500": "US500",
    "dax": "GER30",
    "ger40": "GER30",
    "ger30": "GER30",
    "fra40": "FRA40",
    "cac40": "FRA40",
    # metals / energy
    "gold": "XAUUSD",
    "xauusd": "XAUUSD",
    "silver": "XAGUSD",
    "argent": "XAGUSD",
    "xagusd": "XAGUSD",
    "wti": "USOIL",
    "oil": "USOIL",
    "petrole": "USOIL",
}

# FX / crypto pairs are normalised by removing the slash (EUR/USD -> EURUSD).
# We only have local data for a subset; the replay flags the rest as untestable.
LOCAL_TICKERS = {
    "AUDJPY", "BTCUSD", "ETHUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD",
    "US30", "US500", "USDCHF", "USDJPY", "XAUUSD", "XAGUSD", "USOIL", "GER30",
}

_BUY_KW = ("achat", "achète", "achete", "j'achète", "j’achète", "long", " buy", "acheter")
_SELL_KW = ("vente", "vends", "je vends", "short", " sell", "vendre")

_PAIR_RE = re.compile(r"\b([A-Z]{2,4})\s*/\s*([A-Z]{2,4})\b")
# A price token = contiguous digits with an optional decimal part. The data
# never uses spaces as thousands separators, so forbidding spaces avoids gluing
# the asset name to the price (e.g. 'Nasdaq100 11785' -> '100' + '11785').
_NUM = r"([0-9]+(?:[.,][0-9]+)?)"
_SL_RE = re.compile(r"SL\D{0,4}" + _NUM)
_TP_RE = re.compile(r"TP\s*([1-4])?\D{0,3}" + _NUM)
_MGMT_KW = (
    "je sors", "je clôture", "je cloture", "je ferme", "résumé des trades",
    "resume des trades", "place mon sl", "place le sl", "sortie de", "sortie du",
    "on sort", "clôture", "cloture", "sécurise", "secur",
)
_RATIO_RE = re.compile(r"[Rr]atio\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*/\s*([0-9]+)")
_TYPE_RE = re.compile(r"[Tt]ype de trade\s*[:\-]?\s*([^\n]+)")


def _to_float(tok: str) -> float | None:
    """French-formatted number -> float. '1647,350'->1647.35, '11 373'->11373."""
    s = tok.strip().replace(" ", "")
    # if both '.' and ',' present, assume '.'=thousands, ','=decimal
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    # collapse multiple dots (e.g. thousands '1.647.350' -> keep last as decimal)
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s)
    except ValueError:
        return None


def _detect_direction(text: str) -> str | None:
    low = " " + text.lower()
    buy = any(k in low for k in _BUY_KW)
    sell = any(k in low for k in _SELL_KW)
    if buy and not sell:
        return "BUY"
    if sell and not buy:
        return "SELL"
    return None  # ambiguous -> resolve by geometry


def _strip_accents(s: str) -> str:
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# crypto stable-quote normalisation: only BTC/ETH have local data
_CRYPTO_USDT_TO_USD = {"BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD"}


def _detect_asset(text: str) -> tuple[str | None, str | None]:
    """Return (raw_label, normalised_ticker)."""
    # 1) explicit pair like EUR/USD, ETH/USDT
    mp = _PAIR_RE.search(text)
    if mp:
        raw = f"{mp.group(1)}/{mp.group(2)}"
        tic = (mp.group(1) + mp.group(2)).upper()
        tic = _CRYPTO_USDT_TO_USD.get(tic, tic)
        return raw, tic
    # 2) named instrument (accent-insensitive, '&' removed -> 'S&P500'->'sp500')
    low = _strip_accents(text.lower()).replace("&", "")
    for key, tic in ASSET_MAP.items():
        if re.search(r"\b" + re.escape(_strip_accents(key)) + r"\b", low):
            return key, tic
    return None, None


@dataclass
class Signal:
    msg_id: int
    file: str
    dt_utc: str
    asset_raw: str | None
    ticker: str | None
    testable: bool
    direction: str | None
    dir_source: str
    entry: float | None
    sl: float | None
    tps: list[float] = field(default_factory=list)
    rr_stated: float | None = None
    trade_type: str | None = None
    has_photo: bool = False
    sanity_ok: bool = True
    sanity_note: str = ""
    raw_text: str = ""


@dataclass
class FollowUp:
    msg_id: int
    file: str
    dt_utc: str
    reply_to: int | None
    tags: list[str]
    raw_text: str


@dataclass
class Recap:
    """One self-reported outcome line from a weekly-recap message."""

    msg_id: int
    file: str
    dt_utc: str
    asset_raw: str
    ticker: str | None
    outcome: str  # 'WIN' (TPx), 'LOSS' (SL), 'BE' (breakeven)
    detail: str   # e.g. 'TP2', 'SL -19 points', 'BE'


# recap line e.g. '🔴NZD/CAD : SL ❌ -19 points' / '🟢CAD/CHF : TP2 ✅' / '🟡AUD/JPY : Sorti BE'
_RECAP_LINE = re.compile(
    r"(?P<asset>[A-Za-zÀ-ÿ]{2,12}(?:\s*/\s*[A-Za-zÀ-ÿ]{2,6})?)\s*:\s*(?P<out>[^\n]+)"
)


def _parse_recap(mid: int, fname: str, dt_iso: str, text: str) -> list[Recap]:
    out: list[Recap] = []
    for line in text.split("\n"):
        m = _RECAP_LINE.search(line)
        if not m:
            continue
        asset = m.group("asset").strip()
        rest = m.group("out")
        low = _strip_accents(rest.lower())
        if "sl" in low and ("❌" in rest or "loss" in low or "point" in low):
            outcome = "LOSS"
        elif "be" in low or "break" in low or "entr" in low:
            outcome = "BE"
        elif "tp" in low or "✅" in rest:
            outcome = "WIN"
        else:
            continue
        _, tic = _detect_asset(asset)
        out.append(Recap(mid, fname, dt_iso, asset, tic, outcome, rest.strip()[:40]))
    return out


# Follow-up outcome tags
_FU_PATTERNS = {
    "tp1": re.compile(r"TP\s*1\b.*?(✅|atteint|touch|hit|pris)", re.I),
    "tp2": re.compile(r"TP\s*2\b.*?(✅|atteint|touch|hit|pris)", re.I),
    "tp3": re.compile(r"TP\s*3\b.*?(✅|atteint|touch|hit|pris)", re.I),
    "tp4": re.compile(r"TP\s*4\b.*?(✅|atteint|touch|hit|pris)", re.I),
    "sl_hit": re.compile(r"\bSL\b.*?(touch|atteint|hit|pris)|stop\s*loss\s*(touch|atteint)", re.I),
    "breakeven": re.compile(r"point d[’']entr|break\s*even|seuil de rentab|SL au.*entr", re.I),
    "closed": re.compile(r"je\s+(sors|cl[ôo]ture|ferme)|cl[ôo]tur|sortie", re.I),
    "tp_generic": re.compile(r"\bTP\b.*✅", re.I),
}


def parse_file(path: Path) -> tuple[list[Signal], list[FollowUp], list["Recap"]]:
    html = path.read_text(encoding="utf-8")
    signals: list[Signal] = []
    follows: list[FollowUp] = []
    recaps: list[Recap] = []
    for m in _MSG_RE.finditer(html):
        body = m.group("body")
        mid = int(m.group("id"))
        dtm = _DATE_RE.search(body)
        dt = _parse_dt(dtm.group("dt")) if dtm else None
        dt_iso = dt.isoformat() if dt else ""
        tm = _TEXT_RE.search(body)
        text = _clean_text(tm.group("t")) if tm else ""
        reply = _REPLY_RE.search(body)
        reply_id = int(reply.group("rid")) if reply else None
        has_photo = bool(_PHOTO_RE.search(body))

        low = _strip_accents(text.lower())
        is_recap = "resume des trades" in low
        is_mgmt = any(k in low for k in _MGMT_KW)
        has_levels = bool(_SL_RE.search(text)) and bool(_TP_RE.search(text))

        if is_recap:
            recaps.extend(_parse_recap(mid, path.name, dt_iso, text))
            follows.append(FollowUp(mid, path.name, dt_iso, reply_id, ["recap"], text[:400]))
            continue
        if has_levels and not is_mgmt:
            sig = _build_signal(mid, path.name, dt_iso, text, has_photo)
            # a genuine new signal needs a resolvable direction + entry + SL + TP
            if sig.direction and sig.entry is not None and sig.sl is not None and sig.tps:
                signals.append(sig)
                continue
        tags = [k for k, rx in _FU_PATTERNS.items() if rx.search(text)]
        if tags or reply_id is not None or is_mgmt:
            follows.append(FollowUp(mid, path.name, dt_iso, reply_id, tags, text[:400]))
    return signals, follows, recaps


def _build_signal(mid: int, fname: str, dt_iso: str, text: str, photo: bool) -> Signal:
    asset_raw, ticker = _detect_asset(text)
    direction = _detect_direction(text)
    dir_source = "keyword" if direction else "geometry"

    sl_m = _SL_RE.search(text)
    sl = _to_float(sl_m.group(1)) if sl_m else None

    # collect TPs, dedup by index
    tp_map: dict[int, float] = {}
    auto_idx = 0
    for tm in _TP_RE.finditer(text):
        idx = int(tm.group(1)) if tm.group(1) else (auto_idx + 1)
        auto_idx = idx
        val = _to_float(tm.group(2))
        if val is not None and idx not in tp_map:
            tp_map[idx] = val
    tps = [tp_map[k] for k in sorted(tp_map)]

    # entry = first number that is NOT part of SL/TP. Strategy: take the number
    # following the asset/direction line, before the SL keyword.
    entry = _extract_entry(text, sl, tps)

    # resolve direction by geometry if keyword absent
    if direction is None and entry is not None and sl is not None:
        direction = "SELL" if sl > entry else "BUY"

    rr = None
    rm = _RATIO_RE.search(text)
    if rm:
        try:
            rr = float(rm.group(1).replace(",", ".")) / float(rm.group(2))
        except (ValueError, ZeroDivisionError):
            rr = None
    tt_m = _TYPE_RE.search(text)
    trade_type = tt_m.group(1).strip()[:40] if tt_m else None

    sig = Signal(
        msg_id=mid, file=fname, dt_utc=dt_iso, asset_raw=asset_raw, ticker=ticker,
        testable=(ticker in LOCAL_TICKERS) if ticker else False,
        direction=direction, dir_source=dir_source, entry=entry, sl=sl, tps=tps,
        rr_stated=rr, trade_type=trade_type, has_photo=photo,
        raw_text=text[:600],
    )
    _sanity(sig)
    return sig


def _extract_entry(text: str, sl: float | None, tps: list[float]) -> float | None:
    """Entry = first numeric token appearing before the 'SL' keyword."""
    head = text.split("SL", 1)[0]
    nums = re.findall(_NUM, head)
    cand = [v for v in (_to_float(n) for n in nums) if v is not None]
    # drop obvious noise (years, '50%', tp indices etc.) by preferring a value
    # whose magnitude matches SL/TPs
    ref = sl if sl is not None else (tps[0] if tps else None)
    if ref is not None and cand:
        cand.sort(key=lambda v: abs(v - ref))
        return cand[0]
    return cand[-1] if cand else None


def _sanity(s: Signal) -> None:
    """Validate geometry; flag implausible parses without crashing."""
    if s.entry is None or s.sl is None or not s.tps:
        s.sanity_ok = False
        s.sanity_note = "champ manquant"
        return
    notes = []
    if s.direction == "BUY":
        if not (s.sl < s.entry):
            notes.append("SL>=entry pour BUY")
        if not all(tp > s.entry for tp in s.tps):
            notes.append("TP<=entry pour BUY")
    elif s.direction == "SELL":
        if not (s.sl > s.entry):
            notes.append("SL<=entry pour SELL")
        if not all(tp < s.entry for tp in s.tps):
            notes.append("TP>=entry pour SELL")
    # magnitude sanity: SL/TP within 30% of entry (avoid decimal-place errors)
    for label, v in [("SL", s.sl), *[(f"TP{i+1}", t) for i, t in enumerate(s.tps)]]:
        if s.entry and (v < s.entry * 0.5 or v > s.entry * 2.0):
            notes.append(f"{label} hors plage (x{v / s.entry:.2f})")
    if notes:
        s.sanity_ok = False
        s.sanity_note = "; ".join(notes)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = sorted(HERE.glob("messages*.html"))
    all_sig: list[Signal] = []
    all_fu: list[FollowUp] = []
    all_rc: list[Recap] = []
    for f in files:
        sg, fu, rc = parse_file(f)
        all_sig.extend(sg)
        all_fu.extend(fu)
        all_rc.extend(rc)

    # deletion map across the full id range
    ids = sorted({s.msg_id for s in all_sig} | {f.msg_id for f in all_fu})
    # recompute from raw to include non-signal messages
    all_ids: set[int] = set()
    for f in files:
        for m in re.finditer(r'id="message(\d+)"', f.read_text(encoding="utf-8")):
            all_ids.add(int(m.group(1)))
    lo, hi = min(all_ids), max(all_ids)
    deleted = sorted(set(range(lo, hi + 1)) - all_ids)

    import csv

    with (OUT / "signals.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "msg_id", "file", "dt_utc", "asset_raw", "ticker", "testable",
            "direction", "dir_source", "entry", "sl", "tp1", "tp2", "tp3", "tp4",
            "n_tp", "rr_stated", "trade_type", "has_photo", "sanity_ok",
            "sanity_note", "raw_text",
        ])
        for s in all_sig:
            tps = (s.tps + [None, None, None, None])[:4]
            w.writerow([
                s.msg_id, s.file, s.dt_utc, s.asset_raw, s.ticker, s.testable,
                s.direction, s.dir_source, s.entry, s.sl, *tps, len(s.tps),
                s.rr_stated, s.trade_type, s.has_photo, s.sanity_ok,
                s.sanity_note, s.raw_text.replace("\n", " ⏎ "),
            ])

    with (OUT / "followups.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["msg_id", "file", "dt_utc", "reply_to", "tags", "raw_text"])
        for fu in all_fu:
            w.writerow([
                fu.msg_id, fu.file, fu.dt_utc, fu.reply_to, "|".join(fu.tags),
                fu.raw_text.replace("\n", " ⏎ "),
            ])

    with (OUT / "recaps.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["msg_id", "file", "dt_utc", "asset_raw", "ticker", "outcome", "detail"])
        for rc in all_rc:
            w.writerow([rc.msg_id, rc.file, rc.dt_utc, rc.asset_raw, rc.ticker,
                        rc.outcome, rc.detail])

    (OUT / "deleted.json").write_text(
        json.dumps({"range": [lo, hi], "deleted_ids": deleted, "n_deleted": len(deleted)},
                   indent=2),
        encoding="utf-8",
    )

    _report(all_sig, all_fu, all_rc, deleted, lo, hi)


def _report(sigs, fus, recaps, deleted, lo, hi) -> None:
    ok = [s for s in sigs if s.sanity_ok]
    testable = [s for s in ok if s.testable]
    from collections import Counter

    by_tic = Counter(s.ticker for s in ok)
    by_dir = Counter(s.direction for s in ok)
    rc_out = Counter(r.outcome for r in recaps)
    lines = [
        "=== PARSE REPORT — Trading Family VIP ===",
        f"messages id range: {lo}..{hi}",
        f"deleted (gaps): {len(deleted)}",
        f"signals detected: {len(sigs)}",
        f"  sanity OK: {len(ok)}",
        f"  failed sanity: {len(sigs) - len(ok)}",
        f"  testable (local data): {len(testable)}",
        f"follow-up messages: {len(fus)}",
        f"recap outcome lines: {len(recaps)}  {dict(rc_out)}",
        "",
        "signals by ticker (sanity OK):",
        *[f"  {t or '??':10s} {n}" for t, n in by_tic.most_common()],
        "",
        f"direction split: {dict(by_dir)}",
    ]
    txt = "\n".join(str(x) for x in lines)
    (OUT / "parse_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
