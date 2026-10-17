"""Daily snapshot of the listed S&P 500 Dec-2029 calls (backlog B2; hypotheses H9, H13, H17, H19).

How the capture works
- A Claude scheduled task asks the TradingView connection for the delayed bid and ask of each call in STRIKES
  (plus SP:SPX and AMEX:SPY) and saves the reply, unchanged, as a JSON file.
- `store()` checks that reply and files it:
  1. The close it belongs to: the latest NYSE session already closed (16:15 New York time) at the moment of reading.
  2. Protocol P2's window: a quote counts only if read between 16:15 and 20:15 New York time on that close date
     (after the regular session, before Cboe's overnight session). Reads outside the window are kept and flagged,
     never counted; they serve comparisons such as the overnight test (BACKLOG B2b).
  3. The raw reply goes to data/raw/tradingview_chain/<close date>/<read time>.json, and one row per call goes to
     data/tables/listed_chain.csv. Nothing is ever overwritten.
- TradingView's data is display-only: everything stays in the git-ignored data/ folder; only derived aggregates are
  published.
"""
import csv
import json
from datetime import time, timezone
from zoneinfo import ZoneInfo

from .. import calendar
from . import fetch

NEW_YORK = ZoneInfo("America/New_York")
WINDOW_START, WINDOW_END = time(16, 15), time(20, 15)          # protocol P2, New York time
EXPIRY_CODE = "291221"                                         # OPRA code for 2029-12-21
STRIKES = (8000, 9000, 9500, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 12000, 13000, 14000, 15000)
REFERENCE = ("SP:SPX", "AMEX:SPY")
COLUMNS = ["close_date", "read_utc", "in_window", "symbol", "strike", "bid", "ask", "spx", "spy"]


def option_symbol(strike):
    """TradingView's name for the listed Dec-2029 S&P 500 call at `strike`, e.g. OPRA:SPX291221C10000.0."""
    return f"OPRA:SPX{EXPIRY_CODE}C{strike}.0"


def symbols():
    """Everything the capture asks for: the calls, then the two reference prices."""
    return [option_symbol(k) for k in STRIKES] + list(REFERENCE)


def close_for(read_utc):
    """(close date, read inside P2's window?) for a reading taken at `read_utc` (a timezone-aware datetime).

    The close is the latest NYSE session whose options had closed (16:15 New York time) when the quotes were read.
    """
    ny = read_utc.astimezone(NEW_YORK)
    day = ny.date()
    if calendar.is_session(day) and ny.time() >= WINDOW_START:
        close = day
    else:
        close = calendar.previous_session(day)
    in_window = ny.date() == close and WINDOW_START <= ny.time() <= WINDOW_END
    return close, in_window


# Draft protocol P3 (NOT frozen; evidence gathered 2026-10-19 to 2026-10-30, BACKLOG B2): a quote counts only if its
# market time (read time minus the feed's stated delay) falls between 16:00 and 16:15 New York time on its close date:
# after the S&P 500 closes, while SPX options still trade. TradingView's OPRA feed reports update_mode
# "delayed_streaming_900" (15 minutes); with it, readings between 16:15 and 16:30 show the quotes of 16:00-16:15.
P3_START, P3_END = time(16, 0), time(16, 15)


def feed_delay_seconds(reply_text):
    """The option quotes' delay in seconds from a saved reply ('delayed_streaming_900' -> 900, 'streaming' -> 0), or
    None if the reply did not include update_mode (readings before 2026-10-17). Every option row must agree."""
    modes = {row.get("update_mode") for sym, row in json.loads(reply_text).get("data", {}).items()
             if sym not in REFERENCE}
    if not modes or None in modes:
        return None
    if len(modes) > 1:
        raise ValueError(f"option rows report different feed modes: {sorted(modes)}")
    mode = modes.pop()
    if mode.startswith("delayed_streaming_"):
        return int(mode.rsplit("_", 1)[1])
    if mode in ("streaming", "realtime"):
        return 0
    raise ValueError(f"unknown feed mode: {mode}")


def p3_market_time(read_utc, delay_seconds):
    """(close date, market time in New York, inside the draft P3 window?) for a reading with a known feed delay."""
    from datetime import timedelta
    market = (read_utc - timedelta(seconds=delay_seconds)).astimezone(NEW_YORK)
    day = market.date()
    ok = calendar.is_session(day) and P3_START <= market.time() < P3_END
    close = day if ok or (calendar.is_session(day) and market.time() >= P3_START) else calendar.previous_session(day)
    return close, market, ok


def parse(reply_text):
    """{symbol: (bid, ask) or (close, None) for references} plus the list of symbols TradingView did not return.

    Refuses a reply that is not a successful TradingView batch reply. A call without a usable two-sided quote (bid or
    ask missing, bid or ask of zero or below, or bid above ask) counts as missing for that strike: after the 16:15
    close many long-dated quotes turn one-sided (ask 0), so one bad strike must not throw away the others.
    """
    reply = json.loads(reply_text)
    if not reply.get("success") or not isinstance(reply.get("data"), dict):
        raise ValueError("not a successful TradingView batch reply")
    quotes = {}
    for sym, row in reply["data"].items():
        if sym in REFERENCE:
            quotes[sym] = (float(row["close"]), None)
            continue
        bid, ask = row.get("bid"), row.get("ask")
        if bid is None or ask is None:
            continue                                            # no quote today: counts as missing
        bid, ask = float(bid), float(ask)
        if bid <= 0 or ask <= 0 or bid > ask:
            continue                                            # one-sided or crossed: counts as missing
        quotes[sym] = (bid, ask)
    missing = [s for s in symbols() if s not in quotes]
    return quotes, missing


def store(reply_text, read_utc, root=None):
    """Check and file one reply. Returns a summary dict; a repeated reply for the same read time changes nothing."""
    quotes, missing = parse(reply_text)
    close, in_window = close_for(read_utc)
    root = fetch.data_dir(root)
    raw_dir = root / "raw" / "tradingview_chain" / close.isoformat()
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = read_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"{stamp}.json"
    summary = {"close": close, "in_window": in_window, "quoted": sum(1 for s in quotes if s not in REFERENCE),
               "missing": missing, "raw": raw_path, "new": not raw_path.exists()}
    if not summary["new"]:
        return summary
    raw_path.write_text(reply_text, encoding="utf-8")
    table = root / "tables" / "listed_chain.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    first = not table.exists()
    spx = quotes.get("SP:SPX", (None,))[0]
    spy = quotes.get("AMEX:SPY", (None,))[0]
    with open(table, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(COLUMNS)
        for k in STRIKES:
            sym = option_symbol(k)
            if sym in quotes:
                bid, ask = quotes[sym]
                w.writerow([close.isoformat(), stamp, in_window, sym, k, bid, ask, spx, spy])
    return summary


def load(root=None):
    """All stored rows (a list of dicts), or [] before the first capture."""
    table = fetch.data_dir(root) / "tables" / "listed_chain.csv"
    if not table.exists():
        return []
    with open(table, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def compare(close_date, root=None):
    """For one close: does a reading outside P2's window give the same quotes as the one inside it?

    Returns (strikes compared, strikes identical, list of differences) between the first in-window reading and the
    last out-of-window reading of that close (the overnight test, BACKLOG B2b).
    """
    rows = [r for r in load(root) if r["close_date"] == str(close_date)]
    inside = [r for r in rows if r["in_window"] == "True"]
    outside = [r for r in rows if r["in_window"] == "False"]
    if not inside or not outside:
        return 0, 0, []
    first_in = min(r["read_utc"] for r in inside)
    last_out = max(r["read_utc"] for r in outside)
    a = {r["strike"]: (r["bid"], r["ask"]) for r in inside if r["read_utc"] == first_in}
    b = {r["strike"]: (r["bid"], r["ask"]) for r in outside if r["read_utc"] == last_out}
    both = sorted(set(a) & set(b), key=int)
    diffs = [(k, a[k], b[k]) for k in both if a[k] != b[k]]
    return len(both), len(both) - len(diffs), diffs
