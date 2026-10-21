"""Daily snapshot of the listed S&P 500 calls at three expiries on one strike grid (hypotheses H20 and H21).

Why a second capture
- H20 measures how the Dec-2029 at-the-money-forward (ATMF) vol moves with the index; H21 measures how the ATMF skew
  falls from one to three years. Both need, on each close, the two strikes that bracket each expiry's forward. The
  Dec-2029 capture in `listed.py` holds the strikes XX needs (8,000 and 9,000-15,000), not those near the forwards.
- One TradingView call reads every strike of all three expiries together, so the three skews of a close come from the
  same moment.

The grid (fixed 2026-10-21, after H20 and H21 were frozen and before their test window opens on 2026-11-02): calls at
6,500 to 10,000 in steps of 500 for Dec-2027 (2027-12-17), Dec-2028 (2028-12-15) and Dec-2029 (2029-12-21), plus
SP:SPX and AMEX:SPY. With the S&P 500 at 7,712 (2026-10-21) the three forwards sit near 7,900-8,400, so the grid
brackets them after a fall of about 15% or a rise of about 20%.

How a reply is filed (the same rules as `listed.py`)
- The close it belongs to and protocol P2's read-time window come from `listed.close_for`; the feed's delay comes
  from the reply (`listed.feed_delay_seconds`), so P3's market-time rule can be applied once P3 is frozen.
- A strike without a usable two-sided quote (bid or ask missing, zero or below, or bid above ask) counts as missing.
- The raw reply goes to data/raw/tradingview_strips/<close date>/<read time>.json; one row per quoted call goes to
  data/tables/listed_strips.csv. Nothing is ever overwritten.
- TradingView's data is display-only: everything stays in the git-ignored data/ folder.
"""
import csv
import json
from datetime import date, timezone

from . import fetch, listed

EXPIRIES = {"271217": date(2027, 12, 17), "281215": date(2028, 12, 15), "291221": date(2029, 12, 21)}
STRIKES = (6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000)
REFERENCE = listed.REFERENCE
COLUMNS = ["close_date", "read_utc", "in_window", "feed_delay_s", "symbol", "expiry", "strike", "bid", "ask", "spx",
           "spy"]


def option_symbol(code, strike):
    """TradingView's name for a listed S&P 500 call, e.g. OPRA:SPX271217C8000.0."""
    return f"OPRA:SPX{code}C{strike}.0"


def symbols():
    """Everything the capture asks for (26 symbols): each expiry's calls, then the two reference prices."""
    return [option_symbol(code, k) for code in EXPIRIES for k in STRIKES] + list(REFERENCE)


def parse(reply_text):
    """({symbol: (bid, ask), reference: (close, None)}, [missing symbols]). Refuses anything but a successful reply."""
    reply = json.loads(reply_text)
    if not reply.get("success") or not isinstance(reply.get("data"), dict):
        raise ValueError("not a successful TradingView batch reply")
    wanted = set(symbols())
    quotes = {}
    for sym, row in reply["data"].items():
        if sym not in wanted:
            continue                                            # never file a symbol the grid did not ask for
        if sym in REFERENCE:
            if row.get("close") is not None:
                quotes[sym] = (float(row["close"]), None)
            continue
        bid, ask = row.get("bid"), row.get("ask")
        if bid is None or ask is None:
            continue
        bid, ask = float(bid), float(ask)
        if bid <= 0 or ask <= 0 or bid > ask:
            continue
        quotes[sym] = (bid, ask)
    return quotes, [s for s in symbols() if s not in quotes]


def store(reply_text, read_utc, root=None):
    """Check and file one reply. Returns a summary dict; a repeated reply for the same read time changes nothing."""
    quotes, missing = parse(reply_text)
    delay = listed.feed_delay_seconds(reply_text)
    close, in_window = listed.close_for(read_utc)
    root = fetch.data_dir(root)
    raw_dir = root / "raw" / "tradingview_strips" / close.isoformat()
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = read_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"{stamp}.json"
    quoted = {code: sum(1 for k in STRIKES if option_symbol(code, k) in quotes) for code in EXPIRIES}
    summary = {"close": close, "in_window": in_window, "feed_delay_s": delay, "quoted": quoted, "missing": missing,
               "raw": raw_path, "new": not raw_path.exists()}
    if not summary["new"]:
        return summary
    with open(raw_path, "x", encoding="utf-8") as fh:            # exclusive: never overwrite a raw reply
        fh.write(reply_text)
    table = root / "tables" / "listed_strips.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    first = not table.exists()
    spx = quotes.get("SP:SPX", (None,))[0]
    spy = quotes.get("AMEX:SPY", (None,))[0]
    with open(table, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(COLUMNS)
        for code, expiry in EXPIRIES.items():
            for k in STRIKES:
                sym = option_symbol(code, k)
                if sym in quotes:
                    bid, ask = quotes[sym]
                    w.writerow([close.isoformat(), stamp, in_window, delay, sym, expiry.isoformat(), k, bid, ask,
                                spx, spy])
    return summary


def load(root=None):
    """All stored rows (a list of dicts), or [] before the first capture."""
    table = fetch.data_dir(root) / "tables" / "listed_strips.csv"
    if not table.exists():
        return []
    with open(table, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
