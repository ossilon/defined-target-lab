"""Listed-option capture: close dates, protocol P2's read window, reply checks, filing, comparison.

Synthetic replies only (no TradingView data). Times are written in UTC; the comments give New York time.
"""
import json
import tempfile
from datetime import date, datetime, timezone

from _harness import check, raises

from definedtarget.forward import listed


def utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# 1. Which close, and inside the window? (New York on daylight time until 2026-11-01, standard time after.)
cases = [
    # read time (UTC)          expected close        inside?  why
    (utc(2026, 10, 19, 22, 15), date(2026, 10, 19), True,  "Mon 18:15 New York"),
    (utc(2026, 10, 20, 4, 0),   date(2026, 10, 19), False, "Tue midnight New York: overnight session"),
    (utc(2026, 10, 16, 22, 0),  date(2026, 10, 16), True,  "Fri 18:00 New York"),
    (utc(2026, 10, 17, 5, 0),   date(2026, 10, 16), False, "Sat 01:00 New York"),
    (utc(2026, 10, 19, 20, 0),  date(2026, 10, 16), False, "Mon 16:00 New York: options still open, last close Fri"),
    (utc(2026, 11, 2, 22, 15),  date(2026, 11, 2),  True,  "Mon 17:15 New York (standard time)"),
    (utc(2026, 11, 2, 20, 30),  date(2026, 10, 30), False, "Mon 15:30 New York (standard time): before the close"),
    (utc(2026, 11, 26, 22, 15), date(2026, 11, 25), False, "Thanksgiving: no session, last close Wed"),
    (utc(2026, 10, 20, 0, 16),  date(2026, 10, 19), False, "Mon 20:16 New York: one minute after the window"),
]
for when, close, inside, why in cases:
    check(f"close_for: {why} -> close {close}, {'inside' if inside else 'outside'} the window",
          listed.close_for(when) == (close, inside), listed.close_for(when))


# 2. Reply checks.
def reply(quotes, success=True, extra=None):
    data = {listed.option_symbol(k): {"bid": b, "ask": a, "close": None} for k, (b, a) in quotes.items()}
    data["SP:SPX"] = {"close": 7800.0}
    data["AMEX:SPY"] = {"close": 777.0}
    data.update(extra or {})
    return json.dumps({"success": success, "data": data, "missing": []})


good = reply({10000: (530.0, 590.0), 9600: (670.0, 740.0)})
q, missing = listed.parse(good)
check("a good reply gives each call's bid and ask", q[listed.option_symbol(10000)] == (530.0, 590.0))
check("strikes TradingView did not return are listed as missing", listed.option_symbol(15000) in missing
      and listed.option_symbol(10000) not in missing)
raises("a failed reply is refused", lambda: listed.parse(reply({}, success=False)), "not a successful")
for why, quote in (("a crossed quote (bid above ask)", (600.0, 590.0)), ("a negative bid", (-1.0, 590.0)),
                   ("a one-sided quote (ask 0, as seen after the close)", (183.9, 0.0))):
    q_bad, miss_bad = listed.parse(reply({10000: quote, 9600: (670.0, 740.0)}))
    check(f"{why} counts as missing, and the other strikes are kept",
          listed.option_symbol(10000) in miss_bad and listed.option_symbol(9600) in q_bad)
no_quote = reply({}, extra={listed.option_symbol(10000): {"bid": None, "ask": None}})
check("a call with no bid or ask today counts as missing", listed.option_symbol(10000) in listed.parse(no_quote)[1])
check("the symbol list is the calls plus SP:SPX and AMEX:SPY",
      listed.symbols()[-2:] == ["SP:SPX", "AMEX:SPY"] and len(listed.symbols()) == len(listed.STRIKES) + 2)

# 3. Filing: raw reply kept, rows appended, a repeat changes nothing, then the inside/outside comparison.
with tempfile.TemporaryDirectory() as tmp:
    s1 = listed.store(good, utc(2026, 10, 16, 22, 0), root=tmp)            # Fri 18:00 New York: inside
    check("a stored reading names its close and window", s1["close"] == date(2026, 10, 16) and s1["in_window"])
    check("the raw reply is kept unchanged", s1["raw"].read_text(encoding="utf-8") == good)
    check("one table row per quoted call", len(listed.load(tmp)) == 2)
    again = listed.store(good, utc(2026, 10, 16, 22, 0), root=tmp)
    check("storing the same reading twice changes nothing", not again["new"] and len(listed.load(tmp)) == 2)
    moved = reply({10000: (531.0, 590.0), 9600: (670.0, 740.0)})
    listed.store(moved, utc(2026, 10, 17, 5, 0), root=tmp)                 # Sat 01:00 New York: outside
    n, same, diffs = listed.compare(date(2026, 10, 16), root=tmp)
    check("the comparison finds which strikes changed overnight", (n, same) == (2, 1) and diffs[0][0] == "10000",
          (n, same, diffs))
    check("rows read outside the window are flagged, never mixed in",
          sorted(r["in_window"] for r in listed.load(tmp)) == ["False", "False", "True", "True"])

# 5. Draft protocol P3 (not frozen): the feed's delay and each quote's market time.
def reply_mode(mode):
    data = {listed.option_symbol(10000): {"bid": 530.0, "ask": 590.0, "update_mode": mode},
            "SP:SPX": {"close": 7800.0, "update_mode": "delayed_streaming_600"},
            "AMEX:SPY": {"close": 777.0, "update_mode": "streaming"}}
    return json.dumps({"success": True, "data": data})


check("feed delay: 'delayed_streaming_900' is 15 minutes", listed.feed_delay_seconds(reply_mode("delayed_streaming_900")) == 900)
check("feed delay: a live feed is 0", listed.feed_delay_seconds(reply_mode("streaming")) == 0)
check("feed delay: readings saved without update_mode give None (not a guess)",
      listed.feed_delay_seconds(json.dumps({"success": True, "data": {listed.option_symbol(10000): {"bid": 1, "ask": 2}}}))
      is None)
raises("feed delay: an unknown mode is refused", lambda: listed.feed_delay_seconds(reply_mode("weird")), "unknown feed mode")
p3_cases = [
    # read time (UTC), delay, expected close, inside draft P3?, why
    (utc(2026, 10, 19, 20, 16), 900, date(2026, 10, 19), True, "Mon read 16:16 NY, 15-min delay: market 16:01"),
    (utc(2026, 10, 19, 20, 28), 900, date(2026, 10, 19), True, "Mon read 16:28 NY: market 16:13"),
    (utc(2026, 10, 19, 20, 31), 900, date(2026, 10, 19), False, "Mon read 16:31 NY: market 16:16, options closed"),
    (utc(2026, 10, 19, 20, 14), 900, date(2026, 10, 16), False, "Mon read 16:14 NY: market 15:59, S&P not closed"),
    (utc(2026, 10, 19, 20, 5), 0, date(2026, 10, 19), True, "live feed read 16:05 NY"),
    (utc(2026, 11, 2, 21, 20), 900, date(2026, 11, 2), True, "after the clock change: read 16:20 EST, market 16:05"),
    (utc(2026, 10, 17, 20, 20), 900, date(2026, 10, 16), False, "Saturday: no session"),
]
for read, delay, close, inside, why in p3_cases:
    got_close, market, ok = listed.p3_market_time(read, delay)
    check(f"draft P3: {why}", (got_close, ok) == (close, inside), (got_close, market.strftime("%a %H:%M"), ok))
