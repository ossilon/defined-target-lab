"""Three-expiry strike-grid capture (H20, H21): symbols, reply checks, filing, feed delay, cloud import.

Synthetic replies only (no TradingView data). Times are written in UTC; the comments give New York time.
"""
import csv
import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from _harness import check, raises

from definedtarget import calendar
from definedtarget.forward import cloud_import, listed, strips


def utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# 1. The grid.
syms = strips.symbols()
check("the grid asks for 3 expiries x 8 strikes plus SP:SPX and AMEX:SPY (26 symbols, under the batch limit of 50)",
      len(syms) == 26 and len(set(syms)) == 26 and syms[-2:] == ["SP:SPX", "AMEX:SPY"], len(syms))
check("symbol names follow TradingView's OPRA form", strips.option_symbol("271217", 8000) == "OPRA:SPX271217C8000.0")
check("each expiry date matches its OPRA code (YYMMDD)",
      all(d.strftime("%y%m%d") == code for code, d in strips.EXPIRIES.items()))
check("each expiry is the third Friday of December, a NYSE session (standard monthly SPX expiry)",
      all(d.weekday() == 4 and 15 <= d.day <= 21 and calendar.is_session(d) for d in strips.EXPIRIES.values()))
check("the Dec-2029 code is the one the Dec-2029 capture uses", listed.EXPIRY_CODE in strips.EXPIRIES)
check("strikes increase in steps of 500 from 6,500 to 10,000",
      strips.STRIKES == tuple(range(6500, 10001, 500)))


# 2. Reply checks.
def reply(quotes, success=True, mode="delayed_streaming_900", extra=None, spx=7712.57):
    data = {strips.option_symbol(c, k): {"bid": b, "ask": a, "close": None, "update_mode": mode}
            for (c, k), (b, a) in quotes.items()}
    data["SP:SPX"] = {"close": spx, "update_mode": "delayed_streaming_600"}
    data["AMEX:SPY"] = {"close": 769.05, "update_mode": "streaming"}
    data.update(extra or {})
    return json.dumps({"success": success, "data": data, "missing": []})


good = reply({("271217", 8000): (602.6, 605.5), ("281215", 8000): (1018.2, 1043.3),
              ("291221", 8500): (1107.5, 1183.3)})
q, missing = strips.parse(good)
check("a good reply gives each call's bid and ask", q[strips.option_symbol("281215", 8000)] == (1018.2, 1043.3))
check("the reference closes are kept", q["SP:SPX"] == (7712.57, None))
check("strikes not returned are missing", strips.option_symbol("271217", 6500) in missing
      and strips.option_symbol("271217", 8000) not in missing and len(missing) == 21, len(missing))   # 26 - 3 - 2
raises("a failed reply is refused", lambda: strips.parse(reply({}, success=False)), "not a successful")
for why, quote in (("a crossed quote", (606.0, 605.5)), ("a zero ask", (602.6, 0.0)), ("a missing bid", (None, 605.5))):
    qb, mb = strips.parse(reply({("271217", 8000): quote, ("281215", 8000): (1018.2, 1043.3)}))
    check(f"{why} counts as missing; the other strikes are kept",
          strips.option_symbol("271217", 8000) in mb and strips.option_symbol("281215", 8000) in qb)
stray = reply({("271217", 8000): (602.6, 605.5)},
              extra={"OPRA:SPX271217C8250.0": {"bid": 1.0, "ask": 2.0, "update_mode": "delayed_streaming_900"}})
check("a symbol outside the grid is never filed", "OPRA:SPX271217C8250.0" not in strips.parse(stray)[0])
check("the feed delay is read from the option rows (900 s)", listed.feed_delay_seconds(good) == 900)


# 3. Filing.
with tempfile.TemporaryDirectory() as tmp:
    s1 = strips.store(good, utc(2026, 10, 21, 20, 27), root=tmp)          # Wed 16:27 New York (daylight time)
    check("a 16:27 New York reading files under that day's close, inside P2's window, with its feed delay",
          s1["close"] == date(2026, 10, 21) and s1["in_window"] and s1["feed_delay_s"] == 900 and s1["new"], s1)
    check("the summary counts quoted strikes per expiry",
          s1["quoted"] == {"271217": 1, "281215": 1, "291221": 1}, s1["quoted"])
    rows = strips.load(tmp)
    check("one table row per quoted call, with expiry, strike, delay and both reference closes",
          len(rows) == 3 and rows[0]["expiry"] == "2027-12-17" and rows[0]["strike"] == "8000"
          and rows[0]["feed_delay_s"] == "900" and rows[0]["spx"] == "7712.57" and rows[0]["spy"] == "769.05", rows)
    raw = Path(s1["raw"])
    check("the raw reply is kept byte for byte under tradingview_strips/<close>/",
          raw.read_text(encoding="utf-8") == good and raw.parent.name == "2026-10-21"
          and raw.parent.parent.name == "tradingview_strips")
    again = strips.store(good, utc(2026, 10, 21, 20, 27), root=tmp)
    check("the same reading again changes nothing", not again["new"] and len(strips.load(tmp)) == 3)
    s2 = strips.store(good, utc(2026, 11, 2, 21, 27), root=tmp)            # Mon 16:27 New York (standard time)
    check("after the clock change, 21:27 UTC is 16:27 New York: that day's close, inside P2's window",
          s2["close"] == date(2026, 11, 2) and s2["in_window"], s2)
    s3 = strips.store(good, utc(2026, 11, 2, 20, 27), root=tmp)            # Mon 15:27 New York: before the close
    check("after the clock change, 20:27 UTC (15:27 New York) belongs to the previous close and is outside the window",
          s3["close"] == date(2026, 10, 30) and not s3["in_window"], s3)
    with open(Path(tmp) / "tables" / "listed_strips.csv", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    check("the table header is the documented column list", header == strips.COLUMNS, header)

# 4. Draft P3's market-time rule can be applied to the stored readings (16:27 read, 15-minute delay -> 16:12).
close, market, ok = listed.p3_market_time(utc(2026, 10, 21, 20, 27), 900)
check("a 16:27 reading with a 15-minute delay has market time 16:12, inside draft P3's window", ok
      and close == date(2026, 10, 21) and (market.hour, market.minute) == (16, 12))
close, market, ok = listed.p3_market_time(utc(2026, 11, 2, 21, 27), 900)
check("the same holds at 21:27 UTC after the clock change", ok and close == date(2026, 11, 2))

# 5. Cloud import: replies in tradingview/spx_strips/ are filed through strips.store; a bad file name is refused.
with tempfile.TemporaryDirectory() as tmp:
    clone, local = Path(tmp) / "research-captures", Path(tmp) / "data"
    sdir = clone / "tradingview" / "spx_strips"
    sdir.mkdir(parents=True)
    (sdir / "20261021T202700Z.json").write_text(good, encoding="utf-8")
    (sdir / "not-a-time.json").write_text(good, encoding="utf-8")
    g = cloud_import.import_strips(clone, root=local)
    filed = [x for x in g if "refused" not in x]
    check("a cloud grid reply is filed with its read time; a file with a bad name is refused, not fatal",
          len(filed) == 1 and str(filed[0]["close"]) == "2026-10-21"
          and any(x.get("file") == "not-a-time.json" for x in g), g)
    check("importing again files nothing new (only the refusal repeats)",
          [x for x in cloud_import.import_strips(clone, root=local) if "refused" not in x] == [])
    check("the Dec-2029 quote folder is untouched by the grid import", cloud_import.import_quotes(clone, root=local) == [])
