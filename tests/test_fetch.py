"""Capture logic with a fake web server: tests never touch the internet."""
import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from _harness import check

from definedtarget.forward import fetch, roundhill

T0 = datetime(2026, 10, 14, 3, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 14, 5, 0, 0, tzinfo=timezone.utc)


def server(body, status=200, lm="Tue, 13 Oct 2026 23:22:34 GMT"):
    return lambda url: (status, {"Last-Modified": lm, "Content-Type": "text/csv"}, body)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    url = "https://example.test/a.csv"
    r1 = fetch.capture(url, "a.csv", "test", b"Date,", root=root, now=T0, getter=server(b"Date,x\n1,2\n"))
    check("first valid fetch is saved", r1["result"] == "saved" and (root / r1["path"]).exists(), r1)
    check("saved bytes are unchanged", (root / r1["path"]).read_bytes() == b"Date,x\n1,2\n")
    check("manifest keeps the sha256 of the bytes", r1["sha256"] == fetch.sha256(b"Date,x\n1,2\n"))
    r2 = fetch.capture(url, "a.csv", "test", b"Date,", root=root, now=T1, getter=server(b"Date,x\n1,2\n"))
    check("same bytes again: 'unchanged', nothing new saved", r2["result"] == "unchanged" and "path" not in r2, r2)
    r3 = fetch.capture(url, "a.csv", "test", b"Date,", root=root, now=T1, getter=server(b"Date,x\n1,3\n"))
    check("revised bytes on the same day: saved as a second copy, first copy kept",
          r3["result"] == "saved" and r3["path"] != r1["path"] and (root / r1["path"]).exists(), r3)
    r4 = fetch.capture("https://example.test/b.csv", "b.csv", "test", b"Date,", root=root, now=T0,
                       getter=server(b"<html>404</html>"))
    check("an error page served with status 200 is marked invalid", r4["result"] == "invalid", r4)
    r5 = fetch.capture("https://example.test/b.csv", "b.csv", "test", b"Date,", root=root, now=T1,
                       getter=server(b"<html>404</html>"))
    check("the same error page again is not re-saved", r5["result"] == "invalid_repeat" and "path" not in r5, r5)
    r6 = fetch.capture("https://example.test/c.csv", "c.csv", "test", None, root=root, now=T0,
                       getter=server(b"", status=503))
    check("an HTTP error is recorded, not raised", r6["result"] == "http_error" and r6["status"] == 503, r6)

    def boom(url):
        raise OSError("no network")
    r7 = fetch.capture("https://example.test/d.csv", "d.csv", "test", None, root=root, now=T0, getter=boom)
    check("a network failure is recorded, not raised", r7["result"] == "network_error" and "no network" in r7["error"])
    lines = (root / "raw" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    check("manifest is append-only: one line per fetch (7)", len(lines) == 7, len(lines))
    check("every manifest line is valid JSON with a UTC time", all(json.loads(x)["fetched_utc"].endswith("Z")
                                                                    for x in lines))

# Holdings backfill and revision window
recs = [{"source": "roundhill", "name": roundhill.holdings_name(d), "result": "saved"}
        for d in [date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]]
recs.append({"source": "roundhill", "name": roundhill.holdings_name(date(2026, 10, 6)), "result": "invalid"})
need = roundhill.holdings_days_needed(recs, date(2026, 10, 9))
check("backfill: missing weekdays are fetched (10/06 to 10/09)",
      {date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8), date(2026, 10, 9)} <= set(need), need)
check("revision window: the newest 3 saved days are re-checked",
      {date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)} <= set(need), need)
check("older saved days are not re-fetched", date(2026, 9, 30) not in need, need)
check("weekends are never requested", all(d.weekday() < 5 for d in need))
need_late = roundhill.holdings_days_needed(recs, date(2026, 10, 30))
check("a day that stayed absent for over a week is given up (holiday or never published)",
      date(2026, 10, 6) not in need_late)
check("holdings file name uses MMDDYYYY of the close date",
      roundhill.holdings_name(date(2026, 10, 13)) == "holdings_10132026.csv")

# Refresh rules for public series
from definedtarget.forward import external  # noqa: E402

names = {name: rule for _, name, _, _, rule in external.series(T0)}
check("current-year Treasury curve is fetched every run", names["par_yield_curve_2026.csv"] == "every_run")
check("past Treasury years are fetched once", names["par_yield_curve_2024.csv"] == "once")
check("Cboe volatility histories refresh weekly", names["VIX1Y_History.csv"] == "weekly")
seen = [{"url": "u", "result": "saved", "fetched_utc": "2026-10-10T03:00:00Z"}]
check("weekly file fetched 4 days ago is not due", not external.due(seen, "u", T0, "weekly"))
check("weekly file fetched 8 days ago is due",
      external.due([{**seen[0], "fetched_utc": "2026-10-06T03:00:00Z"}], "u", T0, "weekly"))
check("'once' file already saved is never re-fetched", not external.due(seen, "u", T0, "once"))
check("a file never fetched is always due", external.due([], "u", T0, "once"))
