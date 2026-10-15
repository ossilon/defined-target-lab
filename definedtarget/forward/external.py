"""Public market series the engine needs, fetched the same way as the issuer files (raw bytes + manifest).

- Cboe volatility indices (free daily history CSVs on cboe.com): VIX (30 days), VIX3M, VIX6M, VIX1Y.
  Cboe's terms apply; like everything in data/, the project never commits these files. Full histories, so a weekly
  refresh is enough (the newer copy supersedes the older one; both stay on disk).
- US Treasury daily par yield curve (home.treasury.gov, public domain; the primary source of FRED's DGS series).
  One CSV per calendar year. The current year changes every day, so the capture fetches it on every run and
  fetches earlier years once. FRED's own CSV endpoint refused scripted downloads on 2026-10-14 (timeouts /
  connection resets, as with an earlier SOFR attempt), so the code does not use it.
"""
from datetime import datetime, timezone

from . import fetch

CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
TREASURY = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
            "{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
TREASURY_FIRST_YEAR = 2024        # enough for v0.1 pricing and the 2024+ listed-option calibration window
CBOE_SERIES = ("VIX", "VIX3M", "VIX6M", "VIX1Y")
CBOE_REFRESH_DAYS = 7


def series(now):
    """(source, short name, url, expected header start, refresh rule) for every public file we keep."""
    out = [("cboe", f"{s}_History.csv", CBOE.format(s), b"DATE,", "weekly") for s in CBOE_SERIES]
    for y in range(TREASURY_FIRST_YEAR, now.year + 1):
        rule = "every_run" if y == now.year else "once"
        out.append(("treasury", f"par_yield_curve_{y}.csv", TREASURY.format(y=y), b'Date,"1 Mo"', rule))
    return out


def due(records, url, now, rule):
    times = [r["fetched_utc"] for r in records if r["url"] == url and r["result"] in ("saved", "unchanged")]
    if rule == "every_run" or not times:
        return True
    if rule == "once":
        return False
    last = datetime.strptime(max(times), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (now - last).days >= CBOE_REFRESH_DAYS


def capture_all(root=None, now=None, force=False, getter=fetch.http_get, sleep=fetch.pause):
    root = fetch.data_dir(root)
    now = now or datetime.now(timezone.utc)
    out = []
    for source, name, url, prefix, rule in series(now):
        records = fetch.read_manifest(root)
        if force or due(records, url, now, rule):
            out.append(fetch.capture(url, name, source, prefix, root=root, now=now, getter=getter, records=records))
            sleep()
    return out
