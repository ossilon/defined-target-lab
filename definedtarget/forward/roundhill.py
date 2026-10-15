"""Roundhill's public data files: which ones we capture, and the daily capture routine.

We found these on 2026-10-14 from the network calls of https://www.roundhillinvestments.com/etf/xx/ (see FUND_FACTS.md).
Roundhill's site terms allow downloading and using its materials for informational purposes; republishing or
redistributing them needs written permission. So these files stay in the git-ignored data folder.

Three files only ever show the LATEST day (Roundhill overwrites them after each US close), so a missed day never
comes back: the daily NAV snapshot, the instrument-pricing file (Roundhill's own SPY and SPX closes) and the
bid/ask endpoint. Roundhill names holdings files by date and older ones stay downloadable, so the capture can
backfill them.
"""
from datetime import date, datetime, timedelta, timezone

from . import fetch

SOURCE = "roundhill"
BASE = "https://www.roundhillinvestments.com"
DATA = BASE + "/assets/data/filepointroundhill.40ru.ru_"

# (short name, url, header the body must start with). Latest-day-only files first: they are the ones we
# cannot get back later.
DAILY_FILES = [
    ("dailynav.csv", DATA + "dailynav.csv", b"Fund Name,Fund Ticker,CUSIP,Net Assets"),
    ("daily_instrument_pricing.csv", DATA + "daily_instrument_pricing.csv", b"Rate Date,Fund Name,Fund Ticker"),
    # JSON from Solactive; "level" is the 30-day median bid/ask spread in % (1.1821 on 10/14 = the site's 1.18%)
    ("bidask_xx.json", BASE + "/assets/php/solactiveperformancebidaskxx.php", b"{"),
    ("xx_daily.csv", DATA + "xx_daily.csv", b"Fund Name,Fund Ticker,CUSIP,NAV,Market Price"),
]

# Fund documents: the capture fetches them at most once a week, to notice supplements or a new target period.
DOC_FILES = [
    ("XX_Factsheet.pdf", BASE + "/assets/pdfs/XX_Factsheet.pdf", b"%PDF"),
    ("XX_Prospectus.pdf", BASE + "/assets/pdfs/XX_Prospectus.pdf", b"%PDF"),
    ("XX_Summary_Prospectus.pdf", BASE + "/assets/pdfs/XX_Summary_Prospectus.pdf", b"%PDF"),
    ("XX_SAI.pdf", BASE + "/assets/pdfs/XX_SAI.pdf", b"%PDF"),
]

HOLDINGS_HEADER = b"Date,Account,StockTicker,CUSIP,SecurityName,Shares,Price,MarketValue"
FIRST_HOLDINGS_DAY = date(2026, 9, 30)      # XX's seed day (first close with XX holdings)
DOC_REFRESH_DAYS = 7


def holdings_name(day):
    """Holdings file for the close of `day` (the file's own 'Date' column is the NEXT trading day)."""
    return f"holdings_{day.strftime('%m%d%Y')}.csv"


def holdings_url(day):
    return DATA + holdings_name(day)


def weekdays(start, end):
    """Every Monday-Friday from start to end inclusive (holidays included: those files are simply absent)."""
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


REVISION_WINDOW = 3     # re-fetch the newest N saved holdings days: Roundhill revised 10/05, 10/06 and 10/12 a day later


def holdings_days_needed(records, today, first=FIRST_HOLDINGS_DAY, give_up_after_days=7,
                         revision_window=REVISION_WINDOW):
    """Weekdays whose holdings file we should fetch this run.

    - Days never saved, unless they came back absent/invalid more than `give_up_after_days` ago (market holidays
      and future dates are absent; the capture retries recent ones).
    - Plus the newest `revision_window` saved days, because the issuer sometimes rewrites a file the next day;
      the capture just logs an unchanged file as "unchanged" and saves a revised one as a new version.
    """
    done, gave_up = set(), set()
    for rec in records:
        if rec.get("source") != SOURCE or not rec["name"].startswith("holdings_"):
            continue
        day = datetime.strptime(rec["name"][9:17], "%m%d%Y").date()
        if rec["result"] in ("saved", "unchanged"):
            done.add(day)
        elif (today - day).days > give_up_after_days:
            gave_up.add(day)
    recheck = set(sorted(done)[-revision_window:]) if revision_window else set()
    return [d for d in weekdays(first, today) if (d not in done and d not in gave_up) or d in recheck]


def docs_due(records, now):
    """True if no capture has fetched the fund documents (saved or unchanged) in the last week."""
    times = [rec["fetched_utc"] for rec in records
             if rec.get("source") == SOURCE and rec["name"].endswith(".pdf")
             and rec["result"] in ("saved", "unchanged")]
    if not times:
        return True
    last = datetime.strptime(max(times), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (now - last).days >= DOC_REFRESH_DAYS


def capture_all(root=None, now=None, getter=fetch.http_get, docs=None, sleep=fetch.pause):
    """Run one capture: latest-day files, any missing holdings days since launch, and (weekly) the documents.

    Returns the list of manifest records this run wrote.
    """
    root = fetch.data_dir(root)
    now = now or datetime.now(timezone.utc)
    out = []

    def grab(name, url, prefix):
        records = fetch.read_manifest(root)
        rec = fetch.capture(url, name, SOURCE, prefix, root=root, now=now, getter=getter, records=records)
        out.append(rec)
        sleep()

    for name, url, prefix in DAILY_FILES:
        grab(name, url, prefix)
    for day in holdings_days_needed(fetch.read_manifest(root), now.date()):
        grab(holdings_name(day), holdings_url(day), HOLDINGS_HEADER)
    if docs or (docs is None and docs_due(fetch.read_manifest(root), now)):
        for name, url, prefix in DOC_FILES:
            grab(name, url, prefix)
    return out
