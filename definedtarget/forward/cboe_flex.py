"""Cboe's daily FLEX execution reports: the fund's own option trades (backlog B21; the entry cost behind H1 and H9).

Cboe publishes one spreadsheet per trading day at
https://www.cboe.com/us/options/market_statistics/flex_trade_reports/ with every FLEX execution on its exchange:
time, price, size, OCC root, expiry, call/put, strike. XX's creations appear there (2026-10-06: 144 contracts;
2026-10-12: 96; matching the holdings files).

Use: Cboe's site terms allow downloading one copy for personal, non-commercial use. This project keeps the reports
private (git-ignored data/ folder and a private capture repository), extracts only XX's rows, and publishes derived
statistics only, never the raw rows.

- `capture_all()` downloads each trading day's report once (standard library only, so a cloud routine can run it).
- `extract_fills()` reads the saved reports and keeps XX's option only (needs openpyxl, installed locally).
"""
from datetime import date, datetime, timedelta, timezone

from .. import calendar
from . import fetch

SOURCE = "cboe_flex"
BASE = "https://cdn.cboe.com/resources/us/options/market-statistics/flex-trade-reports/cone/"
XLSX_PREFIX = b"PK\x03\x04"                     # an .xlsx file is a zip archive
FIRST_DAY = date(2026, 9, 30)                   # XX's seed day
COLUMNS = ["transact_time", "price", "size", "osi_root", "expire_date", "call_put_flag", "strike_price", "spread"]
XX_OPTION = {"osi_root": "4SPY", "call_put_flag": "C", "expire_date": date(2030, 1, 10), "strike_price": 995.01}


def report_name(day):
    return f"cone_options_rpt_flex_executions_late_{day.strftime('%Y%m%d')}.xlsx"


def report_url(day):
    return BASE + report_name(day)


def days_needed(records, today, first=FIRST_DAY, give_up_after_days=7):
    """NYSE sessions before `today` whose report has not been saved yet (absent ones are retried for a week)."""
    done, gave_up = set(), set()
    for rec in records:
        if rec.get("source") != SOURCE:
            continue
        day = datetime.strptime(rec["name"][-13:-5], "%Y%m%d").date()
        if rec["result"] in ("saved", "unchanged"):
            done.add(day)
        elif (today - day).days > give_up_after_days:
            gave_up.add(day)
    end = today - timedelta(days=1)
    if end < first:
        return []
    return [d for d in calendar.sessions(first, end) if d not in done and d not in gave_up]


def capture_all(root=None, now=None, getter=fetch.http_get, sleep=fetch.pause):
    """Download every missing report since the seed day. Returns the manifest records this run wrote."""
    root = fetch.data_dir(root)
    now = now or datetime.now(timezone.utc)
    out = []
    for day in days_needed(fetch.read_manifest(root), now.date()):
        records = fetch.read_manifest(root)
        out.append(fetch.capture(report_url(day), report_name(day), SOURCE, XLSX_PREFIX, root=root, now=now,
                                 getter=getter, records=records))
        sleep()
    return out


def _is_xx(row):
    exp = row["expire_date"]
    exp = exp.date() if isinstance(exp, datetime) else exp
    return (str(row["osi_root"]).strip() == XX_OPTION["osi_root"]
            and str(row["call_put_flag"]).strip() == XX_OPTION["call_put_flag"]
            and exp == XX_OPTION["expire_date"]
            and abs(float(row["strike_price"]) - XX_OPTION["strike_price"]) < 1e-6)


def extract_fills(root=None):
    """XX's option trades from every saved report: a list of dicts (report_date, transact_time, size, price,
    spread, file). Raises ValueError if a report's columns change (the parser would otherwise misread it)."""
    import openpyxl                                         # local only; the cloud capture does not need it

    root = fetch.data_dir(root)
    latest = {}
    for rec in fetch.read_manifest(root):
        if rec.get("source") == SOURCE and rec["result"] == "saved":
            latest[rec["name"]] = rec["path"]                # the newest saved copy of each report
    fills = []
    for name, path in sorted(latest.items()):
        wb = openpyxl.load_workbook(root / path, read_only=True, data_only=True)
        try:
            rows = list(wb.worksheets[0].iter_rows(values_only=True))
        finally:
            wb.close()                                      # read-only mode keeps the file open (locked on Windows)
        header = [str(c).strip() if c is not None else "" for c in rows[0]] if rows else []
        if header[:len(COLUMNS)] != COLUMNS:
            raise ValueError(f"{name}: unexpected columns {header} (expected {COLUMNS})")
        report_day = datetime.strptime(name[-13:-5], "%Y%m%d").date()
        for values in rows[1:]:
            row = dict(zip(COLUMNS, values))
            if row["osi_root"] is not None and _is_xx(row):
                fills.append({"report_date": report_day.isoformat(), "transact_time": str(row["transact_time"]),
                              "size": int(row["size"]), "price": float(row["price"]), "spread": row["spread"],
                              "file": name})
    return fills
