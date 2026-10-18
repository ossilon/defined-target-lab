"""Cboe FLEX execution reports: which days to fetch, capture, and extracting XX's option rows.

Synthetic data only: a fake downloader and a small spreadsheet built here (no Cboe data in the repository).
"""
import io
import tempfile
from datetime import date, datetime, timezone

from _harness import check, raises

from definedtarget.forward import cboe_flex, fetch

# 1. Which days: NYSE sessions from the seed day to yesterday, minus those saved; absent ones retried for a week.
check("the report name carries the date", cboe_flex.report_name(date(2026, 10, 6))
      == "cone_options_rpt_flex_executions_late_20261006.xlsx")
need = cboe_flex.days_needed([], date(2026, 10, 7))
check("from the seed day to yesterday, sessions only", need == [date(2026, 9, 30), date(2026, 10, 1),
                                                                 date(2026, 10, 2), date(2026, 10, 5),
                                                                 date(2026, 10, 6)], need)
saved = {"source": "cboe_flex", "name": cboe_flex.report_name(date(2026, 10, 1)), "result": "saved"}
missing_long_ago = {"source": "cboe_flex", "name": cboe_flex.report_name(date(2026, 9, 30)), "result": "http_error"}
need2 = cboe_flex.days_needed([saved, missing_long_ago], date(2026, 10, 20))
check("saved days are skipped, and a day absent for over a week is given up",
      date(2026, 10, 1) not in need2 and date(2026, 9, 30) not in need2 and date(2026, 10, 2) in need2)


# 2. A tiny report in Cboe's layout, with XX's rows and look-alikes that must NOT match.
def report_bytes(rows, header=None):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Flex Executions"
    ws.append(header or cboe_flex.COLUMNS)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


EXP = datetime(2030, 1, 10)
rows = [
    [datetime(2026, 10, 6, 14, 47), 50.0, 144, "4SPY", EXP, "C", 995.01, "N"],         # XX
    [datetime(2026, 10, 6, 14, 47), 50.0, 144, "4SPY", EXP, "C", 995.01, "N"],         # XX, other side
    [datetime(2026, 10, 6, 15, 0), 1.0, 10, "4SPY", EXP, "P", 995.01, "N"],            # a put: no
    [datetime(2026, 10, 6, 15, 0), 1.0, 10, "4SPY", EXP, "C", 990.00, "N"],            # other strike: no
    [datetime(2026, 10, 6, 15, 0), 1.0, 10, "4QQQ", EXP, "C", 995.01, "N"],            # other root: no
    [datetime(2026, 10, 6, 15, 0), 1.0, 10, "4SPY", datetime(2029, 1, 10), "C", 995.01, "N"],  # other expiry: no
]


def fake_getter(body):
    return lambda url: (200, {"Content-Type": "application/vnd.openxmlformats"}, body)


def spreadsheet_checks():
    with tempfile.TemporaryDirectory() as tmp:
        now = datetime(2026, 10, 7, 4, 0, tzinfo=timezone.utc)
        recs = cboe_flex.capture_all(root=tmp, now=now, getter=fake_getter(report_bytes(rows)), sleep=lambda: None)
        check("one download per needed day, each saved as a valid spreadsheet",
              len(recs) == 5 and all(r["result"] == "saved" for r in recs), [r["result"] for r in recs])
        again = cboe_flex.capture_all(root=tmp, now=now, getter=fake_getter(report_bytes(rows)), sleep=lambda: None)
        check("a second run fetches nothing already saved", again == [])
        fills = cboe_flex.extract_fills(tmp)
        check("only XX's option rows are kept (2 per synthetic report x 5 reports)", len(fills) == 10, len(fills))
        check("each fill carries its size, price and report date",
              fills[0]["size"] == 144 and fills[0]["price"] == 50.0 and fills[0]["report_date"] == "2026-09-30")

    with tempfile.TemporaryDirectory() as tmp:
        now = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)
        page = b"<html>not found</html>"
        recs = cboe_flex.capture_all(root=tmp, now=now, getter=fake_getter(page), sleep=lambda: None)
        check("an HTML page instead of a spreadsheet is marked invalid", all(r["result"] == "invalid" for r in recs))
        check("invalid downloads give no fills", cboe_flex.extract_fills(tmp) == [])

    with tempfile.TemporaryDirectory() as tmp:
        now = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)
        odd = report_bytes(rows, header=["time", "px", "qty", "root", "exp", "cp", "k", "spread"])
        cboe_flex.capture_all(root=tmp, now=now, getter=fake_getter(odd), sleep=lambda: None)
        raises("changed report columns stop the extraction", lambda: cboe_flex.extract_fills(tmp), "unexpected columns")


# Reading the spreadsheets needs the optional extra (pip install .[flex]); without it these checks are skipped,
# not failed (external audit A12). The capture itself needs no extra.
try:
    import openpyxl  # noqa: F401
except ModuleNotFoundError:
    print("skip FLEX spreadsheet checks: openpyxl is not installed (pip install .[flex])")
else:
    spreadsheet_checks()
