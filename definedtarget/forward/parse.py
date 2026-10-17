"""Turn the saved raw files into tidy tables. Pure reading: nothing here downloads or edits raw files.

The code rebuilds every table from the raw files the same way every time, so you can simply re-run a parsing fix.
Numbers keep their raw text too (e.g. "49.920000"), because the number of decimals in an option mark is
evidence: a 2-decimal mark looks like a traded price, a 4-decimal mark like a pricing vendor's model value.
"""
import csv
import io
import re
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import fetch

# OCC-style option identifier as Roundhill writes it, e.g. "4SPY  300110C00995010":
# root (4SPY), expiry yymmdd (300110 = 2030-01-10), C/P, strike x 1000 (00995010 = 995.010).
OCC = re.compile(r"^(?P<root>\S+)\s+(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

HOLDINGS_COLUMNS = ["Date", "Account", "StockTicker", "CUSIP", "SecurityName", "Shares", "Price", "MarketValue",
                    "Weightings", "NetAssets", "SharesOutstanding", "CreationUnits", "MoneyMarketFlag"]


def decimals(text):
    """Decimals that carry information: '49.920000' -> 2, '49.677800' -> 4, '57.27' -> 2, '288' -> 0."""
    text = text.strip()
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1].rstrip("0"))


def parse_occ(ticker):
    """Split an OCC-style option identifier; returns None for anything that is not an option."""
    m = OCC.match(ticker.strip())
    if not m:
        return None
    root = m["root"]
    return {"root": root,
            "underlying": root.lstrip("0123456789"),           # FLEX roots carry a leading digit (4SPY -> SPY)
            "flex_prefix": root[: len(root) - len(root.lstrip("0123456789"))] or None,
            "expiry": datetime.strptime(m["yymmdd"], "%y%m%d").date(),
            "cp": m["cp"],
            "strike": int(m["strike"]) / 1000.0}


def _num(text):
    text = text.strip().replace(",", "")
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text) if text else float("nan")


_CAPTURED_BY = [None]          # a cutoff set by captured_by(); None = every saved version


@contextmanager
def captured_by(cutoff_utc):
    """Inside this block, every loader sees only files downloaded at or before `cutoff_utc` ('YYYY-MM-DDTHH:MM:SSZ').

    Protocol P2: "each review uses the data captured by its review date". Every issuer file, Treasury curve and Cboe
    history reaches the code through saved_versions(), so one cutoff here applies to all of them.
    """
    old = _CAPTURED_BY[0]
    _CAPTURED_BY[0] = cutoff_utc
    try:
        yield
    finally:
        _CAPTURED_BY[0] = old


def saved_versions(root, prefix):
    """Manifest records of saved (valid) files whose short name starts with prefix, oldest first.

    Sorted by download time, not by position in the manifest: captures imported from the cloud can be appended
    after newer local ones, and "latest" must still mean the most recent download. Inside captured_by(), files
    downloaded after the cutoff are left out.
    """
    cutoff = _CAPTURED_BY[0]
    recs = [r for r in fetch.read_manifest(root) if r["name"].startswith(prefix) and r["result"] == "saved"
            and (cutoff is None or r["fetched_utc"] <= cutoff)]
    return sorted(recs, key=lambda r: r["fetched_utc"])


def read_text(root, rec):
    return (fetch.data_dir(root) / rec["path"]).read_bytes().decode("utf-8-sig")


# Hand-maintained inputs in data/external (closes you add, SPY dividends, void-event notices). They are edited in
# place, so each capture run also saves a dated copy of their bytes (raw/external/<UTC date>/, manifest record,
# never overwritten). A review then reads the copy that existed by its cutoff, like every downloaded file (P2).
EXTERNAL_SOURCE = "external"
EXTERNAL_FILES = ("spy_dividends.csv", "closes.csv", "tv_closes.csv", "void_events.csv")


def snapshot_external(root=None, now=None, names=EXTERNAL_FILES):
    """Save the current bytes of each hand-maintained file that exists. An unchanged file adds an 'unchanged' line."""
    root = fetch.data_dir(root)
    out = []
    for name in names:
        path = root / "external" / name
        if path.exists():
            body = path.read_bytes()
            out.append(fetch.capture(f"local:external/{name}", name, EXTERNAL_SOURCE, None, root=root, now=now,
                                     getter=lambda url, b=body: (200, {}, b)))
    return out


def external_text(root, name):
    """Text of a hand-maintained file, or None if it does not exist. Outside captured_by(): the working file in
    data/external. Inside: the latest saved copy captured by the cutoff (a later edit never reaches an earlier review)."""
    if _CAPTURED_BY[0] is None:
        path = fetch.data_dir(root) / "external" / name
        return path.read_bytes().decode("utf-8-sig") if path.exists() else None
    recs = [r for r in saved_versions(root, name) if r["name"] == name and r.get("source") == EXTERNAL_SOURCE]
    return read_text(root, recs[-1]) if recs else None


def holdings_rows(text, as_of, version, accounts=None):
    """Rows of one holdings file as dicts (all accounts, or only those listed)."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    if header[: len(HOLDINGS_COLUMNS)] != HOLDINGS_COLUMNS:
        raise ValueError(f"holdings schema changed: {header}")
    out = []
    for row in reader:
        if not row or len(row) < len(HOLDINGS_COLUMNS):
            continue
        r = dict(zip(HOLDINGS_COLUMNS, row))
        if accounts and r["Account"] not in accounts:
            continue
        occ = parse_occ(r["StockTicker"])
        out.append({
            "as_of": as_of, "version": version,
            "effective_date": datetime.strptime(r["Date"], "%m/%d/%Y").date(),
            "account": r["Account"], "ticker": r["StockTicker"].strip(), "cusip": r["CUSIP"].strip(),
            "name": r["SecurityName"].strip(),
            "shares": _num(r["Shares"]), "price": _num(r["Price"]), "price_text": r["Price"].strip(),
            "price_decimals": decimals(r["Price"]), "market_value": _num(r["MarketValue"]),
            "weight": _num(r["Weightings"]), "net_assets": _num(r["NetAssets"]),
            "shares_outstanding": _num(r["SharesOutstanding"]), "creation_units": _num(r["CreationUnits"]),
            "money_market": r["MoneyMarketFlag"].strip() == "Y",
            "is_option": occ is not None,
            **({f"opt_{k}": v for k, v in occ.items()} if occ else {}),
        })
    return out


def load_holdings(root=None, accounts=("XX",), all_versions=True):
    """Tidy holdings table from every saved holdings file. as_of = the close date in the file NAME."""
    rows = []
    recs = saved_versions(root, "holdings_")
    latest = {}
    for rec in recs:
        latest[rec["name"]] = rec["fetched_utc"]
    for rec in recs:
        if not all_versions and latest[rec["name"]] != rec["fetched_utc"]:
            continue
        as_of = datetime.strptime(rec["name"][9:17], "%m%d%Y").date()
        rows += holdings_rows(read_text(root, rec), as_of, rec["fetched_utc"], set(accounts) if accounts else None)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["latest_version"] = df.apply(lambda r: latest[f"holdings_{r.as_of.strftime('%m%d%Y')}.csv"] == r.version,
                                        axis=1)
    return df


def load_fund_daily(root=None, fund="XX"):
    """The issuer's NAV / market-price history (latest saved copy of <fund>_daily.csv)."""
    recs = saved_versions(root, f"{fund.lower()}_daily")
    if not recs:
        raise FileNotFoundError(f"no saved {fund.lower()}_daily.csv yet - run scripts/capture.py")
    df = pd.read_csv(io.StringIO(read_text(root, recs[-1])), dtype=str)
    df = df[df["Fund Ticker"].str.strip() == fund]
    out = pd.DataFrame({
        "as_of": pd.to_datetime(df["Rate Date"], format="%m/%d/%Y").dt.date,
        "nav": df["NAV"].astype(float), "market_price": df["Market Price"].astype(float)})
    out["premium"] = out["market_price"] / out["nav"] - 1.0
    return out.reset_index(drop=True)


def load_snapshots(root=None, name="dailynav.csv", fund="XX"):
    """Every saved copy of a latest-day-only file, as one table (one row per copy for the fund)."""
    rows = []
    for rec in saved_versions(root, name):
        df = pd.read_csv(io.StringIO(read_text(root, rec)), dtype=str)
        if "Fund Ticker" in df.columns and name == "dailynav.csv":
            df = df[df["Fund Ticker"].str.strip() == fund]
        df["fetched_utc"] = rec["fetched_utc"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
