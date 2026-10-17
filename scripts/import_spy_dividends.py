#!/usr/bin/env python3
"""Write data/external/spy_dividends.csv (ex_date, amount, source) from a dividend source you already have.

    py -I scripts\\import_spy_dividends.py <path to a Yahoo chart JSON for SPY>
    py -I scripts\\import_spy_dividends.py --csv <path to a CSV with columns ex_date,amount>

Public users: State Street publishes SPY's distribution history on its SPY fund page; save it as a CSV with
columns ex_date (YYYY-MM-DD) and amount, and use the --csv form. The script needs only ex-dates and amounts.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import fetch, parse  # noqa: E402


def from_yahoo(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ev = data["chart"]["result"][0]["events"]["dividends"]
    rows = []
    for v in ev.values():
        d = datetime.fromtimestamp(int(v["date"]), timezone.utc)
        # Yahoo stamps the ex-date at the US open (13:30-14:30 UTC), so the UTC date is the ex-date
        rows.append((d.date().isoformat(), float(v["amount"])))
    return sorted(rows), f"Yahoo chart JSON {Path(path).name}"


def from_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [(r["ex_date"].strip(), float(r["amount"])) for r in csv.DictReader(fh)]
    return sorted(rows), f"CSV {Path(path).name}"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    rows, src = from_csv(argv[1]) if argv[0] == "--csv" else from_yahoo(argv[0])
    out = fetch.data_dir() / "external" / "spy_dividends.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ex_date", "amount", "source"])
        for d, a in rows:
            w.writerow([d, f"{a:.6f}", f"{src}, imported {stamp}"])
    print(f"wrote {len(rows)} dividends {rows[0][0]} .. {rows[-1][0]} to {out}")
    print("last 4:", rows[-4:])
    snap = parse.snapshot_external(names=("spy_dividends.csv",))     # dated copy, so reviews see it from now on
    print("saved copy:", snap[0]["result"], snap[0].get("path") or snap[0].get("same_as"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
