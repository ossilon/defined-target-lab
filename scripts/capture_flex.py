#!/usr/bin/env python3
"""Cboe FLEX execution reports for XX.  From the repository folder:

    py -I scripts\\capture_flex.py             download any missing daily reports, then rebuild XX's fills table
    py -I scripts\\capture_flex.py --extract   rebuild the table only (no downloads)

Reports stay in the git-ignored data/raw/cboe_flex/; XX's rows go to data/tables/xx_flex_fills.csv (private).
Prints only a count per report day: contracts traded in XX's option, never prices.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import cboe_flex, fetch  # noqa: E402


def main(argv):
    root = fetch.data_dir()
    if "--extract" not in argv:
        recs = cboe_flex.capture_all(root=root)
        by_result = defaultdict(int)
        for r in recs:
            by_result[r["result"]] += 1
        print("downloads:", dict(by_result) or "none needed")
    try:
        fills = cboe_flex.extract_fills(root)
    except ValueError as err:
        print(f"STOP: {err}")
        return 1
    out = root / "tables" / "xx_flex_fills.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["report_date", "transact_time", "size", "price", "spread", "file"])
        w.writeheader()
        w.writerows(fills)
    per_day = defaultdict(list)
    for f in fills:
        per_day[f["report_date"]].append(f["size"])
    print(f"XX's option in {len(per_day)} report day(s):")
    for day, sizes in sorted(per_day.items()):
        print(f"  {day}: {len(sizes)} row(s), sizes {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
