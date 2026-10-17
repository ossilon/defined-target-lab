#!/usr/bin/env python3
"""Capture today's Roundhill files (and backfill missing holdings days).  From target_funds/:

    py -I scripts\\capture.py            (add --docs to force a re-fetch of the fund documents)

Runs with -I (isolated mode) because it handles downloaded files; this script adds only the repository folder
to sys.path, never the data folder.
"""
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import external, parse, roundhill  # noqa: E402


def main(argv):
    records = roundhill.capture_all(docs=True if "--docs" in argv else None)
    records += external.capture_all(force="--external" in argv)     # weekly unless --external
    records += parse.snapshot_external()        # dated copies of closes / dividends / void notices (P2: by the cutoff)
    for rec in records:
        print(f"{rec['result']:<13} {rec.get('status', '-')!s:<4} {rec['name']:<34} "
              f"last-modified={rec.get('last_modified')}  {rec.get('path') or rec.get('same_as') or rec.get('error', '')}")
    counts = Counter(rec["result"] for rec in records)
    print("\nsummary:", dict(counts))
    failed = [r for r in records if r["result"] in ("network_error", "invalid", "invalid_repeat", "http_error")]
    stop = [r for r in failed if r["source"] == roundhill.SOURCE and not r["name"].startswith("holdings_")]
    warn = [r for r in failed if r not in stop and not r["name"].startswith("holdings_")]
    if warn:
        print("WARNING: these public series did not refresh (the next run retries):", ", ".join(r["name"] for r in warn))
    if stop:
        print("STOP: a latest-day Roundhill file failed (no later run can recover it):",
              ", ".join(r["name"] for r in stop))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
