#!/usr/bin/env python3
"""Verdicts for the pre-registered hypotheses, computed from the frozen rules (P1, P2).  From the repository folder:

    py -I scripts\\verdicts.py                      every hypothesis's status today (provisional before its review)
    py -I scripts\\verdicts.py --as-of 2027-01-20   the statuses as they stand on a given date
    py -I scripts\\verdicts.py --record H5          append H5's verdict to the ledger (only if its review is done)
    py -I scripts\\verdicts.py --record due         append every completed verdict not yet in the ledger

The ledger (forward/predictions.jsonl) is append-only and hash-chained: a recorded verdict stays forever, so
--record asks for nothing but refuses anything not final. After recording, re-render forward/HYPOTHESES.md and push.
Each review reads only the files downloaded by its review date (P2). Prints aggregates only.
"""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget import __version__, ledger, verdicts  # noqa: E402

FINAL = {"pass", "fail", "inconclusive", "void"}


def cached(holdings):
    memo = {}

    def provider(cutoff):
        if cutoff not in memo:
            memo[cutoff] = verdicts.load(cutoff=cutoff, holdings=holdings)
        return memo[cutoff]
    return provider


def recorded(entries):
    """{(id, checkpoint or None)} already in the ledger."""
    return {(e["id"], (e.get("evidence") or {}).get("checkpoint")) for e in entries if e.get("kind") == "verdict"}


def to_record(results, entries):
    done, out = recorded(entries), []
    for r in results:
        if r.id == "H7a":
            for year, word in r.stats.items():
                if word in ("pass", "fail") and (r.id, year) not in done:
                    out.append((r, year, word))
            if r.status in FINAL and (r.id, "overall") not in done:
                out.append((r, "overall", r.status))
            continue
        if r.status in FINAL and (r.id, None) not in done:
            out.append((r, None, r.status))
    return out


def main(argv):
    today = datetime.now(timezone.utc).date()
    record = None
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--as-of":
            today = date.fromisoformat(args.pop(0))
        elif a == "--record":
            record = args.pop(0)
        else:
            print(__doc__)
            return 2
    results = verdicts.all_results(today, cached("latest"), cached("first"))
    print(f"Verdicts as of {today} (definedtarget {__version__}); provisional numbers are not verdicts\n")
    for r in results:
        print(r.line())
    entries = ledger.read()
    due = to_record(results, entries)
    if due:
        print("\nready to record: " + ", ".join(f"{r.id}{'/' + cp if cp else ''} {w}" for r, cp, w in due))
    if record is None:
        return 0
    wanted = [d for d in due if record == "due" or d[0].id == record]
    if not wanted:
        print(f"\nnothing to record for '{record}' (not final yet, or already in the ledger)")
        return 1
    ok, msg = ledger.verify()
    if not ok:
        print(f"STOP: ledger check failed ({msg}); nothing recorded")
        return 1
    for r, cp, word in wanted:
        evidence = {"review": r.review, "n": r.n, "stats": r.stats, "window": r.window, "note": r.note,
                    "computed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "code": f"definedtarget {__version__}, scripts/verdicts.py"}
        if cp:
            evidence["checkpoint"] = cp
        ledger.append({"kind": "verdict", "id": r.id, "result": word, "evidence": evidence})
        print(f"recorded {r.id}{'/' + cp if cp else ''}: {word}")
    print("Now re-render forward/HYPOTHESES.md (py -I scripts\\freeze_hypotheses.py) and push.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
