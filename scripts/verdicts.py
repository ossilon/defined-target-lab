#!/usr/bin/env python3
"""Verdicts for the pre-registered hypotheses, computed from the frozen rules (P1, P2).  From the repository folder:

    py -I scripts\\verdicts.py                      every hypothesis's status today (provisional before its review)
    py -I scripts\\verdicts.py --as-of 2027-01-20   the statuses as they stand on a given date
    py -I scripts\\verdicts.py --record H5          append H5's verdict to the ledger (only if its review is done)
    py -I scripts\\verdicts.py --record due         append every completed verdict not yet in the ledger

The ledger (forward/predictions.jsonl) is append-only and hash-chained: a recorded verdict stays forever, so
--record asks for nothing but refuses anything not final, any --as-of date after today, and any verdict whose
deciding day has not ended yet (UTC; a later capture could still change it). After recording, re-render
forward/HYPOTHESES.md and push. Each review reads only the files downloaded by its review date (P2). Prints
aggregates only.
"""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget import __version__, ledger, verdicts  # noqa: E402


def cached(holdings):
    memo = {}

    def provider(cutoff):
        if cutoff not in memo:
            memo[cutoff] = verdicts.load(cutoff=cutoff, holdings=holdings)
        return memo[cutoff]
    return provider


def main(argv, ledger_path=ledger.DEFAULT_PATH, now=None, results_fn=None):
    """`ledger_path`, `now` and `results_fn` exist for the tests; the command line uses the defaults."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    record = None
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--as-of" and args:
            today = date.fromisoformat(args.pop(0))
        elif a == "--record" and args:
            record = args.pop(0)
        else:
            print(__doc__)
            return 2
    if record is not None and today > now.date():
        print(f"STOP: --record with --as-of {today}, after today ({now.date()}): a verdict is recorded only from data "
              "that exists; nothing recorded")
        return 1
    results = (results_fn or (lambda d: verdicts.all_results(d, cached("latest"), cached("first"))))(today)
    print(f"Verdicts as of {today} (definedtarget {__version__}); provisional numbers are not verdicts\n")
    for r in results:
        print(r.line())
    entries = ledger.read(ledger_path)
    due, waiting = verdicts.to_record(results, entries, now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    if due:
        print("\nready to record: " + ", ".join(f"{r.id}{'/' + cp if cp else ''} {w}" for r, cp, w in due))
    for r, cp, w, decided in waiting:
        when = f"after {decided} ends (UTC)" if decided else "(no deciding date: not recordable)"
        print(f"waiting: {r.id}{'/' + cp if cp else ''} {w}, record {when}")
    if record is None:
        return 0
    wanted = [d for d in due if record == "due" or d[0].id == record]
    if not wanted:
        print(f"\nnothing to record for '{record}' (not final yet, its deciding day has not ended, or already in the "
              "ledger)")
        return 1
    ok, msg = ledger.verify(ledger_path)
    if not ok:
        print(f"STOP: ledger check failed ({msg}); nothing recorded")
        return 1
    for r, cp, word in wanted:
        evidence = {"review": r.review, "n": r.n, "stats": r.stats, "window": r.window, "note": r.note,
                    "computed_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "code": f"definedtarget {__version__}, scripts/verdicts.py"}
        decided = verdicts.H7A_CHECKPOINTS[int(cp)] if r.id == "H7a" and cp not in (None, "overall") else r.decided
        evidence["data_cutoff"] = verdicts.cutoff_for(decided)
        if cp:
            evidence["checkpoint"] = cp
        ledger.append({"kind": "verdict", "id": r.id, "result": word, "evidence": evidence}, path=ledger_path)
        print(f"recorded {r.id}{'/' + cp if cp else ''}: {word}")
    print("Now re-render forward/HYPOTHESES.md (py -I scripts\\freeze_hypotheses.py) and push.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
