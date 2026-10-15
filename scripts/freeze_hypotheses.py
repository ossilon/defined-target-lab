#!/usr/bin/env python3
"""Freeze the approved hypotheses into the hash-chained ledger and render forward/HYPOTHESES.md.

    py -I scripts\\freeze_hypotheses.py            (shows what it would freeze; changes nothing)
    py -I scripts\\freeze_hypotheses.py --freeze   (appends every entry not already in the ledger)

Run --freeze only after you approve the wording in forward/hypotheses_draft.json. Once you push them, never edit
frozen entries: add results later as verdict entries.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget import ledger  # noqa: E402

DRAFT = REPO / "forward" / "hypotheses_draft.json"


def main(argv):
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    have = {e["id"] for e in ledger.read() if e["kind"] != "verdict"}
    todo = [e for e in draft if e["id"] not in have]
    for e in todo:
        label = e.get("claim") or f"{len(e.get('rules', []))} rules"
        print(f"{'FREEZE' if '--freeze' in argv else 'would freeze'} {e['kind']:<10} {e['id']}: {label[:90]}")
    if "--freeze" in argv:
        for e in todo:
            ledger.append(e)
        (REPO / "forward" / "HYPOTHESES.md").write_text(ledger.render(), encoding="utf-8")
        print(ledger.verify())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
