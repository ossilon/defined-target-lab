#!/usr/bin/env python3
"""File one TradingView reply for the listed S&P 500 Dec-2029 calls (BACKLOG B2).  From the repository folder:

    py -I scripts\\capture_listed.py --symbols              print the symbols to request, as JSON
    py -I scripts\\capture_listed.py <reply.json>           file a saved reply, read now
    py -I scripts\\capture_listed.py <reply.json> --read-utc 2026-10-16T22:15:00Z
    py -I scripts\\capture_listed.py --compare 2026-10-16   inside vs outside P2's window, for one close

The scheduled task calls `--symbols`, asks the TradingView connection for those symbols, saves the reply unchanged,
then calls this script with the file. Prints the close date, whether the read fell inside protocol P2's window, and
how many calls had quotes. Exit code 1 if the reply is refused or no call had a quote.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import listed  # noqa: E402


def main(argv):
    if argv[:1] == ["--symbols"]:
        print(json.dumps(listed.symbols()))
        return 0
    if argv[:1] == ["--compare"]:
        n, same, diffs = listed.compare(argv[1])
        if not n:
            print(f"{argv[1]}: needs one reading inside and one outside the window")
            return 1
        print(f"{argv[1]}: {same} of {n} strikes identical outside the window")
        for k, a, b in diffs:
            print(f"  strike {k}: inside {a}, outside {b}")
        return 0
    read_utc = datetime.now(timezone.utc)
    if "--read-utc" in argv:
        read_utc = datetime.fromisoformat(argv[argv.index("--read-utc") + 1].replace("Z", "+00:00"))
    try:
        s = listed.store(Path(argv[0]).read_text(encoding="utf-8"), read_utc)
    except (ValueError, KeyError) as err:
        print(f"REFUSED: {err}")
        return 1
    print(f"close {s['close']}; {'INSIDE' if s['in_window'] else 'OUTSIDE (kept, not counted)'} P2's window; "
          f"{s['quoted']} of {len(listed.STRIKES)} calls quoted"
          + (f"; missing: {', '.join(s['missing'])}" if s["missing"] else "")
          + ("" if s["new"] else "; already filed (no change)"))
    return 0 if s["quoted"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
