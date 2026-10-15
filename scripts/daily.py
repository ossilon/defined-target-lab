#!/usr/bin/env python3
"""The daily forward-test routine, which the scheduled task runs.  From the repository folder:

    py -I scripts\\daily.py

See definedtarget/forward/routine.py for what it does. Prints the daily note and any "STOP: ..." lines;
exit code 1 if there is a STOP. Runs with -I (isolated mode) because it reads downloaded files.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import routine  # noqa: E402


def main():
    note, stops = routine.run()
    print("\n".join(note))
    for s in stops:
        print("STOP:", s)
    return 1 if stops else 0


if __name__ == "__main__":
    sys.exit(main())
