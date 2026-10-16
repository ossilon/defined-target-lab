#!/usr/bin/env python3
"""Import the cloud routines' captures into the local data folder.  From the repository folder:

    py -I scripts\\import_captures.py [path to the local clone of the private capture repository]

Default path: the folder `research-captures` next to this repository. Update the clone first (GitHub Desktop:
Fetch, then Pull). Safe to run any number of times. Prints counts only.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import cloud_import  # noqa: E402


def main(argv):
    clone = Path(argv[0]) if argv else REPO.parent / "research-captures"
    if not (clone / ".git").exists():
        print(f"STOP: no clone of the capture repository at {clone}")
        return 1
    files = cloud_import.import_files(clone)
    quotes = cloud_import.import_quotes(clone)
    print(f"files: {files['imported']} imported, {files['already']} already here, {files['refused']} refused")
    for q in quotes:
        if "refused" in q:
            print(f"quotes {q['file']}: REFUSED ({q['refused']})")
        else:
            print(f"quotes close {q['close']}: {'INSIDE' if q['in_window'] else 'outside'} P2's window, "
                  f"{q['quoted']} of 15 calls quoted")
    if not quotes:
        print("quotes: nothing new")
    return 1 if files["refused"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
