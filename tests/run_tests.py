#!/usr/bin/env python3
"""Run every test. From target_funds/:   py tests\\run_tests.py         (must end with "0 failure(s)")

Optional: name parts to run a subset, e.g.  py tests\\run_tests.py kernel   (runs test_kernel.py only).

What it does
1. Imports each tests/test_*.py in name order. Their test code runs on import and records results through
   tests/_harness.py (see its docstring for the contract).
2. Imports each tests/*_local.py (local cross-checks against private sibling folders; git-ignored, so absent
   in a public copy). If there are none it prints "local cross-check: skipped (file absent)".
3. Prints "N failure(s)" and exits with code 1 if anything failed (so scripts and CI can stop on it).
"""
import importlib
import sys
import time
import traceback
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _harness  # noqa: E402  (also puts the repository root on sys.path)


def run_module(path):
    """Import one test module; a crash during import counts as one failure, and the run carries on."""
    print(f"\n===== {path.name} =====")
    t0 = time.perf_counter()
    try:
        importlib.import_module(path.stem)
    except Exception:  # noqa: BLE001
        print(traceback.format_exc())
        _harness.check(f"{path.name} ran to the end without crashing", False, "see traceback above")
    print(f"----- {path.name}: {time.perf_counter() - t0:.2f} s")


def main(selectors):
    def wanted(p):
        return not selectors or any(s in p.stem for s in selectors)

    t0 = time.perf_counter()
    for path in sorted(TESTS.glob("test_*.py")):
        if wanted(path):
            run_module(path)

    local = [p for p in sorted(TESTS.glob("*_local.py")) if wanted(p)]
    if local:
        for path in local:
            run_module(path)
    elif not selectors:
        print("\nlocal cross-check: skipped (file absent)")

    n = len(_harness.failures)
    if n:
        print("\nFailed:\n  " + "\n  ".join(_harness.failures))
    print(f"\ntotal time {time.perf_counter() - t0:.1f} s")
    print(f"\n{n} failure(s)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

