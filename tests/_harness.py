"""Shared test helpers. No pytest needed.

CONTRACT for every tests/test_*.py module (and tests/*_local.py cross-checks):

    from _harness import check, raises

    check("what must be true, in plain words", condition, detail_shown_only_on_failure)
    raises("what must be refused", lambda: some_call(...), "text the error message must contain")

- Test code sits at module level and runs when the runner imports the module.
- check() prints "ok   <name>" or "FAIL <name>  (<detail>)" and records failures in `failures`.
- raises() passes only if the call raises AND the message contains `must_contain` (case-insensitive), so a test
  cannot pass on the wrong error.
- tests/run_tests.py imports every test module in name order, then counts `failures` once at the end. Do not call
  sys.exit() in a test module. A module that crashes on import counts as one failure and the runner carries on.
- Importing this module also puts the repository root on sys.path, so `import definedtarget` works from a test
  run in place, without installing the package.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the repository folder that contains definedtarget/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

failures = []


def check(name, cond, detail=""):
    """Record one test: `cond` must be truthy."""
    cond = bool(cond)
    detail = str(detail)                     # str() first: details can be numbers, tuples or arrays
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def raises(name, fn, must_contain):
    """Record one test: fn() must raise an exception whose message contains `must_contain`."""
    try:
        fn()
    except Exception as err:  # noqa: BLE001 - any exception type is fine; the message is what we check
        check(name, must_contain.lower() in str(err).lower(), f"wrong error: {type(err).__name__}: {err}")
        return
    check(name, False, "did not raise")

