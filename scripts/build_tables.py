#!/usr/bin/env python3
"""Rebuild the tidy tables from the saved raw files and check the accounting identities.  From target_funds/:

    py -I scripts\\build_tables.py

Writes (git-ignored): data/tables/xx_holdings.csv, data/tables/xx_state.csv, data/tables/vendor_marks.csv.
Exit code 1 if any identity fails (the daily log treats that as a STOP).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from definedtarget.forward import fetch, parse, reconcile  # noqa: E402

UNDERLYINGS = ("SPY", "IVV", "VOO", "SPYM")


def main():
    root = fetch.data_dir()
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)

    xx = parse.load_holdings(root, accounts=("XX",))
    daily = parse.load_fund_daily(root, "XX")
    state = reconcile.fund_state(xx, daily)
    xx.to_csv(out / "xx_holdings.csv", index=False)
    state.to_csv(out / "xx_state.csv", index=False)

    every = parse.load_holdings(root, accounts=None, all_versions=False)
    marks = every[every["is_option"] & every["opt_underlying"].isin(UNDERLYINGS)]
    marks.to_csv(out / "vendor_marks.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["as_of", "shares_outstanding", "contracts", "contracts_per_share", "mark_text", "nav_published",
            "nav_net_assets", "lines_gap_usd", "cash_weight", "premium", "flow_day"]
    print(state[cols].to_string(index=False))
    rev = reconcile.revisions(xx)
    if not rev.empty:
        print("\nrevised holdings files (same close date saved more than once):")
        print(rev.to_string(index=False))
    print(f"\nvendor_marks: {len(marks)} option lines on {', '.join(UNDERLYINGS)} across "
          f"{marks['account'].nunique()} funds, {marks['as_of'].nunique()} days")
    fails, explained = reconcile.failures(state)
    for as_of, name in explained:
        print(f"known exception {as_of} {name}: {reconcile.KNOWN_EXCEPTIONS[(as_of, name)]}")
    for as_of, name in fails:
        print(f"FAIL {as_of} {name}")
    print(f"\n{len(fails)} unexplained identity failure(s) over {len(state)} days")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
