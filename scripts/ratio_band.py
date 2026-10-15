#!/usr/bin/env python3
"""Effective S&P 500 trigger for XX's SPY strike, from past January S&P/SPY ratios.  From target_funds/:

    py -I scripts\\ratio_band.py <folder containing your price library>

Needs a local price library: a folder holding a `marketdata.py` module whose load(symbol) returns a DataFrame with
a "close" column for SP:SPX and AMEX:SPY daily closes (the author builds theirs from their own TradingView exports;
display-only data, so the project only ever publishes the summary below).
Caveat (ASSUMPTIONS.md A5): that library's AMEX:SPY rows come from a BATS export, not SPY's official close, so
single days can be off; the 7-session median damps this.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget import ratio  # noqa: E402


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    sys.path.insert(0, str(Path(argv[0]).resolve()))
    import marketdata as md  # the user's own library (trusted code in their own folder)
    spx, spy = md.load("SP:SPX")["close"], md.load("AMEX:SPY")["close"]
    print(md.banner("SP:SPX"))
    print(md.banner("AMEX:SPY"))
    jan = ratio.january_ratios(spx, spy, first_year=1994)
    for y, r in jan.items():
        print(f"  Jan {y}: S&P/SPY = {r:.4f}  -> SPY 995.01 = S&P {995.01 * r:,.0f}")
    print(f"\nstrike implies ratio {10000 / 995.01:.4f}. Effective S&P trigger at 2030-01-10 by sample start "
          "(overlapping spans: a description of history, not a confidence interval):")
    print("  start  spans  4y change min..max     no drift  min drift  max drift  fitted line")
    for start in (1994, 2005, 2008, 2011, 2015):
        b = ratio.trigger_band({y: r for y, r in jan.items() if y >= start})
        print(f"  {start}   {b['n_spans']:>4}  {b['change_min']:+.3%}..{b['change_max']:+.3%}   "
              f"{b['trigger_no_drift']:>8,.0f}  {b['trigger_min_drift']:>9,.0f}  {b['trigger_max_drift']:>9,.0f}"
              f"  {b['trigger_trend']:>11,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
