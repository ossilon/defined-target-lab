#!/usr/bin/env python3
"""Is the slide-rule effect XX's own, or its option's? Effective leverage and the martingale gap, three instruments.

    py -I scripts\\effective_leverage_study.py [--paths N]      From target_funds/.

Fixed before the first run (2026-10-17). Risk-neutral lognormal paths at 16.6% realized vol on the real calendar;
rate 4.86498%, dividend yield from XX's 2026-10-13 example. Instruments: (1) XX (0.48 SPY calls at 995.01 per share,
0.2474 cash, 0.99% fee, to 2030-01-10); (2) the same call with no cash and no fee (the option inside XX); (3) the
listed S&P 500 call, strike 10,000 on an S&P of 7,805.45 (2026-10-13), expiring 2029-12-21. Implied vol 16.6% with
the slide at 0, 1 and 3.5 points per -10% (3.5: the VIX1Y-sized slide measured in F18). Reported per case: effective
leverage on up days and on down days (realized move per 1% move / the elasticity at the previous close, pooled), and
the discounted mean mark / start - 1 after 400 sessions (the martingale gap; 0 for a consistent price).
Changed after the first run (disclosed): effective leverage counts a day only while the previous mark is at least
25% of the start (xxpaths.LEV_FLOOR). Bare calls fall toward zero near expiry, where their elasticity is unbounded,
and those days dominated the pooled ratio (29 and 174 on the first run); XX's cash floor had hidden this.
Aggregates only; research, not investment advice.
"""
import argparse
import math
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from definedtarget import calendar as nyse, sim  # noqa: E402
from definedtarget import xxpaths as xp  # noqa: E402
from definedtarget.montecarlo import paths as lp  # noqa: E402

R = 0.0486498


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=400_000)
    a = ap.parse_args(argv)
    t_xx = lp.times_from_dates(nyse.sessions(date(2026, 10, 13), date(2030, 1, 10)))
    q = R - math.log(882.2603 / 778.01) / float(t_xx[-1])
    t_l = lp.times_from_dates(nyse.sessions(date(2026, 10, 13), date(2029, 12, 21)))
    cases = [("XX (cash, fee)", sim.Fund((sim.Leg("call", 1, 995.01, 0.48),), cash=0.2474, fee=0.0099), 778.01, t_xx),
             ("XX's option alone", sim.Fund((sim.Leg("call", 1, 995.01, 0.48),)), 778.01, t_xx),
             ("listed S&P 10,000 call, Dec 2029", sim.Fund((sim.Leg("call", 1, 10000.0, 1.0),)), 7805.45, t_l)]
    print(f"{a.paths:,} risk-neutral paths per row; vol 16.6%\n")
    print("| instrument | slide (points per -10%) | effective leverage, up days | down days | martingale gap at "
          "session 400 (+/- 1 SE) |")
    print("|---|---|---|---|---|")
    for label, fund, s0, t in cases:
        m = sim.Market(R, q, float(t[-1]))
        for slide in (0.0, 0.01, 0.035):
            o = xp.simulate(17, a.paths, fund, m, s0, t, 0.166, 0.166, slide=slide, checkpoints=(400,))
            up, dn = xp.effective_leverage(o)
            d = math.exp(-R * t[400]) * o["nav_at_checkpoint"][:, 0] / o["nav0"]
            print(f"| {label} | {slide * 100:.1f} | {up:.3f} | {dn:.3f} | {100 * (d.mean() - 1):+.2f}% "
                  f"(+/- {100 * d.std(ddof=1) / math.sqrt(d.size):.2f}%) |")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
