#!/usr/bin/env python3
"""What a buyer of XX lives through, path by path (v0.3 XX layer).  From target_funds/:

    py -I scripts\\xx_paths_study.py [--paths N]

Fixed before the first run (2026-10-17). Inputs: the README's 2026-10-13 example (SPY 778.01; XX = 0.48 calls at
995.01 + 0.2474 cash, 0.99% fee; rate 4.86498%; dividend yield from the forward 882.2603; 815 real sessions to
2030-01-10). The ETF follows a lognormal path at 16.6% realized vol; XX is repriced every session at 16.6% implied
vol, with the slide rule off and at 1 point per -10% (F12). ETF drift: the risk-neutral rate (3.87% a year price
growth) and two illustrative real-world cases, 7% and 10% a year price growth (FINDINGS F13: the S&P's median
3.25-year price growth since 1950 was 8.8% a year). These are assumptions, not forecasts.
Reported per case (1 million paths, standard errors on probabilities about 0.05 points): P(end NAV above the
starting NAV), P(end NAV below 10% of start), median end NAV / start, P(NAV reached 2x, 5x, 10x at any session),
P(2x reached and then lost again by expiry), mean largest NAV drawdown, median longest spell below a previous high
(sessions), median sessions spent below the starting NAV, P(XX's end return beats SPY's price return).
Added 2026-10-18 (after the v0.3 audit; no table above changes): the bare option's martingale gap at sessions 100,
400, 700 and expiry, constant vol and the slide rule, as FINDINGS F17 quotes it.
Prints aggregates only; research, not investment advice.
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
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.montecarlo import paths as lp  # noqa: E402

S0, R = 778.01, 0.0486498


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1_000_000)
    a = ap.parse_args(argv)
    times = lp.times_from_dates(nyse.sessions(date(2026, 10, 13), date(2030, 1, 10)))
    T = float(times[-1])
    q = R - math.log(882.2603 / S0) / T
    m = sim.Market(R, q, T)
    xx = sim.Fund((sim.Leg("call", 1, 995.01, 0.48),), cash=0.2474, fee=0.0099)
    # mu passed to the engine is the MEAN growth rate (E[S_t] = S_0 e^{mu t}). Risk-neutral: mu = r - q. The two
    # real-world cases are MEDIAN growth (as in F13), so their mean rate adds vol^2 / 2.
    cases = [("risk-neutral (mean 3.87%/yr)", R - q), ("median 7%/yr price growth", math.log(1.07) + 0.5 * 0.166 ** 2),
             ("median 10%/yr", math.log(1.10) + 0.5 * 0.166 ** 2)]
    print(f"XX, {times.size - 1} sessions to 2030-01-10; {a.paths:,} paths per row; vol 16.6% realized and implied\n")
    print("| ETF drift | slide | P(end > start) | P(end < 10% of start) | median end / start | P(reach 2x / 5x / 10x) | "
          "P(2x, then end below 2x) | mean max drawdown | median longest underwater | median sessions below start | "
          "P(beats SPY) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, mu in cases:
        for slide in (0.0, 0.01):
            o = xp.simulate(2026, a.paths, xx, m, S0, times, 0.166, 0.166, slide=slide, mu=mu)
            n0 = o["nav0"]
            r = o["nav_end"] / n0
            fm = o["first_multiple"]
            reach = [(fm[:, j] >= 0).mean() for j in range(3)]
            lost = ((fm[:, 0] >= 0) & (r < 2.0)).mean()
            beats = (r > o["etf_end"] / S0).mean()
            print(f"| {label} | {slide * 100:.0f} pt | {(r > 1).mean():.1%} | {(r < 0.1).mean():.1%} | {np.median(r):.2f} | "
                  f"{reach[0]:.1%} / {reach[1]:.1%} / {reach[2]:.1%} | {lost:.1%} | {o['nav_max_drawdown'].mean():.1%} | "
                  f"{np.median(o['longest_underwater']):.0f} | {np.median(o['sessions_below_start']):.0f} | {beats:.1%} |")
    print(f"\nstarting NAV (model) {n0:.4f}; the 2026-10-13 published NAV was 26.51")
    # Added 2026-10-18 after the v0.3 audit, so F17's martingale-gap paragraph has a reproducible source: the bare
    # option (no fee, no cash) under the risk-neutral drift; its discounted mark must average to today's price.
    bare = sim.Fund((sim.Leg("call", 1, 995.01, 0.48),))
    ck = (100, 400, 700, times.size - 1)
    disc = np.exp(-R * times[list(ck)])
    print(f"\nmartingale gap, bare option, risk-neutral: mean of discounted mark / today's price - 1 (+/- 1 SE) at "
          f"sessions {', '.join(str(k) for k in ck)}")
    for slide in (0.0, 0.01):
        o = xp.simulate(2026, a.paths, bare, m, S0, times, 0.166, 0.166, slide=slide, checkpoints=ck)
        d = disc * o["nav_at_checkpoint"] / o["nav0"]
        cells = [f"{100 * (c.mean() - 1):+.2f}% (+/- {100 * c.std(ddof=1) / math.sqrt(c.size):.2f}%)" for c in d.T]
        print(f"| slide {slide * 100:.0f} pt | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    with manifest.run("xx_paths_study", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/xx_paths_study.py",), seeds=[2026]) as run:
        rc = main(sys.argv[1:])
    print(f"run manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
