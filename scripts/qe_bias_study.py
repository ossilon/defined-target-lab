#!/usr/bin/env python3
"""How large is the simulation's own error at XX's strike, and is the marked option a martingale?

    py -I scripts\\qe_bias_study.py [--paths N]      From target_funds/.

Fixed before the first run (2026-10-23):
- Model: Gatheral (2006) Table 3.2 (v0 0.0174, theta 0.0354, kappa 1.3253, eta 0.3877, rho -0.7165), in forward
  terms (forward 100 at every date). Option: the listed S&P 500 call, strike 10,000, expiring 2029-12-21, at a
  forward of 8,370 (strike 119.47 per 100 of forward). Time grid: the NYSE sessions from 2026-10-23 to 2029-12-21.
- Part 1, the bias of the scheme: the call's price by Monte Carlo minus its exact price (volmodel.pricers, two
  pricers that must agree), for the schemes QE-M, QE and Euler, on four grids: every session (daily), every 5th,
  every 21st and every 63rd session (each grid keeps the last session). Reported with the standard error, and the
  bias in vol points (bias / vega).
- Part 2, the martingale identity (the property the slide rule failed, FINDINGS F17 and F19): mark the call along
  daily QE-M paths with volmodel.pricetable at 0.5, 1.5 and 2.5 years; its expected forward value must equal
  today's price. Reported: the gap in % of the price, its standard error and the z-score. The accuracy target of
  v0.3.1 (V031 brief): standard error at most 0.05% of the price, gap within 3 standard errors.
- 2^24 paths per case, seed 20261023 for every case (common random numbers), in chunks of 2^22 paths.
Synthetic paths only; aggregates only; research, not investment advice.
"""
import argparse
import math
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from definedtarget import calendar as nyse  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.montecarlo import paths as mcpaths  # noqa: E402
from definedtarget.volmodel import black, pricers, pricetable, stochvol  # noqa: E402
from definedtarget.volmodel.heston import Heston  # noqa: E402

SEED = 20261023
CHUNK = 1 << 22
MODEL = Heston(0.0174, 0.0354, 1.3253, 0.3877, -0.7165)
K_REL = 10000.0 / 8370.0 * 100.0


def chunked(npaths, fn):
    """Sum and sum of squares of fn(paths dict) over chunks; returns (mean, standard error)."""
    s = s2 = 0.0
    for p0 in range(0, npaths, CHUNK):
        y = fn(p0, min(CHUNK, npaths - p0))
        s += float(y.sum())
        s2 += float((y * y).sum())
    mean = s / npaths
    return mean, math.sqrt(max(s2 / npaths - mean * mean, 0.0) / (npaths - 1))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1 << 24)
    a = ap.parse_args(argv)
    t_all = mcpaths.times_from_dates(nyse.sessions(date(2026, 10, 23), date(2029, 12, 21)))
    T = float(t_all[-1])
    exact = float(pricers.prices(MODEL, 100.0, K_REL, T, check=True)[0])
    vol = float(black.implied_vols(exact, 100.0, K_REL, T)[0])
    vega = black.call_vega(100.0, K_REL, 1.0, T, vol) / 100.0                 # per vol point
    print(f"exact price {exact:.6f} per 100 of forward (implied vol {100 * vol:.3f}%, vega {vega:.4f} per vol point); "
          f"{len(t_all) - 1} sessions; {a.paths:,} paths per case\n")
    print("| grid | steps | scheme | Monte Carlo price | bias (MC - exact) | SE | bias in vol points |")
    print("|---|---|---|---|---|---|---|")
    for every, label in ((1, "daily"), (5, "every 5th session"), (21, "every 21st session"), (63, "every 63rd session")):
        idx = list(range(0, len(t_all) - 1, every)) + [len(t_all) - 1]
        t = t_all[sorted(set(idx))]
        for scheme in ("qe-m", "qe", "euler"):
            def pay(p0, n, t=t, scheme=scheme):
                o = stochvol.simulate_model(MODEL, SEED, n, t, scheme=scheme, path0=p0)
                return np.maximum(100.0 * np.exp(o["log_end"]) - K_REL, 0.0)
            mean, se = chunked(a.paths, pay)
            print(f"| {label} | {len(t) - 1} | {scheme} | {mean:.4f} | {mean - exact:+.4f} | {se:.4f} | "
                  f"{(mean - exact) / vega:+.4f} |")

    print("\nMartingale identity, daily QE-M paths (the table-marked call; gap = expected value / today's price - 1)\n")
    print("| checkpoint (years) | remaining (years) | gap | SE | z |")
    print("|---|---|---|---|---|")
    steps = [int(np.searchsorted(t_all, y)) for y in (0.5, 1.5, 2.5)]
    k0 = math.log(K_REL / 100.0)
    c0 = exact / 100.0
    tables = [pricetable.build(MODEL, T - float(t_all[s])) for s in steps]
    sums = np.zeros((3, 2))
    for p0 in range(0, a.paths, CHUNK):
        n = min(CHUNK, a.paths - p0)
        o = stochvol.simulate_model(MODEL, SEED, n, t_all, checkpoints=tuple(steps), path0=p0)
        for j, tab in enumerate(tables):
            x = o["at_checkpoint"][:, j]
            mark = np.exp(x) * pricetable.calls(tab, k0 - x, o["v_at_checkpoint"][:, j])
            sums[j] += (mark.sum(), (mark * mark).sum())
    for j, s in enumerate(steps):
        mean = sums[j, 0] / a.paths
        se = math.sqrt(max(sums[j, 1] / a.paths - mean * mean, 0.0) / (a.paths - 1))
        print(f"| {t_all[s]:.2f} | {T - t_all[s]:.2f} | {100 * (mean / c0 - 1):+.3f}% | {100 * se / c0:.3f}% | "
              f"{(mean - c0) / se:+.2f} |")
    return 0


if __name__ == "__main__":
    with manifest.run("qe_bias_study", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/qe_bias_study.py",), seeds=[SEED]) as run:
        rc = main(sys.argv[1:])
    print(f"\nrun manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
