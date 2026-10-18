#!/usr/bin/env python3
"""A billion paths in fixed memory (montecarlo.stream), with its run manifest.  From target_funds/:

    py -I scripts\\billion_paths_demo.py

Added 2026-10-18 after the v0.3 audit, as the recorded source of the billion-path figures in CHANGELOG.md and
docs/learn/06. XX's option in the README's 2026-10-13 example (forward 882.2603, strike 995.01, rate 4.86498%,
3.2465753 years, vol 16.6%); one lognormal end price per path, seed 777, chunks of 10 million paths.
1. run_to_precision: the option's value to a standard error of 0.004 (pilot 100,000 paths), against Black-76.
2. The exact median of 1 billion end prices by histogram + one replay, against the exact lognormal median.
Times are wall-clock on the machine that runs it; the values depend only on the seed. Research, not advice.
"""
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from definedtarget import pricing  # noqa: E402
from definedtarget.montecarlo import manifest, paths, stream  # noqa: E402

F, K, R, T, VOL = 882.2603, 995.01, 0.0486498, 3.2465753, 0.166
DF = math.exp(-R * T)
GRID = np.array([0.0, T])
SEED, TARGET_SE, N_MEDIAN = 777, 0.004, 1_000_000_000


def end_prices(path0, n):
    return F * np.exp(paths.lognormal(SEED, n, GRID, 0.0, VOL, path0=path0)["log_end"])


def payoffs(path0, n):
    return DF * np.maximum(end_prices(path0, n) - K, 0.0)


def main():
    end_prices(0, 1000)                                    # compile first, so the times measure the work
    exact = pricing.call_price(F, K, DF, T, VOL)
    t0 = time.perf_counter()
    m, rep = stream.run_to_precision(payoffs, TARGET_SE)
    print(f"1. value to SE {TARGET_SE}: {rep['paths']:,} paths, {m.mean:.5f} (SE {m.se:.5f}) against Black-76 "
          f"{exact:.5f}; {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    (est, lo, hi), info = stream.quantile(end_prices, N_MEDIAN, 0.5)
    truth = F * math.exp(-0.5 * VOL * VOL * T)
    print(f"2. median of {N_MEDIAN:,} end prices: {est:.5f} (95% interval {lo:.5f} to {hi:.5f}) against the exact "
          f"{truth:.5f}; kept {info['kept_values']:,} values; {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    with manifest.run("billion_paths_demo", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/billion_paths_demo.py",), seeds=[SEED], deterministic=False,
                      settings={"target_se": TARGET_SE, "median_paths": N_MEDIAN}) as run:
        main()
    print(f"run manifest: results/manifests/{run['path'].name}")
