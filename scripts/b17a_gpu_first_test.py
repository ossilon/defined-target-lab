#!/usr/bin/env python3
"""B17a: a first GPU test. Price XX's option by Monte Carlo four ways and time them.  From target_funds/:

    py -I scripts\\b17a_gpu_first_test.py            (CuPy optional: without it the GPU rows are skipped)

What it does, in plain words
- XX holds calls on SPY with strike 995.01 that expire 2030-01-10. Under the simple lognormal model the option's
  value today has an exact formula (Black-76), so a Monte Carlo estimate can be checked against the true answer.
- Monte Carlo: draw N random end prices S_T = F x exp(-vol^2 T / 2 + vol sqrt(T) Z), Z standard normal; the option
  pays max(S_T - K, 0) at expiry; the price is the discounted average payoff. Its standard error (SE) is the
  standard deviation of the discounted payoffs / sqrt(N): 100x more paths make the answer 10x more precise.
- Four engines: numpy on one core; numba on every core (24 threads); CuPy on the RTX 3080 with 64-bit numbers;
  CuPy with 32-bit numbers (a gaming GPU does 32-bit math far faster than 64-bit).
- Each estimate must sit within 3 SE of the exact price. The timings show where each engine wins.

This is a learning test, separate from the engine. Its random numbers are NOT the v0.3 design (counter-based Philox,
identical on any thread count or device): numba's parallel random streams here depend on the thread layout, and
cuRAND's stream differs from numpy's, so the engines agree only within Monte Carlo error, not path by path.
Inputs: the README's dated 2026-10-13 example (continuous-yield form), no market data series.
"""
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numba  # noqa: E402
import numpy as np  # noqa: E402

from definedtarget import pricing  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402

# The README's 2026-10-13 example: SPY 778.01, forward to expiry 882.2603, rate 4.86498%, 3.2465753 years
S0, F, K, R, T, VOL = 778.01, 882.2603, 995.01, 0.0486498, 3.2465753, 0.166
DF = math.exp(-R * T)
DRIFT, SCALE = -0.5 * VOL * VOL * T, VOL * math.sqrt(T)       # log(S_T / F) = DRIFT + SCALE x Z
CHUNK = 10_000_000                                              # paths per batch (memory stays small)


def summary(total, total_sq, n):
    """Price and SE from the running sum and sum of squares of the discounted payoffs."""
    mean = total / n
    var = max(total_sq / n - mean * mean, 0.0) * n / (n - 1)
    return mean, math.sqrt(var / n)


# ---------------------------------------------------------------- 1. numpy, one core

def mc_numpy(n, seed=1):
    rng = np.random.default_rng(seed)
    total = total_sq = 0.0
    left = n
    while left:
        m = min(CHUNK, left)
        z = rng.standard_normal(m)
        pay = DF * np.maximum(F * np.exp(DRIFT + SCALE * z) - K, 0.0)
        total += float(pay.sum())
        total_sq += float((pay * pay).sum())
        left -= m
    return summary(total, total_sq, n)


# ---------------------------------------------------------------- 2. numba, every core

@numba.njit(parallel=True, fastmath=False, cache=False)
def _numba_kernel(n, seed):
    np.random.seed(seed)                 # numba gives each thread its own random stream
    total = 0.0
    total_sq = 0.0
    for _ in numba.prange(n):            # prange splits the loop across the CPU threads; += lines become reductions
        s = F * math.exp(DRIFT + SCALE * np.random.standard_normal())
        pay = DF * max(s - K, 0.0)
        total += pay
        total_sq += pay * pay
    return total, total_sq


def mc_numba(n, seed=1):
    total, total_sq = _numba_kernel(n, seed)
    return summary(total, total_sq, n)


# ---------------------------------------------------------------- 3 and 4. CuPy on the GPU

def make_gpu(dtype):
    import cupy as cp
    # One fused GPU kernel: each GPU thread turns one random number into one discounted payoff (and its square)
    payoff = cp.ElementwiseKernel(
        "T z", "T pay, T pay2",
        f"""T s = (T){F!r} * exp((T){DRIFT!r} + (T){SCALE!r} * z);
            pay = (T){DF!r} * max(s - (T){K!r}, (T)0);
            pay2 = pay * pay;""",
        "xx_payoff")

    def mc_gpu(n, seed=1):
        rng = cp.random.default_rng(seed)                    # cuRAND-backed generator on the card
        total = total_sq = 0.0
        left = n
        while left:
            m = min(CHUNK, left)
            z = rng.standard_normal(m, dtype=dtype)
            pay, pay2 = payoff(z)
            # Sum in 64-bit even for 32-bit payoffs: adding millions of 32-bit numbers in 32 bits loses digits
            total += float(pay.sum(dtype=cp.float64))
            total_sq += float(pay2.sum(dtype=cp.float64))
            left -= m
        cp.cuda.Stream.null.synchronize()                    # the GPU runs asynchronously: wait before timing ends
        return summary(total, total_sq, n)
    return mc_gpu


def timed(fn, n, repeats):
    """Median wall time of `repeats` runs (after one warm-up run, which compiles numba / GPU kernels)."""
    fn(min(n, 100_000))
    times, out = [], None
    for i in range(repeats):
        t0 = time.perf_counter()
        out = fn(n, seed=1 + i)
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


def main():
    exact = pricing.call_price(F, K, DF, T, VOL)
    print(f"Exact Black-76 value of one SPY {K} call to 2030-01-10 (README example): {exact:.6f}")
    print(f"CPU threads used by numba: {numba.get_num_threads()} (of {os.cpu_count()} logical cores)\n")
    engines = [("numpy, 1 core", mc_numpy), ("numba, all cores", mc_numba)]
    try:
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        engines += [(f"CuPy float64 ({name})", make_gpu(cp.float64)), (f"CuPy float32 ({name})", make_gpu(cp.float32))]
    except Exception as err:  # noqa: BLE001 - the GPU is optional
        print(f"GPU rows skipped: {type(err).__name__}: {err}\n")

    sizes = [10_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    print(f"{'engine':<34}{'paths':>15}{'price':>12}{'SE':>11}{'err/SE':>8}{'seconds':>10}{'M paths/s':>11}")
    for label, fn in engines:
        for n in sizes:
            if label.startswith("numpy") and n > 100_000_000:
                continue                                     # one core would take minutes; the trend is clear
            (price, se), sec = timed(fn, n, repeats=3 if n <= 100_000_000 else 1)
            z = (price - exact) / se
            flag = "" if abs(z) <= 3 else "  <-- outside 3 SE"
            print(f"{label:<34}{n:>15,}{price:>12.5f}{se:>11.5f}{z:>8.2f}{sec:>10.3f}{n / sec / 1e6:>11.1f}{flag}")
        print()


if __name__ == "__main__":
    with manifest.run("b17a_gpu_first_test", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("definedtarget/**/*.py", "scripts/b17a_gpu_first_test.py"), deterministic=False) as run:
        main()
    print(f"run manifest: results/manifests/{run['path'].name}")
