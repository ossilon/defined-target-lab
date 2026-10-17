"""Fixed-memory results (montecarlo.stream): exact percentiles by replay, histograms, precision on demand.

Judges: a full in-memory sort of the same values (the replay must return the identical order statistics, bit for
bit); stats.quantile_interval on the full sample; the exact lognormal percentile and Black-76 value; and the target
standard error itself.
"""
import math

import numpy as np
from scipy.special import ndtri

from _harness import check, raises

from definedtarget import pricing
from definedtarget.montecarlo import paths, stats, stream

F, K, R, T, VOL = 882.2603, 995.01, 0.0486498, 3.2465753, 0.166
DF = math.exp(-R * T)
GRID = np.array([0.0, T])                                  # one step to expiry: the end price only


def end_prices(path0, n, seed=777):
    return F * np.exp(paths.lognormal(seed, n, GRID, 0.0, VOL, path0=path0)["log_end"])


def payoffs(path0, n):
    return DF * np.maximum(end_prices(path0, n) - K, 0.0)


# 1. Exact percentiles by replay equal a full sort, bit for bit
N = 3_000_000
raw = end_prices(0, N)
full = np.sort(raw)
got = {}
for q in (0.001, 0.1, 0.5, 0.9, 0.999):
    got[q], info = stream.quantile(end_prices, N, q, chunk=400_000)
    want = stats.quantile_interval(full, q, presorted=True)
    check(f"replayed {q:.1%} percentile and its 95% interval equal a full sort of {N:,} values, bit for bit",
          got[q] == tuple(float(v) for v in want), (got[q], want))
    check(f"... and the replay kept few values in memory ({info['kept_values']:,} of {N:,})",
          info["kept_values"] < N / 100, info)
truth = F * math.exp(-0.5 * VOL * VOL * T + VOL * math.sqrt(T) * float(ndtri(0.9)))
check(f"the replayed 90th percentile's interval contains the exact lognormal value {truth:.4f}",
      got[0.9][1] <= truth <= got[0.9][2], (got[0.9], truth))
r = [1, 2, N // 2, N - 1, N]
vals, _ = stream.order_statistics(end_prices, N, r, chunk=1_000_000)
check("order statistics at the extremes (min, 2nd, median, 2nd largest, max) match the full sort",
      [vals[k] for k in r] == [float(full[k - 1]) for k in r])

# 2. Histograms: chunks merge exactly; values outside the range are still counted
h1 = stream.Histogram(600.0, 1400.0, 1000).add(raw[:1_000_000])
h2 = stream.Histogram(600.0, 1400.0, 1000).add(raw[:400_000]).merge(stream.Histogram(600.0, 1400.0, 1000)
                                                                     .add(raw[400_000:1_000_000]))
check("histograms built in two chunks and merged equal one pass", np.array_equal(h1.counts, h2.counts))
check("every value is counted, including below and above the range", h1.n == 1_000_000 and h1.counts[0] > 0
      and h1.counts[-1] > 0)
slot, before = h1.slot_of_rank(500_000)
a, b = h1.edges(slot)
check("the bin found for the median contains the true sorted median", a <= float(np.sort(raw[:1_000_000])[499_999]) < b)
raises("histograms with different bins refuse to merge",
       lambda: stream.Histogram(0, 1, 10).merge(stream.Histogram(0, 2, 10)), "cannot merge")

# 3. Precision on demand: ask for a standard error, get it, and stay honest
exact = pricing.call_price(F, K, DF, T, VOL)
for target in (0.05, 0.01):
    m, rep = stream.run_to_precision(payoffs, target, pilot=50_000, chunk=2_000_000)
    check(f"target SE {target}: met with {rep['paths']:,} paths (SE {m.se:.5f})", rep["met"] and m.se <= target, rep)
    check(f"... and the estimate {m.mean:.4f} is within 3 SE of Black-76 {exact:.4f}", abs(m.mean - exact) <= 3 * m.se)
    predicted = (math.sqrt(m.var) / target) ** 2
    check("... using close to the predicted (sd / target)^2 paths, not far more", m.n <= 1.25 * predicted + 50_000,
          (m.n, predicted))
m, rep = stream.run_to_precision(payoffs, 1e-6, pilot=50_000, chunk=2_000_000, max_paths=300_000)
check("a path limit stops an impossible target and says so", not rep["met"] and rep["paths"] == 300_000, rep)
same = stream.moments(payoffs, 1_000_000, chunk=150_000)
one = stats.Moments().add(payoffs(0, 1_000_000))
check("streamed moments (7 uneven chunks) equal one pass", abs(same.mean - one.mean) < 1e-12 * one.mean
      and abs(same.var - one.var) < 1e-10 * one.var)
