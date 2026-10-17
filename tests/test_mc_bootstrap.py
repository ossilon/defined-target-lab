"""The circular stationary bootstrap (montecarlo.bootstrap) judged by exact properties, on synthetic history.

Judges:
- every history row is equally likely at every step (the circular wrap): chi-square of row counts;
- run lengths are geometric with mean equal to the mean block length; new blocks start with probability 1 / mean;
- E[sum of n steps] = n x history's mean, exactly;
- Var[sum of n steps] = n g(0) + 2 sum_k (n - k)(1 - p)^k g(k) (g = circular autocovariance), exactly: checked for
  i.i.d. resampling (block 1), short and long blocks, on a history with strong serial correlation (AR(1), phi 0.4);
- very long blocks copy a circular window of history; the measures equal a recomputation from the chosen rows;
- replays are identical on any thread count or chunk split; equal_moments sets mean and spread exactly.
"""
import math

import numba
import numpy as np

from _harness import check, raises

from definedtarget.montecarlo import bootstrap as bs
from definedtarget.montecarlo import paths as lp

rs = np.random.default_rng(11)                           # synthetic history only (the bootstrap's own draws are Philox)
n_hist = 3000
eps = rs.standard_t(4, n_hist) * 0.01                    # fat tails
ar = np.empty(n_hist)
ar[0] = eps[0]
for t in range(1, n_hist):
    ar[t] = 0.4 * ar[t - 1] + eps[t]                     # strong serial correlation
ar += 0.0003
HIST = np.column_stack([ar, rs.standard_normal(n_hist)])  # a second column travels with the first

# 1. Uniform rows and geometric runs
idx = bs.indices(5, 40_000, 25, 1000, mean_block=8.0)
counts = np.bincount(idx.ravel(), minlength=1000)
expected = idx.size / 1000
chi2 = float(((counts - expected) ** 2 / expected).sum())
check("every history row is equally likely (chi-square of 1,000 row counts within 6 sd of 999)",
      abs(chi2 - 999) < 6 * math.sqrt(2 * 999), chi2)
for k in (0, 1, 24):
    c = np.bincount(idx[:, k], minlength=1000)
    check(f"... at step {k} too (the circular wrap keeps the end of history as likely as the middle)",
          abs(float(((c - 40.0) ** 2 / 40.0).sum()) - 999) < 6 * math.sqrt(2 * 999))
new_block = (np.diff(idx, axis=1) != 1) & ~((idx[:, :-1] == 999) & (idx[:, 1:] == 0))
rate = new_block.mean()
check("a new block starts with probability 1 / mean block (1/8), allowing for jumps that land on the next row",
      abs(rate - (1 / 8) * (1 - 1 / 1000)) < 4 * math.sqrt(rate * (1 - rate) / new_block.size), rate)

# 2. Exact mean and variance of the sum of steps
for block in (1.0, 5.0, 40.0):
    o = bs.paths(21, 400_000, 60, HIST, block)
    s = o["log_end"]
    m, v = s.mean(), s.var(ddof=1)
    m4 = float(((s - m) ** 4).mean())                    # SE of a variance from the sample's own 4th moment
    se_m, se_v = math.sqrt(v / s.size), math.sqrt((m4 - v * v) / s.size)  # (fat tails make it wider than normal's)
    exact_m, exact_v = 60 * ar.mean(), bs.sum_variance(ar, 60, block)
    check(f"block {block:g}: E[sum of 60 steps] within 3 SE of 60 x history's mean", abs(m - exact_m) < 3 * se_m,
          (m, exact_m, se_m))
    check(f"block {block:g}: Var[sum] within 4 SE of the exact formula {exact_v:.6f} (i.i.d. would give "
          f"{60 * ar.var():.6f})", abs(v - exact_v) < 4 * se_v, (v, exact_v, se_v, (v - exact_v) / se_v))
    check(f"block {block:g}: the second column's total has mean 60 x its history mean",
          abs(o["column_end"][:, 1].mean() - 60 * HIST[:, 1].mean()) < 3 * o["column_end"][:, 1].std() / math.sqrt(s.size))
check("longer blocks keep more of the serial correlation: Var[sum] rises with block length (phi = 0.4)",
      bs.sum_variance(ar, 60, 1.0) < bs.sum_variance(ar, 60, 5.0) < bs.sum_variance(ar, 60, 40.0))

# 3. Long blocks copy circular windows; measures match a recomputation from the rows
o = bs.paths(3, 2000, 50, HIST, 1e15)
rows = bs.indices(3, 2000, 50, n_hist, 1e15)
start = rows[:, 0]
check("with very long blocks each path is a circular window of history (rows start, start+1, ...)",
      np.array_equal(rows, (start[:, None] + np.arange(50)) % n_hist))
o2 = bs.paths(9, 500, 40, HIST, 6.0, barrier=math.exp(0.02), checkpoints=(10, 40))
rows = bs.indices(9, 500, 40, n_hist, 6.0)
cum = np.cumsum(ar[rows], axis=1)
top = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
hit = np.where((cum >= 0.02).any(axis=1), (cum >= 0.02).argmax(axis=1), -1)
check("measures equal a recomputation from the chosen rows (end, high, drawdowns, first hit, checkpoints)",
      np.allclose(o2["log_end"], cum[:, -1], atol=1e-12, rtol=0) and np.allclose(o2["log_max"], top[:, -1], atol=1e-12, rtol=0)
      and np.allclose(o2["log_max_drawdown"], (top - cum).max(axis=1), atol=1e-12, rtol=0)
      and np.array_equal(o2["hit_step"], hit) and np.allclose(o2["at_checkpoint"][:, 1], cum[:, 39], atol=1e-12, rtol=0)
      and np.allclose(o2["at_checkpoint"][:, 0], cum[:, 9], atol=1e-12, rtol=0))

# 4. Replays: threads and chunks change nothing
a = bs.paths(4, 3000, 30, HIST, 10.0)
default_threads = numba.get_num_threads()
numba.set_num_threads(1)
b = bs.paths(4, 3000, 30, HIST, 10.0)
numba.set_num_threads(default_threads)
c = np.concatenate([bs.paths(4, 1000, 30, HIST, 10.0)["log_end"], bs.paths(4, 2000, 30, HIST, 10.0, path0=1000)["log_end"]])
check("thread invariance: 1 thread equals all threads, every measure", all(np.array_equal(a[k], b[k]) for k in a
                                                                          if k != "at_checkpoint"))
check("chunk invariance: 1,000 + 2,000 paths equal 3,000 paths", np.array_equal(a["log_end"], c))

# 5. Equal moments: a bootstrap and a lognormal with the same step mean and spread have the same E[log end]
x = bs.equal_moments(ar, 0.0002, 0.0105)
check("equal_moments sets the mean and spread exactly", abs(x.mean() - 0.0002) < 1e-15 and abs(x.std() - 0.0105) < 1e-15)
dt = 1 / 252
vol = 0.0105 / math.sqrt(dt)
mu = 0.0002 / dt + 0.5 * vol * vol
boot = bs.paths(8, 300_000, 252, x, 21.0)["log_end"]
logn = lp.lognormal(8, 300_000, np.arange(253) * dt, mu, vol)["log_end"]
se = math.sqrt(boot.var() / boot.size + logn.var() / logn.size)
check("same inputs, same mean: E[log end] of bootstrap and lognormal agree within 3 SE",
      abs(boot.mean() - logn.mean()) < 3 * se, (boot.mean(), logn.mean(), se))
check("... but not the same spread: history's serial correlation widens the bootstrap's one-year spread",
      boot.var() > 1.2 * logn.var(), (boot.var(), logn.var()))
raises("a mean block below 1 is refused", lambda: bs.paths(1, 1, 1, ar, 0.5), "at least 1")
raises("history with missing values is refused", lambda: bs.paths(1, 1, 1, np.array([0.1, np.nan]), 2.0), "finite")
