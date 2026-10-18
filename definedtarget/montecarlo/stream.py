"""Streaming: results from any number of paths in little memory, chunk by chunk.

Plain words
- A *chunk function* simulates paths path0 .. path0 + n - 1 and returns one number per path (an end price, a
  drawdown, a payoff). Because every random number has an address (montecarlo.rng), calling it again for the same
  paths returns exactly the same numbers: a *replay*.
- Means: running moments (stats.Moments), merged chunk by chunk.
- Percentiles: a fixed histogram (65,536 bins by default) counts every path; it locates the q-th percentile inside one
  bin. A replay of the same paths then keeps only the values in that bin, sorts that handful and returns the exact
  order statistic: the same number a full sort of all n values would give, with memory for one chunk plus one bin.
- Precision on demand: run_to_precision() runs a pilot, estimates how many paths a target standard error needs,
  runs the rest and stops once the target is met (or at a path limit), reporting what it achieved.
"""
import math

import numpy as np

from .stats import Moments, z_for


class Histogram:
    """Counts of values in `bins` equal bins on [lo, hi), plus counts below lo and at or above hi. Mergeable."""

    def __init__(self, lo, hi, bins=65_536):
        if not hi > lo or bins < 1:
            raise ValueError("need hi > lo and at least one bin")
        self.lo, self.hi, self.bins = float(lo), float(hi), int(bins)
        self.width = (self.hi - self.lo) / self.bins
        self.counts = np.zeros(self.bins + 2, dtype=np.int64)          # [below, bin 0 .. bin bins-1, above]

    def index(self, x):
        """Slot of each value: 0 = below lo, 1 .. bins = the bins, bins + 1 = at or above hi (one rule, used twice)."""
        k = np.floor((np.asarray(x, dtype=np.float64) - self.lo) / self.width)
        return (np.clip(k, -1, self.bins) + 1).astype(np.int64)

    def add(self, x):
        self.counts += np.bincount(self.index(x).ravel(), minlength=self.bins + 2)
        return self

    def merge(self, other):
        if (other.lo, other.hi, other.bins) != (self.lo, self.hi, self.bins):
            raise ValueError("histograms with different bins cannot merge")
        self.counts += other.counts
        return self

    @property
    def n(self):
        return int(self.counts.sum())

    def slot_of_rank(self, rank):
        """(slot, values before that slot) for the rank-th smallest value (rank 1 = the minimum)."""
        cum = np.cumsum(self.counts)
        slot = int(np.searchsorted(cum, rank))
        return slot, int(cum[slot - 1]) if slot else 0

    def edges(self, slot):
        if slot == 0:
            return -math.inf, self.lo
        if slot == self.bins + 1:
            return self.hi, math.inf
        return self.lo + (slot - 1) * self.width, self.lo + slot * self.width


def chunks(npaths, chunk):
    """(path0, n) pairs that cover paths 0 .. npaths - 1."""
    return [(p, min(chunk, npaths - p)) for p in range(0, npaths, chunk)]


def histogram_for(chunk_fn, npaths, chunk=10_000_000, bins=65_536, spread=0.5):
    """A histogram of all npaths values; its range is the first chunk's range widened by `spread` on each side
    (values outside still count, in the below / above slots)."""
    hist = None
    for p0, n in chunks(npaths, chunk):
        x = chunk_fn(p0, n)
        if hist is None:
            span = float(x.max() - x.min()) or 1.0
            hist = Histogram(float(x.min()) - spread * span, float(x.max()) + spread * span, bins)
        hist.add(x)
    return hist


def order_statistics(chunk_fn, npaths, ranks, chunk=10_000_000, hist=None):
    """The rank-th smallest values (ranks 1-based) of all npaths values, exactly, by histogram + one replay."""
    hist = hist or histogram_for(chunk_fn, npaths, chunk)
    wanted = {r: hist.slot_of_rank(r) for r in ranks}
    slots = sorted({s for s, _ in wanted.values()})
    # The replay keeps each needed bin's values as (distinct value, count) pairs, so a large atom (a payoff of exactly
    # zero for most paths) costs one entry, not one per path (audit of v0.3.0, D07). Same order statistics, bit for bit.
    kept = {s: ([], []) for s in slots}
    for p0, n in chunks(npaths, chunk):                          # the replay: same paths, same numbers
        x = chunk_fn(p0, n)
        idx = hist.index(x)
        for s in slots:
            u, c = np.unique(x[idx == s], return_counts=True)
            kept[s][0].append(u)
            kept[s][1].append(c)
    merged = {}
    for s, (us, cs) in kept.items():
        u, inv = np.unique(np.concatenate(us), return_inverse=True)
        merged[s] = (u, np.cumsum(np.bincount(inv, weights=np.concatenate(cs)).astype(np.int64)))
    out = {}
    for r, (s, before) in wanted.items():
        u, cum = merged[s]
        out[r] = float(u[np.searchsorted(cum, r - before)])      # the first distinct value whose count reaches the rank
    return out, {"kept_values": int(sum(u.size for u, _ in merged.values())), "paths": npaths}


def quantile(chunk_fn, npaths, q, level=0.95, chunk=10_000_000):
    """(estimate, low, high) for the q-th percentile from npaths paths: the same order statistics as
    stats.quantile_interval on all values, computed keeping only one histogram bin's distinct values."""
    h = z_for(level) * math.sqrt(npaths * q * (1 - q))
    r_est = max(math.ceil(npaths * q), 1)
    r_lo = min(max(math.floor(npaths * q - h), 1), npaths)
    r_hi = min(max(math.ceil(npaths * q + h) + 1, 1), npaths)
    vals, info = order_statistics(chunk_fn, npaths, (r_lo, r_est, r_hi), chunk)
    return (vals[r_est], vals[r_lo], vals[r_hi]), info


def moments(chunk_fn, npaths, chunk=10_000_000, path0=0):
    """Running moments of all values from paths path0 .. path0 + npaths - 1."""
    m = Moments()
    for p0, n in chunks(npaths, chunk):
        m.add(chunk_fn(path0 + p0, n))
    return m


def run_to_precision(chunk_fn, target_se, pilot=100_000, chunk=10_000_000, max_paths=10_000_000_000):
    """Moments of the mean, with enough paths for standard error <= target_se (or max_paths, whichever first).

    A pilot estimates the spread; the paths needed are (sd / target)^2; the next paths continue the same numbering
    (pilot paths included in the result), and the loop tops up if the full-run spread turns out larger.
    Returns (Moments, report dict)."""
    if not (math.isfinite(target_se) and target_se > 0) or pilot < 2 or max_paths < 2 or chunk < 1:
        raise ValueError("target_se must be positive and finite; pilot and max_paths at least 2; chunk at least 1")
    m = moments(chunk_fn, min(pilot, max_paths), chunk)     # the pilot obeys max_paths and chunk too (D08)
    planned = []
    while m.se > target_se and m.n < max_paths:
        need = min(max(math.ceil(m.var / target_se ** 2 * 1.01), m.n + m.n // 10 + 1), max_paths)  # always grows
        planned.append(need)
        m.merge(moments(chunk_fn, need - m.n, chunk, path0=m.n))
    return m, {"paths": m.n, "se": m.se, "target_se": target_se, "planned": planned, "met": m.se <= target_se}
