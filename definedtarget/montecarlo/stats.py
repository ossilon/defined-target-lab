"""Error bars for Monte Carlo results: every number the engine reports comes with one.

Plain words
- A Monte Carlo number wanders from run to run. Its *standard error* (SE) says by how much: about 2/3 of runs land
  within 1 SE of the truth and 95% within 1.96 SE. A *95% interval* is the estimate +/- 1.96 SE (or its equivalent).
- An error bar is only useful if it is honest: across many independent runs, a 95% interval must contain the true
  value about 95% of the time. tests/test_mc_coverage.py checks exactly that against closed-form answers.
- Moments can be built in chunks and merged (Chan, Golub and LeVeque, 1979), so memory does not grow with the number
  of paths and chunks from different threads or devices can be combined.

Three kinds of result:
- a mean (a price, an expected payoff): estimate +/- z x sd / sqrt(n);
- a probability (P(S_T > K)): the Wilson interval, which stays honest near 0 and 1 where the textbook (Wald)
  interval p +/- z sqrt(p(1-p)/n) is too narrow (wald() is kept to show the difference);
- a percentile: a distribution-free interval from the order statistics (the count of draws below the true q-th
  percentile is binomial(n, q), whatever the distribution).
"""
import math

import numpy as np

from .rng import inv_norm


def z_for(level):
    """The two-sided normal multiplier for a confidence level (0.95 -> 1.959963984540054)."""
    return inv_norm(0.5 + level / 2.0)


class Moments:
    """Count, mean and sum of squared deviations of a stream of numbers, built chunk by chunk (float64)."""

    def __init__(self, n=0, mean=0.0, m2=0.0):
        self.n, self.mean, self.m2 = int(n), float(mean), float(m2)

    def add(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size:
            m = float(x.mean())
            self.merge(Moments(x.size, m, float(((x - m) ** 2).sum())))   # two passes within a chunk: accurate
        return self

    def merge(self, other):
        if other.n:
            n = self.n + other.n
            d = other.mean - self.mean
            self.mean += d * other.n / n
            self.m2 += other.m2 + d * d * self.n * other.n / n
            self.n = n
        return self

    @property
    def var(self):
        return self.m2 / (self.n - 1) if self.n > 1 else float("nan")

    @property
    def se(self):
        return math.sqrt(self.var / self.n) if self.n > 1 else float("nan")

    def interval(self, level=0.95):
        h = z_for(level) * self.se
        return self.mean - h, self.mean + h


def wilson(hits, n, level=0.95):
    """(estimate, low, high) for a probability from `hits` successes in `n` draws (Wilson score interval)."""
    z = z_for(level)
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, center - half, center + half


def wald(hits, n, level=0.95):
    """(estimate, low, high): the textbook interval p +/- z sqrt(p(1-p)/n). Too narrow for rare events; see wilson()."""
    p = hits / n
    h = z_for(level) * math.sqrt(p * (1 - p) / n)
    return p, p - h, p + h


def quantile_interval(x, q, level=0.95, presorted=False):
    """(estimate, low, high) for the q-th percentile of the distribution behind sample x (distribution-free).

    The estimate is the ceil(nq)-th smallest draw. The number of draws below the true percentile is binomial(n, q);
    its normal approximation picks the order statistics that bound a `level` interval (n q (1 - q) should be large,
    say above 100)."""
    s = np.asarray(x, dtype=np.float64).ravel()
    n = s.size
    h = z_for(level) * math.sqrt(n * q * (1 - q))
    k = [min(max(math.floor(n * q - h) - 1, 0), n - 1), max(math.ceil(n * q) - 1, 0), min(max(math.ceil(n * q + h), 0), n - 1)]
    if not presorted:
        s = np.partition(s, k)                       # only these three order statistics are needed: linear time
    return s[k[1]], s[k[0]], s[k[2]]
