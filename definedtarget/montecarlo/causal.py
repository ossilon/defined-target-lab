"""Look-ahead checks: a rule may use only what was known when it decided.

Plain words
- A "rule" here is any function that reads a series (rows = sessions, oldest first) and returns one output per
  session: a signal, an exposure, a running measure. Output k is the decision made at session k's close, so it may
  depend on rows 0..k and nothing later.
- `check` proves that the way it can be proven from outside: at many cut points k, it replaces everything after k
  with different data (the future reversed, and the future replaced by unrelated numbers) and requires outputs
  0..k to stay identical, bit for bit. A rule that reads even one future row changes some output at or before its
  cut, and the check reports where.
- It does not check execution timing (a decision at k's close may only earn returns from k + 1 on); that is the
  backtest's job: apply exposure[k] to return[k + 1].
- numpy only, like the rest of the engine.
"""
import numpy as np


def _futures(series, k, rng):
    """Two different futures after row k: the rest reversed, and unrelated values with the same spread."""
    head, tail = series[:k + 1], series[k + 1:]
    reversed_tail = tail[::-1].copy()
    noise = rng.standard_normal(tail.shape) * (np.std(series) + 1.0) + np.mean(series)
    return [np.concatenate([head, reversed_tail]), np.concatenate([head, noise.astype(series.dtype)])]


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    floats = a.dtype.kind in "fc" and b.dtype.kind in "fc"       # NaN equals NaN only for float outputs
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=floats)


def check(rule, series, cuts=None, seed=0):
    """[(cut, first output index that changed)] for every cut at which the rule read the future; [] if causal.

    series: 1-D or 2-D array (rows = sessions). cuts: rows after which the future is replaced (default: about 25
    points spread over the series, plus the first and the second-to-last rows). The rule must return one output per
    row (arrays with any number of columns)."""
    x = np.asarray(series)
    n = x.shape[0]
    if n < 3:
        raise ValueError("the series needs at least 3 rows")
    base = np.asarray(rule(x))
    if base.shape[0] != n:
        raise ValueError(f"the rule returned {base.shape[0]} outputs for {n} rows: it must return one per row")
    if cuts is None:
        cuts = sorted({0, n - 2, *np.linspace(1, n - 2, 25).astype(int).tolist()})
    rng = np.random.default_rng(seed)
    bad = []
    for k in cuts:
        if not 0 <= k < n - 1:
            raise ValueError(f"cut {k} outside 0..{n - 2}")
        for future in _futures(x, k, rng):
            out = np.asarray(rule(future))
            same = np.array([_same(base[i], out[i]) for i in range(k + 1)])
            if not same.all():
                bad.append((int(k), int(np.argmin(same))))
                break
    return bad
