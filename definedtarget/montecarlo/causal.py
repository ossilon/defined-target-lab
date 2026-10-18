"""Look-ahead checks: a rule may use only what was known when it decided.

Plain words
- A "rule" here is any function that reads a series (rows = sessions, oldest first) and returns one output per
  session: a signal, an exposure, a running measure. Output k is the decision made at session k's close, so it may
  depend on rows 0..k and nothing later.
- `check` tests that from outside (it cannot prove it; audit of v0.3.0, D11): at cut points k, it replaces everything after k with
  different futures and requires outputs 0..k to stay identical, bit for bit. A rule that reads even one future row
  changes some output at or before its cut, and the check reports where. An empty result means none was found at
  the cuts and futures tried: a rule can still peek at an untried cut or for a future unlike all eight.
- The futures (FUTURES): the rest reversed, shuffled, flat (the last known row repeated), a crash and a spike (ten
  spreads below and above the mean), unrelated noise, no future at all (the series ends at k) and a longer one
  (twice the rows). Harsh futures catch rules that peek only when something extreme is coming; the two lengths
  catch rules that read how many rows there are; every cut point (the default for series up to 1,000 rows)
  catches rules that peek only on some dates, such as month-ends.
- What it does NOT check: execution timing (a decision at k's close may only earn returns from k + 1 on), parameters
  fitted on the whole history and then fixed (the rule then looks causal), and data that was revised or published
  late (the check sees the final series). The causal backtest toolkit (v0.3.2 design) covers those.
- numpy only, like the rest of the engine.
"""
import numpy as np

FUTURES = ("reversed", "shuffled", "flat", "crash", "spike", "noise", "none", "longer")
ALL_CUTS_UP_TO = 1000            # series up to this many rows get every cut point by default
SPREAD_CUTS = 100                # longer series: this many cut points spread evenly, plus the first and last


def _futures(series, k, rng):
    """The futures after row k, each the same length as the rest of the series."""
    head, tail = series[:k + 1], series[k + 1:]
    x = series.astype(float)
    mean, sd = x.mean(axis=0), x.std(axis=0) + 1e-12
    shape, dt = tail.shape, series.dtype
    out = {"reversed": tail[::-1].copy(),
           "shuffled": tail[rng.permutation(len(tail))],
           "flat": np.repeat(series[k:k + 1], len(tail), axis=0),
           "crash": np.broadcast_to(mean - 10.0 * sd, shape).astype(dt),
           "spike": np.broadcast_to(mean + 10.0 * sd, shape).astype(dt),
           "noise": (rng.standard_normal(shape) * (sd + 1.0) + mean).astype(dt),
           # different lengths: the series ends at k, or runs on for twice as many rows (catches a rule that reads
           # how many rows there are, which is also the future; audit of v0.3)
           "none": tail[:0],
           "longer": np.concatenate([tail, tail[::-1]])}
    return {name: np.concatenate([head, out[name]]) for name in FUTURES}


def _first_change(base, out, k):
    """The first output index 0..k that differs (NaN equals NaN), or -1 if outputs 0..k are identical."""
    a, b = base[:k + 1], out[:k + 1]
    if a.shape != b.shape:
        return 0
    same = a == b
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        same |= np.isnan(a) & np.isnan(b)
    rows = same.reshape(k + 1, -1).all(axis=1)
    return -1 if rows.all() else int(np.argmin(rows))


def default_cuts(n):
    if n <= ALL_CUTS_UP_TO:
        return list(range(n - 1))
    return sorted({0, n - 2, *np.linspace(1, n - 2, SPREAD_CUTS).astype(int).tolist()})


def check(rule, series, cuts=None, seed=0):
    """[(cut, first output index that changed, future)] for every cut at which the rule read the future; [] if
    causal.

    series: 1-D or 2-D array (rows = sessions). cuts: rows after which the future is replaced (default
    default_cuts: every row for series up to 1,000 rows, else 100 spread evenly). The rule must return one output
    per row (arrays with any number of columns) and accept a series of any length from 1 row up (the checker also
    runs it on shorter and longer series); a rule that needs some history should return NaN until it has it."""
    x = np.asarray(series)
    n = x.shape[0]
    if n < 3:
        raise ValueError("the series needs at least 3 rows")
    base = np.asarray(rule(x))
    if base.shape[0] != n:
        raise ValueError(f"the rule returned {base.shape[0]} outputs for {n} rows: it must return one per row")
    rng = np.random.default_rng(seed)
    bad = []
    for k in (default_cuts(n) if cuts is None else cuts):
        if not 0 <= k < n - 1:
            raise ValueError(f"cut {k} outside 0..{n - 2}")
        for name, future in _futures(x, k, rng).items():
            try:
                out = np.asarray(rule(future))
            except Exception as err:                          # noqa: BLE001  (say why, then stop)
                raise ValueError(f"the rule failed on a {len(future)}-row series (future '{name}' after cut {k}): "
                                 f"{type(err).__name__}: {err}. Rules must accept any length from 1 row up; "
                                 "return NaN until enough history exists") from err
            first = _first_change(base, out, k)
            if first >= 0:
                bad.append((int(k), first, name))
                break
    return bad
