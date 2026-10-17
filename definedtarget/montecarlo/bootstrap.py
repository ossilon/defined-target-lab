"""The circular stationary bootstrap (Politis and Romano, 1994): futures built from blocks of real history.

Plain words
- History is a list of daily steps (for example log returns), one row per day; several columns can travel together
  (returns, rate changes, vol changes), because a row is always copied whole.
- A path starts on a random day. Each following step either moves to the next day of history (probability 1 - p) or
  jumps to a new random day (probability p). Runs of consecutive days therefore have random lengths with mean 1 / p
  (the "mean block length"), which keeps crashes, rebounds and calm spells in their real order.
- Circular: the day after the last day of history is the first one. With that wrap every day of history has the
  same chance of appearing at every step, so the bootstrap's average step equals history's average exactly (the
  property its error-bar and control-variate formulas rely on). The wrap joins the end of history to its start; it
  is harmless for returns and changes, but a column of *levels* (a rate level) should be turned into changes first.
- equal_moments() shifts and scales a column so its mean and spread equal chosen values: a bootstrap and a
  lognormal with the same mean and spread then differ only by history's *shape* (fat tails, clustering, order).
- Random numbers come from montecarlo.rng: step k of a path uses one Philox call at draw-pair k (the first uniform
  decides jump or continue, the second picks the new day), so paths replay exactly on any thread count or chunk.

Measures (one pass, causal) are those of montecarlo.paths for column 0, plus each column's total over the path.
Judged in tests/test_mc_bootstrap.py, including the exact variance formula of the bootstrap's sum.
"""
import numba
import numpy as np

from .rng import _address, _check, philox4x32, u52

NO_BARRIER = np.inf


@numba.njit(cache=True)
def _step(seed, path, k, stream, idx, n, p):
    """The history row used at step k, given the row used at step k - 1 (idx; ignored at k = 0)."""
    c0, c1, c2, c3, k0, k1 = _address(seed, path, k, stream)
    w0, w1, w2, w3 = philox4x32(c0, c1, c2, c3, k0, k1)
    if k == 0 or u52(w0, w1) < p:
        return int(u52(w2, w3) * n)                  # a new block starts on a uniformly chosen day
    idx += 1
    return 0 if idx == n else idx                    # circular: after the last day comes the first


@numba.njit(parallel=True, cache=True)
def _indices(seed, path0, npaths, nsteps, n, p, stream, out):
    for i in numba.prange(npaths):
        path = path0 + np.uint64(i)
        idx = 0
        for k in range(nsteps):
            idx = _step(seed, path, k, stream, idx, n, p)
            out[i, k] = idx


@numba.njit(parallel=True, cache=True)
def _paths(seed, path0, npaths, nsteps, data, p, barrier, check, stream,
           log_end, log_max, dd_end, mdd, hit, at_check, col_end):
    n, ncols = data.shape
    for i in numba.prange(npaths):
        path = path0 + np.uint64(i)
        idx = 0
        x = 0.0
        top = 0.0
        worst = 0.0
        first = -1
        j = 0
        for c in range(ncols):
            col_end[i, c] = 0.0
        for k in range(nsteps):
            idx = _step(seed, path, k, stream, idx, n, p)
            x += data[idx, 0]
            for c in range(ncols):
                col_end[i, c] += data[idx, c]
            if first < 0 and x >= barrier:
                first = k
            if x > top:
                top = x
            if top - x > worst:
                worst = top - x
            while j < check.size and check[j] == k + 1:
                at_check[i, j] = x
                j += 1
        log_end[i] = x
        log_max[i] = top
        dd_end[i] = top - x
        mdd[i] = worst
        hit[i] = first


def _prep(data, mean_block):
    d = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2 or d.shape[0] < 2 or not np.isfinite(d).all():
        raise ValueError("history must be a 1-D or 2-D array of finite numbers with at least 2 rows")
    if not mean_block >= 1.0:
        raise ValueError("mean_block (the mean run length, in steps) must be at least 1")
    return d, 1.0 / mean_block


def indices(seed, npaths, nsteps, nrows, mean_block, path0=0, stream=0):
    """(npaths, nsteps) history rows chosen by the bootstrap (the same choices bootstrap_paths makes)."""
    _check(seed, path0, npaths, nsteps, stream)
    _, p = _prep(np.zeros((max(nrows, 2), 1)), mean_block)
    out = np.empty((npaths, nsteps), dtype=np.int64)
    _indices(np.uint64(seed), np.uint64(path0), npaths, nsteps, nrows, p, np.uint64(stream), out)
    return out


def paths(seed, npaths, nsteps, history, mean_block, barrier=None, checkpoints=(), path0=0, stream=0):
    """Bootstrap npaths paths of nsteps steps from `history` (rows = days; column 0 = the log price step).

    Returns the measures of montecarlo.paths.lognormal for column 0 (log_end, log_max, log_drawdown_end,
    log_max_drawdown, hit_step, at_checkpoint; no p_no_touch: between closes there is no model) and 'column_end',
    each column's total over the path (npaths x columns)."""
    d, p = _prep(history, mean_block)
    _check(seed, path0, npaths, nsteps, stream)
    if nsteps < 1:
        raise ValueError("nsteps must be at least 1")
    check = np.asarray(sorted(checkpoints), dtype=np.int64)
    if check.size and (check[0] < 1 or check[-1] > nsteps):
        raise ValueError("checkpoints are step numbers from 1 to nsteps")
    b = NO_BARRIER if barrier is None else float(np.log(barrier))
    out = {k: np.empty(npaths) for k in ("log_end", "log_max", "log_drawdown_end", "log_max_drawdown")}
    out["hit_step"] = np.empty(npaths, dtype=np.int64)
    out["at_checkpoint"] = np.full((npaths, check.size), np.nan)
    out["column_end"] = np.empty((npaths, d.shape[1]))
    _paths(np.uint64(seed), np.uint64(path0), npaths, nsteps, d, p, b, check, np.uint64(stream),
           out["log_end"], out["log_max"], out["log_drawdown_end"], out["log_max_drawdown"], out["hit_step"],
           out["at_checkpoint"], out["column_end"])
    return out


def equal_moments(steps, mean, sd):
    """The series shifted and scaled so that its mean is `mean` and its spread (population sd) is `sd`, exactly:
    a uniformly drawn step then has the chosen mean and variance, like the lognormal's step with the same inputs."""
    x = np.asarray(steps, dtype=np.float64)
    c = x - x.mean()
    return c * (sd / c.std()) + mean


def sum_variance(history_column, nsteps, mean_block):
    """The exact variance of the bootstrap's sum of nsteps steps (circular stationary bootstrap):
    n g(0) + 2 sum_k (n - k) (1 - p)^k g(k), with g the circular autocovariance of the history column."""
    x = np.asarray(history_column, dtype=np.float64)
    c = x - x.mean()
    n, p = c.size, 1.0 / mean_block
    lags = np.arange(1, nsteps)
    g = np.array([float((c * np.roll(c, -k)).mean()) for k in range(nsteps)])
    return float(nsteps * g[0] + 2.0 * ((nsteps - lags) * (1.0 - p) ** lags * g[1:]).sum())
