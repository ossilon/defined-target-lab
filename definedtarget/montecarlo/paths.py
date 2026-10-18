"""Paths: the lognormal (geometric Brownian motion) on any time grid, with every path statistic measured on the way.

Plain words
- A path is one simulated future, close by close. For the lognormal, log S moves each step by
  (mu - vol^2 / 2) dt + vol sqrt(dt) Z, with Z a standard normal from montecarlo.rng (draw k of a path = step k).
- The time grid can be anything increasing: equal steps, or the real trading calendar (times_from_dates turns session
  dates into year fractions, so a long weekend is a longer step). The end distribution is exact for any grid.
- Measures are made in one pass, step by step, using only the path so far (causal by construction):
  log_end (log S_T / S_0), log_max (highest close, start included), log_drawdown_end (fall from the highest close to
  the last), log_max_drawdown (largest fall from any earlier high), hit_step (first step whose close reaches the
  barrier, -1 if none), p_no_touch (the exact chance that the continuous path never touched the barrier between the
  closes, from the Brownian bridge), and the log level at chosen checkpoint steps.
- Paths are stored as these few numbers each, never as whole paths.

Judged in tests/test_mc_paths.py by exact formulas (Spitzer's identity, the reflection principle, the martingale
property) and two approximations with stated allowances. This module imports only numpy, numba and montecarlo.rng.
"""
import math

import numba
import numpy as np

from .rng import _address, _check, inv_norm, philox4x32, u52

NO_BARRIER = np.inf


@numba.njit(parallel=True, cache=True)
def _lognormal(seed, path0, npaths, drift, vsd, barrier, check, stream,
               log_end, log_max, dd_end, mdd, hit, p_no, at_check):
    nsteps = drift.size
    for i in numba.prange(npaths):
        path = path0 + np.uint64(i)
        x = 0.0
        top = 0.0
        worst = 0.0
        first = -1
        alive = 1.0 if barrier > 0.0 else 0.0           # chance of no touch so far (a barrier at or below 0 is hit)
        j = 0
        for b in range((nsteps + 1) // 2):
            c0, c1, c2, c3, k0, k1 = _address(seed, path, b, stream)
            w0, w1, w2, w3 = philox4x32(c0, c1, c2, c3, k0, k1)
            for half in range(2):
                k = 2 * b + half
                if k >= nsteps:
                    break
                z = inv_norm(u52(w0, w1) if half == 0 else u52(w2, w3))
                x_new = x + drift[k] + vsd[k] * z
                if barrier < NO_BARRIER:
                    if first < 0 and x_new >= barrier:
                        first = k
                    if alive > 0.0:
                        if x_new >= barrier:
                            alive = 0.0
                        else:                           # Brownian bridge: chance the path crossed between closes
                            alive *= 1.0 - math.exp(-2.0 * (barrier - x) * (barrier - x_new) / (vsd[k] * vsd[k]))
                x = x_new
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
        p_no[i] = alive


def times_from_dates(dates, basis=365.0):
    """Year fractions (ACT/basis) of a list of increasing dates from the first one: the time grid of a calendar."""
    d0 = dates[0]
    t = np.array([(d - d0).days / basis for d in dates], dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("dates must be strictly increasing")
    return t


def prepare(seed, npaths, times, mu, vol, barrier=None, checkpoints=(), path0=0, stream=0):
    """Checked kernel inputs (step drifts, step spreads, checkpoint steps, log barrier); shared with montecarlo.gpu so
    the CPU and GPU kernels receive identical numbers."""
    t = np.asarray(times, dtype=np.float64)
    # "not x > 0" rather than "x <= 0", so NaN is refused too (audit: NaN inputs used to return all-NaN measures)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or t[0] != 0.0 or not (np.diff(t) > 0).all():
        raise ValueError("times must be finite, start at 0 and increase strictly")
    if not (math.isfinite(vol) and vol > 0):
        raise ValueError("vol must be positive and finite")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    nsteps = t.size - 1
    _check(seed, path0, npaths, nsteps, stream)
    dt = np.diff(t)
    check = np.asarray(sorted(checkpoints), dtype=np.int64)
    if check.size and (check[0] < 1 or check[-1] > nsteps):
        raise ValueError("checkpoints are step numbers from 1 to nsteps")
    b = NO_BARRIER if barrier is None else math.log(barrier)
    return (mu - 0.5 * vol * vol) * dt, vol * np.sqrt(dt), check, b


def lognormal(seed, npaths, times, mu, vol, barrier=None, checkpoints=(), path0=0, stream=0):
    """Simulate npaths lognormal paths on `times` (year fractions, starting at 0) and return their measures.

    mu: the log-price drift is mu - vol^2 / 2 (so E[S_t] = S_0 exp(mu t)); vol: annual volatility.
    barrier: a level as a multiple of S_0 (e.g. 1.28) for hit_step and p_no_touch; None for no barrier.
    checkpoints: step numbers k (1 .. nsteps) whose log level is returned in 'at_checkpoint' (npaths x len)."""
    drift, vsd, check, b = prepare(seed, npaths, times, mu, vol, barrier, checkpoints, path0, stream)
    out = {k: np.empty(npaths) for k in ("log_end", "log_max", "log_drawdown_end", "log_max_drawdown", "p_no_touch")}
    out["hit_step"] = np.empty(npaths, dtype=np.int64)
    out["at_checkpoint"] = np.full((npaths, check.size), np.nan)
    _lognormal(np.uint64(seed), np.uint64(path0), npaths, drift, vsd, b, check, np.uint64(stream),
               out["log_end"], out["log_max"], out["log_drawdown_end"], out["log_max_drawdown"], out["hit_step"],
               out["p_no_touch"], out["at_checkpoint"])
    return out
