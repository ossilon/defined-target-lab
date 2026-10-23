"""Heston paths: the index and its variance simulated together, on the Monte Carlo engine's random-number streams.

Plain words
- Each step moves the variance v and then the log index x, on any increasing time grid (the real NYSE calendar
  included). Only a few numbers per path are kept, as in montecarlo.paths: the end log level and variance, the
  integrated variance, the highest close, the drawdowns, the first close at or above a barrier, and the log level
  and the variance at chosen checkpoint steps.
- The default scheme is Andersen's (2008) quadratic-exponential scheme with its martingale correction ("qe-m"):
  1. The next variance matches the exact conditional mean m and variance s^2 of the square-root process (his eq.
     17-18). With psi = s^2 / m^2 at most psi_c = 1.5 it is a (b + Z)^2 with a normal Z (eq. 23, 27-28); above it,
     it is 0 with probability p and exponential otherwise (eq. 24-26, 29-30), which handles the mass near 0 that
     breaks Euler schemes when the Feller condition fails.
  2. The log index uses the exact relation between the index and the variance increments (eq. 11), with the time
     integral of v by the trapezoid rule (eq. 32-33, gamma_1 = gamma_2 = 1/2).
  3. The martingale correction replaces the constant K0 so that E[S_{t+dt} | S_t] = S_t e^{mu dt} exactly (his
     Propositions 7 and 9). It needs A < 1/(2a) (quadratic branch) or A < beta (exponential branch), always true for
     rho <= 0; for rho > 0 and large steps the simulation refuses instead of returning biased paths.
  "qe" is the same without the correction; "euler" is the full-truncation Euler scheme (his eq. 6-7), kept for
  comparison only.
- eta = 0 is the lognormal limit: the variance follows its mean path, theta + (v0 - theta) e^{-kappa t}, and each
  step is lognormal with that step's mean variance. With v0 = theta it is montecarlo.paths.lognormal, number for
  number (tests/test_vol_stochvol.py checks this bit for bit).

Random numbers: draw k of a path on stream s drives the index at step k, exactly as in montecarlo.paths (so the
lognormal limit reuses the same numbers); the variance uses stream s + 1. Every draw has an address (seed, path, draw,
stream), so the paths do not depend on threads or chunks. Imports: numpy, numba and montecarlo only.
"""
import math

import numba
import numpy as np

from ..montecarlo.paths import NO_BARRIER
from ..montecarlo.rng import _address, _check, inv_norm, philox4x32, u52

SCHEMES = ("qe-m", "qe", "euler")
PSI_C = 1.5
_CODE = {"qe-m": 0, "qe": 1, "euler": 2}
_LOGNORMAL = 3


@numba.njit(parallel=True, cache=True)
def _heston(seed, path0, npaths, code, v0, theta, kappa, eps, rho, psi_c, dt, mudt, ek, c1, c2, K0, K1, K2, K3, K4, A,
            ln_drift, ln_sd, ln_v, ln_w, barrier, check, stream,
            log_end, v_end, int_var, log_max, dd_end, mdd, hit, at_check, v_check, bad):
    nsteps = dt.size
    sv = stream + np.uint64(1)
    for i in numba.prange(npaths):
        path = path0 + np.uint64(i)
        x, v, w = 0.0, v0, 0.0
        top, worst = 0.0, 0.0
        first, j, nbad = -1, 0, 0
        for blk in range((nsteps + 1) // 2):
            a0, a1, a2, a3, ka, kb = _address(seed, path, blk, stream)
            x0, x1, x2, x3 = philox4x32(a0, a1, a2, a3, ka, kb)          # index draws (as montecarlo.paths)
            if code == 3:                                                # lognormal limit: no variance draws
                y0, y1, y2, y3 = x0, x1, x2, x3                          # (never read; code review, 2026-10-23)
            else:
                b0, b1, b2_, b3, kc, kd = _address(seed, path, blk, sv)
                y0, y1, y2, y3 = philox4x32(b0, b1, b2_, b3, kc, kd)    # variance draws (stream + 1)
            for half in range(2):
                k = 2 * blk + half
                if k >= nsteps:
                    break
                z = inv_norm(u52(x0, x1) if half == 0 else u52(x2, x3))
                if code == 3:                                            # lognormal limit (eta = 0)
                    x_new = x + ln_drift[k] + ln_sd[k] * z
                    w += ln_w[k]
                    v = ln_v[k]
                else:
                    uv = u52(y0, y1) if half == 0 else u52(y2, y3)
                    if code == 2:                                        # full-truncation Euler (comparison only)
                        vp = v if v > 0.0 else 0.0
                        zv = inv_norm(uv)
                        sq = math.sqrt(vp * dt[k])
                        x_new = x + mudt[k] - 0.5 * vp * dt[k] + sq * (rho * zv + math.sqrt(1.0 - rho * rho) * z)
                        w += vp * dt[k]
                        v = v + kappa * (theta - vp) * dt[k] + eps * sq * zv
                    else:                                                # QE, with or without the correction
                        m = theta + (v - theta) * ek[k]
                        psi = (v * c1[k] + c2[k]) / (m * m)
                        k0 = K0[k]
                        if psi <= psi_c:
                            inv = 2.0 / psi
                            bb = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
                            aa = m / (1.0 + bb)
                            t = math.sqrt(bb) + inv_norm(uv)
                            vn = aa * t * t
                            if code == 0:
                                q = 1.0 - 2.0 * A[k] * aa
                                if q > 0.0:
                                    k0 = -A[k] * bb * aa / q + 0.5 * math.log(q) - (K1[k] + 0.5 * K3[k]) * v
                                else:
                                    nbad += 1
                        else:
                            p = (psi - 1.0) / (psi + 1.0)
                            beta = (1.0 - p) / m
                            vn = 0.0 if uv <= p else math.log((1.0 - p) / (1.0 - uv)) / beta
                            if code == 0:
                                if A[k] < beta:
                                    k0 = -math.log(p + beta * (1.0 - p) / (beta - A[k])) - (K1[k] + 0.5 * K3[k]) * v
                                else:
                                    nbad += 1
                        x_new = x + mudt[k] + k0 + K1[k] * v + K2[k] * vn + math.sqrt(K3[k] * v + K4[k] * vn) * z
                        w += 0.5 * dt[k] * (v + vn)
                        v = vn
                if first < 0 and barrier < NO_BARRIER and x_new >= barrier:
                    first = k
                x = x_new
                if x > top:
                    top = x
                if top - x > worst:
                    worst = top - x
                while j < check.size and check[j] == k + 1:
                    at_check[i, j] = x
                    v_check[i, j] = v
                    j += 1
        log_end[i] = x
        v_end[i] = v
        int_var[i] = w
        log_max[i] = top
        dd_end[i] = top - x
        mdd[i] = worst
        hit[i] = first
        bad[i] = nbad


def _number(name, v, low=None, strict=True):
    if not isinstance(v, (int, float, np.floating)) or isinstance(v, bool) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if low is not None and (v <= low if strict else v < low):
        raise ValueError(f"{name} must be {'above' if strict else 'at least'} {low}, got {v}")
    return float(v)


def simulate(seed, npaths, times, v0, theta, kappa, eta, rho, mu=0.0, scheme="qe-m", barrier=None, checkpoints=(),
             path0=0, stream=0, psi_c=PSI_C):
    """Simulate npaths Heston paths on `times` (year fractions from 0) and return their measures (a dict of arrays):
    log_end, v_end, int_var (integrated variance), log_max, log_drawdown_end, log_max_drawdown, hit_step (-1 if the
    barrier, a multiple of the start level, is never reached at a close), at_checkpoint and v_at_checkpoint
    (npaths x len(checkpoints)). mu: E[S_t] = S_0 e^{mu t} (mu = r - q under the pricing measure; 0 in forward terms).
    eta = 0 gives the lognormal limit (see the module docstring)."""
    v0 = _number("v0 (variance now)", v0, 0.0, strict=False)
    theta = _number("theta (long-run variance)", theta, 0.0)
    kappa = _number("kappa (mean-reversion speed)", kappa, 0.0)
    eps = _number("eta (vol of variance)", eta, 0.0, strict=False)
    rho = _number("rho (correlation)", rho)
    mu = _number("mu (drift)", mu)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must lie strictly between -1 and 1, got {rho}")
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if not (isinstance(psi_c, (int, float)) and 1.0 <= psi_c <= 2.0):
        raise ValueError(f"psi_c must lie in [1, 2] (Andersen 2008, section 3.2.3), got {psi_c!r}")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or t[0] != 0.0 or not (np.diff(t) > 0).all():
        raise ValueError("times must be finite, start at 0 and increase strictly")
    nsteps = t.size - 1
    if not (isinstance(stream, (int, np.integer)) and 0 <= stream < 2 ** 32 - 1):
        raise ValueError("stream must be a whole number below 2^32 - 1 (the variance uses stream + 1)")
    _check(seed, path0, npaths, nsteps, stream)
    _check(seed, path0, npaths, nsteps, stream + 1)
    if not all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in checkpoints):
        raise ValueError("checkpoints are whole step numbers (int)")
    check = np.asarray(sorted(checkpoints), dtype=np.int64)
    if check.size and (check[0] < 1 or check[-1] > nsteps):
        raise ValueError("checkpoints are step numbers from 1 to nsteps")
    if barrier is not None and not (isinstance(barrier, (int, float)) and math.isfinite(barrier) and barrier > 0):
        raise ValueError("barrier must be a positive, finite multiple of the start level (or None)")
    b = NO_BARRIER if barrier is None else math.log(barrier)

    dt = np.diff(t)
    ek = np.exp(-kappa * dt)
    zeros = np.zeros(nsteps)
    if eps > 0.0:
        code = _CODE[scheme]
        c1 = eps * eps * ek * (1.0 - ek) / kappa
        c2 = theta * eps * eps * (1.0 - ek) ** 2 / (2.0 * kappa)
        g1 = g2 = 0.5                                                    # central (trapezoid) integral of v
        K0 = -rho * kappa * theta * dt / eps
        K1 = g1 * dt * (kappa * rho / eps - 0.5) - rho / eps
        K2 = g2 * dt * (kappa * rho / eps - 0.5) + rho / eps
        K3 = g1 * dt * (1.0 - rho * rho)
        K4 = g2 * dt * (1.0 - rho * rho)
        A = K2 + 0.5 * K4
        ln_drift = ln_sd = ln_v = ln_w = zeros
    else:
        code = _LOGNORMAL
        c1 = c2 = K0 = K1 = K2 = K3 = K4 = A = zeros
        e0, e1 = np.exp(-kappa * t[:-1]), np.exp(-kappa * t[1:])
        vbar = theta + (v0 - theta) * (e0 - e1) / (kappa * dt)          # mean variance over each step (exact)
        vol = np.sqrt(vbar)
        ln_drift = (mu - 0.5 * vol * vol) * dt                           # the same arithmetic as montecarlo.paths
        ln_sd = vol * np.sqrt(dt)
        ln_v = theta + (v0 - theta) * e1
        ln_w = vbar * dt
    out = {k: np.empty(npaths) for k in ("log_end", "v_end", "int_var", "log_max", "log_drawdown_end",
                                         "log_max_drawdown")}
    out["hit_step"] = np.empty(npaths, dtype=np.int64)
    out["at_checkpoint"] = np.full((npaths, check.size), np.nan)
    out["v_at_checkpoint"] = np.full((npaths, check.size), np.nan)
    bad = np.zeros(npaths, dtype=np.int64)
    _heston(np.uint64(seed), np.uint64(path0), npaths, code, v0, theta, kappa, eps, rho, float(psi_c), dt, mu * dt,
            ek, c1, c2, K0, K1, K2, K3, K4, A, ln_drift, ln_sd, ln_v, ln_w, b, check, np.uint64(stream),
            out["log_end"], out["v_end"], out["int_var"], out["log_max"], out["log_drawdown_end"],
            out["log_max_drawdown"], out["hit_step"], out["at_checkpoint"], out["v_at_checkpoint"], bad)
    nbad = int(bad.sum())
    if nbad:
        raise ArithmeticError(f"the martingale correction does not exist on {nbad} path steps (Andersen 2008, "
                              "Proposition 9: rho > 0 with steps too large); use smaller steps, or scheme='qe' "
                              "without the correction")
    return out


def simulate_model(model, seed, npaths, times, **kwargs):
    """simulate() with the parameters of a heston.Heston model."""
    return simulate(seed, npaths, times, model.v0, model.theta, model.kappa, model.eta, model.rho, **kwargs)
