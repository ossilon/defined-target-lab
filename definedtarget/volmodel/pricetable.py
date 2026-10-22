"""Price tables: Heston call prices for any state (log-moneyness, variance) at one remaining maturity, fast enough for
a path kernel.

Plain words
- Repricing an option along a million paths cannot call a Fourier pricer at every step. Instead, for one remaining
  maturity tau, the table stores the model's implied vol on a grid of log-moneyness k = ln(K / F) and vol s = sqrt(v)
  (the state variance's square root), computed once by pricers.cos_call_surface.
- A call is then Black-76 at the interpolated implied vol: implied vols are smooth in k and s even near expiry,
  where prices bend sharply at the strike, so a local cubic (Catmull-Rom) interpolation of the vol is accurate where
  one of the price would need a much finer grid.
- Values are undiscounted and per unit forward: an option's value is DF(tau) F_t c(k_t, v_t). Below the strike range
  the call is its intrinsic value, above it 0 (the range defaults to k from -3 to 2, an index between 5% and 739% of
  the strike's forward); a vol above the grid's top is clamped to it. Where the model's time value is too small for
  an implied vol (far wings at short maturities) the vol of the nearest valid strike is used, which gives a time
  value no larger than that negligible one.
tests/test_vol_pricetable.py checks the table against direct COS prices and the martingale identity of the marked
option along simulated paths.
"""
import math
from dataclasses import dataclass, field

import numba
import numpy as np

from . import black, pricers


@dataclass(frozen=True)
class CallTable:
    """Implied vols iv[r, c] at vol s = r ds (r = 0 .. ns) and log-moneyness k = k0 + (c - 1) dk (c = 0 .. nk + 1):
    the grid of ns x nk points plus one ghost row above it and one ghost column on each side, so the cubic has its
    full stencil in every cell (first run, 2026-10-23: clamped stencils in the edge cells gave errors up to 4.6e-4).
    Below s = 0 the row is mirrored: the vol depends on v = s^2, so it is symmetric in s."""
    tau: float
    k0: float
    dk: float
    ds: float
    nk: int
    ns: int
    iv: np.ndarray = field(repr=False)


def build(model, tau, k_min=-3.0, k_max=2.0, nk=401, s_max=1.2, ns=61):
    """The table for `model` (its theta, kappa, eta and rho; v0 is not used) at remaining maturity tau (years)."""
    if not (isinstance(tau, (int, float, np.floating)) and math.isfinite(tau) and tau > 0):
        raise ValueError(f"a table needs a positive remaining maturity tau (years), got {tau!r}")
    if not (k_min < k_max and nk >= 4 and ns >= 4 and s_max > 0):
        raise ValueError("the grid needs k_min < k_max, s_max > 0 and at least 4 points on each axis")
    dk, ds = (k_max - k_min) / (nk - 1), s_max / (ns - 1)
    k = k_min + dk * np.arange(-1, nk + 1)                    # one ghost column on each side
    s = ds * np.arange(ns + 1)                                 # one ghost row above s_max
    calls = pricers.cos_call_surface(model, k, s * s, tau)
    K = np.broadcast_to(np.exp(k), calls.shape)
    iv = black.implied_vols(calls.ravel(), 1.0, K.ravel(), tau).reshape(calls.shape)
    for row in iv:                                      # fill wings without a vol from the nearest valid strike
        ok = np.flatnonzero(np.isfinite(row))
        if ok.size == 0:
            raise ArithmeticError(f"no implied vol on a whole row of the table (tau = {tau})")
        nearest = ok[np.clip(np.searchsorted(ok, np.arange(row.size)), 0, ok.size - 1)]
        left = np.clip(np.searchsorted(ok, np.arange(row.size)) - 1, 0, ok.size - 1)
        pick = np.where(np.abs(ok[left] - np.arange(row.size)) <= np.abs(nearest - np.arange(row.size)),
                        ok[left], nearest)
        row[:] = row[pick]
    return CallTable(tau=float(tau), k0=float(k_min), dk=float(dk), ds=float(ds), nk=nk, ns=ns, iv=iv)


@numba.njit(cache=True)
def _cubic(p0, p1, p2, p3, t):
    """Catmull-Rom cubic through p1 (t = 0) and p2 (t = 1), with tangents from the neighbors."""
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))


@numba.njit(parallel=True, cache=True)
def _calls(iv, k0, dk, ds, nk, ns, tau, k, v, out):
    kmax = k0 + dk * (nk - 1)
    for n in numba.prange(k.size):
        kn = k[n]
        if kn < k0:
            out[n] = 1.0 - math.exp(kn)                 # deep in the money: intrinsic
            continue
        if kn > kmax:
            out[n] = 0.0                                # far out of the money
            continue
        s = math.sqrt(v[n]) if v[n] > 0.0 else 0.0
        fs = min(s / ds, ns - 1.0)                      # a vol above the grid is clamped to its top
        fk = (kn - k0) / dk
        i = min(int(fs), ns - 2)
        j = min(int(fk), nk - 2)
        ts, tk = fs - i, fk - j
        rowv = np.empty(4)
        p = np.empty(4)
        for a in range(4):
            r = abs(i - 1 + a)                          # row -1 is row 1 (the vol is symmetric in s)
            for b in range(4):
                p[b] = iv[r, j + b]                     # columns j - 1 .. j + 2, stored with an offset of 1
            rowv[a] = _cubic(p[0], p[1], p[2], p[3], tk)
        sig = _cubic(rowv[0], rowv[1], rowv[2], rowv[3], ts)
        out[n] = black.call_price(1.0, math.exp(kn), 1.0, tau, sig)


def calls(table, k, v):
    """Undiscounted calls per unit forward at log-moneyness k and state variance v (arrays of equal length)."""
    k = np.ascontiguousarray(np.asarray(k, dtype=float).ravel())
    v = np.ascontiguousarray(np.asarray(v, dtype=float).ravel())
    if k.size != v.size:
        raise ValueError("k and v must have the same length")
    out = np.empty(k.size)
    _calls(table.iv, table.k0, table.dk, table.ds, table.nk, table.ns, table.tau, k, v, out)
    return out
