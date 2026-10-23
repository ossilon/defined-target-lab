"""Price tables: Heston call prices for any state (log-moneyness, variance) at one remaining maturity, fast enough for
a path kernel.

Plain words
- Repricing an option along a million paths cannot call a Fourier pricer at every step. Instead, for one remaining
  maturity tau, the table stores the model's implied vol on a grid of log-moneyness k = ln(K / F) and vol s = sqrt(v)
  (the state variance's square root), computed once by pricers.cos_call_surface.
- A call is then Black-76 at the interpolated implied vol: implied vols are smooth in k and s even near expiry,
  where prices bend sharply at the strike, so a local cubic (Catmull-Rom) interpolation of the vol is accurate where
  one of the price would need a much finer grid.
- Values are undiscounted and per unit forward: an option's value is DF(tau) F_t c(k_t, v_t).
- The grid defaults to k from -3 to 3 (an index between 5% and 2,000% of the strike's forward, so a fall of 95%
  stays inside) and vols up to 200%. A state outside the grid is priced exactly by the COS pricer instead, one by
  one; calls(..., return_outside=True) reports how many. (The first version returned the intrinsic value below the
  range, 0 above it and the top row's vol above 120%: a code review on 2026-10-23 found that this undervalues crash
  states, where the index has fallen far and the variance is high.)
- Where the model's time value is too small for an implied vol (far wings at short maturities) the vol of the
  nearest valid strike is used, which gives a time value no larger than that negligible one.
tests/test_vol_pricetable.py checks the table against direct COS prices, the exact prices outside the grid, and the
martingale identity of the marked option along simulated paths.
"""
import math
from dataclasses import dataclass, field

import numba
import numpy as np

from . import black, pricers
from .heston import Heston


@dataclass(frozen=True)
class CallTable:
    """Implied vols iv[r, c] at vol s = r ds (r = 0 .. ns) and log-moneyness k = k0 + (c - 1) dk (c = 0 .. nk + 1):
    the grid of ns x nk points plus one ghost row above it and one ghost column on each side, so the cubic has its
    full stencil in every cell (first run, 2026-10-23: clamped stencils in the edge cells gave errors up to 4.6e-4).
    Below s = 0 the row is mirrored: the vol depends on v = s^2, so it is symmetric in s. `params` holds theta,
    kappa, eta and rho, for the exact prices outside the grid."""
    tau: float
    k0: float
    dk: float
    ds: float
    nk: int
    ns: int
    params: tuple
    iv: np.ndarray = field(repr=False)

    @property
    def k_max(self):
        return self.k0 + self.dk * (self.nk - 1)

    @property
    def s_max(self):
        return self.ds * (self.ns - 1)


def build(model, tau, k_min=-3.0, k_max=3.0, nk=481, s_max=2.0, ns=101):
    """The table for `model` (its theta, kappa, eta and rho; v0 is not used) at remaining maturity tau (years).
    The defaults keep the grid steps of the first version (0.0125 in k, 0.02 in vol)."""
    if not (isinstance(tau, (int, float, np.floating)) and math.isfinite(tau) and tau > 0):
        raise ValueError(f"a table needs a positive remaining maturity tau (years), got {tau!r}")
    if not (k_min < k_max and nk >= 4 and ns >= 4 and s_max > 0):
        raise ValueError("the grid needs k_min < k_max, s_max > 0 and at least 4 points on each axis")
    dk, ds = (k_max - k_min) / (nk - 1), s_max / (ns - 1)
    k = k_min + dk * np.arange(-1, nk + 1)                    # one ghost column on each side
    s = ds * np.arange(ns + 1)                                 # one ghost row above s_max
    calls_ = pricers.cos_call_surface(model, k, s * s, tau)
    K = np.broadcast_to(np.exp(k), calls_.shape)
    iv = black.implied_vols(calls_.ravel(), 1.0, K.ravel(), tau).reshape(calls_.shape)
    idx = np.arange(iv.shape[1])
    for row in iv:                                      # fill wings without a vol from the nearest valid strike
        ok = np.flatnonzero(np.isfinite(row))
        if ok.size == 0:
            raise ArithmeticError(f"no implied vol on a whole row of the table (tau = {tau})")
        right = ok[np.clip(np.searchsorted(ok, idx), 0, ok.size - 1)]
        left = ok[np.clip(np.searchsorted(ok, idx) - 1, 0, ok.size - 1)]
        row[:] = row[np.where(np.abs(left - idx) <= np.abs(right - idx), left, right)]
    return CallTable(tau=float(tau), k0=float(k_min), dk=float(dk), ds=float(ds), nk=nk, ns=ns,
                     params=(model.theta, model.kappa, model.eta, model.rho), iv=iv)


@numba.njit(cache=True)
def _cubic(p0, p1, p2, p3, t):
    """Catmull-Rom cubic through p1 (t = 0) and p2 (t = 1), with tangents from the neighbors."""
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))


@numba.njit(cache=True)
def _row(iv, r, j, tk):
    """The cubic along k in row r (row -1 is row 1: the vol is symmetric in s), over stored columns j .. j + 3."""
    r = abs(r)
    return _cubic(iv[r, j], iv[r, j + 1], iv[r, j + 2], iv[r, j + 3], tk)


@numba.njit(parallel=True, cache=True)
def _calls(iv, k0, dk, ds, nk, ns, tau, k, v, out):
    """Table prices; NaN for a state outside the grid (k outside [k0, k_max] or vol above the top row)."""
    kmax = k0 + dk * (nk - 1)
    smax = ds * (ns - 1)
    for n in numba.prange(k.size):
        kn = k[n]
        s = math.sqrt(v[n]) if v[n] > 0.0 else 0.0
        if kn < k0 or kn > kmax or s > smax:
            out[n] = np.nan
            continue
        fs = s / ds
        fk = (kn - k0) / dk
        i = min(int(fs), ns - 2)
        j = min(int(fk), nk - 2)
        ts, tk = fs - i, fk - j
        sig = _cubic(_row(iv, i - 1, j, tk), _row(iv, i, j, tk), _row(iv, i + 1, j, tk), _row(iv, i + 2, j, tk), ts)
        out[n] = black.call_price(1.0, math.exp(kn), 1.0, tau, sig)


def calls(table, k, v, return_outside=False):
    """Undiscounted calls per unit forward at log-moneyness k and state variance v (arrays of equal length). States
    outside the grid are priced exactly by pricers.cos_prices; with return_outside=True also returns their count."""
    k = np.ascontiguousarray(np.asarray(k, dtype=float).ravel())
    v = np.ascontiguousarray(np.asarray(v, dtype=float).ravel())
    if k.size != v.size:
        raise ValueError("k and v must have the same length")
    out = np.empty(k.size)
    _calls(table.iv, table.k0, table.dk, table.ds, table.nk, table.ns, table.tau, k, v, out)
    outside = np.flatnonzero(np.isnan(out))
    theta, kappa, eta, rho = table.params
    for n in outside:
        model = Heston(max(float(v[n]), 0.0), theta, kappa, eta, rho)
        out[n] = pricers.cos_prices(model, 1.0, math.exp(k[n]), table.tau)[0]
    return (out, int(outside.size)) if return_outside else out
