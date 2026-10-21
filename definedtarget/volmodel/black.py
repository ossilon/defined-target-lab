"""Black-76 on the forward and its implied volatility, numba-compiled.

A copy of definedtarget/pricing.py's formulas so that volmodel imports nothing else from definedtarget (other
projects can then import volmodel unchanged). tests/test_vol_black.py requires the two to give identical numbers;
change both together or neither.

    C = DF (F N(d1) - K N(d2)),   d1 = (ln(F/K) + sigma^2 tau / 2) / (sigma sqrt(tau)),   d2 = d1 - sigma sqrt(tau)

F forward, K strike, DF discount factor to expiry, tau years to expiry, sigma volatility, N() the standard normal
distribution function. Implied volatility is the sigma that reproduces a given call price.
"""
import math

import numpy as np
from numba import njit

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
VOL_LO, VOL_HI = 1e-6, 5.0


@njit(cache=True)
def norm_cdf(x):
    return 0.5 * math.erfc(-x / SQRT2)


@njit(cache=True)
def norm_pdf(x):
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True)
def call_price(F, K, DF, tau, sigma):
    """Black-76 call value; at tau <= 0 or sigma <= 0 the discounted intrinsic value DF max(F - K, 0)."""
    sd = sigma * math.sqrt(tau) if tau > 0.0 else 0.0
    if sd <= 0.0:
        return DF * max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    return DF * (F * norm_cdf(d1) - K * norm_cdf(d1 - sd))


@njit(cache=True)
def put_price(F, K, DF, tau, sigma):
    """Black-76 put by put-call parity: P = C - DF (F - K)."""
    return call_price(F, K, DF, tau, sigma) - DF * (F - K)


@njit(cache=True)
def call_vega(F, K, DF, tau, sigma):
    """dC/dsigma per 1.00 of volatility."""
    sd = sigma * math.sqrt(tau) if tau > 0.0 else 0.0
    if sd <= 0.0:
        return 0.0
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    return DF * F * norm_pdf(d1) * math.sqrt(tau)


@njit(cache=True)
def implied_vol(price, F, K, DF, tau, tol=1e-12, max_iter=100):
    """Volatility that reproduces a call price; NaN outside the no-arbitrage range (at or below the discounted
    intrinsic value, at or above DF F) or for tau <= 0. Newton's method kept inside a bracket, bisecting whenever a
    Newton step would leave it; the stopping rule is relative to the time value (price minus intrinsic)."""
    if tau <= 0.0 or not (price > DF * max(F - K, 0.0)) or not (price < DF * F):
        return np.nan
    lo, hi = VOL_LO, VOL_HI
    if call_price(F, K, DF, tau, hi) < price:
        return np.nan
    if call_price(F, K, DF, tau, lo) > price:
        return lo
    sigma = math.sqrt(2.0 * math.pi / tau) * price / (DF * F)
    sigma = min(max(sigma, 0.05), 1.0)
    for _ in range(max_iter):
        p = call_price(F, K, DF, tau, sigma)
        vega = call_vega(F, K, DF, tau, sigma)
        diff = p - price
        if abs(diff) <= tol * (price - DF * max(F - K, 0.0)):
            return sigma
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        new = sigma - diff / vega if vega > 1e-14 else 0.5 * (lo + hi)
        if not (lo < new < hi):
            new = 0.5 * (lo + hi)
        sigma = new
        if hi - lo < 1e-15:
            return sigma
    return sigma


@njit(cache=True)
def call_price_array(F, K, DF, tau, sigma):
    """call_price over equal-length float arrays."""
    out = np.empty(F.shape[0])
    for i in range(F.shape[0]):
        out[i] = call_price(F[i], K[i], DF[i], tau[i], sigma[i])
    return out


@njit(cache=True)
def implied_vol_array(price, F, K, DF, tau):
    """implied_vol over equal-length float arrays (NaN where no valid answer)."""
    out = np.empty(price.shape[0])
    for i in range(price.shape[0]):
        out[i] = implied_vol(price[i], F[i], K[i], DF[i], tau[i])
    return out


def implied_vols(prices, F, K, T, DF=1.0):
    """Implied vols of call prices at strikes K (arrays broadcast together); NaN where no vol reproduces the price."""
    prices, F, K, T, DF = (np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=float), np.broadcast(
        prices, F, K, T, DF).shape)).ravel() for a in (prices, F, K, T, DF))
    return implied_vol_array(prices, F, K, DF, T)
