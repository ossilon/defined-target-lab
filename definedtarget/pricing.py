"""Black-76: the option formula, its sensitivities ("Greeks") and the implied-volatility solver. numba-compiled.

Plain words
- Black-76 is the Black-Scholes formula written with the FORWARD price F (what SPY "should" cost at expiry once
  you account for interest and the dividends a call holder misses). A European call is worth
      C = DF * ( F * N(d1) - K * N(d2) ),   d1 = (ln(F/K) + sigma^2 tau / 2) / (sigma sqrt(tau)),  d2 = d1 - sigma sqrt(tau)
  where K is the strike, DF the discount factor to expiry, tau the years to expiry, sigma the volatility and
  N() the standard normal cumulative probability. Under the formula's own (risk-neutral) assumptions, N(d2) is
  the probability the option finishes above the strike.
- Implied volatility = the sigma that makes the formula equal a price we observe. It is the market's (or the
  pricing vendor's) "price of uncertainty", and the single number that summarizes how expensive an option is.
- Everything here works on plain floats or float arrays and numba compiles it, so pricing a million options
  takes milliseconds. Input checking lives in the Python wrappers (valuation.py), not here.

Greeks in forward terms (per one option on one SPY share):
  dC/dF   = DF * N(d1)                       (delta with respect to the forward)
  d2C/dF2 = DF * n(d1) / (F sigma sqrt(tau)) (gamma with respect to the forward; n() = normal density)
  dC/dsigma = DF * F * n(d1) * sqrt(tau)     (vega per 1.00 of volatility, i.e. per 100 vol points)
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
    """Black-76 call value. At tau <= 0 or sigma <= 0 it is the discounted intrinsic value DF*max(F-K, 0)."""
    sd = sigma * math.sqrt(tau) if tau > 0.0 else 0.0
    if sd <= 0.0:
        return DF * max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    return DF * (F * norm_cdf(d1) - K * norm_cdf(d1 - sd))


@njit(cache=True)
def put_price(F, K, DF, tau, sigma):
    """Black-76 put via put-call parity: P = C - DF*(F - K)."""
    return call_price(F, K, DF, tau, sigma) - DF * (F - K)


@njit(cache=True)
def call_greeks(F, K, DF, tau, sigma):
    """(price, delta_F, gamma_F, vega, prob_itm) for a call; vega per 1.00 vol; prob_itm = N(d2)."""
    sd = sigma * math.sqrt(tau) if tau > 0.0 else 0.0
    if sd <= 0.0:
        itm = 1.0 if F > K else 0.0
        return DF * max(F - K, 0.0), DF * itm, 0.0, 0.0, itm
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    nd1 = norm_cdf(d1)
    price = DF * (F * nd1 - K * norm_cdf(d2))
    pdf = norm_pdf(d1)
    return price, DF * nd1, DF * pdf / (F * sd), DF * F * pdf * math.sqrt(tau), norm_cdf(d2)


@njit(cache=True)
def implied_vol(price, F, K, DF, tau, tol=1e-12, max_iter=100):
    """Volatility that reproduces a call price. NaN if the price is outside the no-arbitrage range
    (below the discounted intrinsic value DF*max(F-K,0), or at/above DF*F) or tau <= 0.

    Method: Newton's method on sigma (fast), kept inside a bracket [lo, hi] that always contains the answer;
    whenever a Newton step would leave the bracket, it bisects instead (safe).
    """
    if tau <= 0.0 or not (price > DF * max(F - K, 0.0)) or not (price < DF * F):
        return np.nan
    lo, hi = VOL_LO, VOL_HI
    if call_price(F, K, DF, tau, hi) < price:
        return np.nan
    if call_price(F, K, DF, tau, lo) > price:
        return lo
    # starting guess (Brenner-Subrahmanyam style, clipped into the bracket)
    sigma = math.sqrt(2.0 * math.pi / tau) * price / (DF * F)
    sigma = min(max(sigma, 0.05), 1.0)
    for _ in range(max_iter):
        p, _d, _g, vega, _q = call_greeks(F, K, DF, tau, sigma)
        diff = p - price
        # relative tolerance on the TIME VALUE (price minus intrinsic): an absolute tolerance would stop at the
        # first guess for very cheap options, and the time value is the only part that carries vol information
        if abs(diff) <= tol * (price - DF * max(F - K, 0.0)):
            return sigma
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        step_ok = vega > 1e-14
        new = sigma - diff / vega if step_ok else 0.5 * (lo + hi)
        if not (lo < new < hi):
            new = 0.5 * (lo + hi)
        sigma = new
        if hi - lo < 1e-15:
            return sigma
    return sigma


@njit(cache=True)
def call_price_array(F, K, DF, tau, sigma):
    """Vectorized call_price over equal-length float arrays."""
    out = np.empty(F.shape[0])
    for i in range(F.shape[0]):
        out[i] = call_price(F[i], K[i], DF[i], tau[i], sigma[i])
    return out


@njit(cache=True)
def implied_vol_array(price, F, K, DF, tau):
    """Vectorized implied_vol over equal-length float arrays (NaN where no valid answer)."""
    out = np.empty(price.shape[0])
    for i in range(price.shape[0]):
        out[i] = implied_vol(price[i], F[i], K[i], DF[i], tau[i])
    return out
