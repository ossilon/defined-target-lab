"""Variance swaps and the VIX family: the log-strip of option prices and its Heston closed form.

Plain words
- A variance swap pays the realized variance of the index over its life. Its fair rate needs no model: for any
  diffusion it equals the value of a strip of out-of-the-money puts and calls weighted by 1 / K^2,
      E[W_T] = -2 E[ln(S_T / F)] = (2 / DF) [ int_0^F P(K) / K^2 dK + int_F^inf C(K) / K^2 dK ]
  (Demeterfi, Derman, Kamal and Zou 1999; Gatheral 2006, ch. 11). W_T is the total variance from now to T.
- Cboe's VIX is 100 times the square root of a 30-day variance-swap rate computed by a discrete version of this
  strip from listed S&P 500 options; VIX9D, VIX3M, VIX6M and VIX1Y do the same at other tenors. Cboe's exact
  discretization and day counts (the VIX white paper) are applied when the family is fitted (v0.3.1 step 4).
- In Heston the rate is linear in today's variance: VIX^2(tau) = a + b v0, with b = (1 - e^{-kappa tau}) /
  (kappa tau) and a = theta (1 - b). It does not depend on eta or rho, so the VIX family alone can pin down v0,
  theta and kappa but says nothing about the vol of vol or the correlation (those need the smile).

strip_variance is model-agnostic: it takes any function that returns discounted option prices for an array of
strikes. tests/test_vol_varswap.py checks it on Heston prices from both pricers (against the closed form) and on
Black-76 prices (against sigma^2 T).
"""
import math

import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W
PANELS = 32
X_MAX = 600.0                     # |ln(K/F)| beyond which a strip that has not converged is refused (e^600 ~ 1e260)


def _positive(name, v):
    if not isinstance(v, (int, float, np.floating)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
        raise ValueError(f"the {name} must be a finite positive number, got {v!r}")


def strip_variance(price_fn, F, T, df=1.0, tol=1e-13, max_blocks=400, halves=False):
    """E[W_T] from option prices: the log-strip above. price_fn(K, kind) returns discounted prices for an array of
    strikes K and kind "call" or "put". Integrated in x = ln(K / F), outward from the forward on each side, with
    24-point Gauss-Legendre panels whose width grows with the distance once past four standard deviations (the
    scale comes from the at-the-money price), until a block of panels adds less than tol of the running total.
    Strikes stop at |ln(K/F)| = X_MAX; a side that has not converged by then is refused. With halves=True, returns
    (calls' part, puts' part), which add up to E[W_T]."""
    _positive("forward F", F)
    _positive("maturity T (years)", T)
    _positive("discount factor df", df)
    atm = float(np.asarray(price_fn(np.array([float(F)]), "call"), dtype=float)[0])
    scale = max(atm * math.sqrt(2.0 * math.pi) / (df * F), 1e-4)    # about sigma sqrt(T)
    parts = []
    for side, kind in ((1.0, "call"), (-1.0, "put")):
        start, acc, block = 0.0, 0.0, float("nan")
        for _ in range(max_blocks):
            if start >= X_MAX:
                raise ArithmeticError(f"the {kind} side of the strip still adds {block:.1e} at ln(K/F) = "
                                      f"{side * start:.0f}: the prices do not decay (noise or a wrong price function)")
            edges = [start]
            for _j in range(PANELS):
                edges.append(min(X_MAX, edges[-1] + 0.25 * scale * max(1.0, edges[-1] / (4.0 * scale))))
            edges = np.unique(np.array(edges))                                # clipped at X_MAX: drop empty panels
            lo, wid = edges[:-1], np.diff(edges)
            x = (lo[:, None] + wid[:, None] * _GL_X[None, :]).ravel()          # distance from the forward
            prices = np.asarray(price_fn(F * np.exp(side * x), kind), dtype=float)
            block = float(np.sum((wid[:, None] * _GL_W[None, :]).ravel() * prices * np.exp(-side * x))) / F
            acc += block
            start = float(edges[-1])
            if abs(block) <= tol * abs(acc):
                break
        else:
            raise ArithmeticError(f"the {kind} side of the strip did not converge within {max_blocks} blocks")
        parts.append(2.0 * acc / df)
    return tuple(parts) if halves else parts[0] + parts[1]


def vix_squared_coefficients(model, tau):
    """(a, b) with variance-swap rate(tau) = a + b v0 in Heston: b = (1 - e^{-kappa tau}) / (kappa tau),
    a = theta (1 - b). tau in years."""
    _positive("tenor tau (years)", tau)
    b = -math.expm1(-model.kappa * tau) / (model.kappa * tau)
    return model.theta * (1.0 - b), b
