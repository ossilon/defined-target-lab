"""European option prices in the Heston model: two independent pricers that must agree.

Inputs are in forward terms: F the forward for the expiry, K strikes (array or number), T years to expiry, df the
discount factor to the payment date. Prices are per unit of the underlying, like definedtarget.pricing's Black-76.

1. cos_prices: the COS method (Fang and Oosterlee 2008). The density of X_T = ln(S_T / F) on a range [A, A + w] is
   replaced by its Fourier-cosine series, whose coefficients come straight from the characteristic function
   (heston.cf), and the payoff's cosine coefficients are known in closed form (their eq. 22-25). The number of terms
   N doubles from 64 until the added terms fall below the tolerance. Puts are computed and calls follow by put-call
   parity (their Remark 5.2: a call's payoff grows without bound, which costs accuracy in a truncated range).
   The range differs from the paper's. Fang and Oosterlee use the mean plus or minus 12 standard deviations of X_T
   (section 5.3). Here that range is far too narrow for Heston's exponential tails: 0.38 too low for Andersen's
   Case I at XX's strike, and 4e-5 too low at the paper's own one-year test, where its Table 4 shows convergence to
   about 3e-7 (so the paper's implementation must differ in a way it does not describe; a term-by-term replication
   with its range and its N did not reproduce the table, 2026-10-21). The range used here holds all but 1e-15 of the
   probability on each side, by Chernoff's bound from the model's moment generating function (chernoff_range); the
   two pricers then agree to about 1e-14 of the forward (tests/test_vol_pricers.py).
2. integral_prices: the Lewis-Lipton single integral (Lewis 2001, Option Valuation Under Stochastic Volatility;
   Lipton 2002, Risk) in Andersen's (2008, Proposition 3) notation,
       C = DF ( F - (K / pi) int_0^inf Re[ exp((1/2 - i k) ln(F/K) + h1(k) - (k^2 + 1/4) h2(k) v0) ] / (k^2 + 1/4) dk ),
   with its own algebra for h1 and h2 (no code shared with heston.cf). In h2 the sign is the one for which the
   integrand decays and T -> 0 gives Black-76 (h2 -> T/2): the PDF's text layer leaves that sign ambiguous, and the
   two limits fix it independently of the COS pricer. Integrated with 24-point Gauss-Legendre panels of width 1
   until the integrand's envelope, with its estimated decay, is below the tolerance. Its puts come from calls by
   parity, so deep out-of-the-money puts carry an absolute error of about 1e-15 F, large relative to their tiny
   value far in the left wing: use COS there (tests/test_vol_varswap.py measures this).
3. prices(..., check=True): the COS prices, verified against the integral; a disagreement above rtol (of the
   discounted forward) raises instead of returning a number. implied_vols: Black-76 vols of the COS calls.

tests/test_vol_pricers.py holds the judges: published reference values, the two pricers against each other over a
grid that includes XX's strike, a third quadrature (scipy), no-arbitrage shape, the lognormal limit, the range and
term checks, and Lee's (2004) wing bound.
"""
import math

import numpy as np

from . import black, heston

N_START, N_MAX = 64, 1 << 17
CHUNK = 4096                                   # cosine terms per matrix block (bounds memory for many strikes)
KINDS = ("call", "put")
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W          # nodes and weights on [0, 1]
PANELS_PER_BLOCK = 64
K_MAX = 1e6


def _inputs(F, K, T, df, kind):
    for name, v in (("forward F", F), ("maturity T (years)", T), ("discount factor df", df)):
        if not isinstance(v, (int, float, np.floating)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"the {name} must be a finite positive number, got {v!r}")
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if K.ndim != 1 or K.size == 0 or not np.all(np.isfinite(K)) or np.any(K <= 0):
        raise ValueError("every strike must be a finite positive number")
    if kind not in KINDS:
        raise ValueError(f"option kind must be one of {KINDS}, got {kind!r}")
    return float(F), K, float(T), float(df)


def _put_coefficients(om, a, d, w):
    """(2/w)(psi_k(a, d) - chi_k(a, d)) for the put payoff (1 - e^y)^+ on [a, d], d <= 0, in the cosine basis of
    [a, a + w] (Fang and Oosterlee 2008, eq. 22, 23 and 25). om: (n,) frequencies; a, d: (M,). Returns (n, M)."""
    om = om[:, None]
    s = d - a                                   # length of the payoff region inside the range, >= 0
    c, sn = np.cos(om * s), np.sin(om * s)
    ed, ea = np.exp(d), np.exp(a)
    chi = (c * ed - ea + om * sn * ed) / (1.0 + om * om)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(om == 0.0, s, sn / np.where(om == 0.0, 1.0, om))
    return (2.0 / w) * (psi - chi)


# exponents s for the moment generating function E[e^{sX}]: fine near 0 (extreme models explode early), then 0.05
S_GRID = np.concatenate((np.geomspace(1e-3, 0.04, 12), np.arange(0.05, 40.0001, 0.05)))


def _valid_prefix(lnm):
    """How many leading grid points carry a valid log-moment: finite, real, and convex with ln M(0) = 0 (a log
    moment generating function is convex). Beyond the moment explosion the closed form wraps around (a jump down,
    or a complex logarithm); the first such point and everything after it are dropped."""
    ok = np.isfinite(lnm) & (np.abs(lnm.imag) <= 1e-9 * np.maximum(1.0, np.abs(lnm.real)))
    stop = np.flatnonzero(~ok)
    n = int(stop[0]) if stop.size else lnm.size
    with np.errstate(all="ignore"):
        slope = np.diff(np.concatenate(([0.0], lnm.real))) / np.diff(np.concatenate(([0.0], S_GRID)))
    bad = np.flatnonzero(np.diff(slope) < -1e-9 * np.maximum(1.0, np.abs(slope[1:])))   # convex: slopes never fall
    if bad.size:
        n = min(n, int(bad[0]) + 1)
    return n


def chernoff_range(model, T, tail=1e-15):
    """[A, B] with P(X_T < A) <= tail and P(X_T > B) <= tail, from Chernoff's bound P(X > B) <= E[e^{sX}] e^{-sB}
    (s > 0, and the mirror image for the left tail), minimized over s up to the moment explosion. A distribution
    free bound: it needs no shape assumption, so it is safe for Heston's exponential tails, where a range of L
    standard deviations is too narrow (tests/test_vol_pricers.py, first run 2026-10-21: at L = 12 the COS price of
    Andersen's Case I at XX's strike was 0.38 too low)."""
    lt = math.log(tail)
    up = heston.log_cf(model, -1j * S_GRID, T)          # ln E[e^{sX}] for s > 0
    dn = heston.log_cf(model, 1j * S_GRID, T)           # ln E[e^{-sX}]
    nu, nd = _valid_prefix(up), _valid_prefix(dn)
    if nu == 0 or nd == 0:
        raise ArithmeticError("no valid moment of X_T near 0: the model's moment generating function failed")
    B = float(np.min((up.real[:nu] - lt) / S_GRID[:nu]))
    A = float(np.max((lt - dn.real[:nd]) / S_GRID[:nd]))
    return A, B


def cos_prices(model, F, K, T, df=1.0, kind="call", tail=1e-15, tol=1e-14, return_info=False, L=None, N=None):
    """Heston European option prices by the COS method (see the module docstring). Returns an array like K;
    with return_info=True also {"N": terms used, "converged": bool, "range": (A, w)}.

    The range [A, A + w] for X_T = ln(S_T / F) holds all but `tail` of the probability on each side (chernoff_range).
    For comparison and replication only: L replaces it with Fang and Oosterlee's mean +/- L standard deviations,
    and N fixes the number of terms instead of doubling until convergence."""
    F, K, T, df = _inputs(F, K, T, df, kind)
    if L is None:
        A, B = chernoff_range(model, T, tail)
    else:
        c1, c2 = model.cumulants(T)
        A, B = c1 - L * math.sqrt(c2), c1 + L * math.sqrt(c2)
    w = B - A
    x = np.log(F / K)
    a = A + x                                   # the same range in y = ln(S_T / K) = X_T + x, per strike
    d = np.clip(0.0, a, a + w)                  # the put pays on y < 0: [a, min(0, a + w)]
    total = np.zeros(K.size)
    fixed = N is not None
    if fixed and (not isinstance(N, int) or N < 1):
        raise ValueError(f"the number of terms N must be a positive integer, got {N!r}")
    done, nxt, converged = 0, (N if fixed else N_START), False
    while done < N_MAX:
        block = np.zeros(K.size)
        for lo in range(done, nxt, CHUNK):
            k = np.arange(lo, min(nxt, lo + CHUNK))
            om = k * math.pi / w
            phi = heston.cf(model, om, T)
            term = (phi * np.exp(-1j * om * A)).real
            if lo == 0:
                term[0] *= 0.5                  # the first term of a cosine series counts half
            block += term @ _put_coefficients(om, a, d, w)
        total += block
        tail = float(np.max(np.abs(phi)))
        done, nxt = nxt, 2 * nxt
        if fixed:
            break
        if np.max(np.abs(block) * K) <= tol * F and tail <= tol:
            converged = True
            break
    N = done
    puts = df * K * total
    beyond = a + w <= 0.0                       # the whole range lies below the strike: the call cannot pay
    puts[beyond] = df * (K[beyond] - F)         # (parity would leave rounding noise of order 1e-16 K there)
    out = puts if kind == "put" else puts + df * (F - K)
    if kind == "call":
        out[beyond] = 0.0
    if return_info:
        return out, {"N": N, "converged": converged, "range": (A, w)}
    return out


def lewis_exponent(model, k, T):
    """h1(k) - (k^2 + 1/4) h2(k) v0 for real k (array): Andersen (2008), Proposition 3."""
    kap, th, eps, rho, v0 = model.kappa, model.theta, model.eta, model.rho, model.v0
    kh = kap - rho * eps / 2.0
    k = np.asarray(k, dtype=float)
    xi = np.sqrt(k * k * eps * eps * (1.0 - rho * rho) + 2j * k * eps * rho * kh + kh * kh + eps * eps / 4.0)
    shift = 1j * k * rho * eps + kh
    dm, dp = xi + shift, xi - shift              # Andersen's d-minus and d-plus
    e = np.exp(-xi * T)
    h2 = (1.0 - e) / (dm + dp * e)
    h1 = -kap * th / (eps * eps) * (dp * T + 2.0 * np.log((dm + dp * e) / (2.0 * xi)))
    return h1 - (k * k + 0.25) * h2 * v0


def integral_prices(model, F, K, T, df=1.0, kind="call", tol=1e-14):
    """Heston European option prices by the Lewis-Lipton integral (see the module docstring)."""
    F, K, T, df = _inputs(F, K, T, df, kind)
    x = np.log(F / K)
    total = np.zeros(K.size)
    start = 0.0
    prev_env = None
    while start < K_MAX:
        nodes = (start + np.arange(PANELS_PER_BLOCK)[:, None] + _GL_X[None, :]).ravel()
        weights = np.tile(_GL_W, PANELS_PER_BLOCK)
        expo = lewis_exponent(model, nodes, T)
        vals = np.exp((0.5 - 1j * nodes)[:, None] * x[None, :] + expo[:, None]).real / (nodes * nodes + 0.25)[:, None]
        total += weights @ vals
        start += PANELS_PER_BLOCK
        # envelope of |integrand| at the block's end, and its decay per unit k over the block
        env = math.exp(float(expo[-1].real)) / (start * start + 0.25)
        first = math.exp(float(expo[0].real)) / (nodes[0] ** 2 + 0.25)
        rate = max(math.log(first / env) / PANELS_PER_BLOCK, 1e-3) if env > 0 and first > env else 1e-3
        if env == 0.0 or np.max(K * np.exp(0.5 * x)) * env / (math.pi * rate) <= tol * F:
            break
        prev_env = env
    del prev_env
    calls = df * (F - K / math.pi * total)
    return calls if kind == "call" else calls - df * (F - K)


def prices(model, F, K, T, df=1.0, kind="call", check=False, rtol=1e-9):
    """COS prices; with check=True the Lewis-Lipton integral must agree within rtol of the discounted forward."""
    p = cos_prices(model, F, K, T, df=df, kind=kind)
    if check:
        q = integral_prices(model, F, K, T, df=df, kind=kind)
        gap = float(np.max(np.abs(p - q))) / (df * F)
        if not gap <= rtol:
            raise ArithmeticError(f"the two Heston pricers disagree by {gap:.2e} of the discounted forward (allowed "
                                  f"{rtol:.0e}) at T = {T}: do not use these prices")
    return p


def implied_vols(model, F, K, T, check=False):
    """Black-76 implied vols of the model's calls (no discounting needed: vols are the same in forward terms)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    calls = prices(model, F, K, T, df=1.0, check=check)
    return black.implied_vols(calls, F, K, T, 1.0)
