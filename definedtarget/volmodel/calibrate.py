"""Fit the Heston parameters to implied vols and variance-swap rates, and say how well the data pins each one down.

Plain words
- A quote is either a listed option's implied vol (strike K, forward F, maturity T) or a vol index such as the VIX
  (no strike: its square is a variance-swap rate, varswap.py). Each quote carries a noise standard deviation sd in
  vol units (for example 0.002 = 0.2 vol points); the fit minimizes the sum of ((model - market) / sd)^2.
- The parameters are moved on an unconstrained scale (logarithms of v0, theta, kappa, eta; the inverse hyperbolic
  tangent of rho), so every trial point is a valid model. Any of them can be held fixed.
- The solver is Levenberg-Marquardt (Levenberg 1944; Marquardt 1963): Gauss-Newton steps on a linearized model,
  damped toward small steepest-descent steps whenever a step fails to lower the cost.
- Standard errors come from the Jacobian at the answer: cov = (J^T J)^{-1} on the free scale, carried to the
  natural scale by the chain rule. They are local (linearized) and assume the stated sd is the real noise. A
  parameter that no quote depends on (eta and rho, given only variance-swap rates) gets an infinite standard error.
  The correlation matrix of the estimates shows which parameters trade off against each other.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import black, pricers
from .heston import Heston

PARAMS = ("v0", "theta", "kappa", "eta", "rho")


@dataclass(frozen=True)
class Quotes:
    """Market quotes as equal-length arrays. K = NaN marks a vol-index quote (vol = sqrt of a variance-swap rate)."""
    T: np.ndarray
    K: np.ndarray
    F: np.ndarray
    vol: np.ndarray
    sd: np.ndarray

    def __post_init__(self):
        arrays = {}
        for name in ("T", "K", "F", "vol", "sd"):
            arrays[name] = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
            object.__setattr__(self, name, arrays[name])
        n = arrays["T"].size
        if any(a.ndim != 1 or a.size != n for a in arrays.values()):
            raise ValueError("every quote array must have the same length (one entry per quote)")
        if n == 0:
            raise ValueError("no quotes")
        strikes = arrays["K"][~np.isnan(arrays["K"])]               # NaN strikes are vol-index quotes
        for name, a in (("maturity T", arrays["T"]), ("forward F", arrays["F"]), ("quoted vol", arrays["vol"]),
                        ("noise sd", arrays["sd"]), ("strike K", strikes)):
            if not np.all(np.isfinite(a) & (a > 0)):
                raise ValueError(f"every {name} must be a finite positive number")

    @property
    def n(self):
        return self.T.size


def model_vols(model, quotes):
    """The model's vol for each quote: Black-76 implied vols of COS calls for options (grouped by maturity and
    forward, so each group is one pricer call), sqrt(variance-swap rate) for vol indices."""
    out = np.empty(quotes.n)
    idx = np.isnan(quotes.K)
    for i in np.flatnonzero(idx):
        out[i] = math.sqrt(model.variance_swap_rate(quotes.T[i]))
    opt = np.flatnonzero(~idx)
    if opt.size:
        keys = np.stack([quotes.T[opt], quotes.F[opt]], axis=1)
        for key in np.unique(keys, axis=0):
            sel = opt[np.all(keys == key, axis=1)]
            T, F = float(key[0]), float(key[1])
            calls = pricers.cos_prices(model, F, quotes.K[sel], T)
            out[sel] = black.implied_vols(calls, F, quotes.K[sel], T)
    return out


def layout(vix_tenors, options, model, sd_vix=0.001, sd_option=0.002):
    """Synthetic noise-free quotes from `model`: vol indices at vix_tenors (years) and options as (T, F, K) triples.
    Used to test recovery and to measure what a given layout of real quotes can identify."""
    T = [float(t) for t in vix_tenors] + [float(o[0]) for o in options]
    F = [1.0] * len(vix_tenors) + [float(o[1]) for o in options]
    K = [float("nan")] * len(vix_tenors) + [float(o[2]) for o in options]
    sd = [sd_vix] * len(vix_tenors) + [sd_option] * len(options)
    q = Quotes(T=T, K=K, F=F, vol=np.ones(len(T)), sd=sd)
    return Quotes(T=q.T, K=q.K, F=q.F, vol=model_vols(model, q), sd=q.sd)


def to_free(model, names):
    """The unconstrained coordinates of the named parameters."""
    return np.array([math.atanh(model.rho) if p == "rho" else math.log(getattr(model, p)) for p in names])


def from_free(z, template, names):
    """A Heston model with the named parameters set from unconstrained coordinates z, the rest from template."""
    values = {p: getattr(template, p) for p in PARAMS}
    for p, v in zip(names, z):
        values[p] = math.tanh(v) if p == "rho" else math.exp(v)
    return Heston(**values)


def residual_function(quotes, template, names=PARAMS):
    """z -> (model vols - market vols) / sd; an infinite vector where the trial model is invalid or a vol fails."""
    def fun(z):
        try:
            with np.errstate(all="ignore"):                # trial models far from the answer may overflow
                model = from_free(z, template, names)
                r = (model_vols(model, quotes) - quotes.vol) / quotes.sd
        except (ValueError, ArithmeticError, OverflowError):
            return np.full(quotes.n, np.inf)
        return np.where(np.isfinite(r), r, np.inf)
    return fun


def _jacobian(fun, x, r):
    """Central differences, step 1e-6 per unit of x. Forward differences (first run, 2026-10-21) carried a rounding
    error of 5e-9, which moves the answer of any fit whose residuals do not vanish (real, noisy quotes)."""
    J = np.empty((r.size, x.size))
    for j in range(x.size):
        h = 1e-6 * max(1.0, abs(x[j]))
        up, dn = x.copy(), x.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (fun(up) - fun(dn)) / (2 * h)
    return J


def levenberg_marquardt(fun, x0, max_iter=200, xtol=1e-12, ftol=1e-15):
    """Minimize sum(fun(x)^2). Returns (x, info) with info: iterations, cost, converged, message."""
    x = np.array(x0, dtype=float)
    r = fun(x)
    if not np.all(np.isfinite(r)):
        raise ValueError("the starting point gives non-finite residuals")
    cost, lam = float(r @ r), 1e-3
    message, converged, it = "iteration limit", False, 0          # it = 0 if max_iter is 0 (code review, 2026-10-23)
    for it in range(1, max_iter + 1):
        J = _jacobian(fun, x, r)
        A, g = J.T @ J, J.T @ r
        if np.max(np.abs(g)) <= 1e-15 * (1.0 + cost):
            message, converged = "gradient is zero", True
            break
        diag = np.maximum(np.diag(A), 1e-12 * max(float(np.max(np.diag(A))), 1e-300))
        while lam <= 1e16:
            try:
                dx = np.linalg.solve(A + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            rn = fun(x + dx)
            cn = float(rn @ rn) if np.all(np.isfinite(rn)) else math.inf
            if cn < cost:
                break
            lam *= 10.0
        else:
            message, converged = "no step lowers the cost (at the minimum to rounding)", True
            break
        decrease = (cost - cn) / max(cost, 1e-300)
        x, r, cost = x + dx, rn, cn
        lam = max(lam / 10.0, 1e-12)
        size = np.max(np.abs(dx)) / (1.0 + np.max(np.abs(x)))
        # a small decrease alone is not enough: near a minimum the cost is flat (an error d in x changes it by d^2),
        # so it must come with a small step (first run, 2026-10-21: stopped 5e-10 short on a linear problem)
        if size <= xtol or (decrease <= ftol and size <= 1e-8) or cost == 0.0:
            message, converged = "step or decrease below tolerance", True
            break
    return x, {"iterations": it, "cost": cost, "converged": converged, "message": message}


@dataclass
class Fit:
    """A Heston fit: the model, its fit quality (RMSE in vol units, unweighted), standard errors on the natural
    scale (NaN for fixed parameters, inf for unidentified ones), the correlation matrix of the estimates in PARAMS
    order, and the solver's report."""
    model: Heston
    rmse: float
    cost: float
    se: dict
    corr: np.ndarray
    free: tuple
    iterations: int
    converged: bool
    message: str
    residuals: np.ndarray = field(repr=False)


def fit_heston(quotes, start, fixed=(), max_iter=200):
    """Fit the Heston parameters not named in `fixed` to the quotes, starting from `start` (a Heston model that
    also supplies the fixed values)."""
    unknown = set(fixed) - set(PARAMS)
    if unknown:
        raise ValueError(f"unknown parameter name(s) in fixed: {sorted(unknown)}; the parameters are {PARAMS}")
    free = tuple(p for p in PARAMS if p not in fixed)
    if not free:
        raise ValueError("every parameter is fixed: there is nothing to fit")
    fun = residual_function(quotes, start, free)
    z, info = levenberg_marquardt(fun, to_free(start, free), max_iter=max_iter)
    model = from_free(z, start, free)
    r = fun(z)
    J = _jacobian(fun, z, r)
    se = {p: float("nan") for p in PARAMS}
    corr = np.full((len(PARAMS), len(PARAMS)), np.nan)
    norms = np.linalg.norm(J, axis=0)
    ident = norms > 1e-10 * max(float(np.max(norms)), 1e-300)
    for p, ok in zip(free, ident):
        se[p] = math.inf if not ok else se[p]
    if ident.any():
        Ji = J[:, ident]
        try:
            cov_z = np.linalg.inv(Ji.T @ Ji)
        except np.linalg.LinAlgError:
            cov_z = np.full((Ji.shape[1], Ji.shape[1]), np.inf)
        names = [p for p, ok in zip(free, ident) if ok]
        grad = np.array([1.0 - model.rho ** 2 if p == "rho" else getattr(model, p) for p in names])
        cov = cov_z * np.outer(grad, grad)
        var = np.diag(cov)
        # a near-singular J^T J can invert to negative or NaN variances: those directions are not identified
        sd = np.where(np.isfinite(var) & (var > 0), np.sqrt(np.where(var > 0, var, 1.0)), np.inf)
        for p, s in zip(names, sd):
            se[p] = float(s)
        pos = [PARAMS.index(p) for p in names]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr[np.ix_(pos, pos)] = cov / np.outer(sd, sd)
    resid_vol = r * quotes.sd
    return Fit(model=model, rmse=float(math.sqrt(np.mean(resid_vol ** 2))), cost=float(r @ r), se=se, corr=corr,
               free=free, iterations=info["iterations"], converged=info["converged"], message=info["message"],
               residuals=resid_vol)
