"""The Heston variance-swap term structure fitted to vol indices (the VIX family), with a band on its long-run level.

Plain words
- A vol index at tenor tau (in years) is, squared, the fair variance-swap rate for that tenor (varswap.py). In Heston
  that rate is linear in the day's variance v:  rate(tau) = a(tau) + b(tau) v, with b = (1 - e^{-kappa tau}) /
  (kappa tau) and a = theta (1 - b). It depends on kappa and theta only, not on eta or rho.
- One speed kappa and one long-run variance theta hold for the whole history; each day has its own variance v.
  For given kappa and theta, the day's v is a weighted least-squares solution in closed form (weights from the
  delta method, so that errors count in vol units: 1 / (2 vol sd)^2), clipped at 0. The fit then moves only kappa
  and theta (Levenberg-Marquardt, calibrate.py), with every day's v solved inside each step.
- The band on theta (the long-run anchor beyond the longest index) comes from a moving-block bootstrap of the days:
  blocks of consecutive days keep the slow drift of the vol level, which an ordinary bootstrap would break.
Indices are discrete versions of the variance-swap strip (Cboe drops strikes after two consecutive zero bids), so
they sit slightly below the true rate in fat-tailed markets; the fit treats them as the rate.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .calibrate import levenberg_marquardt


def coefficients(kappa, theta, tau):
    """(a, b) arrays with variance-swap rate(tau) = a + b v in Heston."""
    tau = np.asarray(tau, dtype=float)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError("every tenor must be a positive number of years")
    b = -np.expm1(-kappa * tau) / (kappa * tau)
    return theta * (1.0 - b), b


def _check(tenors, vols, sd):
    tenors = np.atleast_1d(np.asarray(tenors, dtype=float))
    vols = np.atleast_2d(np.asarray(vols, dtype=float))
    sd = np.broadcast_to(np.asarray(sd, dtype=float), tenors.shape)
    if vols.shape[1] != tenors.size:
        raise ValueError(f"vols must have one column per tenor ({tenors.size}), got {vols.shape[1]}")
    if np.any(sd <= 0) or np.any(~np.isfinite(sd)):
        raise ValueError("every noise sd must be a positive number (vol units)")
    if np.any(vols[np.isfinite(vols)] <= 0):
        raise ValueError("every quoted vol must be positive (NaN marks a missing index)")
    return tenors, vols, sd


def states(kappa, theta, tenors, vols, sd):
    """Each day's variance v (closed-form weighted least squares in variance units, clipped at 0; NaN for a day with
    no index). vols: (days x tenors) annual vols, NaN where an index is missing."""
    tenors, vols, sd = _check(tenors, vols, sd)
    a, b = coefficients(kappa, theta, tenors)
    obs = np.isfinite(vols)
    v2 = np.where(obs, vols, 1.0) ** 2
    w = np.where(obs, 1.0 / (2.0 * np.where(obs, vols, 1.0) * sd[None, :]) ** 2, 0.0)
    num = (w * b[None, :] * (v2 - a[None, :])).sum(axis=1)
    den = (w * b[None, :] ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(den > 0, num / den, np.nan)
    return np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)


def predict(kappa, theta, v, tau):
    """Model vol at tenor(s) tau for variance(s) v (broadcast together)."""
    a, b = coefficients(kappa, theta, tau)
    return np.sqrt(np.maximum(a + b * np.asarray(v, dtype=float), 0.0))


@dataclass
class TermFit:
    """kappa, theta, the daily variances v, the RMSE by tenor (vol units, unweighted) and the solver's report."""
    kappa: float
    theta: float
    v: np.ndarray = field(repr=False)
    rmse: np.ndarray
    cost: float
    iterations: int
    converged: bool
    message: str


def _residuals(z, tenors, vols, sd):
    """Weighted residuals in vol units; an infinite vector for a trial point too extreme to evaluate (the solver
    then rejects the step: a long-run level and a speed can drift together when only short tenors are seen)."""
    if not np.all(np.abs(z) < 50.0):
        return np.full(int(np.isfinite(vols).sum()), np.inf)
    kappa, theta = math.exp(z[0]), math.exp(z[1])
    with np.errstate(all="ignore"):
        v = states(kappa, theta, tenors, vols, sd)
        model = predict(kappa, theta, v[:, None], tenors[None, :])
    r = (model - vols) / sd[None, :]
    return np.where(np.isfinite(r), r, np.inf)[np.isfinite(vols)]


def fit(tenors, vols, sd, start=(1.0, 0.04), max_iter=200):
    """Fit kappa and theta (and each day's v) to a history of vol indices: vols (days x tenors), NaN where missing."""
    tenors, vols, sd = _check(tenors, vols, sd)
    z, info = levenberg_marquardt(lambda z: _residuals(z, tenors, vols, sd), np.log(np.asarray(start, dtype=float)),
                                  max_iter=max_iter)
    kappa, theta = math.exp(z[0]), math.exp(z[1])
    v = states(kappa, theta, tenors, vols, sd)
    err = predict(kappa, theta, v[:, None], tenors[None, :]) - vols
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))
    return TermFit(kappa=kappa, theta=theta, v=v, rmse=rmse, cost=info["cost"], iterations=info["iterations"],
                   converged=info["converged"], message=info["message"])


def bootstrap(tenors, vols, sd, block=63, reps=200, seed=0, start=(1.0, 0.04), return_converged=False):
    """(reps x 2) array of (kappa, theta) refits on moving-block resamples of the days (blocks of `block` days).
    With return_converged=True also a boolean array: which refits the solver reported as converged (code review,
    2026-10-23: degenerate refits, kappa near 0 with theta running off, must be counted, not silently pooled)."""
    tenors, vols, sd = _check(tenors, vols, sd)
    n = vols.shape[0]
    if not (isinstance(block, int) and 1 <= block <= n):
        raise ValueError("block must be a whole number of days between 1 and the number of days")
    rng = np.random.default_rng(seed)
    nblocks = -(-n // block)
    out = np.empty((reps, 2))
    ok = np.zeros(reps, dtype=bool)
    for r in range(reps):
        starts = rng.integers(0, n - block + 1, nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        f = fit(tenors, vols[idx], sd, start=start)
        out[r] = f.kappa, f.theta
        ok[r] = f.converged
    return (out, ok) if return_converged else out
