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

Two factors (heston.DoubleHeston)
- The rate is the sum of the two factors' one-factor rates:
      rate(tau) = theta1 (1 - b1) + b1 v1 + c2 g2 + b2 v2,   g2 = (1 - b2) / kappa2,   c2 = kappa2 theta2,
  with b_i the one-factor b at speed kappa_i. The slow factor carries its drift c2 instead of its long-run level:
  as kappa2 -> 0, g2 -> tau / 2 and every formula stays finite (g2 uses a series below kappa2 tau = 1e-3). The fit
  moves kappa1 > kappa2, theta1 and c2; each day's (v1, v2) is the closed-form weighted least-squares solution with
  both variances at 0 or above (fit_two).
- Why these choices (fit-window runs, 2026-10-23). With the day's variances left free in sign, any theta1 gave the
  same rates (the variances absorbed it), the slow speed ran to 1e-21 and its long-run level to 461,000 from one start
  and 7,600 from another, at the same cost and the same drift; the fitted variances were negative on 1,868 of 2,587
  days. With both variances at 0 or above, as the model requires, theta1 is identified (the same value from three
  starts, a clear minimum in its profile), and the slow speed still goes to its floor KAPPA2_MIN: on 2009-2019 the
  slow factor does not revert. The floor (1e-6 a year, a half-life of 690,000 years) changes no rate by more than
  1e-6 relative over 3.5 years and keeps the pricing and simulation formulas accurate.
- vol_of_variance measures each factor's eta from the time series: factor i's innovation over a step h has the
  exact conditional variance eta_i^2 (v_i A + kappa_i theta_i hp^2 / 2), with hp = (1 - e^{-kappa h}) / kappa and A =
  e^{-kappa h} hp (Andersen 2008, eq. 18). A change of measure moves drifts, never the diffusion, so a diffusion model
  must show the same eta in the record of the indices and in the option prices; a gap is a misfit of the model.
  Weekly steps limit the effect of index noise, which adds to every squared change.
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


def _block_resamples(n, block, reps, seed):
    """Row indices of `reps` moving-block resamples of n days (blocks of `block` consecutive days)."""
    if not (isinstance(block, int) and 1 <= block <= n):
        raise ValueError("block must be a whole number of days between 1 and the number of days")
    rng = np.random.default_rng(seed)
    nblocks = -(-n // block)
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, nblocks)
        yield (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap(tenors, vols, sd, block=63, reps=200, seed=0, start=(1.0, 0.04), return_converged=False):
    """(reps x 2) array of (kappa, theta) refits on moving-block resamples of the days (blocks of `block` days).
    With return_converged=True also a boolean array: which refits the solver reported as converged (code review,
    2026-10-23: degenerate refits, kappa near 0 with theta running off, must be counted, not silently pooled)."""
    tenors, vols, sd = _check(tenors, vols, sd)
    out = np.empty((reps, 2))
    ok = np.zeros(reps, dtype=bool)
    for r, idx in enumerate(_block_resamples(vols.shape[0], block, reps, seed)):
        f = fit(tenors, vols[idx], sd, start=start)
        out[r] = f.kappa, f.theta
        ok[r] = f.converged
    return (out, ok) if return_converged else out


# ----- two factors -----

def _phi(y):
    """(1 - e^{-y}) / y for y > 0 (the one-factor b at y = kappa tau)."""
    return -np.expm1(-y) / y


def _half_tail(y):
    """(1 - phi(y)) / y, which tends to 1/2 as y -> 0: a series below y = 1e-3 (truncation below 1e-15 relative),
    where 1 - phi(y) would lose digits."""
    y = np.asarray(y, dtype=float)
    small = y < 1e-3
    ys = np.where(small, 1.0, y)
    return np.where(small, 0.5 - y / 6.0 + y * y / 24.0 - y ** 3 / 120.0, (1.0 - _phi(ys)) / ys)


KAPPA2_MIN = 1e-6


def coefficients_two(kappa1, theta1, kappa2, c2, tau):
    """(a, b1, b2) arrays with variance-swap rate(tau) = a + b1 v1 + b2 v2 in two-factor Heston (c2 = kappa2 theta2)."""
    a1, b1 = coefficients(kappa1, theta1, tau)
    tau = np.asarray(tau, dtype=float)
    return a1 + c2 * tau * _half_tail(kappa2 * tau), b1, _phi(kappa2 * tau)


def states_two(kappa1, theta1, kappa2, c2, tenors, vols, sd):
    """Each day's (v1, v2), both at 0 or above: the weighted least-squares solution in variance units with the weights
    of states(), in closed form (the interior solution if both are positive, else the best one on an axis or at 0;
    for a convex problem in two unknowns one of these is the minimum). (days x 2); NaN for a day with fewer than two
    indices."""
    tenors, vols, sd = _check(tenors, vols, sd)
    a, b1, b2 = coefficients_two(kappa1, theta1, kappa2, c2, tenors)
    obs = np.isfinite(vols)
    vv = np.where(obs, vols, 1.0)
    w = np.where(obs, 1.0 / (2.0 * vv * sd[None, :]) ** 2, 0.0)
    r = np.where(obs, vv * vv - a[None, :], 0.0)
    s11, s12, s22 = (w * b1 * b1).sum(axis=1), (w * b1 * b2).sum(axis=1), (w * b2 * b2).sum(axis=1)
    q1, q2 = (w * b1 * r).sum(axis=1), (w * b2 * r).sum(axis=1)
    det = s11 * s22 - s12 * s12
    ok = (obs.sum(axis=1) >= 2) & (det > 1e-12 * s11 * s22)
    with np.errstate(invalid="ignore", divide="ignore"):
        cands = (np.column_stack(((s22 * q1 - s12 * q2) / det, (s11 * q2 - s12 * q1) / det)),
                 np.column_stack((np.maximum(q1 / s11, 0.0), np.zeros_like(q1))),
                 np.column_stack((np.zeros_like(q2), np.maximum(q2 / s22, 0.0))))
    best, cost = np.zeros((vols.shape[0], 2)), (w * r * r).sum(axis=1)     # start from (0, 0)
    for cand in cands:
        feasible = np.all(cand >= 0.0, axis=1)
        safe = np.where(feasible[:, None], cand, 0.0)
        res = r - safe[:, :1] * b1[None, :] - safe[:, 1:] * b2[None, :]
        c = np.where(feasible, (w * res * res).sum(axis=1), np.inf)
        better = c < cost
        best[better], cost[better] = safe[better], c[better]
    best[~ok] = np.nan
    return best


def predict_two(kappa1, theta1, kappa2, c2, v, tau):
    """Model vol at tenor(s) tau for variances v (last axis: v1, v2), broadcast together."""
    a, b1, b2 = coefficients_two(kappa1, theta1, kappa2, c2, tau)
    v = np.asarray(v, dtype=float)
    return np.sqrt(np.maximum(a + b1 * v[..., 0] + b2 * v[..., 1], 0.0))


@dataclass
class TwoFactorFit:
    """kappa1 > kappa2 >= KAPPA2_MIN, theta1, the slow drift c2 = kappa2 theta2, each day's variances v (days x 2),
    the RMSE by tenor (vol units, unweighted) and the solver's report."""
    kappa1: float
    theta1: float
    kappa2: float
    c2: float
    v: np.ndarray = field(repr=False)
    rmse: np.ndarray
    cost: float
    iterations: int
    converged: bool
    message: str

    @property
    def theta2(self):
        return self.c2 / self.kappa2

    @property
    def params(self):
        return self.kappa1, self.theta1, self.kappa2, self.c2


def _two_params(z):
    """(kappa1, theta1, kappa2, c2) from the free coordinates (ln(kappa1 - kappa2), ln theta1, ln(kappa2 - KAPPA2_MIN),
    ln c2)."""
    kappa2 = KAPPA2_MIN + math.exp(z[2])
    return kappa2 + math.exp(z[0]), math.exp(z[1]), kappa2, math.exp(z[3])


def _residuals_two(z, tenors, vols, sd):
    if not np.all(np.abs(z) < 50.0):
        return np.full(int(np.isfinite(vols).sum()), np.inf)
    p = _two_params(z)
    with np.errstate(all="ignore"):
        v = states_two(*p, tenors, vols, sd)
        model = predict_two(*p, v[:, None, :], tenors[None, :])
    r = (model - vols) / sd[None, :]
    return np.where(np.isfinite(r), r, np.inf)[np.isfinite(vols)]


def to_free_two(kappa1, theta1, kappa2, c2):
    if not (kappa1 > kappa2 > KAPPA2_MIN and theta1 > 0 and c2 > 0):
        raise ValueError(f"need kappa1 > kappa2 > {KAPPA2_MIN}, theta1 > 0 and c2 > 0, got "
                         f"{(kappa1, theta1, kappa2, c2)}")
    return np.array([math.log(kappa1 - kappa2), math.log(theta1), math.log(kappa2 - KAPPA2_MIN), math.log(c2)])


def fit_two(tenors, vols, sd, start=(5.0, 0.02, 0.5, 0.01), max_iter=200, fixed=()):
    """Fit kappa1 > kappa2, theta1 and the slow drift c2 (and each day's v1, v2) to a history of vol indices;
    start = (kappa1, theta1, kappa2, c2). fixed: "theta1" and/or "kappa2", held at start's values (for profiles)."""
    tenors, vols, sd = _check(tenors, vols, sd)
    if set(fixed) - {"theta1", "kappa2"}:
        raise ValueError(f"fixed may name theta1 and kappa2 only, got {fixed!r}")
    z_all = to_free_two(*(float(s) for s in start))
    free = [i for i, n in enumerate(("kappa1", "theta1", "kappa2", "c2")) if n not in fixed]

    def full(zf):
        z = z_all.copy()
        z[free] = zf
        return z

    zf, info = levenberg_marquardt(lambda zf: _residuals_two(full(zf), tenors, vols, sd), z_all[free],
                                   max_iter=max_iter)
    p = _two_params(full(zf))
    v = states_two(*p, tenors, vols, sd)
    err = predict_two(*p, v[:, None, :], tenors[None, :]) - vols
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))
    return TwoFactorFit(kappa1=p[0], theta1=p[1], kappa2=p[2], c2=p[3], v=v, rmse=rmse, cost=info["cost"],
                        iterations=info["iterations"], converged=info["converged"], message=info["message"])


def bootstrap_two(tenors, vols, sd, block=63, reps=200, seed=0, start=(5.0, 0.02, 0.5, 0.01)):
    """((reps x 4) array of (kappa1, theta1, kappa2, c2) refits on moving-block resamples, converged flags)."""
    tenors, vols, sd = _check(tenors, vols, sd)
    out = np.empty((reps, 4))
    ok = np.zeros(reps, dtype=bool)
    for r, idx in enumerate(_block_resamples(vols.shape[0], block, reps, seed)):
        f = fit_two(tenors, vols[idx], sd, start=start)
        out[r] = f.params
        ok[r] = f.converged
    return out, ok


def vol_of_variance(kappa1, theta1, kappa2, c2, v, t, step=5):
    """Each factor's eta from the time series of its variance (see the module docstring): v (rows of consecutive
    observation days, columns v1, v2) at times t (years), sampled every `step` rows. Returns a dict: eta1, eta2,
    eta (both factors pooled), corr (the correlation of the two factors' innovations: 0 in the model) and n."""
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or t.shape != (v.shape[0],):
        raise ValueError("v must be (days x 2) and t one time per day")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0)):
        raise ValueError("v and t must be finite, with t strictly increasing")
    if not (isinstance(step, int) and 1 <= step and v.shape[0] // step >= 3):
        raise ValueError("step must be a whole number of rows that leaves at least 3 observations")
    vs, h = v[::step], np.diff(t[::step])
    E, D, inn = [], [], []
    for i, (kappa, drift) in enumerate(((kappa1, kappa1 * theta1), (kappa2, c2))):
        ek = np.exp(-kappa * h)
        hp = h * _phi(kappa * h)                                  # (1 - e^{-kappa h}) / kappa
        e = vs[1:, i] - vs[:-1, i] * ek - drift * hp
        E.append(float(e @ e))
        D.append(float((vs[:-1, i] * ek * hp + 0.5 * drift * hp * hp).sum()))
        inn.append(e)
    return {"eta1": math.sqrt(E[0] / D[0]), "eta2": math.sqrt(E[1] / D[1]), "eta": math.sqrt(sum(E) / sum(D)),
            "corr": float(np.corrcoef(inn[0], inn[1])[0, 1]), "n": int(h.size)}
