"""volmodel.termstructure: the Heston variance-swap term structure fitted to vol indices, judged on synthetic data.

Judges (tolerances fixed 2026-10-23, before the first run):
1. The state of one day from its indices (closed form) equals a brute-force least-squares search (1e-10), refuses
   nothing for missing indices, and is clipped at 0.
2. Recovery: 600 synthetic days from kappa = 1.5, theta = 0.035 at the tenors 30, 93, 184 and 366 days, without
   noise and with one missing index on some days: the fit recovers kappa and theta within 1e-6 relative and every
   day's variance within 1e-8.
3. With noise (0.3 vol points), the moving-block bootstrap's 5-95% band for theta contains the truth, and the fit is
   deterministic for a given seed.
4. predict() at a tenor equals sqrt(a + b v), and bad inputs are refused.
"""
import math

import numpy as np

from _harness import check, raises

from definedtarget.volmodel import termstructure as ts
from definedtarget.volmodel.heston import Heston

TENORS = np.array([30, 93, 184, 366]) / 365.0
K_TRUE, TH_TRUE = 1.5, 0.035
rng = np.random.default_rng(2026)
v_true = 0.035 * np.exp(rng.normal(0, 0.6, 600) - 0.18)            # a spread of daily variances around theta
a, b = ts.coefficients(K_TRUE, TH_TRUE, TENORS)
vols = np.sqrt(a[None, :] + b[None, :] * v_true[:, None])
holes = vols.copy()
holes[::7, 2] = np.nan                                              # some days miss one index

# 1. One day's state.
day = holes[7]
v1 = ts.states(K_TRUE, TH_TRUE, TENORS, day[None, :], np.full(4, 0.005))[0]
grid = np.linspace(0, 0.2, 200001)
obs = ~np.isnan(day)
w = 1.0 / (2.0 * day[obs] * 0.005) ** 2
sse = ((a[obs][None, :] + b[obs][None, :] * grid[:, None] - day[obs] ** 2) ** 2 * w).sum(axis=1)
check("one day's state (closed form) equals a brute-force search over v (to the grid's step), with an index missing",
      abs(v1 - grid[np.argmin(sse)]) <= 1e-6 and abs(v1 - v_true[7]) < 1e-12, (v1, grid[np.argmin(sse)], v_true[7]))
low = np.sqrt(np.maximum(a + b * -0.01, 1e-8))[None, :]             # indices that imply a negative variance
check("a day whose indices imply a negative variance gets 0", ts.states(K_TRUE, TH_TRUE, TENORS, low, 0.005)[0] == 0.0)

# 2. Recovery without noise.
fit = ts.fit(TENORS, holes, np.full(4, 0.005), start=(0.5, 0.06))
check("noise-free recovery: kappa and theta within 1e-6 relative", abs(fit.kappa / K_TRUE - 1) < 1e-6
      and abs(fit.theta / TH_TRUE - 1) < 1e-6 and fit.converged, (fit.kappa, fit.theta, fit.message))
check("... and every day's variance within 1e-8", np.max(np.abs(fit.v - v_true)) < 1e-8, np.max(np.abs(fit.v - v_true)))
check("... the RMSE by tenor is below 1e-9 vol", np.max(fit.rmse) < 1e-9, fit.rmse)
check("the coefficients agree with varswap's Heston identity",
      abs(a[1] + b[1] * 0.02 - Heston(0.02, TH_TRUE, K_TRUE, 0.4, -0.7).variance_swap_rate(TENORS[1])) < 1e-15)

# 3. With noise: the bootstrap band holds the truth; the fit is deterministic.
noisy = holes + rng.normal(0, 0.003, holes.shape)
f1 = ts.fit(TENORS, noisy, np.full(4, 0.003))
boot = ts.bootstrap(TENORS, noisy, np.full(4, 0.003), block=63, reps=60, seed=7)
lo, hi = np.percentile(boot[:, 1], [5, 95])
check(f"with 0.3-point noise the bootstrap 5-95% band of theta ({lo:.4f} to {hi:.4f}) contains the truth",
      lo <= TH_TRUE <= hi, (lo, hi, f1.theta))
boot2 = ts.bootstrap(TENORS, noisy, np.full(4, 0.003), block=63, reps=60, seed=7)
check("the bootstrap is deterministic for a given seed", np.array_equal(boot, boot2))

# 4. Prediction and refusals.
p = ts.predict(K_TRUE, TH_TRUE, np.array([0.02, 0.05]), np.array([30 / 365, 3.0]))
check("predict() is sqrt(a + b v) at each tenor", np.allclose(
    p, np.sqrt(np.array([ts.coefficients(K_TRUE, TH_TRUE, t)[0] + ts.coefficients(K_TRUE, TH_TRUE, t)[1] * v
                         for v, t in ((0.02, 30 / 365), (0.05, 3.0))])), rtol=0, atol=1e-15))
raises("vols with the wrong number of tenors are refused", lambda: ts.fit(TENORS, vols[:, :3], np.full(4, 0.005)),
       "tenor")
raises("a non-positive tenor is refused", lambda: ts.coefficients(1.0, 0.04, np.array([0.0, 1.0])), "tenor")
