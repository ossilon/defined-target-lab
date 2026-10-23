"""volmodel.calibrate: the Levenberg-Marquardt solver and the Heston fit, judged by known answers.

Judges (tolerances fixed 2026-10-21, before the first run):
1. The solver: Rosenbrock's valley (residuals 10 (y - x^2), 1 - x from (-1.2, 1)) ends at (1, 1) within 1e-8; a
   linear least-squares problem ends at numpy's lstsq answer within 1e-10.
2. Parameter recovery: quotes laid out like this project's data on 2026-10-21 (the VIX family at 9 days, 30 days,
   3, 6 and 12 months; listed calls at 6,500-10,000 for Dec-2027, Dec-2028 and Dec-2029; and XX's strikes
   10,000-15,000 for Dec-2029), generated without noise from Gatheral's (2006) Table 3.2 parameters. From three
   different starts the fit recovers all five parameters within 1e-6 relative, with RMSE below 1e-9 (vol units).
3. scipy's least_squares on the same residuals from the same start lands on the same parameters (1e-6 relative).
4. What the VIX family alone identifies: variance-swap rates do not depend on eta or rho (varswap), so with eta
   and rho fixed at wrong values a fit to the VIX family alone still recovers v0, theta and kappa (1e-6), and a fit
   of all five reports eta and rho as unidentified (standard error reported as infinite).
5. Standard errors and correlations are finite where identified, the correlation matrix is symmetric with a unit
   diagonal, and bad inputs are refused.
"""
import math

import numpy as np
from scipy.optimize import least_squares

from _harness import check, raises

from definedtarget.volmodel import calibrate
from definedtarget.volmodel.heston import Heston

# 1. The solver on known problems.
x, info = calibrate.levenberg_marquardt(lambda z: np.array([10 * (z[1] - z[0] ** 2), 1 - z[0]]), [-1.2, 1.0])
check("Levenberg-Marquardt solves Rosenbrock's valley to (1, 1) within 1e-8", np.max(np.abs(x - 1)) < 1e-8
      and info["converged"], (x, info))
rng = np.random.default_rng(2026)
A, b = rng.normal(size=(30, 4)), rng.normal(size=30)
x, info = calibrate.levenberg_marquardt(lambda z: A @ z - b, np.zeros(4))
check("... and a linear least-squares problem to numpy's lstsq answer within 1e-10",
      np.max(np.abs(x - np.linalg.lstsq(A, b, rcond=None)[0])) < 1e-10, x)

# 2. Parameter recovery from quotes laid out like the project's data.
TRUE = Heston(v0=0.0174, theta=0.0354, kappa=1.3253, eta=0.3877, rho=-0.7165)       # Gatheral (2006) Table 3.2
S, CARRY = 7712.57, 0.02                       # index level on 2026-10-21 and an illustrative r - q for forwards
EXPIRIES = (1.157, 2.150, 3.167)               # Dec-2027, Dec-2028, Dec-2029 from 2026-10-21 (years)
quotes = calibrate.layout(
    vix_tenors=(9 / 365, 30 / 365, 0.25, 0.5, 1.0),
    options=[(T, S * math.exp(CARRY * T), K) for T in EXPIRIES for K in range(6500, 10001, 500)]
    + [(3.167, S * math.exp(CARRY * 3.167), K) for K in (11000, 12000, 13000, 14000, 15000)],
    model=TRUE, sd_vix=0.001, sd_option=0.002)
check("the layout holds 5 vol-index quotes and 29 option quotes", quotes.n == 34 and np.isnan(quotes.K).sum() == 5,
      quotes.n)
starts = (Heston(0.04, 0.04, 3.0, 0.8, -0.3), Heston(0.01, 0.06, 0.5, 0.2, -0.9), Heston(0.03, 0.02, 1.0, 1.0, 0.0))
fits = [calibrate.fit_heston(quotes, s) for s in starts]
for i, f in enumerate(fits):
    rel = max(abs(getattr(f.model, p) / getattr(TRUE, p) - 1) for p in calibrate.PARAMS)
    check(f"start {i + 1}: the fit recovers all five parameters within 1e-6 relative, RMSE below 1e-9",
          rel < 1e-6 and f.rmse < 1e-9 and f.converged, (rel, f.rmse, f.model, f.message))

# 3. scipy agrees.
start = starts[0]
fun = calibrate.residual_function(quotes, start)
z0 = calibrate.to_free(start, calibrate.PARAMS)
sol = least_squares(fun, z0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
sc = calibrate.from_free(sol.x, start, calibrate.PARAMS)
rel = max(abs(getattr(sc, p) / getattr(fits[0].model, p) - 1) for p in calibrate.PARAMS)
check("scipy's least_squares on the same residuals lands on the same parameters (1e-6 relative)", rel < 1e-6,
      (rel, sc))

# 4. What the VIX family alone identifies.
vix_only = calibrate.layout(vix_tenors=(9 / 365, 30 / 365, 0.25, 0.5, 1.0), options=[], model=TRUE, sd_vix=0.001)
wrong = Heston(0.03, 0.05, 2.0, 0.9, -0.2)
f = calibrate.fit_heston(vix_only, wrong, fixed=("eta", "rho"))
rel = max(abs(getattr(f.model, p) / getattr(TRUE, p) - 1) for p in ("v0", "theta", "kappa"))
check("VIX family alone, eta and rho fixed at wrong values: v0, theta and kappa are still recovered (1e-6)",
      rel < 1e-6 and f.model.eta == 0.9 and f.model.rho == -0.2, (rel, f.model))
f5 = calibrate.fit_heston(vix_only, wrong)
check("VIX family alone, all five free: eta and rho are reported as unidentified (infinite standard error)",
      math.isinf(f5.se["eta"]) and math.isinf(f5.se["rho"]), f5.se)

# 4b. Many days, each with its own state variance (fit_heston_days, added 2026-10-23 for step 4): 40 synthetic days of
#     Dec-2029-like quotes (maturity 3.5 down to 2.4 years, 4 strikes a day) from Gatheral's parameters; with kappa
#     fixed, theta, eta and rho are recovered within 1e-6 relative; with theta fixed too, eta and rho.
rng4 = np.random.default_rng(11)
days = []
for i in range(40):
    T = 3.5 - i * 0.0275
    v = float(0.0174 * math.exp(rng4.normal(0, 0.4)))
    K = np.sort(rng4.choice(np.arange(6500.0, 15001.0, 500.0), 4, replace=False))
    F = 7600.0 + 20.0 * i
    iv = calibrate.black.implied_vols(calibrate.pricers.cos_prices(
        Heston(v, TRUE.theta, TRUE.kappa, TRUE.eta, TRUE.rho), F, K, T), F, K, T)
    days.append(calibrate.Day(v=v, F=F, T=T, K=K, iv=iv))
fd = calibrate.fit_heston_days(days, Heston(0.02, 0.05, TRUE.kappa, 0.7, -0.4), fixed=("v0", "kappa"))
rel = max(abs(getattr(fd.model, p) / getattr(TRUE, p) - 1) for p in ("theta", "eta", "rho"))
check("many days, each with its own variance: theta, eta and rho recovered within 1e-6 relative (kappa fixed)",
      rel < 1e-6 and fd.converged and fd.rmse < 1e-9, (rel, fd.model, fd.rmse))
fd2 = calibrate.fit_heston_days(days, Heston(0.02, TRUE.theta, TRUE.kappa, 0.7, -0.4), fixed=("v0", "kappa", "theta"))
rel = max(abs(getattr(fd2.model, p) / getattr(TRUE, p) - 1) for p in ("eta", "rho"))
check("... and eta and rho with theta fixed too", rel < 1e-6, (rel, fd2.model))
raises("fit_heston_days refuses a free v0 (each day has its own)",
       lambda: calibrate.fit_heston_days(days, TRUE, fixed=("kappa",)), "v0")

# 5. Standard errors, correlations and refusals.
f = fits[0]
check("with the full layout every standard error is finite and positive",
      all(math.isfinite(v) and v > 0 for v in f.se.values()), f.se)
C = f.corr
check("the correlation matrix of the estimates is symmetric with a unit diagonal",
      np.allclose(C, C.T, atol=1e-12) and np.allclose(np.diag(C), 1, atol=1e-12), C)
raises("a negative quoted vol is refused", lambda: calibrate.Quotes(T=[1.0], K=[100.0], F=[100.0], vol=[-0.1],
                                                                       sd=[0.01]), "vol")
raises("quote arrays of different lengths are refused", lambda: calibrate.Quotes(T=[1.0, 2.0], K=[100.0], F=[100.0],
                                                                                    vol=[0.2], sd=[0.01]), "length")
raises("an unknown parameter name in fixed is refused", lambda: calibrate.fit_heston(quotes, start, fixed=("nu",)),
       "unknown")
raises("fixing every parameter is refused", lambda: calibrate.fit_heston(quotes, start, fixed=calibrate.PARAMS),
       "nothing")
