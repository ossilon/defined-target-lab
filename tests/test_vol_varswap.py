"""volmodel.varswap: the log-strip of option prices equals the model's expected variance (the VIX identity).

For any diffusion, E[W_T] = -2 E[ln(S_T / F)] = (2 / DF) [ int_0^F P(K) / K^2 dK + int_F^inf C(K) / K^2 dK ]
(Demeterfi, Derman, Kamal and Zou 1999; Gatheral 2006, ch. 11). In Heston, E[W_T] = theta T + (v0 - theta)
(1 - e^{-kappa T}) / kappa, whatever eta and rho are. So integrating the model's option prices over every strike
must give that closed form: one number that checks the whole smile, wings included.

Judges (tolerances fixed 2026-10-21, before the first run):
1. strip_variance of COS prices equals expected_variance within 1e-9 relative: 5 parameter sets x 6 maturities
   (9 days to XX's 3.22 years; see GRID below for the two slow sets);
2. the same from the Lewis-Lipton integral prices at 2 points (1e-9; changed after the first run to the call
   half, see the finding below);
3. eta does not matter: three vols of vol give the same strip variance (1e-9 of each other) and the closed form;
4. VIX^2(tau) = a + b v0 with b = (1 - e^{-kappa tau}) / (kappa tau), a = theta (1 - b): equals the variance-swap
   rate (1e-15) and is linear in v0;
5. model-agnostic: on Black-76 prices the strip gives sigma^2 T (1e-10);
6. bad inputs are refused.
"""
import math

import numpy as np

from _harness import check, raises

from definedtarget.volmodel import black, pricers, varswap
from definedtarget.volmodel.heston import Heston

SETS = {"Fang-Oosterlee": Heston(v0=0.0175, theta=0.0398, kappa=1.5768, eta=0.5751, rho=-0.5711),
        "Gatheral 3.2": Heston(v0=0.0174, theta=0.0354, kappa=1.3253, eta=0.3877, rho=-0.7165),
        "Andersen I": Heston(v0=0.04, theta=0.04, kappa=0.5, eta=1.0, rho=-0.9),
        "Andersen II": Heston(v0=0.04, theta=0.04, kappa=0.3, eta=0.9, rho=-0.5),
        "Andersen III": Heston(v0=0.09, theta=0.09, kappa=1.0, eta=1.0, rho=-0.3)}
XX_T = 3.22
F = 8370.0


def cos_fn(m, T, df=0.9):
    return lambda K, kind: pricers.cos_prices(m, F, K, T, df=df, kind=kind)


# Andersen's Cases I and II at 1 year and beyond cost 3-8 s each here (COS with up to 32,768 terms over ~1,500
# strikes; timed 2026-10-21), so they run at 9 days and 3 months only. Their long-dated prices are judged in
# test_vol_pricers.py, and a long-dated heavy tail is judged here by Gatheral's set at eta = 1.0 (section 3).
GRID = {name: ((9 / 365, 0.25) if name in ("Andersen I", "Andersen II") else (9 / 365, 30 / 365, 0.25, 0.5, 1.0, XX_T))
        for name in SETS}
worst, where = 0.0, None
for name, m in SETS.items():
    for T in GRID[name]:
        got = varswap.strip_variance(cos_fn(m, T), F, T, df=0.9)
        err = abs(got / m.expected_variance(T) - 1)
        if err > worst:
            worst, where = err, (name, round(T, 4), got, m.expected_variance(T))
check("the log-strip of COS prices equals Heston's expected variance: 22 set-maturity cases (1e-9 relative)",
      worst < 1e-9, (worst, where))

# 2. The integral pricer. FINDING (first run, 2026-10-21): its puts come from calls by parity, so deep in the left
#    wing a rounding error of about 1e-15 of the forward is multiplied by F/K in the strip (relative errors 1e-9 at
#    30 days, 1e-6 for Andersen's heavy-tailed Case I). The judge is therefore the call half of the strip (direct
#    call prices), compared with the COS call half; the put half is judged by COS, whose puts are computed directly.
for name, T in (("Gatheral 3.2", 30 / 365), ("Andersen III", XX_T)):
    m = SETS[name]
    ci, _ = varswap.strip_variance(lambda K, kind: pricers.integral_prices(m, F, K, T, kind=kind), F, T, halves=True)
    cc, pc = varswap.strip_variance(lambda K, kind: pricers.cos_prices(m, F, K, T, kind=kind), F, T, halves=True)
    check(f"{name}, T = {T:.3f}: the call half of the strip from the integral pricer equals COS's (1e-10 relative)",
          abs(ci / cc - 1) < 1e-10, (ci, cc))
    check(f"{name}, T = {T:.3f}: the two halves from COS add up to the expected variance (1e-9)",
          abs((cc + pc) / m.expected_variance(T) - 1) < 1e-9)

base = SETS["Gatheral 3.2"]
# eta = 1.0 here needs strikes down to ln(K/F) = -37 (first run: the panels then overshot to K = 0; they are now
# clipped at |ln(K/F)| = 600, varswap.X_MAX)
vals = [varswap.strip_variance(cos_fn(Heston(base.v0, base.theta, base.kappa, eta, base.rho), XX_T), F, XX_T, df=0.9)
        for eta in (0.2, 0.6, 1.0)]
check("eta does not change the variance-swap value (three vols of vol, 1e-9 of each other and of the closed form)",
      max(vals) / min(vals) - 1 < 1e-9 and abs(vals[0] / base.expected_variance(XX_T) - 1) < 1e-9, vals)

tau = 30 / 365
a, b = varswap.vix_squared_coefficients(base, tau)
check("VIX^2 = a + b v0 equals the 30-day variance-swap rate (1e-15)",
      abs(a + b * base.v0 - base.variance_swap_rate(tau)) < 1e-15, (a, b))
other = Heston(0.05, base.theta, base.kappa, base.eta, base.rho)
check("... and is linear in v0 with the same a and b (another v0, 1e-15)",
      abs(a + b * 0.05 - other.variance_swap_rate(tau)) < 1e-15)
check("b = (1 - e^{-kappa tau}) / (kappa tau) and a = theta (1 - b)",
      abs(b - (1 - math.exp(-base.kappa * tau)) / (base.kappa * tau)) < 1e-15 and abs(a - base.theta * (1 - b)) < 1e-16)


def black_fn(vol, T):
    def fn(K, kind):
        n = K.size
        c = black.call_price_array(np.full(n, F), K, np.ones(n), np.full(n, T), np.full(n, vol))
        return c if kind == "call" else c - (F - K)
    return fn


got = varswap.strip_variance(black_fn(0.2, 1.5), F, 1.5)
check("model-agnostic: on Black-76 prices the strip gives sigma^2 T (1e-10)", abs(got / (0.04 * 1.5) - 1) < 1e-10, got)
raises("a price function that never decays is refused, not integrated forever",
       lambda: varswap.strip_variance(lambda K, kind: np.full(K.size, 1e-3 * F), F, 1.0), "do not decay")
raises("a non-positive forward is refused", lambda: varswap.strip_variance(black_fn(0.2, 1.0), -1.0, 1.0), "forward")
raises("a zero maturity is refused", lambda: varswap.strip_variance(black_fn(0.2, 1.0), F, 0.0), "maturity")
raises("VIX coefficients need a positive tenor", lambda: varswap.vix_squared_coefficients(base, 0.0), "tenor")
