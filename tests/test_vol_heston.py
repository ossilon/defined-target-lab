"""volmodel.heston: the Heston model's parameters, characteristic function and moments, judged by closed forms.

Judges (tolerances fixed 2026-10-21, before the first run):
- the characteristic function phi(u) = E[exp(i u X_T)] with X_T = ln(S_T / F_T): phi(0) = 1 and phi(-i) = 1 (the
  forward is a martingale), phi(-u) = conj(phi(u)) and |phi(u)| <= 1 on the real line; to 1e-13;
- the first two cumulants three ways: Fang and Oosterlee (2008, Table 11, with zero drift), Andersen (2008,
  Table A.1, written out below independently of the package) and numerical derivatives of log phi at 0
  (Richardson-extrapolated central differences); the closed forms to 1e-11 relative, the numerical ones to 1e-7;
  and c1 = -E[W_T] / 2, with E[W_T] = Gatheral's (2006, ch. 11) expected total variance;
- the lognormal limit: with rho = 0 and eta -> 0 the variance path is deterministic, so phi tends to the normal
  characteristic function with mean -W/2 and variance W; the gap shrinks like eta^2 (ratio 100 +/- 10% for a
  10x smaller eta);
- continuity on the real line at long maturities (the "little Heston trap"): the form used here (Gatheral eq. 2.12,
  Fang-Oosterlee section 3.3) has no jumps for Andersen's Case I at T = 10; the original form (Heston 1993, Gatheral
  eq. 2.14) with the principal logarithm does. Test: the largest step |phi(u + h) - phi(u)| on [0, 60] falls about
  10x when h falls 10x for a continuous function (ratio below 0.2) but not across a jump;
- the moments of the variance: E[v_t] and Var[v_t] (Andersen Corollary 1) against numerical derivatives of the joint
  transform are left to step 3 (simulation); here E[v_t] only, against its closed form;
- import guard: volmodel imports only the standard library, numpy, numba and scipy (and from step 3 the sibling
  package montecarlo), so other projects can import it unchanged.
"""
import ast
import math
import sys
from pathlib import Path

import numpy as np

from _harness import check, raises

from definedtarget.volmodel import heston
from definedtarget.volmodel.heston import Heston

PKG = Path(heston.__file__).resolve().parent

# Parameter sets used across the volmodel tests (sources in brackets).
FO = Heston(v0=0.0175, theta=0.0398, kappa=1.5768, eta=0.5751, rho=-0.5711)     # Fang-Oosterlee (2008) eq. 53
GATHERAL = Heston(v0=0.0174, theta=0.0354, kappa=1.3253, eta=0.3877, rho=-0.7165)  # Gatheral (2006) Table 3.2
CASE_I = Heston(v0=0.04, theta=0.04, kappa=0.5, eta=1.0, rho=-0.9)              # Andersen (2008) Table 1
CASE_II = Heston(v0=0.04, theta=0.04, kappa=0.3, eta=0.9, rho=-0.5)
CASE_III = Heston(v0=0.09, theta=0.09, kappa=1.0, eta=1.0, rho=-0.3)
SETS = {"Fang-Oosterlee": FO, "Gatheral 3.2": GATHERAL, "Andersen I": CASE_I, "Andersen II": CASE_II,
        "Andersen III": CASE_III}
TS = (9 / 365, 30 / 365, 0.25, 1.0, 3.2, 10.0, 15.0)


# 1. Parameters are checked; the Feller ratio is reported, not required (real fits usually break it).
raises("a negative variance is refused", lambda: Heston(-0.01, 0.04, 1.0, 0.5, -0.5), "v0")
raises("a zero mean-reversion speed is refused", lambda: Heston(0.04, 0.04, 0.0, 0.5, -0.5), "kappa")
raises("a correlation of 1 or more is refused", lambda: Heston(0.04, 0.04, 1.0, 0.5, 1.0), "rho")
raises("a NaN parameter is refused", lambda: Heston(0.04, float("nan"), 1.0, 0.5, -0.5), "theta")
raises("a zero vol of vol is refused (use the lognormal limit instead)", lambda: Heston(0.04, 0.04, 1.0, 0.0, 0.0),
       "eta")
check("Feller ratio 2 kappa theta / eta^2: Gatheral's fit breaks the condition (0.62 < 1)",
      abs(GATHERAL.feller - 2 * 1.3253 * 0.0354 / 0.3877 ** 2) < 1e-15 and GATHERAL.feller < 1, GATHERAL.feller)

# 2. The characteristic function's identities.
u = np.linspace(-80.0, 80.0, 1601)
for name, m in SETS.items():
    for T in TS:
        phi0 = heston.cf(m, 0.0, T)
        mart = heston.cf(m, -1j, T)
        ph = heston.cf(m, u, T)
        ok = (abs(phi0 - 1) < 1e-13 and abs(mart - 1) < 1e-13 and np.all(np.abs(ph) <= 1 + 1e-13)
              and np.allclose(heston.cf(m, -u, T), np.conj(ph), rtol=0, atol=1e-13))
        if not ok:
            break
    check(f"{name}: phi(0) = 1, phi(-i) = 1 (martingale), |phi| <= 1 and phi(-u) = conj(phi(u)) at 7 maturities",
          ok, (T, phi0, mart))


# 3. Cumulants three ways, and c1 = -E[W]/2. The package's c2 is Andersen's (2008, Table A.1); written out here
#    independently is Fang and Oosterlee's (2008, Table 11) form, as printed and as corrected. FINDING (2026-10-21):
#    the printed form has theta (6 e^{-kT} - 7) where theta (4 e^{-kT} - 5) is right. The printed form fails the
#    numerical derivatives, has a term of order T that no variance can have at short maturities, and turns negative
#    (FO's own remark that "c2 may become negative"); the corrected form equals Andersen's to rounding.
def fo_c2(m, T, corrected):
    k, th, eta, rho, v0 = m.kappa, m.theta, m.eta, m.rho, m.v0
    e = math.exp(-k * T)
    a, b = (4, 5) if corrected else (6, 7)
    return (eta * T * k * e * (v0 - th) * (8 * k * rho - 4 * eta) + k * rho * eta * (1 - e) * (16 * th - 8 * v0)
            + 2 * th * k * T * (-4 * k * rho * eta + eta ** 2 + 4 * k ** 2)
            + eta ** 2 * ((th - 2 * v0) * e * e + th * (a * e - b) + 2 * v0) + 8 * k ** 2 * (v0 - th) * (1 - e)) / (8 * k ** 3)


def numeric_cumulants(m, T, scale=0.001):
    """c1, c2 from log phi near 0: central differences, Richardson-extrapolated over h and 2h (error O(h^4)).
    The step is h = scale / sqrt(E[W_T]), fixed in units of the distribution's own width. First runs (2026-10-21):
    a fixed h = 0.002 was rounding-limited at 9 days (2.0e-7; smaller h made it worse); scale 0.01 was
    truncation-limited at 15 years (8e-5: the expansion's radius shrinks at long maturities, where Heston's moments
    explode sooner); scale 0.001 gives below 1e-8 at both ends."""
    h = scale / math.sqrt(m.expected_variance(T))

    def c12(s):
        lp, lm = heston.log_cf(m, s, T), heston.log_cf(m, -s, T)
        return ((lp - lm) / (2 * s)).imag, -((lp + lm) / s ** 2).real
    a1, a2 = c12(h)
    b1, b2 = c12(2 * h)
    return (4 * a1 - b1) / 3, (4 * a2 - b2) / 3


worst = {"numeric c1": 0.0, "numeric c2": 0.0, "FO corrected": 0.0, "FO printed": 0.0, "c1 vs -E[W]/2": 0.0}
for name, m in SETS.items():
    for T in TS:
        c1, c2 = m.cumulants(T)
        n1, n2 = numeric_cumulants(m, T)
        worst["numeric c1"] = max(worst["numeric c1"], abs(n1 / c1 - 1))
        worst["numeric c2"] = max(worst["numeric c2"], abs(n2 / c2 - 1))
        worst["FO corrected"] = max(worst["FO corrected"], abs(fo_c2(m, T, True) / c2 - 1))
        worst["FO printed"] = max(worst["FO printed"], abs(fo_c2(m, T, False) / c2 - 1))
        worst["c1 vs -E[W]/2"] = max(worst["c1 vs -E[W]/2"], abs(c1 / (-m.expected_variance(T) / 2) - 1))
check("c1 and c2 (Andersen's form) equal the numerical derivatives of log phi at 0 (35 cases, 1e-7)",
      worst["numeric c1"] < 1e-7 and worst["numeric c2"] < 1e-7, worst)
check("c2 equals Fang-Oosterlee's Table 11 form with theta (4 e^{-kT} - 5) (1e-11)",
      worst["FO corrected"] < 1e-11, worst)
check("Fang-Oosterlee's c2 as printed, theta (6 e^{-kT} - 7), is off (by more than 1% in some case)",
      worst["FO printed"] > 0.01, worst)
check("c1 equals -E[W_T]/2, Gatheral's expected total variance (1e-11)", worst["c1 vs -E[W]/2"] < 1e-11, worst)
check("the variance of X_T is positive in every case (Fang-Oosterlee's |c2| guard is never needed here)",
      all(m.cumulants(T)[1] > 0 for m in SETS.values() for T in TS))
check("the printed c2 does turn negative in one of these cases (the symptom Fang and Oosterlee guarded against)",
      any(fo_c2(m, T, False) < 0 for m in SETS.values() for T in TS),
      min(fo_c2(m, T, False) for m in SETS.values() for T in TS))
# Code review, 2026-10-23: the closed form loses digits as kappa shrinks (relative error 2e-8 at kappa = 0.001, a
# negative variance at kappa = 1e-6); below kappa = 0.01 or kappa T = 0.001 the package uses log phi's second
# derivative instead.
small = [(Heston(0.04, 0.04, kap, 0.5, -0.7), T) for kap in (1e-3, 1e-4, 1e-6) for T in (9 / 365, 1.0)]
gaps = [abs(m.cumulants(T)[1] / numeric_cumulants(m, T)[1] - 1) for m, T in small]
check("tiny kappa (1e-3 to 1e-6): c2 stays positive and equals the numerical derivative (1e-7)",
      max(gaps) < 1e-7 and all(m.cumulants(T)[1] > 0 for m, T in small), gaps)
edge = Heston(0.04, 0.04, 0.01, 0.5, -0.7)
check("at the switch (kappa = 0.01) the closed form and the numerical derivative agree (1e-9)",
      abs(edge.cumulants(1.0)[1] / heston._variance_numeric(edge, 1.0) - 1) < 1e-9)
EV = GATHERAL.expected_variance(3.2)
check("expected total variance: theta T + (v0 - theta)(1 - e^{-kappa T}) / kappa (Gatheral ch. 11)",
      abs(EV - (0.0354 * 3.2 + (0.0174 - 0.0354) * (1 - math.exp(-1.3253 * 3.2)) / 1.3253)) < 1e-16, EV)
check("E[v_t] = theta + (v0 - theta) e^{-kappa t} (Andersen Corollary 1)",
      abs(GATHERAL.mean_variance(2.0) - (0.0354 + (0.0174 - 0.0354) * math.exp(-2 * 1.3253))) < 1e-16)
check("the variance-swap rate is the expected variance per year",
      abs(GATHERAL.variance_swap_rate(3.2) - EV / 3.2) < 1e-17)


# 4. The lognormal limit (rho = 0): the gap to the normal characteristic function shrinks like eta^2.
def lognormal_gap(eta, T=3.2):
    m = Heston(v0=0.0174, theta=0.0354, kappa=1.3253, eta=eta, rho=0.0)
    W = m.expected_variance(T)
    uu = np.linspace(-20, 20, 401)
    return np.max(np.abs(heston.cf(m, uu, T) - np.exp(-0.5 * W * (uu * uu + 1j * uu))))


g1, g2 = lognormal_gap(1e-2), lognormal_gap(1e-3)
check("lognormal limit: the gap to the normal characteristic function is small (eta = 0.001: below 1e-6)",
      g2 < 1e-6, g2)
check("lognormal limit: the gap shrinks like eta^2 (10x smaller eta, 100x smaller gap, +/- 10%)",
      90 < g1 / g2 < 110, g1 / g2)


# 5. The little Heston trap: the form used here is continuous at long maturities; the original form jumps.
def cf_original(m, uu, T):
    """Heston (1993) / Gatheral eq. 2.14: the same function, written with e^{+dT}; principal logarithm."""
    k, th, eta, rho, v0 = m.kappa, m.theta, m.eta, m.rho, m.v0
    beta = k - 1j * rho * eta * uu
    d = np.sqrt(beta * beta + eta * eta * (uu * uu + 1j * uu))
    g = (beta + d) / (beta - d)
    e = np.exp(d * T)
    D = (beta + d) / eta ** 2 * (1 - e) / (1 - g * e)
    C = k * th / eta ** 2 * ((beta + d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + D * v0)


def jump_ratio(f, hi=60.0):
    """max |f(u+h) - f(u)| at h = 0.001 over the same at h = 0.01: about 0.1 if f is continuous, near 1 at a jump.
    The grid starts just above 0 because the original form divides 0 by 0 at u = 0 exactly."""
    fine, coarse = f(np.arange(0.0005, hi, 0.001)), f(np.arange(0.005, hi, 0.01))
    return np.max(np.abs(np.diff(fine))) / np.max(np.abs(np.diff(coarse)))


r_safe = jump_ratio(lambda x: heston.cf(CASE_I, x, 10.0))
r_orig = jump_ratio(lambda x: cf_original(CASE_I, x, 10.0))
check("the characteristic function used here is continuous (Andersen Case I, T = 10: step ratio below 0.2)",
      r_safe < 0.2, r_safe)
check("the original form with the principal logarithm jumps there (the trap this form avoids: ratio above 0.5)",
      r_orig > 0.5, r_orig)
uu = np.linspace(0.01, 0.19, 10)            # the original form's first jump is near u = 0.2 here (first run)
check("before its first jump the original form is the same function (u in [0.01, 0.19], 1e-12)",
      np.allclose(cf_original(CASE_I, uu, 10.0), heston.cf(CASE_I, uu, 10.0), rtol=0, atol=1e-12))
check("after it, the original form is a different, wrong function (gap above 0.1 at u = 0.5)",
      abs(cf_original(CASE_I, 0.5, 10.0) - heston.cf(CASE_I, 0.5, 10.0)) > 0.1)

# 6. Import guard: volmodel stays project-agnostic.
allowed = {"numpy", "numba", "scipy"} | set(sys.stdlib_module_names)
bad = []
for f in sorted(PKG.glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            bad += [f"{f.name}: {a.name}" for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module.split(".")[0] not in allowed:
            bad.append(f"{f.name}: from {node.module}")
        elif isinstance(node, ast.ImportFrom) and node.level >= 2 and (node.level > 2 or not (node.module or "")
                                                                         .startswith("montecarlo")):
            bad.append(f"{f.name}: from {'.' * node.level}{node.module or ''}")
check("volmodel imports only the standard library, numpy, numba, scipy and the sibling montecarlo", not bad, bad)
