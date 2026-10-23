"""The two-factor Heston model (heston.DoubleHeston): pricing, the term structure and its level split, the many-day
smile fit and the simulation, judged by exact identities, by an independent simulation and by recovery.

Judges (tolerances fixed 2026-10-23, before the first run; SE = Monte Carlo standard error):
1. The model: the variance now, the long-run level, E[W_T] and the mean and variance of X_T are the sums over the
   factors; the variance of X_T equals the numerical second derivative of log phi (1e-8 relative); bad factors are
   refused; cos_call_surface refuses a two-factor model.
2. The reduction: with equal kappa, eta and rho, log phi equals one-factor Heston's with v0 = v1 + v2 and
   theta = theta1 + theta2 (1e-12 relative, real and complex u), and so do the COS prices (1e-12 of F).
3. The two pricers agree: COS and the Lewis-Lipton integral, for a fast-and-slow model with separate eta and rho
   and for a shared one, over strikes from 0.5 F to 2 F (XX's 1.195 F included) and maturities from 30 days to 10
   years: within 1e-12 of F.
4. The variance-swap identity: the log strip of COS prices equals E[W_T] (1e-9 relative) at 30 days, 1 year and XX's
   3.2 years; coefficients_two gives the same rate as the model (1e-15); as kappa2 -> 0 the slow factor adds
   c2 tau / 2 (1e-11), and the small-kappa series meets the closed form (1e-12).
5. The term structure (judges rewritten 2026-10-23 for the constrained fit, before its first run): an interior day's
   variances equal a direct weighted least-squares solve (1e-12); a day whose free solution has v1 < 0 gets v1 = 0
   and beats every point of a grid over v >= 0; noise-free recovery of kappa1, theta1, kappa2 and c2 (1e-6 relative)
   and of every day's variances (1e-8) from days that include zeros and a missing index; a profile point costs more;
   at the boundary (kappa2 = 1e-7) kappa1, theta1 and c2 are recovered (1e-5) with kappa2 at the floor; with 0.3-point
   noise the bootstrap band of theta1 holds the truth, and the bootstrap is deterministic.
6. vol_of_variance: on 20 simulated 20-year daily paths, read through noise-free indices (so states_two is in the
   loop), the means of the weekly eta1, eta2 and pooled eta are within 3 SE of the truth and the innovation
   correlation within 3 SE of 0, inside (kappa2 = 0.4) and at the boundary (kappa2 = 1e-6).
7. The many-day smile fit: 12 synthetic Dec-2029-like days from a known model; theta2, eta and rho recovered within
   1e-6 relative (kappas and theta1 fixed); a start whose factors do not share eta is refused.
8. The simulation: (a) both eta = 0 with v_i = theta_i is montecarlo.paths.lognormal bit for bit on the NYSE
   calendar; (b) factor 1's variance path equals one-factor stochvol's variance path bit for bit (same stream, same
   QE step); (c) E[v_i(T)] and Var[v_i(T)] equal Andersen's Corollary 1 within 3 SE at 1-year steps; (d) the
   martingale identity E[S_T] = S_0 e^{mu T} within 3 SE with the correction at 1-year steps; (e) on the daily NYSE
   calendar, the Dec-2029 10,000 call and an at-the-money call within 3 SE of COS; (f) one call or two chunks, 1
   thread or all: bit-identical; (g) one factor alone at eta = 0, the Euler scheme and the last stream are refused.
"""
import math
from datetime import date

import numba
import numpy as np

from _harness import check, raises

from definedtarget import calendar as nyse
from definedtarget.montecarlo import paths
from definedtarget.volmodel import calibrate, heston, pricers, stochvol, termstructure as ts, varswap
from definedtarget.volmodel.heston import DoubleHeston, Heston

SEED = 20261023
N6 = 1 << 20
FAST = Heston(0.022, 0.012, 6.0, 0.9, -0.6)
SLOW = Heston(0.025, 0.030, 0.4, 0.6, -0.85)
MODEL = DoubleHeston(FAST, SLOW)                                    # separate eta and rho
SHARED = DoubleHeston(Heston(0.022, 0.012, 6.0, 0.8, -0.8), Heston(0.025, 0.030, 0.4, 0.8, -0.8))


def mean_se(x):
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(x.size))


# 1. The model's sums.
T = 3.2
check("v0, theta and E[W_T] are sums over the factors",
      MODEL.v0 == FAST.v0 + SLOW.v0 and MODEL.theta == FAST.theta + SLOW.theta
      and MODEL.expected_variance(T) == FAST.expected_variance(T) + SLOW.expected_variance(T))
c1, c2 = MODEL.cumulants(T)
num = heston._variance_numeric(MODEL, T)
check("the variance of X_T (sum of the factors' closed forms) equals -d2 log phi / du2 at 0 (1e-8 relative)",
      abs(c2 / num - 1) < 1e-8 and abs(c1 + MODEL.expected_variance(T) / 2) < 1e-15, (c2, num))
raises("a factor that is not a Heston factor is refused", lambda: DoubleHeston(FAST, (0.02, 0.03, 0.4, 0.6, -0.8)),
       "Heston factor")
raises("cos_call_surface takes one-factor models only", lambda: pricers.cos_call_surface(MODEL, [0.0], [0.02], 1.0),
       "one-factor")

# 2. The reduction to one factor.
eq = DoubleHeston(Heston(0.011, 0.015, 1.3, 0.5, -0.7), Heston(0.014, 0.022, 1.3, 0.5, -0.7))
one = Heston(0.011 + 0.014, 0.015 + 0.022, 1.3, 0.5, -0.7)
u = np.concatenate((np.linspace(0.01, 60.0, 300), np.linspace(0.01, 60.0, 50) - 0.5j, -1j * np.linspace(0.05, 3, 20)))
lt, lo = heston.log_cf(eq, u, 2.0), heston.log_cf(one, u, 2.0)
check("equal kappa, eta and rho: log phi equals one-factor Heston's with summed v and theta (1e-12 relative)",
      np.max(np.abs(lt - lo) / np.maximum(1.0, np.abs(lo))) < 1e-12, np.max(np.abs(lt - lo)))
K = 100.0 * np.array([0.5, 0.8, 1.0, 1.195, 1.5, 2.0])
gap = max(float(np.max(np.abs(pricers.cos_prices(eq, 100.0, K, t) - pricers.cos_prices(one, 100.0, K, t))))
          for t in (0.25, 1.0, 3.2))
check("... and the COS prices are equal (1e-12 of F)", gap < 1e-10, gap)

# 3. The two pricers agree.
worst = (0.0, None)
for name, m in (("separate eta and rho", MODEL), ("shared eta and rho", SHARED)):
    for t in (30 / 365, 0.5, 1.0, 3.2, 10.0):
        for kind in ("call", "put"):
            g = float(np.max(np.abs(pricers.cos_prices(m, 100.0, K, t, kind=kind)
                                    - pricers.integral_prices(m, 100.0, K, t, kind=kind)))) / 100.0
            if g > worst[0]:
                worst = (g, (name, round(t, 3), kind))
check(f"COS and the Lewis-Lipton integral agree within 1e-12 of F (worst {worst[0]:.1e} at {worst[1]})",
      worst[0] < 1e-12, worst)

# 4. The variance-swap identity.
errs = []
for t in (30 / 365, 1.0, 3.2):
    got = varswap.strip_variance(lambda k, kind, t=t: pricers.cos_prices(MODEL, 100.0, k, t, df=0.9, kind=kind),
                                 100.0, t, df=0.9)
    errs.append(abs(got / MODEL.expected_variance(t) - 1))
check("the log strip of COS prices equals E[W_T] at 30 days, 1 year and 3.2 years (1e-9 relative)", max(errs) < 1e-9,
      errs)
tau2 = np.array([30 / 365, 3.2])
a2, b1, b2 = ts.coefficients_two(FAST.kappa, FAST.theta, SLOW.kappa, SLOW.kappa * SLOW.theta, tau2)
check("coefficients_two: a + b1 v1 + b2 v2 is the model's variance-swap rate (1e-15)",
      np.max(np.abs(a2 + b1 * FAST.v0 + b2 * SLOW.v0 - [MODEL.variance_swap_rate(t) for t in tau2])) < 1e-15)
tau4 = np.array([30 / 365, 1.0, 3.2])
a_lim = ts.coefficients_two(6.0, 0.012, 1e-12, 0.01, tau4)[0] - ts.coefficients(6.0, 0.012, tau4)[0]
lo3, hi3 = ts._half_tail(np.array([1e-3 * (1 - 1e-12), 1e-3]))
check("as kappa2 -> 0 the slow factor adds c2 tau / 2 (1e-11 relative), and the series meets the closed form at "
      "kappa tau = 1e-3 (1e-12)", np.max(np.abs(a_lim / (0.01 * tau4 / 2) - 1)) < 1e-11 and abs(lo3 - hi3) < 1e-12,
      (a_lim, lo3, hi3))

# 5. The term structure. Synthetic days where some variances sit at 0: only such days identify theta1.
TENORS = np.array([30, 93, 184, 366]) / 365.0
P_TRUE = (6.0, 0.015, 0.4, 0.4 * 0.03)                              # kappa1, theta1, kappa2, c2
rng = np.random.default_rng(2026)
v_true = np.column_stack((0.015 * np.exp(rng.normal(0, 0.8, 600)), 0.03 * np.exp(rng.normal(0, 0.4, 600))))
v_true[::5, 0] = 0.0                                                # calm days: no fast-factor variance
v_true[3::11, 1] = 0.0


def synthetic(p, v):
    aa, bb1, bb2 = ts.coefficients_two(*p, TENORS)
    return np.sqrt(aa[None, :] + bb1[None, :] * v[:, :1] + bb2[None, :] * v[:, 1:])


vols = synthetic(P_TRUE, v_true)
holes = vols.copy()
holes[::7, 2] = np.nan
sd = np.full(4, 0.005)
aa, bb1, bb2 = ts.coefficients_two(*P_TRUE, TENORS)
day = holes[7]
obs = np.isfinite(day)
w = 1.0 / (2.0 * day[obs] * 0.005) ** 2
A = np.column_stack((bb1[obs], bb2[obs])) * np.sqrt(w)[:, None]
direct = np.linalg.lstsq(A, (day[obs] ** 2 - aa[obs]) * np.sqrt(w), rcond=None)[0]
got = ts.states_two(*P_TRUE, TENORS, day[None, :], sd)[0]
check("an interior day's variances equal a direct weighted least-squares solve, with an index missing (1e-12)",
      np.max(np.abs(got - direct)) < 1e-12 and np.max(np.abs(got - v_true[7])) < 1e-12, (got, direct, v_true[7]))
low = synthetic(P_TRUE, np.array([[0.0, 0.03]])) * np.array([0.8, 1.0, 1.0, 1.0])     # short end below the model's
g1 = np.linspace(0.0, 0.01, 401)
g2 = np.linspace(0.02, 0.04, 2001)
grid = np.stack(np.meshgrid(g1, g2, indexing="ij"), axis=-1).reshape(-1, 2)
ww = 1.0 / (2.0 * low[0] * 0.005) ** 2
objective = lambda vv: ((ww * (low[0] ** 2 - aa - bb1 * vv[..., :1] - bb2 * vv[..., 1:]) ** 2).sum(axis=-1))  # noqa
gl = ts.states_two(*P_TRUE, TENORS, low, sd)[0]
check("a day whose free solution has v1 < 0 gets v1 = 0, and no point of a grid over v >= 0 does better",
      gl[0] == 0.0 and gl[1] > 0 and objective(gl) <= objective(grid).min() + 1e-12, (gl, objective(gl),
                                                                                       objective(grid).min()))
f2 = ts.fit_two(TENORS, holes, sd, start=(3.0, 0.03, 1.0, 0.03))
rel = max(abs(g / t - 1) for g, t in zip(f2.params, P_TRUE))
check("noise-free recovery of kappa1, theta1, kappa2 and c2 (1e-6 relative)", rel < 1e-6 and f2.converged,
      (f2.params, f2.message))
check("... and of every day's variances, zeros included (1e-8)", np.nanmax(np.abs(f2.v - v_true)) < 1e-8,
      np.nanmax(np.abs(f2.v - v_true)))
fp = ts.fit_two(TENORS, holes, sd, start=(3.0, 0.03, 1.0, 0.03), fixed=("theta1",))
# (first run, 2026-10-23: the test compared theta1 == 0.03 exactly; it comes back as exp(log(0.03)), one ulp off)
check("a profile point (theta1 held at a wrong value) costs more and keeps theta1 fixed",
      abs(fp.theta1 - 0.03) < 1e-15 and fp.cost > f2.cost + 1.0, (fp.theta1, fp.cost, f2.cost))
P_B = (6.0, 0.015, 1e-7, 0.0115)
fb_ = ts.fit_two(TENORS, synthetic(P_B, v_true), sd, start=(3.0, 0.03, 0.5, 0.03))
check("at the boundary (data from kappa2 = 1e-7): kappa1, theta1 and c2 recovered (1e-5 relative), kappa2 at the "
      "floor (below 1e-5)", abs(fb_.kappa1 / 6.0 - 1) < 1e-5 and abs(fb_.theta1 / 0.015 - 1) < 1e-5
      and abs(fb_.c2 / 0.0115 - 1) < 1e-5 and fb_.kappa2 < 1e-5, fb_.params)
noisy = holes + rng.normal(0.0, 0.003, holes.shape)
boot, conv = ts.bootstrap_two(TENORS, noisy, np.full(4, 0.003), block=63, reps=40, seed=7)
lo_, hi_ = np.percentile(boot[:, 1], [5, 95])
check(f"with 0.3-point noise the bootstrap 5-95% band of theta1 ({lo_:.4f} to {hi_:.4f}) holds the truth",
      lo_ <= P_TRUE[1] <= hi_ and conv.all(), (lo_, hi_, int(conv.sum())))
boot2, _ = ts.bootstrap_two(TENORS, noisy, np.full(4, 0.003), block=63, reps=40, seed=7)
check("the two-factor bootstrap is deterministic for a given seed", np.array_equal(boot, boot2))
raises("a start with kappa1 <= kappa2 is refused",
       lambda: ts.fit_two(TENORS, holes, sd, start=(0.4, 0.02, 0.4, 0.04)), "kappa1 > kappa2")

# 6. Each factor's eta from the time series, through noise-free indices: inside (kappa2 = 0.4) and at the boundary
#    (kappa2 = 1e-6, theta2 = c2 / kappa2).
days = 20 * 252
t20 = np.arange(days + 1) / 252.0
for label, k2s, th2 in (("kappa2 = 0.4", 0.4, 0.028), ("kappa2 = 1e-6", 1e-6, 0.0115 / 1e-6)):
    model6 = DoubleHeston(Heston(0.012, 0.012, 6.0, 0.5, -0.7), Heston(0.028, th2, k2s, 0.5, -0.7))
    p6 = (6.0, 0.012, k2s, k2s * th2)
    o = stochvol.simulate_two_model(model6, SEED, 20, t20, checkpoints=tuple(range(1, days + 1)))
    ests = []
    for p in range(20):
        v = o["v_at_checkpoint"][p]                                  # (days, 2)
        ests.append([ts.vol_of_variance(*p6, ts.states_two(*p6, TENORS, synthetic(p6, v), sd), t20[1:])[k]
                     for k in ("eta1", "eta2", "eta", "corr")])
    ests = np.array(ests)
    mean, se = ests.mean(axis=0), ests.std(axis=0, ddof=1) / math.sqrt(20)
    check(f"{label}: weekly eta1, eta2 and pooled eta recovered, innovation correlation 0 (3 SE; means "
          f"{np.round(mean, 4).tolist()}, SE {np.round(se, 4).tolist()})",
          np.all(np.abs(mean - [0.5, 0.5, 0.5, 0.0]) < 3 * se), (mean, se))

# 7. The many-day smile fit.
rng7 = np.random.default_rng(11)
days7 = []
for i in range(12):
    v1, v2 = 0.012 * math.exp(rng7.normal(0, 0.5)), 0.026 * math.exp(rng7.normal(0, 0.3))
    Tm, F = 3.5 - 0.1 * i, 7600.0 + 20.0 * i
    Km = np.array([8000.0, 9000.0, 10000.0, 12000.0])
    iv = calibrate.black.implied_vols(pricers.cos_prices(
        calibrate.double_model({"theta1": 0.012, "theta2": 0.030, "eta": 0.8, "rho": -0.8}, 6.0, 0.4, v1, v2),
        F, Km, Tm), F, Km, Tm)
    days7.append(calibrate.Day2(v1=v1, v2=v2, F=F, T=Tm, K=Km, iv=iv))
start7 = DoubleHeston(Heston(0.01, 0.012, 6.0, 0.5, -0.5), Heston(0.02, 0.05, 0.4, 0.5, -0.5))
fd = calibrate.fit_double_days(days7, start7)
m = fd.model
rel = max(abs(m.f2.theta / 0.030 - 1), abs(m.f1.eta / 0.8 - 1), abs(m.f1.rho / -0.8 - 1))
check("many days with their own v1, v2: theta2, eta and rho recovered within 1e-6 relative",
      rel < 1e-6 and fd.converged and fd.rmse < 1e-9 and m.f1.eta == m.f2.eta and m.f1.theta == 0.012,
      (rel, m, fd.rmse))
check("... the fit reports finite standard errors for the free parameters and NaN for the fixed one",
      all(math.isfinite(fd.se[p]) for p in ("theta2", "eta", "rho")) and math.isnan(fd.se["theta1"]), fd.se)
raises("a start whose factors do not share eta and rho is refused",
       lambda: calibrate.fit_double_days(days7, MODEL), "share")

# 8. The simulation.
t_cal = paths.times_from_dates(nyse.sessions(date(2026, 10, 23), date(2030, 1, 10)))
cps = (100, 400, t_cal.size - 1)
f1z, f2z = (0.012, 0.012, 6.0, 0.0, 0.0), (0.0156, 0.0156, 0.4, 0.0, 0.0)
a = stochvol.simulate_two(SEED, 20_000, t_cal, f1z, f2z, mu=0.03, barrier=1.28, checkpoints=cps)
b = paths.lognormal(SEED, 20_000, t_cal, mu=0.03, vol=math.sqrt(0.012 + 0.0156), barrier=1.28, checkpoints=cps)
same = all(np.array_equal(a[k], b[k]) for k in ("log_end", "log_max", "log_drawdown_end", "log_max_drawdown",
                                                "hit_step", "at_checkpoint"))
check("(a) both eta = 0, v_i = theta_i: every shared output equals montecarlo.paths.lognormal bit for bit", same)
check("... and each factor's variance stays at its theta", np.all(a["v_end"] == [0.012, 0.0156]))
t6 = np.linspace(0.0, 3.0, 157)
o2 = stochvol.simulate_two_model(MODEL, SEED, 6000, t6, checkpoints=(52, 156))
o1 = stochvol.simulate_model(FAST, SEED, 6000, t6, checkpoints=(52, 156))
check("(b) factor 1's variance path equals one-factor stochvol's, bit for bit (stream + 1, the same QE step)",
      np.array_equal(o2["v_end"][:, 0], o1["v_end"]) and np.array_equal(o2["v_at_checkpoint"][:, :, 0],
                                                                         o1["v_at_checkpoint"]))
grid10 = np.linspace(0.0, 10.0, 11)
oc = stochvol.simulate_two_model(MODEL, SEED, N6, grid10, mu=0.02)
fails = []
for i, f in enumerate((FAST, SLOW)):
    vv = oc["v_end"][:, i]
    mv, se_m = mean_se(vv)
    x = math.exp(-f.kappa * 10.0)
    var_true = f.v0 * f.eta ** 2 / f.kappa * (x - x * x) + f.theta * f.eta ** 2 / (2 * f.kappa) * (1 - x) ** 2
    sv = float(np.var(vv, ddof=1))
    se_v = math.sqrt(max(float(np.mean((vv - mv) ** 4)) - sv * sv, 0.0) / vv.size)
    if not (abs(mv - f.mean_variance(10.0)) < 3 * se_m and abs(sv - var_true) < 3 * se_v):
        fails.append((i, mv, f.mean_variance(10.0), se_m, sv, var_true, se_v))
check("(c) E[v_i(T)] and Var[v_i(T)] equal Andersen's Corollary 1 for both factors at 1-year steps (3 SE)", not fails,
      fails)
g, se_g = mean_se(np.exp(oc["log_end"] - 0.02 * 10.0))
check("(d) the martingale identity with the correction at 1-year steps: E[S_T] = S_0 e^(mu T) (3 SE)",
      abs(g - 1) < 3 * se_g, (g, se_g))
t_xx = paths.times_from_dates(nyse.sessions(date(2026, 10, 23), date(2029, 12, 21)))
T_xx = float(t_xx[-1])
od = stochvol.simulate_two_model(MODEL, SEED, N6, t_xx)
res = []
for Kr in (100.0, 10000.0 / 8370.0 * 100.0):
    ex = float(pricers.prices(MODEL, 100.0, Kr, T_xx, check=True)[0])
    pr, se_p = mean_se(np.maximum(100.0 * np.exp(od["log_end"]) - Kr, 0.0))
    res.append((round(Kr, 2), round(pr, 4), round(ex, 4), round(se_p, 4)))
check(f"(e) daily NYSE calendar: at-the-money and Dec-2029 10,000 calls within 3 SE of COS {res}",
      all(abs(pr - ex) < 3 * s for _, pr, ex, s in res), res)
whole = stochvol.simulate_two_model(MODEL, SEED, 6000, t6, checkpoints=(52, 156))
parts = [stochvol.simulate_two_model(MODEL, SEED, 3000, t6, checkpoints=(52, 156), path0=p0) for p0 in (0, 3000)]
threads = numba.get_num_threads()
numba.set_num_threads(1)
single = stochvol.simulate_two_model(MODEL, SEED, 6000, t6, checkpoints=(52, 156))
numba.set_num_threads(threads)
check(f"(f) one call or two chunks, 1 thread or {threads}: bit-identical (every output)",
      all(np.array_equal(whole[k], np.concatenate([p[k] for p in parts])) and np.array_equal(whole[k], single[k])
          for k in whole))
raises("(g) one factor alone at eta = 0 is refused",
       lambda: stochvol.simulate_two(SEED, 10, t6, f1z, (0.02, 0.03, 0.4, 0.5, -0.7)), "both")
raises("... the Euler scheme is refused for two factors", lambda: stochvol.simulate_two_model(MODEL, SEED, 10, t6,
                                                                                             scheme="euler"), "scheme")
raises("... the last two stream numbers are refused (the factors need stream + 1 and + 2)",
       lambda: stochvol.simulate_two_model(MODEL, SEED, 10, t6, stream=2 ** 32 - 2), "stream")
