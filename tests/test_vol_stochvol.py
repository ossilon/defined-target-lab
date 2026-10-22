"""volmodel.stochvol: Heston paths by Andersen's (2008) QE scheme, judged by exact identities, by his published
biases, by the two pricers, and by the engine's own rules.

Judges (tolerances fixed 2026-10-23, before the first run; SE = Monte Carlo standard error):
1. The lognormal limit, bit for bit: with eta = 0 and v0 = theta the variance is constant, and every shared output
   (log level at the end and at checkpoints, highest close, drawdowns, barrier step) equals montecarlo.paths.lognormal
   exactly, on the real NYSE calendar. With v0 != theta the end log level is normal with mean mu T - E[W_T]/2 and
   variance E[W_T] (3 SE).
2. QE matches the first two moments of the next variance exactly (Andersen, Propositions 5 and 6): one step from a
   fixed variance, in each of the two branches (quadratic and exponential), mean and variance within 3 SE.
3. Hence the moments of V_T are exact at any step size (both moments are linear in the previous variance):
   E[V_T] and Var[V_T] within 3 SE of Andersen's Corollary 1, at 1-year and 1/8-year steps. The martingale identity:
   with the martingale correction (QE-M), and for the full-truncation Euler scheme, E[S_T] = S_0 e^{mu T} within 3 SE.
4. Andersen's published biases (his Tables 2-4; 10^6 paths each, his count) at coarse steps, where they are large:
   ours equal his within 3 combined SE, sqrt(SE_ours^2 + SE_his^2), for the Euler, QE and QE-M schemes. Exact prices
   come from volmodel.pricers (he did not print them).
5. Fine steps: QE-M at 1/32-year steps prices Andersen's Case I calls (K = 70, 100, 140) within 3 SE of the exact
   price; and the listed Dec-2029 10,000 call (Gatheral's parameters, XX's maturity, the real daily calendar) too.
6. The engine's rules: the same paths in one call or in two chunks, and with 1 thread or all, are bit-identical.
7. The variance never goes below 0 under QE; the martingale correction refuses a step too large for it (rho > 0);
   bad inputs are refused.
"""
import math
from datetime import date

import numba
import numpy as np

from _harness import check, raises

from definedtarget import calendar as nyse
from definedtarget.montecarlo import paths
from definedtarget.volmodel import pricers, stochvol
from definedtarget.volmodel.heston import Heston

SEED = 20261023
CASES = {"I": (Heston(0.04, 0.04, 0.5, 1.0, -0.9), 10.0),           # Andersen (2008) Table 1: model, maturity
         "II": (Heston(0.04, 0.04, 0.3, 0.9, -0.5), 15.0),
         "III": (Heston(0.09, 0.09, 1.0, 1.0, -0.3), 5.0)}
N6 = 1 << 20                                                        # about Andersen's 10^6 paths


def grid(T, dt):
    return np.linspace(0.0, T, int(round(T / dt)) + 1)


def mean_se(x):
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(x.size))


def call_mc(model, T, dt, K, scheme, n=N6, seed=SEED, times=None):
    t = grid(T, dt) if times is None else times
    o = stochvol.simulate_model(model, seed, n, t, scheme=scheme)
    pay = np.maximum(100.0 * np.exp(o["log_end"]) - K, 0.0)
    return mean_se(pay)


# 1. The lognormal limit, bit for bit, on the real calendar.
t_cal = paths.times_from_dates(nyse.sessions(date(2026, 10, 23), date(2030, 1, 10)))
th = 0.166 ** 2
cps = (100, 400, t_cal.size - 1)
a = stochvol.simulate(SEED, 20_000, t_cal, v0=th, theta=th, kappa=1.3, eta=0.0, rho=0.0, mu=0.03, barrier=1.28,
                      checkpoints=cps)
b = paths.lognormal(SEED, 20_000, t_cal, mu=0.03, vol=math.sqrt(th), barrier=1.28, checkpoints=cps)
same = all(np.array_equal(a[k], b[k]) for k in ("log_end", "log_max", "log_drawdown_end", "log_max_drawdown",
                                                "hit_step", "at_checkpoint"))
check("lognormal limit (eta = 0, v0 = theta): every shared output equals montecarlo.paths.lognormal bit for bit",
      same, [k for k in ("log_end", "log_max", "hit_step", "at_checkpoint") if not np.array_equal(a[k], b[k])])
check("... and its variance stays at theta exactly", np.all(a["v_end"] == th) and np.all(a["v_at_checkpoint"] == th))
m = Heston(0.01, 0.04, 2.0, 0.5, 0.0)
T = float(t_cal[-1])
d = stochvol.simulate(SEED, N6, t_cal, v0=0.01, theta=0.04, kappa=2.0, eta=0.0, rho=0.0, mu=0.03)
W = m.expected_variance(T)
mu_, se_ = mean_se(d["log_end"])
var_, var_se = float(np.var(d["log_end"], ddof=1)), W * math.sqrt(2.0 / N6)
check("deterministic variance (v0 != theta): end log level has mean mu T - E[W]/2 (3 SE)",
      abs(mu_ - (0.03 * T - W / 2)) < 3 * se_, (mu_, 0.03 * T - W / 2, se_))
check("... and variance E[W_T] (3 SE)", abs(var_ - W) < 3 * var_se, (var_, W, var_se))
check("... and the integrated variance of every path is E[W_T] (1e-12 relative)",
      np.max(np.abs(d["int_var"] / W - 1)) < 1e-12)

# 2. One QE step matches the conditional mean and variance exactly, in both branches.
case_i = CASES["I"][0]
# (first run, 2026-10-23: the quadratic case started at v = 0.04, where psi = 5.5 puts it in the exponential branch;
#  the test saw that; v = 0.2 gives psi near 1.3)
for label, v, dt in (("quadratic branch", 0.2, 0.25), ("exponential branch", 0.002, 0.25)):
    e = math.exp(-case_i.kappa * dt)
    mm = case_i.theta + (v - case_i.theta) * e
    s2 = (v * case_i.eta ** 2 * e * (1 - e) / case_i.kappa
          + case_i.theta * case_i.eta ** 2 * (1 - e) ** 2 / (2 * case_i.kappa))
    psi = s2 / mm ** 2
    o = stochvol.simulate(SEED, N6, np.array([0.0, dt]), v0=v, theta=case_i.theta, kappa=case_i.kappa,
                          eta=case_i.eta, rho=case_i.rho)
    vv = o["v_end"]
    mv, se_m = mean_se(vv)
    sv = float(np.var(vv, ddof=1))
    se_v = math.sqrt(max(float(np.mean((vv - mv) ** 4)) - sv * sv, 0.0) / vv.size)
    in_branch = psi <= 1.5 if label == "quadratic branch" else psi > 1.5
    check(f"one QE step, {label} (psi = {psi:.2f}): mean and variance of the next variance exact (3 SE)",
          in_branch and abs(mv - mm) < 3 * se_m and abs(sv - s2) < 3 * se_v, (psi, mv, mm, se_m, sv, s2, se_v))

# 3. Moments of V_T exact at any step; the martingale identity for QE-M and Euler.
for dt in (1.0, 0.125):
    o = stochvol.simulate_model(case_i, SEED, N6, grid(10.0, dt))
    mv, se_m = mean_se(o["v_end"])
    ev = case_i.mean_variance(10.0)
    k, thv, eta, v0 = case_i.kappa, case_i.theta, case_i.eta, case_i.v0
    x = math.exp(-k * 10.0)
    var_true = v0 * eta ** 2 / k * (x - x * x) + thv * eta ** 2 / (2 * k) * (1 - x) ** 2
    sv = float(np.var(o["v_end"], ddof=1))
    se_v = math.sqrt(max(float(np.mean((o["v_end"] - mv) ** 4)) - sv * sv, 0.0) / N6)
    check(f"Case I, T = 10, steps of {dt:g} year: E[V_T] and Var[V_T] equal Andersen's Corollary 1 (3 SE)",
          abs(mv - ev) < 3 * se_m and abs(sv - var_true) < 3 * se_v, (mv, ev, se_m, sv, var_true, se_v))
for scheme in ("qe-m", "euler"):
    o = stochvol.simulate_model(case_i, SEED, N6, grid(10.0, 1.0), mu=0.02, scheme=scheme)
    g, se_g = mean_se(np.exp(o["log_end"] - 0.02 * 10.0))
    check(f"martingale identity, {scheme}, 1-year steps: E[S_T] = S_0 e^(mu T) (3 SE)", abs(g - 1) < 3 * se_g,
          (g, se_g))
o = stochvol.simulate_model(case_i, SEED, N6, grid(10.0, 1.0))
check("QE keeps the variance at 0 or above (no truncation needed)", float(o["v_end"].min()) >= 0.0)

# 4. Andersen's published biases at coarse steps (e = exact - Monte Carlo; his Tables 2-4).
PUBLISHED = [("I", 100.0, 1.0, "euler", -6.394, 0.029), ("I", 100.0, 1.0, "qe", -1.022, 0.013),
             ("I", 100.0, 1.0, "qe-m", -0.233, 0.013), ("I", 100.0, 0.5, "qe", -0.311, 0.013),
             ("I", 100.0, 0.5, "qe-m", -0.133, 0.013), ("I", 140.0, 1.0, "qe", 0.077, 0.002),
             ("I", 140.0, 1.0, "qe-m", 0.086, 0.002), ("II", 100.0, 1.0, "euler", -7.039, 0.073),
             ("II", 100.0, 1.0, "qe", 0.459, 0.041), ("II", 100.0, 1.0, "qe-m", 0.528, 0.041),
             ("III", 100.0, 1.0, "euler", -4.365, 0.074), ("III", 100.0, 1.0, "qe", 0.372, 0.052),
             ("III", 100.0, 1.0, "qe-m", 0.492, 0.053)]
exact = {}
misses = []
for case, K, dt, scheme, e_his, se_his in PUBLISHED:
    model, T = CASES[case]
    if (case, K) not in exact:
        exact[(case, K)] = float(pricers.prices(model, 100.0, K, T, check=True)[0])
    price, se = call_mc(model, T, dt, K, scheme)
    e_ours = exact[(case, K)] - price
    if not abs(e_ours - e_his) < 3 * math.hypot(se, se_his):
        misses.append((case, K, dt, scheme, round(e_ours, 3), e_his, round(math.hypot(se, se_his), 3)))
check(f"Andersen's published biases reproduced within 3 combined SE ({len(PUBLISHED)} table entries)", not misses,
      misses)

# 5. Fine steps: QE-M prices equal the exact prices.
model, T = CASES["I"]
fine = []
for K in (70.0, 100.0, 140.0):
    ex = float(pricers.prices(model, 100.0, K, T, check=True)[0])
    price, se = call_mc(model, T, 1 / 32, K, "qe-m")
    fine.append((K, round(price - ex, 4), round(se, 4)))
check("QE-M, Case I, 1/32-year steps: prices of the 70, 100 and 140 calls within 3 SE of exact",
      all(abs(gap) < 3 * se for _, gap, se in fine), fine)
gat = Heston(0.0174, 0.0354, 1.3253, 0.3877, -0.7165)
t_xx = paths.times_from_dates(nyse.sessions(date(2026, 10, 23), date(2029, 12, 21)))
T_xx, K_rel = float(t_xx[-1]), 10000.0 / 8370.0 * 100.0
ex = float(pricers.prices(gat, 100.0, K_rel, T_xx, check=True)[0])
price, se = call_mc(gat, T_xx, None, K_rel, "qe-m", times=t_xx)
check(f"QE-M on the daily NYSE calendar: the Dec-2029 10,000 call (Gatheral's parameters) within 3 SE of exact "
      f"({price:.4f} vs {ex:.4f}, SE {se:.4f})", abs(price - ex) < 3 * se, (price, ex, se))

# 6. Chunking and threads do not change a single bit.
t6 = grid(3.0, 1 / 52)
whole = stochvol.simulate_model(case_i, SEED, 6000, t6, checkpoints=(52, 156))
parts = [stochvol.simulate_model(case_i, SEED, 3000, t6, checkpoints=(52, 156), path0=p0) for p0 in (0, 3000)]
check("the same paths in one call or in two chunks are bit-identical (every output)",
      all(np.array_equal(whole[k], np.concatenate([p[k] for p in parts])) for k in whole))
threads = numba.get_num_threads()
numba.set_num_threads(1)
single = stochvol.simulate_model(case_i, SEED, 6000, t6, checkpoints=(52, 156))
numba.set_num_threads(threads)
check(f"1 thread and {threads} threads give bit-identical paths", all(np.array_equal(whole[k], single[k]) for k in whole))
check("another stream gives other paths (streams never share numbers)",
      not np.array_equal(stochvol.simulate_model(case_i, SEED, 100, t6, stream=2)["log_end"], whole["log_end"][:100]))

# 7. Refusals. FINDING (first run, 2026-10-23): no ordinary input violates the martingale correction's condition. A
#    search over eta 0.5-3, rho 0.5-0.95, kappa 0.1-2, steps 0.1-4 years and variances 0.04-2 found none: Andersen's
#    bound needs rho eta dt >= 2, and steps that large make A negative through its -dt rho^2 / 4 term. The guard
#    stays; it is tested here by forcing A inside the kernel.
def forced_violations():
    n, nsteps = 64, 4
    dt = np.full(nsteps, 0.25)
    one = np.ones(nsteps)
    outs = [np.empty(n) for _ in range(6)]
    hit, bad = np.empty(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    chk = np.zeros(0, dtype=np.int64)
    stochvol._heston(np.uint64(SEED), np.uint64(0), n, 0, 0.04, 0.04, 1.0, 1.0, 0.5, 1.5, dt, 0 * dt,
                     np.exp(-dt), 0.2 * one, 0.001 * one, 0 * one, 0 * one, 0 * one, 0.1 * one, 0.1 * one,
                     1e6 * one, 0 * one, 0 * one, 0 * one, 0 * one, np.inf, chk, np.uint64(0), *outs, hit,
                     np.full((n, 0), np.nan), np.full((n, 0), np.nan), bad)
    return int(bad.sum())


check("the martingale-correction guard counts every step where its condition fails (forced in the kernel)",
      forced_violations() == 64 * 4)
raises("a negative vol of vol is refused", lambda: stochvol.simulate(SEED, 10, t6, 0.04, 0.04, 1.0, -0.1, 0.0), "eta")
raises("an unknown scheme is refused", lambda: stochvol.simulate_model(case_i, SEED, 10, t6, scheme="milstein"),
       "scheme")
raises("times that do not start at 0 are refused", lambda: stochvol.simulate_model(case_i, SEED, 10, t6 + 1.0), "times")
raises("the last stream number is refused (the variance needs stream + 1)",
       lambda: stochvol.simulate_model(case_i, SEED, 10, t6, stream=2 ** 32 - 1), "stream")
