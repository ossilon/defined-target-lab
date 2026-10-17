"""Path statistics of montecarlo.paths judged by exact formulas (and two approximations with allowances fixed here).

Exact judges (each estimate must lie within 3 standard errors):
- end distribution: E[S_T] = S_0 e^{mu T} and P(S_T > K) = N(d2), on the real NYSE calendar (unequal steps);
- Spitzer's identity (1956): for a random walk with i.i.d. steps, E[max(0, S_1, ..., S_n)] = sum_k E[S_k^+] / k,
  exact for daily monitoring; by time reversal the drawdown at the end, max_k S_k - S_n, has the law of the maximum of
  the walk with negated steps, so Spitzer gives its mean too;
- the reflection principle: P(a continuous path touches level b by T) has a closed form; the engine's Brownian-bridge
  probability between closes estimates exactly that;
- the martingale property: with mu = r, E[e^{-rt} S_t] = S_0 at every checkpoint.
Approximations (allowance fixed before the first run, 2026-10-17):
- Broadie, Glasserman and Kou (1997): a barrier checked only at closes acts like a continuous one moved up by
  0.5826 vol sqrt(dt); allowance 3 SE + 0.002;
- Magdon-Ismail et al. (2004): with zero drift, E[maximum drawdown] = sqrt(pi / 2) vol sqrt(T). Daily sampling
  misses part of each fall (a bias close to c x sqrt(dt)); the test removes it by Richardson extrapolation from 1,024
  and 4,096 steps; allowance 3 combined SE.
"""
import math
from datetime import date

import numba
import numpy as np
from scipy.special import ndtr

from _harness import check, raises

from definedtarget import calendar as nyse
from definedtarget.montecarlo import paths, rng

SEED = 31415


def within(est, se, truth, k=3.0, extra=0.0):
    return abs(est - truth) <= k * se + extra


def mse(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def spitzer_max(n, m, s):
    """E[max(0, S_1..S_n)] for a Gaussian walk with step mean m and sd s (Spitzer's identity)."""
    k = np.arange(1, n + 1, dtype=np.float64)
    mu, sd = k * m, np.sqrt(k) * s
    pos = mu * ndtr(mu / sd) + sd * np.exp(-0.5 * (mu / sd) ** 2) / math.sqrt(2 * math.pi)
    return float((pos / k).sum())


# 1. End distribution on the real calendar: 815 NYSE sessions to XX's expiry, ACT/365 steps (weekends are longer)
sessions = nyse.sessions(date(2026, 10, 14), date(2030, 1, 10))
t_cal = paths.times_from_dates(sessions)
mu, vol = 0.0486498, 0.166
T = float(t_cal[-1])
out = paths.lognormal(SEED, 200_000, t_cal, mu, vol, checkpoints=range(68, 815, 68))
s_end = np.exp(out["log_end"])
m, se = mse(s_end)
check(f"calendar grid: {len(t_cal) - 1} steps, unequal (weekends, holidays), summing to T = {T:.6f} years",
      len(t_cal) - 1 == 814 and np.ptp(np.diff(t_cal)) > 0)
check("E[S_T / S_0] = e^{mu T} within 3 SE (unequal steps)", within(m, se, math.exp(mu * T)), (m, se, math.exp(mu * T)))
K = 995.01 / 778.01
d2 = (math.log(1 / K) + (mu - 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
hits = (s_end > K).mean()
check("P(S_T > K) = N(d2) within 3 SE", within(hits, math.sqrt(hits * (1 - hits) / s_end.size), float(ndtr(d2))),
      (hits, float(ndtr(d2))))
ck = np.exp(out["at_checkpoint"])
mart = [mse(np.exp(-mu * t_cal[k]) * ck[:, j]) for j, k in enumerate(range(68, 815, 68))]
check("martingale: E[e^{-rt} S_t] = S_0 at all 11 checkpoints (within 3 SE each)",
      all(within(a, b, 1.0) for a, b in mart), [round((a - 1) / b, 2) for a, b in mart])

# 2. Spitzer: the expected highest close and the expected drawdown at the end, exact for daily monitoring
n, T2 = 815, 815 / 252
t_eq = np.linspace(0.0, T2, n + 1)
for mu2 in (0.10, -0.05):
    o = paths.lognormal(SEED + 1, 200_000, t_eq, mu2, 0.2, stream=int(mu2 > 0))
    step_m, step_s = (mu2 - 0.02) * T2 / n, 0.2 * math.sqrt(T2 / n)
    a, b = mse(o["log_max"])
    check(f"Spitzer, drift {mu2:+.2f}: E[highest log close] within 3 SE of the exact {spitzer_max(n, step_m, step_s):.5f}",
          within(a, b, spitzer_max(n, step_m, step_s)), (a, b))
    a, b = mse(o["log_drawdown_end"])
    check(f"Spitzer by time reversal, drift {mu2:+.2f}: E[drawdown at the end] within 3 SE of "
          f"{spitzer_max(n, -step_m, step_s):.5f}", within(a, b, spitzer_max(n, -step_m, step_s)), (a, b))
check("every path: maximum drawdown >= drawdown at the end, and highest close >= 0 and >= the last close",
      bool((o["log_max_drawdown"] >= o["log_drawdown_end"] - 1e-15).all() and (o["log_max"] >= 0).all()
           and (o["log_max"] >= o["log_end"] - 1e-15).all()))

# 3. Touching a level: continuous (Brownian bridge, exact) and daily closes (Broadie-Glasserman-Kou, approximate)
B = 1.28                                               # the S&P must rise 28% (XX's strike from 2026-10-13, roughly)
o = paths.lognormal(SEED + 2, 400_000, t_cal, mu, vol, barrier=B, stream=7)
nu, bl = mu - 0.5 * vol * vol, math.log(B)


def touch_continuous(b, t):
    return float(ndtr((nu * t - b) / (vol * math.sqrt(t))) + math.exp(2 * nu * b / vol ** 2)
                 * ndtr((-b - nu * t) / (vol * math.sqrt(t))))


a, b = mse(1.0 - o["p_no_touch"])
check(f"continuous touch of +28% before expiry: bridge estimate within 3 SE of the exact {touch_continuous(bl, T):.5f}",
      within(a, b, touch_continuous(bl, T)), (a, b))
daily = (o["hit_step"] >= 0).astype(float)
a2, b2 = mse(daily)
bgk = touch_continuous(bl + 0.5826 * vol * math.sqrt(T / (len(t_cal) - 1)), T)
check(f"touch at a close (daily): within 3 SE + 0.002 of Broadie-Glasserman-Kou {bgk:.5f}", within(a2, b2, bgk, extra=0.002),
      (a2, b2, bgk))
check("a path that touched at a close also has continuous-touch chance 1 (p_no_touch = 0)",
      bool((o["p_no_touch"][o["hit_step"] >= 0] == 0).all()))
check("hit_step is the first close at or above the level", bool(np.all(o["log_max"][o["hit_step"] >= 0] >= bl)
                                                                 and np.all(o["log_max"][o["hit_step"] < 0] < bl)))

# 4. Maximum drawdown, zero drift: Magdon-Ismail's sqrt(pi/2) vol sqrt(T), daily-sampling bias extrapolated away
vol0, T0 = 0.2, 1.0
res = {}
for steps, st in ((1024, 11), (4096, 12)):
    t = np.linspace(0.0, T0, steps + 1)
    o = paths.lognormal(SEED + 3, 100_000, t, 0.5 * vol0 * vol0, vol0, stream=st)   # mu = vol^2/2: zero log drift
    res[steps] = mse(o["log_max_drawdown"])
ext = 2 * res[4096][0] - res[1024][0]                   # the bias halves when sqrt(dt) halves
ext_se = math.sqrt(4 * res[4096][1] ** 2 + res[1024][1] ** 2)
truth = math.sqrt(math.pi / 2) * vol0 * math.sqrt(T0)
check(f"zero drift: E[max drawdown] extrapolated to continuous {ext:.5f} within 3 SE of sqrt(pi/2) vol sqrt(T) = {truth:.5f}",
      within(ext, ext_se, truth), (res, ext, ext_se))
check("daily sampling understates the maximum drawdown (fewer steps, smaller drawdown)", res[1024][0] < res[4096][0] < truth)

# 5. Identities: threads and chunks do not change a single number; draws are the rng's own
p1 = paths.lognormal(5, 1000, t_eq[:50], 0.05, 0.2, barrier=1.1)
default_threads = numba.get_num_threads()
numba.set_num_threads(1)
p2 = paths.lognormal(5, 1000, t_eq[:50], 0.05, 0.2, barrier=1.1)
numba.set_num_threads(default_threads)
p3 = {k: np.concatenate([paths.lognormal(5, 400, t_eq[:50], 0.05, 0.2, barrier=1.1)[k],
                         paths.lognormal(5, 600, t_eq[:50], 0.05, 0.2, barrier=1.1, path0=400)[k]])
      for k in ("log_end", "log_max_drawdown", "p_no_touch", "hit_step")}
check("thread invariance: every measure identical on 1 thread and all threads",
      all(np.array_equal(p1[k], p2[k]) for k in p1 if k != "at_checkpoint"))
check("chunk invariance: 400 + 600 paths equal 1,000 paths, bit for bit", all(np.array_equal(p1[k], p3[k]) for k in p3))
z = rng.normals(5, 3, 49)
dt = np.diff(t_eq[:50])
manual = np.cumsum((0.05 - 0.02) * dt + 0.2 * np.sqrt(dt) * z, axis=1)[:, -1]
check("a path is the rng's draws: step k uses normal draw k of that path", np.allclose(p1["log_end"][:3], manual,
                                                                                      rtol=0, atol=1e-12))
raises("times must start at 0", lambda: paths.lognormal(1, 1, [0.1, 0.2], 0.0, 0.2), "start at 0")
raises("checkpoints must be valid step numbers", lambda: paths.lognormal(1, 1, [0, 0.1], 0.0, 0.2, checkpoints=[2]),
       "step numbers")
