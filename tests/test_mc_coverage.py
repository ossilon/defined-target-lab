"""Are the engine's error bars honest? 1,000 independent runs, each judged against a closed-form answer.

Target fixed in the v0.3 depth pass (V03_DEPTH_PASS.md, rank 2) before this test first ran: across 1,000 runs, the
95% intervals must contain the exact value between 93.2% and 96.8% of the time (the 99% band of a binomial(1000,
0.95) count: 950 +/- 2.576 x 6.9). The pass wrote "93.6-96.4%" but called it the 99% band; the arithmetic was
corrected to the stated 99% before any run (2026-10-17).

Model: the README's 2026-10-13 XX example under the lognormal (S_T = F exp(-vol^2 T/2 + vol sqrt(T) Z)), where the
call value (Black-76), P(S_T > K) and every percentile of S_T have exact formulas. Run r uses Philox stream r, so
the runs are independent and reproducible. Judges: definedtarget.pricing (Black-76, normal CDF) and scipy's ndtri,
not the engine's own functions.
"""
import math

import numpy as np
from scipy.special import ndtri

from _harness import check

from definedtarget import pricing
from definedtarget.montecarlo import rng, stats

F, K, R, T, VOL = 882.2603, 995.01, 0.0486498, 3.2465753, 0.166
DF = math.exp(-R * T)
DRIFT, SCALE = -0.5 * VOL * VOL * T, VOL * math.sqrt(T)
SEED, RUNS = 20261017, 1000
BAND = (0.932, 0.968)

exact_price = pricing.call_price(F, K, DF, T, VOL)
d2 = (math.log(F / K) - 0.5 * VOL * VOL * T) / (VOL * math.sqrt(T))
exact_prob = pricing.norm_cdf(d2)
QS = (0.1, 0.5, 0.9)
exact_q = {q: F * math.exp(DRIFT + SCALE * float(ndtri(q))) for q in QS}
K_RARE = 2 * K
exact_rare = pricing.norm_cdf((math.log(F / K_RARE) - 0.5 * VOL * VOL * T) / (VOL * math.sqrt(T)))


def terminal(n, stream):
    return F * np.exp(DRIFT + SCALE * rng.normals(SEED, n, 1, stream=stream)[:, 0])


def share(hits):
    return hits / RUNS


cover = {"price": 0, "prob": 0, **{q: 0 for q in QS}, "rare_wald": 0, "rare_wilson": 0}
z_scores = []
for r in range(RUNS):
    s = terminal(1_000_000, r)                                   # 1 million paths per run: mean and probability
    m = stats.Moments().add(DF * np.maximum(s - K, 0.0))
    lo, hi = m.interval()
    cover["price"] += lo <= exact_price <= hi
    z_scores.append((m.mean - exact_price) / m.se)
    _, lo, hi = stats.wilson(int((s > K).sum()), s.size)
    cover["prob"] += lo <= exact_prob <= hi
    small = terminal(100_000, RUNS + r)                          # 100 thousand paths: percentiles
    for q in QS:
        _, lo, hi = stats.quantile_interval(small, q)
        cover[q] += lo <= exact_q[q] <= hi
    tiny = terminal(1_000, 2 * RUNS + r)                         # a rare event with few paths: about 2 hits expected
    hits = int((tiny > K_RARE).sum())
    cover["rare_wald"] += stats.wald(hits, tiny.size)[1] <= exact_rare <= stats.wald(hits, tiny.size)[2]
    cover["rare_wilson"] += stats.wilson(hits, tiny.size)[1] <= exact_rare <= stats.wilson(hits, tiny.size)[2]

print(f"info: exact call value {exact_price:.6f}, P(S_T > K) {exact_prob:.6f}, P(S_T > 2K) {exact_rare:.6f}; "
      f"coverage over {RUNS} runs: " + ", ".join(f"{k} {share(v):.1%}" for k, v in cover.items()))
inside = lambda v: BAND[0] <= share(v) <= BAND[1]  # noqa: E731
check("price (mean of discounted payoffs, 1M paths): 95% intervals cover Black-76 93.2-96.8% of 1,000 runs",
      inside(cover["price"]), share(cover["price"]))
check("P(S_T > K) (Wilson, 1M paths): 95% intervals cover the exact N(d2) 93.2-96.8% of runs",
      inside(cover["prob"]), share(cover["prob"]))
for q in QS:
    check(f"{q:.0%} percentile of S_T (order statistics, 100k paths): intervals cover the exact value 93.2-96.8%",
          inside(cover[q]), share(cover[q]))
zs = np.asarray(z_scores)
check("the price errors in SE units look standard normal (mean within 0.15, sd within 0.9-1.1)",
      abs(zs.mean()) < 0.15 and 0.9 < zs.std() < 1.1, (zs.mean(), zs.std()))
check("rare event, 1,000 paths (~2 hits): the textbook Wald interval under-covers (below 93.2%)",
      share(cover["rare_wald"]) < BAND[0], share(cover["rare_wald"]))
check("... while Wilson's interval stays honest (at least 93.2%)", share(cover["rare_wilson"]) >= BAND[0],
      share(cover["rare_wilson"]))

# Merging chunks gives the same moments as one pass (so threads, chunks and devices can be combined)
x = terminal(100_000, 99_999)
a = stats.Moments().add(x)
b = stats.Moments().add(x[:30_000]).merge(stats.Moments().add(x[30_000:70_000])).merge(stats.Moments().add(x[70_000:]))
check("Moments merged from 3 chunks equal one pass (mean and variance to 1e-12 relative)",
      abs(a.mean - b.mean) <= 1e-12 * abs(a.mean) and abs(a.var - b.var) <= 1e-12 * a.var, (a.var, b.var))
check("z for 95% is 1.959963984540054", abs(stats.z_for(0.95) - 1.959963984540054) < 1e-15)
