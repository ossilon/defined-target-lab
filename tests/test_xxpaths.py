"""The XX layer (definedtarget/xxpaths.py) against sim.py, the payoff and the martingale identity.

Judges: day-0 NAV = sim.share_value (the browser simulator's Python reference); end NAV = the payoff; the ETF path =
montecarlo.paths.lognormal's, bit for bit; a recomputation of whole NAV paths with sim.share_value and sim.vol_at
reproduces every path measure; with constant vol, no fee and no cash under the risk-neutral drift, the discounted NAV
averages to today's NAV at every checkpoint (exact identity, within 3 SE). The slide rule's gap from that identity is
reported as information (the mystery the v0.3 depth pass asked to log).
"""
import math
from datetime import date

import numba
import numpy as np

from _harness import check, raises

from definedtarget import calendar as nyse, sim
from definedtarget import xxpaths as xp
from definedtarget.montecarlo import paths as lp
from definedtarget.montecarlo import rng

S0, R = 778.01, 0.0486498
times = lp.times_from_dates(nyse.sessions(date(2026, 10, 13), date(2030, 1, 10)))
T = float(times[-1])
Q = R - math.log(882.2603 / S0) / T
M = sim.Market(R, Q, T)
XX = sim.Fund((sim.Leg("call", 1, 995.01, 0.48),), cash=0.2474, fee=0.0099)
NSTEPS = times.size - 1
CK = (100, 400, 700, NSTEPS)

o = xp.simulate(11, 200_000, XX, M, S0, times, 0.166, 0.166, checkpoints=CK)
kinds, signs, strikes, units = xp.fund_arrays(XX)
nav0_kernel = xp._nav(S0, 0.0, T, kinds, signs, strikes, units, XX.cash, XX.fee, R, Q, 0.166, 0.0, S0)
check(f"day-0 NAV from the kernel equals sim.share_value ({o['nav0']:.6f})", abs(nav0_kernel - o["nav0"]) < 1e-12)
payoff = math.exp(-XX.fee * T) * 0.48 * np.maximum(o["etf_end"] - 995.01, 0.0) + XX.cash
check("end NAV equals the payoff on every path", float(np.abs(o["nav_end"] - payoff).max()) < 1e-12)
lg = lp.lognormal(11, 200_000, times, R - Q, 0.166)
check("the ETF path is montecarlo.paths.lognormal's, bit for bit", np.array_equal(o["etf_end"], S0 * np.exp(lg["log_end"])))
check("the last checkpoint is the end NAV", np.array_equal(o["nav_at_checkpoint"][:, -1], o["nav_end"]))

# Whole NAV paths recomputed in plain Python with sim.share_value and sim.vol_at (short grid, slide on, a few paths)
short = times[:61] * (T / times[60])                     # 60 steps to the same target date
os_ = xp.simulate(3, 6, XX, M, S0, short, 0.4, 0.17, slide=0.012, mu=0.15)
drift, vsd, _, _ = lp.prepare(3, 6, short, 0.15, 0.4)
z = rng.normals(3, 6, 60)
ok = True
for i in range(6):
    x = np.cumsum(drift + vsd * z[i])
    S = S0 * np.exp(x)
    nav = np.array([sim.share_value(XX, S[k], short[k + 1], sim.vol_at(0.17, 0.0, 0.012, S[k], S0), M) for k in range(60)])
    n0 = os_["nav0"]
    hi = np.maximum.accumulate(np.concatenate([[n0], nav]))[1:]
    first = [int(np.argmax(nav >= m * n0)) if (nav >= m * n0).any() else -1 for m in xp.MULTIPLES]
    spell, longest, top = 0, 0, n0
    for v in nav:
        spell = 0 if v > top else spell + 1
        top = max(top, v)
        longest = max(longest, spell)
    ok &= (abs(os_["nav_end"][i] - nav[-1]) < 1e-9 and abs(os_["nav_max"][i] - max(n0, nav.max())) < 1e-9
           and abs(os_["nav_min"][i] - min(n0, nav.min())) < 1e-9
           and abs(os_["nav_max_drawdown"][i] - max(0.0, float((1 - nav / hi).max()))) < 1e-9
           and os_["sessions_below_start"][i] == int((nav < n0).sum()) and list(os_["first_multiple"][i]) == first
           and os_["longest_underwater"][i] == longest)
check("every path measure equals a plain-Python recomputation with sim.share_value and sim.vol_at (6 paths, slide on)",
      ok)

# The martingale identity: constant vol, no fee, no cash, risk-neutral drift
bare = sim.Fund((sim.Leg("call", 1, 995.01, 0.48),))
mo = xp.simulate(21, 400_000, bare, M, S0, times, 0.166, 0.166, checkpoints=CK)
gaps = []
for j, k in enumerate(CK):
    disc = math.exp(-R * times[k]) * mo["nav_at_checkpoint"][:, j]
    gaps.append((disc.mean() - mo["nav0"]) / (disc.std(ddof=1) / math.sqrt(disc.size)))
check("martingale: E[e^{-rt} NAV_t] = NAV_0 at sessions 100, 400, 700 and expiry (within 3 SE each)",
      all(abs(g) < 3 for g in gaps), [round(g, 2) for g in gaps])
so = xp.simulate(21, 400_000, bare, M, S0, times, 0.166, 0.166, slide=0.01, checkpoints=CK)
rel = [(math.exp(-R * times[k]) * so["nav_at_checkpoint"][:, j]).mean() / so["nav0"] - 1 for j, k in enumerate(CK)]
print("info: with the slide rule (1 point per -10%), discounted NAV / NAV_0 - 1 at sessions 100, 400, 700, expiry: "
      + ", ".join(f"{100 * v:+.2f}%" for v in rel) + " (0 would mean the rule is consistent with today's price)")

# Sanity and identities
check("NAV high >= start >= NAV low; drawdown in [0, 1]; a 5x or 10x touch never comes before the 2x touch",
      bool((o["nav_max"] >= o["nav0"]).all() and (o["nav_min"] <= o["nav0"]).all()
           and ((o["nav_max_drawdown"] >= 0) & (o["nav_max_drawdown"] <= 1)).all()
           and ((o["first_multiple"][:, 1] < 0) | (o["first_multiple"][:, 0] <= o["first_multiple"][:, 1])).all()))
a = xp.simulate(5, 2000, XX, M, S0, times, 0.2, 0.17, slide=0.01)
default_threads = numba.get_num_threads()
numba.set_num_threads(1)
b = xp.simulate(5, 2000, XX, M, S0, times, 0.2, 0.17, slide=0.01)
numba.set_num_threads(default_threads)
check("thread invariance: every XX measure identical on 1 thread and all threads",
      all(np.array_equal(a[k], b[k]) for k in a if k not in ("nav_at_checkpoint", "nav0")))
raises("the grid must end at the target date", lambda: xp.simulate(1, 1, XX, sim.Market(R, Q, 3.0), S0, times, 0.2, 0.2),
       "target date")
