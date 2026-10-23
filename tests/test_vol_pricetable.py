"""volmodel.pricetable: Heston call prices from a table of implied vols, and the mark's martingale identity.

Judges (tolerances fixed 2026-10-23, before the first run):
1. pricers.cos_call_surface (many starting variances at once) equals cos_prices row by row (1e-12 per unit forward).
2. The table reprices: at 400 random points inside the grid (log-moneyness -3 to 2, vol 0 to 120%), the table's
   call equals a direct COS price within 2e-6 per unit forward, at 4 remaining maturities from 1 week to 3.2 years
   (Gatheral's parameters).
3. Outside the grid (strikes beyond k = -3 or 3, vols above 200%) the table gives the exact COS price and counts
   those states (changed 2026-10-23 after a code review; see section 3).
4. The martingale identity (the property the slide rule failed, FINDINGS F17): mark the Dec-2029 10,000 call by the
   table along QE-M paths on the daily calendar; its expected forward value at 0.5, 1.5 and 2.5 years equals today's
   price within 3 SE (2^20 paths).
"""
import math
from datetime import date

import numpy as np

from _harness import check, raises

from definedtarget import calendar as nyse
from definedtarget.montecarlo import paths
from definedtarget.volmodel import pricers, pricetable, stochvol
from definedtarget.volmodel.heston import Heston

GAT = Heston(0.0174, 0.0354, 1.3253, 0.3877, -0.7165)

# 1. The surface equals the one-at-a-time pricer.
k = np.linspace(-1.0, 0.8, 19)
v = np.array([0.0, 0.0174, 0.09, 0.4])
surf = pricers.cos_call_surface(GAT, k, v, 1.5)
rows = np.array([pricers.cos_prices(Heston(vi, GAT.theta, GAT.kappa, GAT.eta, GAT.rho), 1.0, np.exp(k), 1.5)
                 for vi in v])
check("cos_call_surface equals cos_prices row by row (1e-12 per unit forward)", np.max(np.abs(surf - rows)) < 1e-12,
      np.max(np.abs(surf - rows)))

# 2. The table reprices inside its grid.
rng = np.random.default_rng(7)
worst = {}
for tau in (7 / 365, 0.25, 1.0, 3.2):
    tab = pricetable.build(GAT, tau)
    kk = rng.uniform(-3.0, 2.0, 400)
    vv = rng.uniform(0.0, 1.2, 400) ** 2
    got = pricetable.calls(tab, kk, vv)
    ref = np.array([pricers.cos_prices(Heston(vi, GAT.theta, GAT.kappa, GAT.eta, GAT.rho), 1.0, math.exp(ki), tau)[0]
                    for ki, vi in zip(kk, vv)])
    worst[round(tau, 4)] = float(np.max(np.abs(got - ref)))
check("the table reprices 400 random points per maturity within 2e-6 per unit forward (1 week to 3.2 years)",
      max(worst.values()) < 2e-6, worst)

# 3. Outside the grid the table prices exactly. (Code review, 2026-10-23: the first version returned the intrinsic
#    value below the strike range, 0 above it and the top row's vol above 120%, which undervalues crash states.)
tab = pricetable.build(GAT, 2.5)
# deep in the money; far out of it at vol 200%; vol 250%; both at once. (First draft of this test used vol 71% for
# the far strike, where the exact call is 2e-12: beyond k = 2 the value matters only at very high vol, for example
# 6.4e-4 per unit forward at k = 2.5 and vol 120%, about 1.8% of the call's value today.)
kk = np.array([-6.0, 3.5, 0.0, 3.5])
vv = np.array([0.04, 4.0, 6.25, 6.25])
got, n_out = pricetable.calls(tab, kk, vv, return_outside=True)
ref = np.array([pricers.cos_prices(Heston(vi, GAT.theta, GAT.kappa, GAT.eta, GAT.rho), 1.0, math.exp(ki), 2.5)[0]
                for ki, vi in zip(kk, vv)])
check("outside the grid (strike or vol) the table gives the exact COS price and counts those states",
      np.array_equal(got, ref) and n_out == 4, (got, ref, n_out))
check("a crash state (index 97% below the strike's forward, vol 200%, 2.5 years left) keeps real value (> 1e-3)",
      got[1] > 1e-3, got[1])
raises("a table needs a positive remaining maturity", lambda: pricetable.build(GAT, 0.0), "maturity")

# 4. The martingale identity for the table-marked option along QE-M paths.
t = paths.times_from_dates(nyse.sessions(date(2026, 10, 23), date(2029, 12, 21)))
T = float(t[-1])
k0 = math.log(10000.0 / 8370.0)
c0 = float(pricers.cos_prices(GAT, 1.0, math.exp(k0), T)[0])
steps = [int(np.searchsorted(t, y)) for y in (0.5, 1.5, 2.5)]
o = stochvol.simulate_model(GAT, 20261023, 1 << 20, t, checkpoints=tuple(steps))
gaps = []
for j, s in enumerate(steps):
    tau = T - float(t[s])
    tab = pricetable.build(GAT, tau)
    x = o["at_checkpoint"][:, j]                        # ln(F_t / F_0) in forward terms
    mark = np.exp(x) * pricetable.calls(tab, k0 - x, o["v_at_checkpoint"][:, j])
    mean, se = float(mark.mean()), float(mark.std(ddof=1) / math.sqrt(mark.size))
    gaps.append((round(float(t[s]), 2), round((mean - c0) / se, 2), round(100 * se / c0, 3)))
check("the table-marked Dec-2029 10,000 call is a martingale along QE-M paths: expected value at 0.5, 1.5 and 2.5 "
      "years equals today's price (3 SE)", all(abs(z) < 3 for _, z, _ in gaps), gaps)
