"""Yield-curve bootstrap and SPY dividend projection."""
import math
from datetime import date

import numpy as np

from _harness import check, raises

from definedtarget import curves, dividends as dv

# 1. Bootstrap on a flat par curve: zero rates equal the par rate (semiannual) at every maturity
flat = {t: 0.05 for t in (1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)}
c = curves.Curve(date(2026, 10, 13), flat)
target = 2 * math.log(1.025)
check("flat 5% par curve -> flat zero curve at 2 ln(1.025) (continuous)",
      max(abs(c.zero(t) - target) for t in (0.5, 1, 2, 3.25, 5, 10)) < 1e-12)


def par_price(curve, maturity, coupon):
    """Price per 1 of a semiannual-coupon note from the curve's discount factors."""
    times = np.arange(0.5, maturity + 1e-9, 0.5)
    return sum(coupon / 2 * curve.df(t) for t in times) + curve.df(maturity)


# 2. A sloped curve: every par note reprices to exactly 1 (the definition of bootstrapping)
sloped = {1 / 12: 0.041, 0.25: 0.0426, 0.5: 0.0434, 1: 0.0448, 2: 0.0479, 3: 0.049, 5: 0.05, 7: 0.0511,
          10: 0.0521, 20: 0.056, 30: 0.0556}                   # Treasury par curve, 2026-10-13 (rounded)
cs = curves.Curve(date(2026, 10, 13), sloped)
worst = max(abs(par_price(cs, m, sloped[m]) - 1.0) for m in (2, 3, 5, 7, 10, 20, 30))
check("bootstrapped curve reprices 2-30 year par notes to 1 (within 1e-12)", worst < 1e-12, worst)
check("3-year zero rate sits near the 3-year par yield (within 10 bp, continuous)",
      abs(cs.zero(3.0) - 2 * math.log(1 + 0.049 / 2)) < 0.001, cs.zero(3.0))
check("a spread shifts every zero rate by exactly the spread",
      abs(cs.shifted(0.0025).zero(3.25) - cs.zero(3.25) - 0.0025) < 1e-15)
check("discount factor of 1 at time zero", cs.df(0.0) == 1.0)
check("box spread rate: a box paying 100 in exactly one year, worth 95 now -> -ln(0.95) = 5.129%",
      abs(curves.box_implied_rate(0.0, 100.0, 95.0, date(2025, 1, 1), date(2026, 1, 1)) + math.log(0.95)) < 1e-12)

# 4. Dividend schedule
check("SPY ex-date rule: third Friday (2026-09-18)", dv.ex_date_for(2026, 9) == date(2026, 9, 18))
check("Juneteenth 2026 (Fri 06-19) moves the ex-date to Thu 06-18 (SPY's actual ex-date)",
      dv.ex_date_for(2026, 6) == date(2026, 6, 18))
check("Juneteenth 2027 observed Fri 06-18 moves the ex-date to Thu 06-17", dv.ex_date_for(2027, 6) == date(2027, 6, 17))
check("last ex-date before expiry is 2029-12-21", dv.ex_date_for(2029, 12) == date(2029, 12, 21))
check("pay date for the 2026-09-18 ex-date is the last session of October (2026-10-30)",
      dv.pay_date_for(date(2026, 9, 18)) == date(2026, 10, 30))
hist = [(date(2025, 12, 19), 1.993), (date(2026, 3, 20), 1.797), (date(2026, 6, 18), 1.904), (date(2026, 9, 18), 1.889)]
proj = dv.project(hist, date(2026, 10, 13), date(2030, 1, 10), growth=0.0)
check("13 dividends go ex between 2026-10-13 and 2030-01-10", len(proj) == 13, len(proj))
check("with zero growth each projected dividend = trailing-4 average (1.89575)",
      all(abs(d.amount - 7.583 / 4) < 1e-12 for d in proj))
check("projection ignores dividends announced after the valuation date",
      dv.project(hist + [(date(2026, 12, 18), 9.99)], date(2026, 10, 13), date(2027, 1, 31), 0.0)[0].amount
      == proj[0].amount)
g = dv.project(hist, date(2026, 10, 13), date(2030, 1, 10), growth=0.05)
check("with 5% growth the last dividend is ~ 1.896 x 1.05^3.26", abs(g[-1].amount - 7.583 / 4 * 1.05 ** ((date(2029, 12, 21) - date(2026, 9, 18)).days / 365)) < 1e-12)
raises("fewer than 4 past dividends is refused", lambda: dv.project(hist[:3], date(2026, 10, 13), date(2027, 1, 1)),
       "at least 4")
check("trailing yield = 7.583 / 780.00", abs(dv.trailing_yield(hist, date(2026, 10, 13), 780.00) - 7.583 / 780.00) < 1e-15)
F_d, DF_d = dv.forward_discrete(780.00, date(2026, 10, 13), date(2030, 1, 10), cs, proj)
F_c, DF_c = dv.forward_continuous(780.00, date(2026, 10, 13), date(2030, 1, 10), cs, 7.583 / 780.00)
check("discrete and continuous dividend forwards agree within 0.3%", abs(F_d / F_c - 1) < 0.003, (F_d, F_c))
