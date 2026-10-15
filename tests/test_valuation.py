"""Valuation wrapper (curve + dividends + Black-76), the prospectus payoff table, and the real XX marks."""
import math
from datetime import date, timedelta
from statistics import NormalDist

from _harness import check, raises

from definedtarget import curves, valuation

N = NormalDist().cdf
AS_OF, EXPIRY, K = date(2026, 10, 13), date(2030, 1, 10), 995.01
SLOPED = {1 / 12: 0.041, 0.25: 0.0426, 0.5: 0.0434, 1: 0.0448, 2: 0.0479, 3: 0.049, 5: 0.05, 7: 0.0511,
          10: 0.0521, 20: 0.056, 30: 0.0556}
CV = curves.Curve(AS_OF, SLOPED)
HIST = [(date(2025, 12, 19), 1.993), (date(2026, 3, 20), 1.797), (date(2026, 6, 18), 1.904), (date(2026, 9, 18), 1.889)]


def inputs(spot=780.00, mode="discrete", curve=CV, as_of=AS_OF):
    return valuation.Inputs(as_of, spot, K, EXPIRY, curve, HIST, mode=mode, growth=0.05)


v = valuation.value(inputs(), 0.166)
# delta and gamma against bumping the spot price itself (full chain: dividends and curve held fixed)
up, dn = valuation.price(inputs(780.00 + 0.5), 0.166), valuation.price(inputs(780.00 - 0.5), 0.166)
check("delta matches a spot bump of the full chain", abs(v["delta"] - (up - dn) / 1.0) < 1e-6, (v["delta"], up - dn))
check("gamma matches a spot bump of the full chain",
      abs(v["gamma"] - (up - 2 * v["price"] + dn) / 0.25) < 1e-6)
vp = valuation.price(inputs(), 0.166 + 1e-4) - valuation.price(inputs(), 0.166 - 1e-4)
check("vega per vol point matches a vol bump (relative 1e-6)", abs(v["vega_pt"] / (vp / 2e-4 / 100) - 1) < 1e-6)
check("rho is positive for a call (higher rates raise the forward)", v["rho_1pct"] > 0)
check("theta is negative with spot, vol and curve held", v["theta_1d"] < 0)
check("elasticity = delta x S / price", abs(v["elasticity"] - v["delta"] * 780.00 / v["price"]) < 1e-12)
check("break-even vs bills is above the cash break-even", v["breakeven_vs_bills"] > v["breakeven_cash"])

# continuous-yield mode equals an independent Black-Scholes-Merton with r = zero(T), q = trailing yield
vc = valuation.value(inputs(mode="continuous"), 0.166)
T = (EXPIRY - AS_OF).days / 365.0
r, q = CV.zero(T), 7.583 / 780.00
d1 = (math.log(780.00 / K) + (r - q + 0.5 * 0.166 ** 2) * T) / (0.166 * math.sqrt(T))
bsm = 780.00 * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d1 - 0.166 * math.sqrt(T))
check("continuous mode = independent Black-Scholes-Merton to 1e-9", abs(vc["price"] - bsm) < 1e-9, vc["price"] - bsm)
check("continuous delta = e^(-qT) N(d1)", abs(vc["delta"] - math.exp(-q * T) * N(d1)) < 1e-9)
check("discrete and continuous dividend prices within 1% of each other",
      abs(v["price"] / vc["price"] - 1) < 0.01, (v["price"], vc["price"]))
# theta convention: discrete theta exceeds continuous theta by about the dividend accrual (q F N(d1) DF / 365)
gap = v["theta_1d"] - vc["theta_1d"]
check("discrete-dividend theta is more negative by roughly the daily dividend accrual (documented convention)",
      -0.02 < gap < -0.005, gap)
raises("expiry before the valuation date is refused",
       lambda: valuation.Inputs(EXPIRY, 778.0, K, AS_OF, CV, HIST), "expiry must be after")
raises("an unknown dividend mode is refused", lambda: inputs(mode="none"), "dividend mode")
raises("a dividend history more than a quarter old is refused (stale file)",
       lambda: valuation.Inputs(date(2027, 3, 1), 778.0, K, EXPIRY, CV, HIST), "stale SPY dividend history")
CURVES = {date(2026, 10, 9): SLOPED, date(2026, 10, 13): SLOPED}
check("curve fallback over a weekend/holiday is allowed (10/12 uses 10/09)",
      curves.curve_on(CURVES, date(2026, 10, 12)).as_of == date(2026, 10, 9))
raises("a Treasury curve more than 5 days old is refused (stale file)",
       lambda: curves.curve_on(CURVES, date(2026, 10, 20)), "stale Treasury curve")

# Prospectus payoff table (p.6): Target Option with $1,000 strike and $50 premium, at expiry
table = [(600, 0, -50), (900, 0, -50), (1000, 0, -50), (1050, 50, 0), (1100, 100, 50), (1250, 250, 200),
         (1500, 500, 450)]
ok = all(max(etf - 1000, 0) == pay and max(etf - 1000, 0) - 50 == prof for etf, pay, prof in table)
check("prospectus payoff table reproduced (payout = max(ETF - 1000, 0), profit = payout - 50)", ok)

# Synthetic round trip: a price made at 17% vol comes back as 17% (real-data pins live in realdata_local.py)
syn = valuation.price(inputs(), 0.17)
check("synthetic mark priced at 17% implies 17% back (1e-9)", abs(valuation.implied_vol(inputs(), syn) - 0.17) < 1e-9)
