"""sim.py (the browser simulator's model) against the audited analytics.py, valuation.py and pricing.py."""
import math
from datetime import date

from _harness import check

from definedtarget import analytics as an, curves, pricing, sim, valuation

# The 2026-10-13 example in the README (continuous-yield form, like the calculator)
S0, R, T, K, OPS, CASH, FEE = 778.01, 0.0486498, 3.2465753, 995.01, 0.48, 0.2474, 0.0099
Q = R - math.log(882.2603 / S0) / T
M = sim.Market(R, Q, T)
XX = sim.Fund((sim.Leg("call", 1, K, OPS),), cash=CASH, fee=FEE)
VOL = 0.166

# Value: today = options x Black-76 + cash; at the target date = analytics.nav_at_expiry
F, DF = S0 * math.exp((R - Q) * T), math.exp(-R * T)
check("today's share value = 0.48 x Black-76 + cash",
      abs(sim.share_value(XX, S0, 0.0, VOL, M) - (OPS * pricing.call_price(F, K, DF, T, VOL) + CASH)) < 1e-12)
for s_t in (800.0, 995.01, 1100.0, 1400.0):
    check(f"value at the target date (SPY {s_t:g}) = analytics.nav_at_expiry with the fee",
          abs(sim.share_value(XX, s_t, T, VOL, M) - an.nav_at_expiry(s_t, K, OPS, CASH, FEE, T)) < 1e-12)

# Delta and leverage against bumps of the full share value
for t in (0.0, 1.0, 3.0):
    h = 0.01
    num = (sim.share_value(XX, S0 + h, t, VOL, M) - sim.share_value(XX, S0 - h, t, VOL, M)) / (2 * h)
    check(f"share delta matches a bump (t = {t:g} yr)", abs(sim.share_delta(XX, S0, t, VOL, M) - num) < 1e-7)
put_fund = sim.Fund((sim.Leg("put", 1, 700.0, 1.0),))
num = (sim.share_value(put_fund, S0 + 0.01, 0.5, VOL, M) - sim.share_value(put_fund, S0 - 0.01, 0.5, VOL, M)) / 0.02
check("put delta matches a bump (negative)", abs(sim.share_delta(put_fund, S0, 0.5, VOL, M) - num) < 1e-7 and num < 0)
check("leverage today is about 5.6x on the example", 5.4 < sim.leverage(XX, S0, 0.0, VOL, M) < 5.8,
      sim.leverage(XX, S0, 0.0, VOL, M))

# Solving the vol from the price paid matches pricing.implied_vol on the option alone
nav = 26.51
iv = sim.solve_vol(XX, nav, S0, M)
iv_ref = pricing.implied_vol((nav - CASH) / OPS, F, K, DF, T)
check("vol solved from the share price = pricing.implied_vol on the option price (1e-10)", abs(iv - iv_ref) < 1e-10,
      (iv, iv_ref))
check("the solved vol is near the discrete-dividend 16.61% (within 0.05 points)", abs(iv - 0.16608) < 5e-4, iv)
short_vol = sim.Fund((sim.Leg("call", -1, K, OPS),), cash=100.0)
check("a fund that is short volatility gets no solved vol (NaN: type one in)",
      math.isnan(sim.solve_vol(short_vol, 90.0, S0, M)))

# Odds rows equal analytics.outcome_table for XX's single call
rows, be1, be2 = an.outcome_table(nav, S0, F, R, iv, T, K, OPS, CASH, FEE)
for ref in rows:
    got = sim.outcome_row(XX, nav, S0, iv, M, ref["mu"])
    worst = max(abs(got[k] - ref[k]) for k in ("p_worthless", "p_profit", "p_double", "multiple_p10",
                                                "multiple_median", "multiple_p90", "median_price_cagr",
                                                "median_tr_cagr"))
    check(f"odds row '{ref['assumption']}' = analytics.outcome_table (1e-9)", worst < 1e-9, worst)
row = sim.outcome_row(XX, nav, S0, iv, M)
check("break-even and doubling levels = analytics.breakeven_spot (1e-9)",
      abs(row["breakeven"] - be1) < 1e-9 and abs(row["double_level"] - be2) < 1e-9, (row["breakeven"], be1))

# One day: decay + carry = flat; along the forward = DF ratio identity (as in valuation.value, continuous yield)
flat, decay, fwd = sim.one_day(XX, S0, iv, M)
v0 = sim.share_value(XX, S0, 0.0, iv, M)
check("one day with SPY flat costs about -0.13% of NAV", -0.0015 < flat < -0.0011, flat)
check("decay is about half of it, and SPY along its forward costs less than flat", flat < decay < 0 and flat < fwd < 0,
      (flat, decay, fwd))
cv = curves.Curve(date(2026, 10, 13), {t: 0.0486 for t in (1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)})
vv = valuation.value(valuation.Inputs(date(2026, 10, 13), S0, K, date(2030, 1, 10), cv,
                                      [(date(2025, 12, 19), 1.993), (date(2026, 3, 20), 1.797),
                                       (date(2026, 6, 18), 1.904), (date(2026, 9, 18), 1.889)], mode="continuous"), iv)
check("the decay share of theta agrees with valuation.value (continuous mode) within 10%",
      abs(decay / flat - vv["theta_decay_1d"] / vv["theta_1d"]) < 0.10, (decay / flat, vv["theta_decay_1d"] / vv["theta_1d"]))

# Vol follows the index
check("vol slide: a 10% fall adds the slide (1.5 points)",
      abs(sim.vol_at(0.166, 0.0, 0.015, 0.9 * S0, S0) - 0.181) < 1e-12)
check("vol slide: no move, no change; shift adds on top",
      abs(sim.vol_at(0.166, 0.03, 0.015, S0, S0) - 0.196) < 1e-12)
check("vol slide: floored at 1%", sim.vol_at(0.05, -0.08, 0.015, 3 * S0, S0) == sim.VOL_FLOOR)

# Several legs: a call spread
spread = sim.Fund((sim.Leg("call", 1, 900.0, 1.0), sim.Leg("call", -1, 1100.0, 1.0)))
v_sp = sim.share_value(spread, S0, 0.0, VOL, M)
ref_sp = pricing.call_price(F, 900.0, DF, T, VOL) - pricing.call_price(F, 1100.0, DF, T, VOL)
check("a call spread = long call minus short call", abs(v_sp - ref_sp) < 1e-12)
check("a call spread pays at most the strike gap at the target date",
      abs(sim.value_at_target(spread, 5000.0, M) - 200.0) < 1e-12)
check("a call spread's break-even lies between its strikes",
      900.0 < sim.level_for_value(spread, v_sp, M) < 1100.0)
check("a target beyond the cap is out of reach (NaN)", math.isnan(sim.level_for_value(spread, 250.0, M)))
pp = sim.Fund((sim.Leg("put", 1, 700.0, 1.0), sim.Leg("call", 1, 900.0, 1.0)))
mean = sim.forward_at_target(S0, M)
check("strangle: worthless only between the put and call strikes",
      abs(sim.p_worthless(pp, mean, VOL, T) - ((1 - sim.prob_above(900.0, mean, VOL, T))
                                                - (1 - sim.prob_above(700.0, mean, VOL, T)))) < 1e-15)
check("index ratio runs in a straight line from today to the target date",
      sim.ratio_at(0.0, 10.03, 10.045, 3.0) == 10.03 and abs(sim.ratio_at(1.5, 10.03, 10.045, 3.0) - 10.0375) < 1e-12)
