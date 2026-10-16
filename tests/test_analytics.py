"""Break-evens, required growth and lognormal odds against hand values and a brute-force simulation."""
import math

import numpy as np

from _harness import check

from definedtarget import analytics as an
from definedtarget import pricing as px

# Payoff and break-even on round numbers: 0.5 options per share, strike 1000, cash 0.25, no fee
check("value at expiry below the strike is just the cash", an.nav_at_expiry(900, 1000, 0.5, 0.25) == 0.25)
check("value at expiry above the strike = 0.5 x (1100 - 1000) + 0.25 = 50.25",
      abs(an.nav_at_expiry(1100, 1000, 0.5, 0.25) - 50.25) < 1e-12)
be = an.breakeven_spot(25.25, 1000, 0.5, 0.25)
check("cash break-even for a 25.25 share: 1000 + (25.25 - 0.25) / 0.5 = 1050", abs(be - 1050) < 1e-12)
check("value at the break-even equals the price paid", abs(an.nav_at_expiry(be, 1000, 0.5, 0.25) - 25.25) < 1e-12)
be2 = an.breakeven_spot(25.25, 1000, 0.5, 0.25, multiple=2.0)
check("doubling level: 1000 + (50.5 - 0.25) / 0.5 = 1100.5", abs(be2 - 1100.5) < 1e-12)
fee_be = an.breakeven_spot(25.25, 1000, 0.5, 0.25, fee=0.0099, tau=3.25)
check("fee paid by selling options raises the break-even", fee_be > be)
check("required growth: 100 -> 121 over 2 years is 10%/yr", abs(an.required_growth(100, 121, 2) - 0.10) < 1e-12)

# Risk-neutral probability equals Black-76's N(d2)
F, K, sig, T = 880.0, 995.01, 0.166, 3.25
_, _, _, _, nd2 = px.call_greeks(F, K, 0.85, T, sig)
check("lognormal P(above strike) at the forward equals Black-76's N(d2)",
      abs(an.lognormal_prob_above(F, K, sig, T) - nd2) < 1e-12)
check("median of a lognormal with mean F is F exp(-sigma^2 T / 2)",
      abs(an.lognormal_quantile(F, sig, T, 0.5) - F * math.exp(-0.5 * sig * sig * T)) < 1e-9)
check("real-world forward at mu = the rate equals the risk-neutral forward",
      abs(an.real_world_forward(F, math.log1p(0.05), 0.05, T) - F) < 1e-9)
check("higher expected return raises the odds",
      an.lognormal_prob_above(an.real_world_forward(F, 0.048, 0.10, T), K, sig, T)
      > an.lognormal_prob_above(an.real_world_forward(F, 0.048, 0.06, T), K, sig, T))

# Brute force: simulate a lognormal S_T and compare odds and the median multiple
rng = np.random.default_rng(7)
z = rng.standard_normal(2_000_000)
s_t = F * np.exp(-0.5 * sig * sig * T + sig * math.sqrt(T) * z)
rows, b1, b2 = an.outcome_table(25.0, 778.0, F, 0.048, sig, T, K, 0.48, 0.25, 0.0, mus=(None,))
r = rows[0]
sim_worthless = float(np.mean(s_t <= K))
sim_profit = float(np.mean(s_t > b1))
nav = 0.48 * np.maximum(s_t - K, 0) + 0.25
check("closed-form P(worthless) matches 2M simulated paths (within 3 standard errors)",
      abs(r["p_worthless"] - sim_worthless) < 3 * math.sqrt(sim_worthless * (1 - sim_worthless) / len(z)),
      (r["p_worthless"], sim_worthless))
check("closed-form P(profit) matches simulation (3 standard errors)",
      abs(r["p_profit"] - sim_profit) < 3 * math.sqrt(sim_profit * (1 - sim_profit) / len(z)))
check("closed-form median multiple matches simulation (0.5%)",
      abs(r["multiple_median"] / (float(np.median(nav)) / 25.0) - 1) < 0.005 or r["multiple_median"] < 0.02,
      (r["multiple_median"], float(np.median(nav)) / 25.0))
check("closed-form 90th percentile multiple matches simulation (0.5%)",
      abs(r["multiple_p90"] / (float(np.quantile(nav, 0.9)) / 25.0) - 1) < 0.005)

# Median-path growth (audit B10a): mu is a MEAN total return; the median path grows by (1 + mu) exp(-sigma^2/2) - 1
rw, _, _ = an.outcome_table(25.0, 778.0, F, 0.048, sig, T, K, 0.48, 0.25, 0.0, mus=(None, 0.10))
check("median price growth = (median S_T / spot)^(1/T) - 1, checked against the simulated median",
      abs(rw[0]["median_price_cagr"] - ((float(np.median(s_t)) / 778.0) ** (1 / T) - 1)) < 2e-4, rw[0])
check("median total return at a 10% mean = 1.10 x exp(-0.166^2 / 2) - 1 = 8.49%",
      abs(rw[1]["median_tr_cagr"] - (1.10 * math.exp(-0.5 * sig * sig) - 1)) < 1e-12
      and abs(rw[1]["median_tr_cagr"] - 0.0849) < 5e-5, rw[1]["median_tr_cagr"])
check("the 10% row is labeled as a mean, and its median price growth is below 10%",
      rw[1]["assumption"] == "mean total return 10%/yr" and rw[1]["median_price_cagr"] < 0.10)

# Odds from call spreads (FINDINGS F10): on synthetic lognormal call prices, the call-spread method must recover the
# true lognormal chances. Strikes and spread pairs match the listed S&P 500 Dec-2029 chain used in F10.
check("a call spread's price / discounted strike gap is the chance: (60 - 40) / (100 x 0.8) = 0.25",
      abs(an.call_spread_prob(60.0, 40.0, 1000, 1100, 0.8) - 0.25) < 1e-15)
PAIRS = ((9600, 10400), (10600, 10800), (11000, 12000), (12000, 13000), (13000, 14000), (14000, 15000))
for vol in (0.166, 0.157):
    F_, DF_, T_ = 8800.0, 0.86, 3.19
    centers = [(a + b) / 2 for a, b in PAIRS]
    probs = [an.call_spread_prob(px.call_price(F_, a, DF_, T_, vol), px.call_price(F_, b, DF_, T_, vol), a, b, DF_)
             for a, b in PAIRS]
    worst = max(abs(an.prob_above_from_spreads(centers, probs, lvl)[0] - an.lognormal_prob_above(F_, lvl, vol, T_))
                for lvl in (9995, 10563, 11136, 11708, 12854, 15719))
    check(f"call-spread odds recover lognormal odds within 0.2 points (vol {vol:.1%}, incl. extrapolation to 15,719)",
          worst < 0.002, worst)
ext = an.prob_above_from_spreads([1.0, 2.0], [0.5, 0.25], 3.0)
check("beyond the last spread center the trend continues (0.5, 0.25 -> 0.125) and is flagged as extrapolated",
      abs(ext[0] - 0.125) < 1e-12 and ext[1] is True, ext)
check("a level between centers interpolates log(chance): halfway between 0.5 and 0.125 gives 0.25",
      abs(an.prob_above_from_spreads([1.0, 2.0], [0.5, 0.125], 1.5)[0] - 0.25) < 1e-12)
