"""Black-76 pricing, Greeks and implied volatility against known answers and independent checks."""
import math
import time
from statistics import NormalDist

import numpy as np

from _harness import check

from definedtarget import pricing as px

N = NormalDist().cdf


def bs_independent(S, K, r, q, sigma, T):
    """Textbook Black-Scholes-Merton written separately (statistics.NormalDist), used as a cross-check."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d1 - sigma * math.sqrt(T))


# 1. Published values
# Hull, Options, Futures and Other Derivatives, Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 -> c=4.76, p=0.81
F, DF = 42 * math.exp(0.05), math.exp(-0.05)
c = px.call_price(F, 40.0, DF, 0.5, 0.2)
check("Hull Ex 15.6 call = 4.76 (to the cent)", abs(c - 4.76) < 0.005, c)
check("Hull Ex 15.6 put = 0.81 (to the cent)", abs(px.put_price(F, 40.0, DF, 0.5, 0.2) - 0.81) < 0.005)
# Haug, Complete Guide to Option Pricing Formulas, Black-76: F=19, K=19, T=0.75, r=10%, sigma=28% -> 1.7011
h = px.call_price(19.0, 19.0, math.exp(-0.075), 0.75, 0.28)
check("Haug Black-76 at-the-money example = 1.7011", abs(h - 1.7011) < 5e-5, h)

# 2. Independent implementation over a grid (BSM with continuous yield is Black-76 with F = S e^{(r-q)T})
worst = 0.0
for S in (600.0, 780.00, 1100.0):
    for K in (500.0, 995.01, 1500.0):
        for T in (0.1, 1.0, 3.25):
            for sig in (0.08, 0.166, 0.45):
                r, q = 0.0486, 0.0097
                a = px.call_price(S * math.exp((r - q) * T), K, math.exp(-r * T), T, sig)
                worst = max(worst, abs(a - bs_independent(S, K, r, q, sig, T)))
check("Black-76 matches an independent Black-Scholes-Merton to 1e-9 over 81 cases", worst < 1e-9, worst)

# 3. Put-call parity and limits
F, K, DF, T = 880.0, 995.01, 0.853, 3.25
check("put-call parity: C - P = DF (F - K)",
      abs(px.call_price(F, K, DF, T, 0.2) - px.put_price(F, K, DF, T, 0.2) - DF * (F - K)) < 1e-10)
check("at expiry the call is worth its payoff", px.call_price(1050.0, K, 1.0, 0.0, 0.2) == 1050.0 - K)
check("with zero volatility the call is the discounted intrinsic value",
      abs(px.call_price(1100.0, K, DF, T, 0.0) - DF * (1100.0 - K)) < 1e-12)
check("call value rises with volatility", px.call_price(F, K, DF, T, 0.2) > px.call_price(F, K, DF, T, 0.15))
check("call value stays below DF x F even at huge volatility", px.call_price(F, K, DF, T, 4.0) < DF * F)

# 4. Greeks against finite differences
p, dF, gF, vega, pitm = px.call_greeks(F, K, DF, T, 0.166)
e = 1e-3
fd_delta = (px.call_price(F + e, K, DF, T, 0.166) - px.call_price(F - e, K, DF, T, 0.166)) / (2 * e)
fd_gamma = (px.call_price(F + 1, K, DF, T, 0.166) - 2 * p + px.call_price(F - 1, K, DF, T, 0.166)) / 1.0
fd_vega = (px.call_price(F, K, DF, T, 0.166 + 1e-5) - px.call_price(F, K, DF, T, 0.166 - 1e-5)) / 2e-5
check("delta (forward) matches finite difference", abs(dF - fd_delta) < 1e-7, (dF, fd_delta))
check("gamma (forward) matches finite difference", abs(gF - fd_gamma) < 1e-6, (gF, fd_gamma))
check("vega matches finite difference", abs(vega - fd_vega) < 1e-5, (vega, fd_vega))
sd = 0.166 * math.sqrt(T)
check("prob_itm is N(d2)", abs(pitm - N((math.log(F / K) - 0.5 * sd * sd) / sd)) < 1e-12)

# 5. Implied volatility round trip
errs, n = [], 0
for KF in (0.5, 0.8, 1.0, 1.13, 1.5, 2.0):
    for T in (0.05, 0.5, 3.25):
        for sig in (0.05, 0.166, 0.5, 1.0):
            Fv, Kv, DFv = 880.0, 880.0 * KF, 0.85
            pv = px.call_price(Fv, Kv, DFv, T, sig)
            if pv - DFv * max(Fv - Kv, 0.0) < 1e-6 * pv:   # time value below float precision: no vol information
                continue
            n += 1
            errs.append(abs(px.implied_vol(px.call_price(Fv, Kv, DFv, T, sig), Fv, Kv, DFv, T) - sig))
check(f"implied vol recovers the input volatility to 1e-8 ({n} cases)", max(errs) < 1e-8, max(errs))
check("implied vol is NaN below the discounted intrinsic value",
      math.isnan(px.implied_vol(DF * (1100 - K) - 0.01, 1100.0, K, DF, T)))
check("implied vol is NaN at or above DF x F", math.isnan(px.implied_vol(DF * F, F, K, DF, T)))
check("implied vol is NaN at expiry", math.isnan(px.implied_vol(10.0, F, K, DF, 0.0)))

# 6. Speed (after compilation): 1M prices and 1M implied vols
m = 1_000_000
rng = np.random.default_rng(1)
Fa = rng.uniform(600, 1200, m)
Ka = np.full(m, 995.01)
Da = np.full(m, 0.85)
Ta = rng.uniform(0.1, 3.3, m)
Sa = rng.uniform(0.1, 0.4, m)
px.call_price_array(Fa[:10], Ka[:10], Da[:10], Ta[:10], Sa[:10])          # compile
t0 = time.perf_counter()
prices = px.call_price_array(Fa, Ka, Da, Ta, Sa)
t_price = time.perf_counter() - t0
px.implied_vol_array(prices[:10], Fa[:10], Ka[:10], Da[:10], Ta[:10])
t0 = time.perf_counter()
ivs = px.implied_vol_array(prices, Fa, Ka, Da, Ta)
t_iv = time.perf_counter() - t0
ok = ~np.isnan(ivs)
check(f"1M call prices in under 0.2 s ({t_price:.3f} s)", t_price < 0.2)
check(f"1M implied vols in under 2 s ({t_iv:.3f} s)", t_iv < 2.0)
check("1M implied vols round-trip to 1e-7 where defined", float(np.max(np.abs(ivs[ok] - Sa[ok]))) < 1e-7,
      float(np.max(np.abs(ivs[ok] - Sa[ok]))))
