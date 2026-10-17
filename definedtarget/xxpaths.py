"""XX along every simulated path: the fund repriced each session, with the measures a buyer cares about.

Plain words
- The ETF (SPY) follows montecarlo's lognormal path on the real trading calendar; the draws are the engine's, so the
  ETF path is exactly paths.lognormal's for the same seed, path and stream.
- Each session the fund share is repriced with sim.py's model (the browser simulator's): legs valued by Black-76 on
  the forward, the fee paid by selling options (exp(-fee x t)), cash flat. Its implied vol follows the index by
  sim.vol_at's slide rule (vol rises when the ETF falls), while the ETF itself moves at a fixed realized vol: the
  two are separate on purpose, because that rule is the model's assumption under test.
- Measures per path, in one causal pass: NAV at the end and at checkpoints; the highest NAV; the largest NAV
  drawdown (as a fraction); the longest spell below a previous NAV high and the sessions spent below the starting
  NAV; the first session at 2x, 5x and 10x the starting NAV; the lowest NAV; and the ETF's own end (for XX vs SPY).
- Effective leverage: on up days and on down days separately, the fund's realized move per 1% ETF move against the
  model elasticity (delta x S / NAV) known at the previous close (effective_leverage).
- This module is the XX layer: it uses montecarlo (which knows nothing about XX) and sim.py; nothing imports it back.

Judged in tests/test_xxpaths.py: day-0 NAV = sim.share_value; the end NAV = the payoff; with constant vol, no fee and
no cash, the discounted option value averages to today's price at every checkpoint (the martingale identity).
"""
import math

import numba
import numpy as np

from . import pricing, sim
from .montecarlo import paths as lp
from .montecarlo.rng import _address, inv_norm, philox4x32, u52

MULTIPLES = (2.0, 5.0, 10.0)
VOL_FLOOR = sim.VOL_FLOOR
# Effective leverage counts a day only while the previous mark is at least this share of the starting mark: a nearly
# worthless option (no cash floor) has an unbounded elasticity, and a few such days would dominate the pooled ratio
# (found on the first run, 2026-10-17: ratios of 29 and 174 for bare calls).
LEV_FLOOR = 0.25
LN_09 = math.log(0.9)


@numba.njit(cache=True)
def _nav(S, t, T, kinds, signs, strikes, units, cash, fee, r, q, base, slide, S0):
    tau = T - t
    vol = base + slide * math.log(S / S0) / LN_09           # sim.vol_at with shift 0, floored
    if vol < VOL_FLOOR:
        vol = VOL_FLOOR
    total = 0.0
    for j in range(kinds.size):
        K = strikes[j]
        if tau <= 0.0:
            v = max(S - K, 0.0) if kinds[j] == 1 else max(K - S, 0.0)
        else:
            F = S * math.exp((r - q) * tau)
            DF = math.exp(-r * tau)
            v = pricing.call_price(F, K, DF, tau, vol) if kinds[j] == 1 else pricing.put_price(F, K, DF, tau, vol)
        total += signs[j] * units[j] * v
    return math.exp(-fee * t) * total + cash


@numba.njit(cache=True)
def _delta(S, t, T, kinds, signs, strikes, units, fee, r, q, base, slide, S0):
    """ETF shares one fund share behaves like at (S, t), at the vol the slide rule gives there (sim.share_delta)."""
    tau = T - t
    if tau <= 0.0:
        return 0.0
    vol = base + slide * math.log(S / S0) / LN_09
    if vol < VOL_FLOOR:
        vol = VOL_FLOOR
    total = 0.0
    dfds = math.exp((r - q) * tau)
    DF = math.exp(-r * tau)
    for j in range(kinds.size):
        _p, delta_f, _g, _v, _q = pricing.call_greeks(S * dfds, strikes[j], DF, tau, vol)
        d = delta_f * dfds
        total += signs[j] * units[j] * (d if kinds[j] == 1 else d - DF * dfds)
    return math.exp(-fee * t) * total


@numba.njit(parallel=True, cache=True)
def _xx(seed, path0, npaths, drift, vsd, times, S0, kinds, signs, strikes, units, cash, fee, r, q, base, slide,
        mult, check, stream, nav_end, nav_max, nav_min, nav_mdd, under_longest, below_start, first_mult, etf_end,
        at_check, lev):
    nsteps = drift.size
    T = times[nsteps]
    nav0 = _nav(S0, 0.0, T, kinds, signs, strikes, units, cash, fee, r, q, base, slide, S0)
    e0 = _delta(S0, 0.0, T, kinds, signs, strikes, units, fee, r, q, base, slide, S0) * S0 / nav0
    for i in numba.prange(npaths):
        s_prev = S0
        n_prev = nav0
        e_prev = e0                                  # model elasticity at the previous close (known before the move)
        up_x = up_m = dn_x = dn_m = 0.0
        path = path0 + np.uint64(i)
        x = 0.0
        hi = nav0
        lo = nav0
        worst = 0.0
        spell = 0
        longest = 0
        below = 0
        for m in range(mult.size):
            first_mult[i, m] = -1
        j = 0
        nav = nav0
        for b in range((nsteps + 1) // 2):
            c0, c1, c2, c3, k0, k1 = _address(seed, path, b, stream)
            w0, w1, w2, w3 = philox4x32(c0, c1, c2, c3, k0, k1)
            for half in range(2):
                k = 2 * b + half
                if k >= nsteps:
                    break
                z = inv_norm(u52(w0, w1) if half == 0 else u52(w2, w3))
                x = x + drift[k] + vsd[k] * z
                s_now = S0 * math.exp(x)
                nav = _nav(s_now, times[k + 1], T, kinds, signs, strikes, units, cash, fee, r, q, base, slide, S0)
                rs = s_now / s_prev - 1.0                # effective leverage: fund move vs ETF move, up and down days
                if n_prev >= LEV_FLOOR * nav0:           # only while the mark is still in play (see LEV_FLOOR)
                    rn = nav / n_prev - 1.0
                    if rs > 0.0:
                        up_x += rn * rs
                        up_m += e_prev * rs * rs
                    else:
                        dn_x += rn * rs
                        dn_m += e_prev * rs * rs
                if nav > 0.0:
                    e_prev = _delta(s_now, times[k + 1], T, kinds, signs, strikes, units, fee, r, q, base, slide,
                                    S0) * s_now / nav
                s_prev = s_now
                n_prev = nav
                if nav > hi:
                    hi = nav
                    spell = 0
                else:
                    spell += 1
                    if spell > longest:
                        longest = spell
                if nav < lo:
                    lo = nav
                if 1.0 - nav / hi > worst:
                    worst = 1.0 - nav / hi
                if nav < nav0:
                    below += 1
                for m in range(mult.size):
                    if first_mult[i, m] < 0 and nav >= mult[m] * nav0:
                        first_mult[i, m] = k
                while j < check.size and check[j] == k + 1:
                    at_check[i, j] = nav
                    j += 1
        nav_end[i] = nav
        nav_max[i] = hi
        nav_min[i] = lo
        nav_mdd[i] = worst
        under_longest[i] = longest
        below_start[i] = below
        etf_end[i] = S0 * math.exp(x)
        lev[i, 0] = up_x
        lev[i, 1] = up_m
        lev[i, 2] = dn_x
        lev[i, 3] = dn_m


def fund_arrays(fund):
    kinds = np.array([1 if leg.kind == "call" else -1 for leg in fund.legs], dtype=np.int64)
    signs = np.array([float(leg.sign) for leg in fund.legs])
    strikes = np.array([float(leg.strike) for leg in fund.legs])
    units = np.array([float(leg.units) for leg in fund.legs])
    return kinds, signs, strikes, units


def simulate(seed, npaths, fund, market, S0, times, realized_vol, implied_vol, slide=0.0, mu=None,
             checkpoints=(), path0=0, stream=0):
    """XX-style fund along npaths lognormal ETF paths on `times` (year fractions from today; times[-1] must equal
    market.horizon). mu: the ETF's drift (None = the risk-neutral rate - dividend yield); realized_vol: the ETF's
    path volatility; implied_vol, slide: the vol used to price the options (sim.vol_at with shift 0).
    Returns per-path arrays: nav_end, nav_max, nav_min, nav_max_drawdown, longest_underwater (sessions),
    sessions_below_start, first_multiple (npaths x 3: first session at 2x, 5x, 10x, or -1), etf_end, nav_at_checkpoint,
    and nav0."""
    t = np.asarray(times, dtype=np.float64)
    if abs(t[-1] - market.horizon) > 1e-12:
        raise ValueError("the time grid must end at the fund's target date (market.horizon)")
    mu = market.rate - market.div_yield if mu is None else mu
    drift, vsd, check, _ = lp.prepare(seed, npaths, t, mu, realized_vol, None, checkpoints, path0, stream)
    kinds, signs, strikes, units = fund_arrays(fund)
    out = {k: np.empty(npaths) for k in ("nav_end", "nav_max", "nav_min", "nav_max_drawdown", "etf_end")}
    out["longest_underwater"] = np.empty(npaths, dtype=np.int64)
    out["sessions_below_start"] = np.empty(npaths, dtype=np.int64)
    out["first_multiple"] = np.empty((npaths, len(MULTIPLES)), dtype=np.int64)
    out["nav_at_checkpoint"] = np.full((npaths, check.size), np.nan)
    out["leverage_sums"] = np.empty((npaths, 4))
    _xx(np.uint64(seed), np.uint64(path0), npaths, drift, vsd, t, float(S0), kinds, signs, strikes, units,
        float(fund.cash), float(fund.fee), float(market.rate), float(market.div_yield), float(implied_vol), float(slide),
        np.asarray(MULTIPLES), check, np.uint64(stream), out["nav_end"], out["nav_max"], out["nav_min"],
        out["nav_max_drawdown"], out["longest_underwater"], out["sessions_below_start"], out["first_multiple"],
        out["etf_end"], out["nav_at_checkpoint"], out["leverage_sums"])
    out["nav0"] = sim.share_value(fund, S0, 0.0, max(implied_vol, sim.VOL_FLOOR), market)
    return out


def effective_leverage(out):
    """(up, down): realized fund move per 1% ETF move on up days and on down days, divided by the model elasticity
    known at the previous close (pooled over all paths and over the days whose previous mark was at least LEV_FLOOR of
    the start: sum(r_fund x r_etf) / sum(elasticity x r_etf^2)).
    1.0 = the fund moved exactly as its elasticity said. Option curvature (gamma) alone lifts the up-day ratio a little
    above 1 and pulls the down-day ratio a little below; a vol that moves against the ETF pulls both down."""
    s = out["leverage_sums"].sum(axis=0)
    return s[0] / s[1], s[2] / s[3]
