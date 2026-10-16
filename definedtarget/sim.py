"""The browser simulator's model, in Python: the reference that `web/engine.js` must match number for number.

Plain words
- A fund is a list of option legs plus cash. Each leg: call or put, bought (+1) or sold (-1), a strike, and how many
  one-share options the fund holds per fund share ("units"; XX: 0.48). A listed option contract is the same thing
  with units = its multiplier (100) and no cash or fee.
- One fund share on a future date t (years from today), with the ETF at S and volatility vol, is worth
      exp(-fee x t) x sum over legs of sign x units x option value  +  cash
  The fee is paid by selling options (the conservative case, as in analytics.py); the cash stays flat.
- Option values use Black-76 with a continuous dividend yield q: forward = S exp((r - q) tau), discount exp(-r tau),
  tau = years left. At the target date the value is the payoff, exactly.
- "Vol follows the index": vol(S) = base + shift + slide x ln(S / S0) / ln(0.9). With slide = 1.5 points, a 10% fall
  adds 1.5 points and a rise of 1/0.9 - 1 = 11.1% takes 1.5 off. Floored at 1%.
- Odds: S_T lognormal with the implied vol; its mean grows at the rate (risk-neutral) or at a mean total return mu
  you choose. This is the same model as analytics.outcome_table, generalized to several legs.

Nothing here reads data. The tests pin sim.py to analytics.py and valuation.py for XX's single call.
"""
import math
from dataclasses import dataclass, field
from statistics import NormalDist

from . import pricing

_N = NormalDist()
DAY = 1.0 / 365.0
VOL_FLOOR = 0.01
LN_09 = math.log(0.9)


@dataclass(frozen=True)
class Leg:
    kind: str            # "call" or "put"
    sign: int            # +1 bought, -1 sold
    strike: float
    units: float         # one-share options per fund share (or per contract)


@dataclass(frozen=True)
class Fund:
    legs: tuple
    cash: float = 0.0    # cash per share
    fee: float = 0.0     # yearly fee, paid by selling options
    tags: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        for leg in self.legs:
            if leg.kind not in ("call", "put") or leg.sign not in (1, -1) or not (leg.strike > 0 and leg.units > 0):
                raise ValueError(f"bad leg: {leg}")


@dataclass(frozen=True)
class Market:
    rate: float          # continuous rate to the target date
    div_yield: float     # continuous dividend yield of the ETF
    horizon: float       # years from today to the target date (T)


def option_value(kind, S, K, tau, vol, m):
    """One option on one ETF share, `tau` years before its expiry."""
    if tau <= 0.0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    F = S * math.exp((m.rate - m.div_yield) * tau)
    DF = math.exp(-m.rate * tau)
    return pricing.call_price(F, K, DF, tau, vol) if kind == "call" else pricing.put_price(F, K, DF, tau, vol)


def option_delta(kind, S, K, tau, vol, m):
    """dValue/dS for one option on one ETF share."""
    if tau <= 0.0:
        if kind == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    F = S * math.exp((m.rate - m.div_yield) * tau)
    DF = math.exp(-m.rate * tau)
    _p, delta_f, _g, _v, _q = pricing.call_greeks(F, K, DF, tau, vol)
    dfds = math.exp((m.rate - m.div_yield) * tau)
    call_delta = delta_f * dfds
    return call_delta if kind == "call" else call_delta - DF * dfds


def share_value(fund, S, t, vol, m):
    """Value of one fund share t years from today (0 <= t <= horizon) with the ETF at S and volatility vol."""
    keep = math.exp(-fund.fee * t)
    tau = m.horizon - t
    opts = sum(leg.sign * leg.units * option_value(leg.kind, S, leg.strike, tau, vol, m) for leg in fund.legs)
    return keep * opts + fund.cash


def share_delta(fund, S, t, vol, m):
    """ETF shares one fund share behaves like (dValue/dS)."""
    keep = math.exp(-fund.fee * t)
    return keep * sum(leg.sign * leg.units * option_delta(leg.kind, S, leg.strike, m.horizon - t, vol, m)
                      for leg in fund.legs)


def leverage(fund, S, t, vol, m):
    """% move in the share per 1% move in the ETF (delta x S / value)."""
    v = share_value(fund, S, t, vol, m)
    return share_delta(fund, S, t, vol, m) * S / v if v > 0 else float("nan")


def vol_at(base, shift, slide, S, S0):
    """Volatility at ETF level S when vol follows the index: `slide` (decimal) per -10% move, log scale."""
    return max(base + shift + slide * math.log(S / S0) / LN_09, VOL_FLOOR)


def solve_vol(fund, price, S, m, lo=1e-4, hi=3.0):
    """Volatility at which one share today is worth `price`. NaN if no answer in [lo, hi] (or vol does not raise
    the value: a fund that is short volatility needs a typed-in vol)."""
    f_lo, f_hi = share_value(fund, S, 0.0, lo, m) - price, share_value(fund, S, 0.0, hi, m) - price
    if not (f_lo < 0.0 < f_hi):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if share_value(fund, S, 0.0, mid, m) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def one_day(fund, S, vol, m):
    """Cost of one calendar day as fractions of today's value: (SPY flat, decay part, SPY along its forward).

    Flat: S unchanged. Decay: forward and discount factor held, only the time left shrinks. Along the forward:
    S grows to S exp((r - q) / 365) so the forward is unchanged. Carry = flat - decay. (Continuous-yield versions of
    valuation.value's theta_1d, theta_decay_1d and theta_fwd_1d.)
    """
    v0 = share_value(fund, S, 0.0, vol, m)
    flat = share_value(fund, S, DAY, vol, m) - v0
    fwd = share_value(fund, S * math.exp((m.rate - m.div_yield) * DAY), DAY, vol, m) - v0
    # decay: same forward and discount factor as today, one day less of volatility
    keep = math.exp(-fund.fee * DAY)
    tau0 = m.horizon
    F = S * math.exp((m.rate - m.div_yield) * tau0)
    DF = math.exp(-m.rate * tau0)
    opts = 0.0
    for leg in fund.legs:
        c = pricing.call_price(F, leg.strike, DF, tau0 - DAY, vol)
        opts += leg.sign * leg.units * (c if leg.kind == "call" else c - DF * (F - leg.strike))
    decay = keep * opts + fund.cash - v0
    return flat / v0, decay / v0, fwd / v0


def forward_at_target(S0, m, mu=None):
    """Mean of S_T: the risk-neutral forward, or the forward under a MEAN total return mu (incl. dividends)."""
    growth = m.rate if mu is None else math.log1p(mu)
    return S0 * math.exp((growth - m.div_yield) * m.horizon)


def prob_above(level, mean_st, vol, T):
    if level <= 0:
        return 1.0
    sd = vol * math.sqrt(T)
    return _N.cdf((math.log(mean_st / level) - 0.5 * sd * sd) / sd)


def quantile(mean_st, vol, T, p):
    sd = vol * math.sqrt(T)
    return mean_st * math.exp(-0.5 * sd * sd + sd * _N.inv_cdf(p))


def value_at_target(fund, S_T, m):
    return share_value(fund, S_T, m.horizon, 0.0, m)


def level_for_value(fund, target_value, m, lo=1e-6, hi=None):
    """Lowest ETF level at the target date where a share is worth `target_value` (bisection; payoffs here rise
    with the ETF up to any cap). NaN if out of reach."""
    hi = hi or 20.0 * max(leg.strike for leg in fund.legs)
    if value_at_target(fund, hi, m) < target_value:
        return float("nan")
    if value_at_target(fund, lo, m) >= target_value:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if value_at_target(fund, mid, m) < target_value:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * hi:
            break
    return hi


def p_worthless(fund, mean_st, vol, T):
    """Chance every option leg expires worthless: S_T between the highest put strike and the lowest call strike."""
    calls = [leg.strike for leg in fund.legs if leg.kind == "call"]
    puts = [leg.strike for leg in fund.legs if leg.kind == "put"]
    below_call = 1.0 - prob_above(min(calls), mean_st, vol, T) if calls else 1.0
    below_put = 1.0 - prob_above(max(puts), mean_st, vol, T) if puts else 0.0
    return max(below_call - below_put, 0.0)


def outcome_row(fund, price, S0, vol, m, mu=None):
    """Odds and outcomes at the target date for a share bought at `price` (same fields as analytics.outcome_table)."""
    mean_st = forward_at_target(S0, m, mu)
    T = m.horizon
    be1, be2 = level_for_value(fund, price, m), level_for_value(fund, 2.0 * price, m)
    mult = {p: value_at_target(fund, quantile(mean_st, vol, T, p), m) / price for p in (0.10, 0.50, 0.90)}
    mean_growth = math.exp(m.rate) if mu is None else 1.0 + mu
    return {
        "mu": mu,
        "p_worthless": p_worthless(fund, mean_st, vol, T),
        "p_profit": prob_above(be1, mean_st, vol, T) if be1 == be1 else 0.0,
        "p_double": prob_above(be2, mean_st, vol, T) if be2 == be2 else 0.0,
        "multiple_p10": mult[0.10], "multiple_median": mult[0.50], "multiple_p90": mult[0.90],
        "median_price_cagr": (quantile(mean_st, vol, T, 0.5) / S0) ** (1.0 / T) - 1.0,
        "median_tr_cagr": mean_growth * math.exp(-0.5 * vol * vol) - 1.0,
        "breakeven": be1, "double_level": be2,
    }


def required_growth(now, level, T):
    return (level / now) ** (1.0 / T) - 1.0


def ratio_at(t, ratio_now, ratio_target, T):
    """Index points per ETF dollar on date t: straight line from today's ratio to the finish-line ratio."""
    return ratio_now + (ratio_target - ratio_now) * (t / T)
