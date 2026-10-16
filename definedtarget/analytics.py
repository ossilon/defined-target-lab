"""Closed-form answers for a defined-target call ETF share: break-evens, required growth, odds and outcomes.

Plain words
- At the target date one fund share is worth about  N_opt x max(S_T - K, 0) + cash, where N_opt = one-share
  options per fund share (XX: 0.48), K = strike, S_T = the ETF's closing price that day, cash = the cash buffer.
- If the fund pays the fee by selling options (ASSUMED until hypothesis H2 says otherwise), N_opt shrinks by the
  fee rate each year: N_T = N_opt x exp(-fee x T). That is the conservative case and the one this module uses.
- "Odds" need an assumption about how fast the market grows. There are two very different kinds:
  * risk-neutral: what the option PRICE implies. It assumes the market grows at the interest rate, because that
    is how the market prices options; it is a price, not a forecast.
  * real-world: what happens if the market's MEAN total return is mu (e.g. 8%/yr including dividends). We show a
    CURVE over several mu, never one number, because nobody knows mu. A mean is not the typical path: with ~16%
    volatility the median path grows about 1.5 points a year slower, so each row also shows its median growth.
- The model for real-world odds is the same lognormal one as Black-76, with the volatility set to the option's
  implied vol (ASSUMED; a choice, shown in the output). Under it the ETF's forward simply grows at mu instead of
  the interest rate: F_real = F_rn x exp((ln(1+mu) - r) T).
- Because the fund's value at expiry only goes up as S_T goes up, every outcome percentile is just the payoff at
  the same percentile of S_T (no simulation needed).
"""
import math
from statistics import NormalDist

_N = NormalDist()


def nav_at_expiry(s_t, strike, options_per_share, cash_per_share, fee=0.0, tau=0.0):
    """Fund share value at the target date for an ETF close s_t (fee paid by selling options: conservative)."""
    n_t = options_per_share * math.exp(-fee * tau)
    return n_t * max(s_t - strike, 0.0) + cash_per_share


def breakeven_spot(price, strike, options_per_share, cash_per_share, fee=0.0, tau=0.0, multiple=1.0):
    """ETF level at the target date where a share bought at `price` is worth `multiple` x price."""
    n_t = options_per_share * math.exp(-fee * tau)
    return strike + max(multiple * price - cash_per_share, 0.0) / n_t


def required_growth(spot_now, level, tau):
    """Average yearly price growth needed to go from spot_now to level in tau years."""
    return (level / spot_now) ** (1.0 / tau) - 1.0


def lognormal_prob_above(forward, level, sigma, tau):
    """P(S_T > level) when S_T is lognormal with mean `forward` and volatility sigma."""
    if level <= 0:
        return 1.0
    sd = sigma * math.sqrt(tau)
    return _N.cdf((math.log(forward / level) - 0.5 * sd * sd) / sd)


def lognormal_quantile(forward, sigma, tau, p):
    """The p-quantile of a lognormal S_T with mean `forward`."""
    sd = sigma * math.sqrt(tau)
    return forward * math.exp(-0.5 * sd * sd + sd * _N.inv_cdf(p))


def call_spread_prob(price_low, price_high, strike_low, strike_high, discount_factor):
    """The market's risk-neutral chance of finishing above a level between two strikes, from two call prices.

    A call spread (buy the low-strike call, sell the high-strike call) pays (strike_high - strike_low) at expiry if
    the index ends above strike_high, nothing below strike_low, and part of it in between. Its price divided by the
    discounted strike gap is therefore the AVERAGE chance of finishing above a level inside [strike_low,
    strike_high]: close to the chance at the midpoint when the strikes sit close together.
    """
    return (price_low - price_high) / ((strike_high - strike_low) * discount_factor)


def prob_above_from_spreads(centers, probs, level):
    """(chance of finishing above `level`, extrapolated?) from call-spread chances at sorted spread centers.

    Inside the centers: linear interpolation of log(chance), since chances fall roughly exponentially with the
    level. Outside them: the nearest segment's trend continues (marked extrapolated, to report as a rougher figure).
    """
    x = [float(c) for c in centers]
    y = [math.log(p) for p in probs]
    if x[0] <= level <= x[-1]:
        for i in range(len(x) - 1):
            if x[i] <= level <= x[i + 1]:
                w = (level - x[i]) / (x[i + 1] - x[i])
                return math.exp(y[i] + w * (y[i + 1] - y[i])), False
    i = 0 if level < x[0] else len(x) - 2
    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    edge = 0 if level < x[0] else len(x) - 1
    return math.exp(y[edge] + slope * (level - x[edge])), True


def real_world_forward(rn_forward, rate_cont, mu_total, tau):
    """Forward under an assumed expected TOTAL return mu (yearly, incl. dividends) instead of the rate."""
    return rn_forward * math.exp((math.log1p(mu_total) - rate_cont) * tau)


def outcome_table(price, spot, forward_rn, rate_cont, sigma, tau, strike, options_per_share, cash_per_share,
                  fee, mus=(None, 0.04, 0.06, 0.08, 0.10, 0.12)):
    """Odds and outcomes for a share bought at `price`, one row per growth assumption.

    mu=None is the risk-neutral row (forward grows at the rate). Each row: P(options finish worthless),
    P(any profit), P(doubling), median and 10th/90th percentile value at the target date (as multiples of price),
    and the yearly growth of the MEDIAN path: price only (comparable with the growth the strike needs) and
    total return. mu is a MEAN (arithmetic) total return: with volatility sigma the median path grows more slowly,
    median total-return growth = (1 + mu) x exp(-sigma^2 / 2) - 1 (audit B10a: the 10% row's median price growth
    is about 7.4%/yr, below the ~7.9%/yr needed just to reach the strike on 2026-10-13).
    """
    be1 = breakeven_spot(price, strike, options_per_share, cash_per_share, fee, tau, 1.0)
    be2 = breakeven_spot(price, strike, options_per_share, cash_per_share, fee, tau, 2.0)
    rows = []
    for mu in mus:
        f = forward_rn if mu is None else real_world_forward(forward_rn, rate_cont, mu, tau)
        q = {p: nav_at_expiry(lognormal_quantile(f, sigma, tau, p), strike, options_per_share, cash_per_share,
                              fee, tau) / price for p in (0.10, 0.50, 0.90)}
        mean_growth = math.exp(rate_cont) if mu is None else 1.0 + mu       # yearly growth factor of the mean
        rows.append({
            "assumption": "risk-neutral (option price)" if mu is None else f"mean total return {mu:.0%}/yr",
            "mu": mu,
            "median_price_cagr": (lognormal_quantile(f, sigma, tau, 0.5) / spot) ** (1.0 / tau) - 1.0,
            "median_tr_cagr": mean_growth * math.exp(-0.5 * sigma * sigma) - 1.0,
            "p_worthless": 1.0 - lognormal_prob_above(f, strike, sigma, tau),
            "p_profit": lognormal_prob_above(f, be1, sigma, tau),
            "p_double": lognormal_prob_above(f, be2, sigma, tau),
            "multiple_p10": q[0.10], "multiple_median": q[0.50], "multiple_p90": q[0.90],
        })
    return rows, be1, be2
