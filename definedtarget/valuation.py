"""Value one SPY call on one date: connects the curve, the dividends and the Black-76 formula.

Everything is per option on ONE SPY share (an exchange contract covers 100 shares). Per-XX-share numbers are
these times the fund's options per share (0.48 SPY shares per XX share; FINDINGS F1).

Greeks returned (all per one-share option, holding everything else fixed):
- delta: change in option value per $1 move in SPY (the "number of SPY shares it behaves like")
- gamma: change in delta per $1 move in SPY
- vega_pt: change in value per 1 volatility point (e.g. 16.6% -> 17.6%)
- theta_1d: change in value from one calendar day passing (SPY, vol and the curve unchanged). If SPY goes
  ex-dividend that next day, "SPY unchanged" means SPY minus the dividend: the price drops by the dividend on the
  ex-date, and the option never receives it (without this the step would show a false gain; audit B10a)
- theta_decay_1d: the part of theta that is pure time decay: the forward price and discount factor held fixed,
  only the time left shrinks. The rest of theta (theta_1d - theta_decay_1d) is carry: with SPY unchanged, the
  forward price drifts down a day's worth of interest net of dividends
- theta_fwd_1d: change in value from one day passing if SPY drifts along its forward (the forward price unchanged),
  which is what the option price itself assumes SPY does on average
- rho_1pct: change in value if all interest rates rise by 1 percentage point
- elasticity: % change in option value per 1% move in SPY = delta * S / C (the option's "leverage")
- prob_itm: N(d2), the formula's (risk-neutral) probability of finishing above the strike; a market price,
  not a forecast
- breakeven_cash / breakeven_vs_bills: SPY level at expiry needed to get the price back, or to match T-bills
"""
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import dividends as dv
from . import pricing

MODES = ("discrete", "continuous")


@dataclass
class Inputs:
    """Everything needed to value the option on one date. `history` = list of (ex_date, amount)."""
    as_of: date
    spot: float
    strike: float
    expiry: date
    curve: object
    history: list
    mode: str = "discrete"
    growth: float = dv.DEFAULT_GROWTH
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"dividend mode must be one of {MODES}")
        if not (self.spot > 0 and self.strike > 0):
            raise ValueError("spot and strike must be positive")
        if self.expiry <= self.as_of:
            raise ValueError("expiry must be after the valuation date")
        dv.check_fresh(self.history, self.as_of)            # stale dividends stop the run (audit A1)


def ex_dividend_next_day(inp):
    """SPY dividend (per share) going ex on the calendar day after as_of; 0.0 if none (or in continuous mode)."""
    if inp.mode != "discrete":
        return 0.0
    nxt = inp.as_of + timedelta(days=1)
    return sum(d.amount for d in dv.project(inp.history, inp.as_of, nxt, inp.growth))


def forward(inp, as_of=None, curve=None, spot=None):
    """(F, DF, tau, dF/dS) for the inputs, optionally at another date, curve or spot (theta and rho use this)."""
    as_of = as_of or inp.as_of
    curve = curve or inp.curve
    spot = inp.spot if spot is None else spot
    tau = (inp.expiry - as_of).days / 365.0
    if inp.mode == "discrete":
        divs = dv.project(inp.history, as_of, inp.expiry, inp.growth)
        F, DF = dv.forward_discrete(spot, as_of, inp.expiry, curve, divs)
        return F, DF, tau, 1.0 / DF
    q = dv.trailing_yield(inp.history, as_of, spot)
    F, DF = dv.forward_continuous(spot, as_of, inp.expiry, curve, q)
    return F, DF, tau, math.exp(-q * tau) / DF


def price(inp, sigma):
    F, DF, tau, _ = forward(inp)
    return pricing.call_price(F, inp.strike, DF, tau, sigma)


def value(inp, sigma):
    """Price and Greeks for volatility sigma (decimal, e.g. 0.166)."""
    F, DF, tau, dfds = forward(inp)
    p, delta_f, gamma_f, vega, prob_itm = pricing.call_greeks(F, inp.strike, DF, tau, sigma)
    # theta: same spot (less any dividend going ex overnight), vol and curve, one calendar day later
    ex_div = ex_dividend_next_day(inp)
    F1, DF1, tau1, _ = forward(inp, as_of=inp.as_of + timedelta(days=1), spot=inp.spot - ex_div)
    theta = pricing.call_price(F1, inp.strike, DF1, tau1, sigma) - p
    theta_decay = pricing.call_price(F, inp.strike, DF, tau1, sigma) - p        # forward and discounting fixed
    theta_fwd = pricing.call_price(F, inp.strike, DF1, tau1, sigma) - p         # SPY drifts along its forward
    # rho: parallel move of all rates by +/- 1 bp, scaled to 1 percentage point
    up = forward(inp, curve=inp.curve.shifted(+1e-4))
    dn = forward(inp, curve=inp.curve.shifted(-1e-4))
    rho = (pricing.call_price(up[0], inp.strike, up[1], tau, sigma)
           - pricing.call_price(dn[0], inp.strike, dn[1], tau, sigma)) / 2.0 * 100.0
    delta = delta_f * dfds
    return {
        "price": p, "forward": F, "df": DF, "tau": tau, "sigma": sigma,
        "delta": delta, "gamma": gamma_f * dfds * dfds, "vega_pt": vega / 100.0,
        "theta_1d": theta, "theta_decay_1d": theta_decay, "theta_fwd_1d": theta_fwd, "ex_div_next_day": ex_div,
        "rho_1pct": rho, "elasticity": delta * inp.spot / p if p > 0 else float("nan"),
        "prob_itm": prob_itm,
        # SPY level at expiry where a buyer at price p gets the money back (prospectus convention) ...
        "breakeven_cash": inp.strike + p,
        # ... and where the buyer matches simply holding Treasury bills instead (p grown at the curve's rate)
        "breakeven_vs_bills": inp.strike + p / DF,
    }


def implied_vol(inp, mark):
    """Volatility at which the formula reproduces `mark` (NaN outside the no-arbitrage range)."""
    F, DF, tau, _ = forward(inp)
    return pricing.implied_vol(mark, F, inp.strike, DF, tau)
