"""'What must the S&P do?' calculator for one share of a defined-target call ETF.

Works with NO downloaded data: give it the price you would pay, SPY's price, the option's implied volatility, an
interest rate and a dividend yield (continuous-yield Black-76; the daily report uses discrete dividends, which
differ by a few hundredths of a vol point). If captured data exists, the CLI fills the defaults from the latest
close instead.

Output (plain text): break-even levels and the growth needed per year, the value at the target date for a range
of S&P levels, and the odds/outcome table across several growth assumptions.
"""
import math
from dataclasses import dataclass

from . import analytics as an

SP_LEVELS = (7000, 8000, 9000, 9500, 10000, 10500, 11000, 12000, 13000, 14000, 15000)


@dataclass
class CalcInputs:
    price: float                 # what you pay for one fund share
    spot: float                  # the ETF's price now (SPY)
    sigma: float                 # the option's implied volatility (decimal)
    rate: float                  # continuously compounded rate to the target date (decimal)
    tau: float                   # years to the target date
    strike: float = 995.01
    options_per_share: float = 0.48
    cash_per_share: float = 0.25
    fee: float = 0.0099
    div_yield: float = 0.0097    # continuous dividend yield (decimal); fwd() ignores it if you give a forward
    forward: float = None        # optional: the forward price (e.g. from the daily model with discrete dividends)
    index_per_etf: float = 9995.0 / 995.01   # S&P points per SPY dollar at the target date (FINDINGS F7, central)
    index_now: float = None      # S&P now; default spot x index_per_etf

    def fwd(self):
        return self.forward if self.forward else self.spot * math.exp((self.rate - self.div_yield) * self.tau)


def calculate(c):
    """Return a dict with break-evens, a payoff table and the outcome table."""
    idx_now = c.index_now or c.spot * c.index_per_etf
    out = {"inputs": c, "index_now": idx_now}
    rows = []
    for mult in (1.0, 2.0, 3.0):
        etf = an.breakeven_spot(c.price, c.strike, c.options_per_share, c.cash_per_share, c.fee, c.tau, mult)
        rows.append((mult, etf, etf * c.index_per_etf, an.required_growth(idx_now, etf * c.index_per_etf, c.tau)))
    out["breakevens"] = rows
    out["strike_growth"] = an.required_growth(idx_now, c.strike * c.index_per_etf, c.tau)
    out["payoff"] = [(lvl, an.nav_at_expiry(lvl / c.index_per_etf, c.strike, c.options_per_share, c.cash_per_share,
                                             c.fee, c.tau)) for lvl in SP_LEVELS]
    out["odds"], _, _ = an.outcome_table(c.price, c.spot, c.fwd(), c.rate, c.sigma, c.tau, c.strike,
                                         c.options_per_share, c.cash_per_share, c.fee)
    return out


def render(res):
    c = res["inputs"]
    L = ["WHAT MUST THE S&P DO?  (price paid: ${:.2f} per share; research tool, not investment advice)".format(c.price),
         "",
         f"Inputs: SPY {c.spot:.2f} (S&P ~{res['index_now']:,.0f}); implied vol {c.sigma:.2%}; rate {c.rate:.2%}; "
         f"{c.tau:.2f} years left; {c.options_per_share:.4f} one-share options + ${c.cash_per_share:.2f} cash per "
         f"share; strike {c.strike}; fee {c.fee:.2%}/yr, which the fund pays by selling options (conservative).",
         "",
         "At the target date the share is worth about: options per share x max(SPY - strike, 0) + cash.",
         "",
         "  Goal                      SPY above     ~S&P      growth needed per year (price only)"]
    names = {1.0: "get your money back", 2.0: "double", 3.0: "triple"}
    for mult, etf, idx, g in res["breakevens"]:
        L.append(f"  {names[mult]:<24}  {etf:>9,.2f}  {idx:>8,.0f}   {g:>6.1%}")
    L.append(f"  {'just reach the strike':<24}  {c.strike:>9,.2f}  {c.strike * c.index_per_etf:>8,.0f}   "
             f"{res['strike_growth']:>6.1%}")
    L += ["", "  S&P at the target date   value per share   multiple of price"]
    for lvl, v in res["payoff"]:
        L.append(f"  {lvl:>12,}           {v:>10.2f}        {v / c.price:>6.2f}x")
    L += ["", f"Odds at the target date (lognormal, volatility {c.sigma:.1%}; read across the rows):",
          "  assumption                    median path/yr: price  total   worthless  profit  doubles   p10    median"
          "   p90"]
    for r in res["odds"]:
        L.append(f"  {r['assumption']:<30}  {r['median_price_cagr']:>20.1%}  {r['median_tr_cagr']:>6.1%}   "
                 f"{r['p_worthless']:>8.0%}  {r['p_profit']:>6.0%}  {r['p_double']:>7.0%}  "
                 f"{r['multiple_p10']:>5.2f}x  {r['multiple_median']:>5.2f}x  {r['multiple_p90']:>5.2f}x")
    L += ["", f"'Mean total return' is an average incl. dividends; the median path grows more slowly. Compare its "
          f"price growth with the {res['strike_growth']:.1%}/yr needed just to reach the strike."]
    L += ["", "S&P levels use the central finish-line ratio (the effective trigger sits at about 9,955-10,040, "
          "depending on the history the estimate uses; central ~9,995)."]
    return "\n".join(L)


PROSPECTUS_TABLE = [(6000, 600), (7000, 700), (8000, 800), (9000, 900), (9500, 950), (10000, 1000), (10500, 1050),
                    (11000, 1100), (11500, 1150), (12000, 1200), (12500, 1250), (13000, 1300), (14000, 1400),
                    (15000, 1500)]


def prospectus_table(strike=1000.0, premium=50.0):
    """The prospectus's illustration (one option, $1,000 strike, $50 premium): payout and profit at expiry."""
    return [(idx, etf, max(etf - strike, 0.0), max(etf - strike, 0.0) - premium) for idx, etf in PROSPECTUS_TABLE]
