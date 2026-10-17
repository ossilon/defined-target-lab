# 5. The browser simulator

`web/index.html` shows what one share of a defined-target call ETF is worth at any S&P 500 level on any date up to
its target date. It runs in your browser: no install, no account, and nothing you type leaves your computer. Open it
from the project's GitHub Pages site, or double-click `web/index.html` in a downloaded copy.

## What you type

- **Price paid**, the ETF's price (SPY for XX) and the S&P 500 level. Type either the ETF or the S&P: the page fills
  in the other with the "S&P per SPY today" ratio (about 10.03 in October 2026; under FUND TERMS).
- **Date**, **rate** (the Treasury zero rate to the target date) and **dividend yield**.
- **Implied volatility**: the page solves it from the price paid. Type one to override; SOLVE VOL undoes that.
- **Fund terms** (folded away): options per share, strike, cash, fee, target date, and the S&P finish line (the S&P
  level the strike matches on the target date; about 9,995 for XX, FINDINGS F7).

The page opens with a dated example (the close of 2026-10-13). Type today's numbers to update every figure. A link
can carry the inputs after a `#` (for example `index.html#paid=24.51&spy=771.79&spx=7752.86&date=2026-10-15`).
Browsers never send the part after `#` to the server. The daily report (`py -m definedtarget daily`) ends with such
a link for its close.

## What you see

1. **Summary**: max loss, the S&P break-even on the target date and the growth per year it needs, the chance the
   options expire worthless and of any profit, today's leverage, and one day's cost split into time decay and carry.
2. **Volatility**: a shift for every cell, CRASH +3 / +6 / +10 and CRUSH -3 presets, and "vol follows the S&P": the
   volatility rises by the slide for each 10% the S&P sits below today. The default slide, +1 point per 10% fall, was
   measured on the listed S&P 500 Dec-2029 10,000 call (FINDINGS F12). One-year volatility (VIX1Y) moves about +4.6.
   For scale, in the April 2025 sell-off the 3-year volatility at this strike rose about 2.3 points.
3. **The grid**: rows are S&P levels, columns are dates, and each cell shows P/L, P/L %, value or leverage. Hover or
   tap a cell for all of them. The last column is the chance the S&P finishes above that row's level. GRAPH draws
   the value against the S&P on four dates.
4. **Odds at the target date**: one row per assumption about the market's mean total return (4% to 24%), the
   risk-neutral row (what the option price implies), and a custom row. Click a row to use it in the grid.

A "mean total return" is an average. A volatile market's typical (median) path grows more slowly: at about 16.6%
volatility, about 1.5 points a year slower. So compare each row's median path with the growth the break-even needs.
For history: every 3¼-year stretch of the S&P since 1871 returned a median 10% a year with dividends, the top 10%
of stretches 22.5% or more, and the best 40% (FINDINGS F13).

## Other funds

- **SPX 10000C DEC-29**: one listed S&P 500 call contract (strike 10,000, expiring 2029-12-21), per contract, with
  no fee and no cash. Type the ask your broker shows, times 100. It answers "XX or the option itself?"
- **+ SPEC**: any fund made of option legs plus cash, described in a few lines:

```json
{"name": "Hypothetical call-spread target fund", "index": "S&P 500", "etf": "SPY", "target": "2030-01-10",
 "index_per_etf_target": 10.045, "cash": 0.3, "fee": 0.95,
 "legs": [{"kind": "call", "sign": 1, "strike": 900, "units": 0.5},
          {"kind": "call", "sign": -1, "strike": 1100, "units": 0.5}]}
```

`sign` is 1 for bought and -1 for sold options, `units` is one-share options per fund share, strikes are in ETF
dollars, `fee` is percent per year, `index` names the index the fund tracks, and `index_per_etf_target` is the index points per ETF dollar on the target date. For an index other than the S&P 500, the page relabels itself and greys out its S&P 500 evidence (the volatility slide of F12, the history of F13 and the skew note of F10): set your own slide, and type the fund's own ETF and index prices.
A call spread's value rises and then falls as volatility grows, so the page solves on the rising side.

## The model, and how it differs from the daily report

The page uses Black-76 with a continuous dividend yield, the same model as `py -m definedtarget calc`. The daily
report spreads SPY's dividends over their actual quarterly dates. The two agree on the option's price but split it
slightly differently: on 2026-10-13 the page shows 5.5x leverage and -0.11% per day (the report: 5.6x and -0.13%;
FINDINGS F8 explains the gap). The fund pays its fee by selling options (the conservative case), and one volatility
prices every strike: listed options price the chance of finishing above the strike about 10 points higher (F10).

## How we know the numbers are right

`web/selftest.html` runs the page's maths on 720 cases and compares each answer with the Python engine's
(`scripts/make_golden.py`). Every case must agree to about nine digits. A test fails if the answers go stale.

This page provides general information only. It gives no investment advice.

Back to [1. What a call option is](01-what-is-a-call.md).
