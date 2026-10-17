# 4. Reading the daily report

`py -m definedtarget daily` downloads the issuer's public files for the latest close, checks them, prices the
option and writes a one-page report (Markdown and HTML) to `forward/daily/`. Section by section:

1. **Today in one line**: NAV, market price, premium or discount to NAV, the option's implied volatility and
   today's leverage.
2. **What one share holds**: options per share, the mark, the cash, and any flow. On a *flow day*, dealers
   created or redeemed shares.
3. **What the option's price implies**: implied volatility and its sensitivities (rate spread, dividend growth),
   the forward price, and the *risk-neutral* chance of finishing above the strike. Risk-neutral pricing assumes the
   market grows at the interest rate, so this chance measures what the option costs. It does not forecast.
4. **How the share moves**: the Greeks per XX share in plain units: % of NAV per 1% SPY move, per volatility
   point, per rate point and per day. The day's cost splits into *time decay* and *carry*, and a second line shows
   the cost if SPY rises along its forward price, as the option price assumes on average.
5. **Break-even for a buyer today**: the SPY and S&P levels a buyer needs at the target date, and the yearly growth
   those levels require.
6. **Odds and outcomes**: one row per assumption about the market's *mean* future return. Read across the rows; no
   row gives "the" answer. Two columns show how fast each row's median path grows, so you can compare it with the
   break-even growth. Each multiple divides the value at the target date by today's NAV.
7. **What moved the price**: splits the day's change into market (delta, gamma), volatility (vega), time
   (theta), rates (rho) and a small residual, each in dollars per share and in % of NAV. On an SPY ex-dividend
   date a dividend line adds back SPY's price drop, which the option never had a claim on.
8. **Sources**: names the source of every input that day.

## What the report cannot tell you

- Whether the market will rise. Nobody knows; the odds table shows the range.
- The fund's trading costs. The issuer marks its options with a pricing vendor's model and publishes no trade
  prices.
- What happens after 2030. The fund will roll into a new target and date under a rule it has not published.

This report provides general information only. It gives no investment advice.

Next: [5. The browser simulator](05-the-simulator.md).
