# 3. Why volatility matters as much as the market

Before the target date, an XX share is worth what its options would sell for today. Five inputs set an option's
price, and SPY's price is only one of them:

| Input | Effect on XX's NAV (around 2026-10-13) |
|---|---|
| SPY's price | about **5.6%** NAV move per 1% SPY move: its "leverage" today |
| Implied volatility (the movement the market expects) | about **10%** NAV move per 1 point of volatility |
| Interest rates | about **15%** NAV move per 1 percentage point of rates |
| Time | about **-0.13%** of NAV per calendar day, if nothing else changes |
| SPY's dividends | small; the price already includes them |

So XX can move a few percent on a day when the S&P stays flat, if the market changes its expectation of future
movement. On 2026-10-13, XX fell 2.3% while SPY held flat. A drop in the option's implied volatility caused 96% of
the move. The daily report breaks down every move this way.

## Implied volatility in one paragraph

Traders quote option prices through a formula: Black-Scholes, or its forward-price form, Black-76. Plug in the
observed price and solve for the volatility that makes the formula match. That volatility is the *implied
volatility*. It measures price, not the future: it tells you how expensive the option is. The issuer marked XX's
option at about 16.6% on 2026-10-13.

## Leverage changes every day

The 5.6× changes daily. If the S&P falls, the option moves further from the strike and its leverage *rises*: each
dollar of NAV controls more SPY, but the chance of a payoff falls. If the S&P rallies well past 10,000, the
leverage falls toward 1×. A daily-reset 3× fund behaves differently: it resets to 3× every day.

Next: [4. Reading the daily report](04-reading-the-daily-report.md).
