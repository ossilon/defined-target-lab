# 1. What a call option is (and what XX holds)

**A call option gives its holder the right, not the obligation, to buy an asset at a fixed price (the *strike*)
on a fixed date (the *expiry*).**

- If the price closes *above* the strike on the expiry date, the call pays the difference. A call with a $995
  strike on an ETF that closes at $1,050 pays $55.
- If the price closes *at or below* the strike, the call pays nothing. Nobody pays $995 for an asset that sells
  for less.

Two terms matter for XX:

- **European**: the holder can exercise only on the expiry date. Before then, only the option's market value
  counts.
- **Cash-settled**: nobody delivers ETF shares. The holder receives the difference in cash.

## What XX holds

XX spends almost all its money on one batch of these calls on SPY, an ETF that tracks the S&P 500:

| Term | XX's option |
|---|---|
| Underlying | SPY |
| Strike | $995.01 (SPY's equivalent of the S&P 500 at 10,000) |
| Expiry | January 10, 2030 |
| Style | European, cash-settled FLEX option (a custom exchange-listed option) |

Each XX share holds calls on **0.48 SPY shares**, plus cash worth about 1% of the share. (The issuer's daily
holdings show 288 contracts of 100 shares for 60,000 fund shares on 2026-10-13.)

These terms make XX a **defined-target call ETF**: one strike (the target), one expiry (the target date), then a
roll into a new target.

Next: [2. The payoff](02-the-payoff.md).
