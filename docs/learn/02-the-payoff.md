# 2. The payoff: what a share is worth on the target date

On January 10, 2030, one XX share will be worth about

> **0.48 × max(SPY − 995.01, 0) + cash**

0.48 counts the options per share today. If the fund pays its fee by selling options, that count shrinks by
about 1% a year. The tools use the lower count to stay conservative.

Run `py -m definedtarget calc` to see the payoff for any price you might pay. The payoff has this shape:

| S&P 500 on the target date | Value of a share |
|---|---|
| Below about 10,000 | Only the cash buffer (about 1% of today's value) |
| Somewhat above 10,000 | Something, but less than the purchase price |
| Well above 10,000 | About $0.46 more per SPY dollar (about $4.6 per 100 S&P points) |

## Three numbers people mix up

1. **The finish line.** The option pays only above SPY $995.01. SPY's price does not equal S&P/10, so the
   matching S&P level drifts. Our measurement puts it near **9,995**: between about 9,955 and 10,040, depending
   on how much history the estimate uses.
2. **The cash break-even.** A buyer must also earn back the purchase price. A buyer at NAV on 2026-10-13 needs the
   S&P well above 10,000: near 10,560.
3. **The odds.** Unless the S&P grows fast, the options more likely than not expire worthless. So the report's
   *median* outcome often reads "only the cash remains", while the *average* outcome can still look attractive:
   a few large wins lift the average. The report's odds use one volatility for every outcome. On 2026-10-13, listed
   S&P 500 option prices put the chance of finishing above the strike about 10 points higher; for very large wins
   the two methods could not be told apart ([FINDINGS F10](../../FINDINGS.md)).

Next: [3. Why volatility matters as much as the market](03-why-volatility-matters.md).
