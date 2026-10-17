# 6. The Monte Carlo engine

A closed formula answers "what is the option worth today" and "what are the odds it finishes above the strike". It
cannot answer questions about the path: how often XX doubles at some point and then loses it, how deep its worst
drawdown runs, how long a holder waits below the price paid. Monte Carlo answers those: it simulates many possible
futures, measures each one, and counts. `definedtarget/montecarlo/` holds the engine; XX is its first user, and it
imports nothing XX-specific, so other projects can use it unchanged.

A simulation is only as good as the proof that its numbers are right. This page lists what the engine guarantees,
how the tests prove each guarantee, and what it cannot do.

## What it guarantees

**1. The same number on any machine.** Most simulators draw random numbers from a stream, so the answer changes
with the thread count or the order of work. This engine computes each random number from its address: (seed, path,
step, stream), through the Philox4x32-10 generator. Path 7,341,002 at step 400 gets the same number on 1 thread or
24, in one batch or a thousand, on the CPU or the GPU. You can replay any single path. Tests check the generator
against its authors' published answers (Random123), and the CPU and GPU against each other: the GPU matches to the
last digit, except about 1 in 40,000 extreme draws, where NVIDIA's logarithm rounds the last digit differently.

**2. Error bars that hold.** Every result comes with a standard error or a 95% interval. A test proves the intervals
honest: 1,000 independent runs against exact answers. The 95% intervals contained the truth 94.0% (price), 94.2%
(a probability) and 95.4-95.8% (percentiles) of the time, inside the 93.2-96.8% band fixed before the run. The test
also shows why the engine uses Wilson intervals for probabilities: for a rare event, the textbook interval covered
the truth only 87.7% of the time; Wilson's covered 94.8%.

**3. Judged by exact answers.** Each path measure has a judge that does not use the simulation: Spitzer's identity
for the expected highest close, the reflection principle for the chance of touching a level (exact 0.54079 for a
+28% touch before expiry), the martingale identity at checkpoints. Two measures have only approximations
(Broadie-Glasserman-Kou for touches checked at closes, Magdon-Ismail's expected maximum drawdown); they must sit
within allowances fixed before the run.

**4. A billion paths in fixed memory.** The engine never stores all paths. It merges running totals and fine
histograms and finds exact percentiles with a second pass. On the test machine, the option's value to a standard
error of 0.004 took 1.03 billion paths in 30 s (54.67715 against the formula's 54.67744), and the exact median of a
billion end prices took 51 s while keeping 235,112 values. `run_to_precision` picks the path count for a target
error, so "how accurate" replaces "how many paths".

**5. Real history, not only the bell curve.** The bootstrap builds futures from blocks of real S&P 500 days (the
circular stationary bootstrap, three mean block lengths: about 1 month, 6 months, 2 years). It can reset history
to any mean and spread, so a comparison isolates history's shape from its average return. FINDINGS F15 shows why
that matters: generators built from 1950-1999 at that era's volatility put 2000-2026's drawdowns above their 90th
percentile in 37-39% of windows. A history bootstrap is a shape, not a forecast; its volatility has to be an input.

**6. No look-ahead.** Every engine reports at session k exactly what a run stopped at k reports, bit for bit, on the
CPU and the GPU. `montecarlo.causal.check(rule, series)` tests any later trading rule the same way: it replaces the
future after every cut point with six different futures (reversed, shuffled, flat, a crash, a spike, noise) and
requires every earlier decision to stay identical. It catches a centered average, tomorrow's value, a full-sample
z-score, and rules that peek only on month-ends or only before a crash. It does not check execution timing, rules
whose parameters were fitted on the whole history, or data revised or published late; the planned backtest toolkit
covers those.

**7. Reproducible results.** Each study script writes a run manifest to `results/manifests/`: its arguments,
seeds, the fingerprint (SHA-256) of every input file and code file, the library versions, and the fingerprint of
everything it printed. Run it again on the same inputs and code, and the output fingerprint must match
(`manifest.same_output`). The four published studies (F15, F17, F18, F19) reproduce this way.

## Speed

Measured on an Intel Core i7-13700K (24 threads) and an NVIDIA RTX 3080 (FINDINGS F14, F16):

| Job | 24 CPU threads | GPU (optional, CuPy) |
|---|---|---|
| XX's option, one end price per path | 498 million paths/s | 2.9 billion paths/s (64-bit) |
| Lognormal paths, 814 real sessions, with a barrier | 0.57 billion path-steps/s | 1.36 billion |
| Bootstrap paths, 814 sessions | 1.26 billion path-steps/s | 19.7 billion |

Small jobs belong on the CPU: below about a million paths, starting a GPU job costs more than the work. The GPU
module loads only if CuPy and an NVIDIA card are present; everything runs without it.

## Try it

From the repository folder (or after `pip install .`):

```python
import math
from datetime import date

from definedtarget import calendar as nyse
from definedtarget.montecarlo import paths, stats

t = paths.times_from_dates(nyse.sessions(date(2026, 10, 14), date(2030, 1, 10)))
o = paths.lognormal(seed=1, npaths=1_000_000, times=t, mu=0.0387, vol=0.166, barrier=1.279)
above = int((o["log_end"] > math.log(1.279)).sum())
p, lo, hi = stats.wilson(above, o["log_end"].size)
print(f"P(finish above): {p:.4f} (95% interval {lo:.4f} to {hi:.4f})")
print(f"P(touch at a close): {(o['hit_step'] >= 0).mean():.4f}")
```

It prints P(finish above) 0.2909 (95% interval 0.2900 to 0.2917) and P(touch at a close) 0.4924 for XX's
2026-10-13 example: strike 1.279 x spot, 16.6% volatility, the risk-neutral drift, 814 real sessions.

## What it found so far

- **F15**: real history's shape lowers XX's model value 13-24% against the bell curve at the same daily volatility.
- **F17**: XX doubles at some point in about half of all futures but ends up in about a quarter (risk-neutral).
- **F18**: real one-year index volatility moves 3.5 times as much as the simulator's "vol follows the index" rule;
  the choice changes XX's drawdowns and peaks by several points.
- **F19**: that effect belongs to the option itself: XX, its bare SPY call and the listed S&P 10,000 call lose the
  same leverage and show the same mid-life value gap.

## The monthly forecast check

The engine also forecasts XX one month ahead, every month to expiry, under four models (protocol FC1, frozen before
its first month). A month later the log records where the real outcome landed inside each forecast. Honest
forecasts spread those percentiles evenly between 0 and 1. It is the only test of the engine on XX itself with data
the engine has not seen. The log (`forward/forecast_log.jsonl`) holds percentiles and fingerprints only.

## What it cannot do

- Paths follow the model you choose. A lognormal path has no crashes beyond the bell curve; a bootstrap cannot
  produce a day worse than history's worst. The engine measures a model's consequences; it does not make the model
  right.
- Few separate 3-year spans exist in any history (about 23 since 1950), so history-based results carry wide
  uncertainty that no path count removes.
- Volatility in the XX layer follows a rule (constant, or the slide rule); a fitted volatility model comes in v0.3.1.

Back to [1. What a call option is](01-what-is-a-call.md).
