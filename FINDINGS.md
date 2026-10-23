# Findings and build log

A result counts only after this file records it, with the code that produced it, its data and its date.

## Build log

**S0, 2026-10-14 (Opus 5.5, high).**
- First capture: 4 latest-day Roundhill files, 10 holdings days (2026-09-30 to 10-13), 4 fund
  documents, Cboe VIX/VIX3M/VIX6M/VIX1Y and Treasury par curves for 2024-2026. A second run returned "unchanged"
  for every file (idempotent).
- FRED refused scripted CSV downloads (timeouts and connection resets); the code switched to Treasury's own file.
- Tests at the end of S0: 60 checks, 0 failures.
- With the user's OK, a daily routine (`scripts/daily.py`) and a local scheduled task (Tuesday to Saturday, after
  Roundhill's updates) went live; the first run succeeded.

**S1, 2026-10-14 (Opus 5.5, xhigh; audit A1 on Sonnet 5.5, xhigh).**
- Built: curves (Treasury par-to-zero bootstrap), dividends (SPY ex/pay dates, projection, escrowed and continuous
  forwards), pricing (numba Black-76, Greeks, implied-vol solver), valuation, ratio, spec + funds/xx.json, marks.
- Speed (after compilation; single-threaded, one core of an Intel Core i7-13700K with 16 cores / 24 threads;
  median of 7 runs on 2026-10-14; numba 0.68, numpy 2.5.3, Python 3.13.15): 1M option prices in 0.033 s
  (range 0.033-0.035); 1M implied vols in 0.34 s (range 0.338-0.342).
- **Audit A1** ran on **Sonnet 5.5 at xhigh** instead of the agreed Fable 5.1, because the Fable usage limit was
  reached (user decision). Verdict: **usable with caveats**. Independent code (its own calendar, bootstrap and
  Black-76) reproduced the 2026-10-13 implied vol (16.60845% vs 16.60847%), every Greek, and all 10 days' implied
  vols to 2e-7. The corrections appear below (F1, F4, F6, F7, F8, F9) and in the code: the box rate now uses the
  effective date; a dividend-growth sensitivity runs every day; a stale curve (> 5 days), stale dividends
  (> 100 days), a missing spot or a NaN implied vol now stop the run; tests pin the headline values; real-data
  tests moved to the git-ignored `tests/realdata_local.py`; `scripts/scan_identifiers.py` removes personal
  identifiers. A Fable re-audit of the pricing core runs before v0.2 builds on it.
- Tests after the fixes: 148 checks, 0 failures (130 public checks on synthetic numbers + 18 real-data checks that
  run locally from the git-ignored `tests/realdata_local.py`).

**S2, 2026-10-14 (Opus 5.5, high; no audit at this step).**
- Renamed the package `targetfund` → `definedtarget` (user decision); tests unchanged.
- Built: `analytics` (break-evens, required growth, lognormal odds and outcome percentiles; checked against a
  2M-path simulation), `report` (daily one-page report, Markdown + HTML, with daily attribution), `calc` + CLI
  (`py -m definedtarget calc | daily | report | prospectus`), `ledger` (hash-chained predictions),
  `forward/routine` (the daily routine, which now also prices and reports; stale or missing inputs trigger STOPs),
  `docs/learn/01-04`, README.
- Drafted hypotheses H2-H8 and parked H1, H9 in `forward/hypotheses_draft.json`; they await the user's approval
  before freezing.
- Tests: 184 checks, 0 failures. Identifier scan: 0 problems in 56 publishable files.

**S3, 2026-10-14/15 (Opus 5.5, high, then xhigh for the audit fixes; audit A2 on Sonnet 5.5, xhigh).**
- Froze protocol P1 and 20 hypotheses on 2026-10-14 (the ledger records the exact UTC time); OpenTimestamps
  stamped the frozen bytes.
- Release prep: MIT license, `pyproject.toml`, CHANGELOG, CITATION.cff, a license guard in the test suite, an
  active-voice pass over code comments, version 0.1.0. Added F10 and F11 (exploratory).
- **Audit A2** ran on **Sonnet 5.5 at xhigh** (the Fable limit was reached again; user decision). Verdict: safe for
  a private repository, not yet public. It re-derived the 2026-10-13 implied vol (0.1660847), every exploratory
  number in the ledger and the F10 ladder with its own code. Fixed after it: the ledger keeps its line endings
  and `.gitattributes` protects them (the timestamp proof would otherwise fail after upload); a test pins the
  frozen bytes; the daily tracker prices the days it can and lists the rest (a fresh copy stopped on day one);
  the scanner matches .gitignore the way git does and allows the author's handle only in the attribution lines;
  F10/F11 and README wording no longer claim more than one snapshot shows; the F10 method moved into tested code
  (`analytics.call_spread_prob`, `prob_above_from_spreads`); two calculator crashes; a repeated download failure
  now stops the run; stale "H1 parked" text; PEP 639 license metadata. Protocol P2 records the clarifications a
  verdict script needs, without changing any band.
- Tests: 251 checks, 0 failures (233 public + 18 local real-data). Identifier scan: 0 problems in 65 publishable
  files.

## Results

### S0: what the issuer's own files show (2026-10-14; `scripts/build_tables.py`, data to the 2026-10-13 close)

**F1. Each XX share holds calls on exactly 0.48 SPY shares.** Contracts per share equal 0.0048 on every close
from 10/01 to 10/13, through a 1-share redemption (10/01) and creations of +30,000 shares (10/06) and +20,000
shares (10/12) (seed day: 48/10,001). Creations buy options in proportion, so fund flows do not change a buyer's
exposure per share.

**F2. NetAssets ÷ shares gives the published NAV, and the holdings lines account for it to within half a cent per
share.** NetAssets/shares rounds to the published NAV on all 10 days. The itemized lines (option + money market +
Cash & Other) differ from NetAssets by −$81 to +$185 after the seed day. The gap changes sign, so the accrued fee
alone cannot explain it; its cause stays OPEN. Seed day: NetAssets equals exactly 10,001 × $24.20, while the lines
give $24.206 per share; cause unexplained.

**F3. The cash buffer holds about 1% (0.90-1.02% of net assets).** It sits in a money-market fund plus a
"Cash & Other" line. On a creation day the two lines can swing large and opposite while the trade settles (10/07:
+$834,837 and −$824,764), so the code always adds them together.

**F4. Two-decimal option marks: evidence, but confounded.** The marks carry 2 decimals on 9/30 (seed), 10/06 and
10/12 (the two creation days) and 4 decimals otherwise. But audit A1 found that Roundhill rewrote exactly the
10/05, 10/06 and 10/12 files the next day, and each rewrite strips trailing zeros from every field (10/05's
"54.924" reads 54.9240). So the data cannot separate "a 2-decimal mark is the fund's own trade price" from "the
revision replaced the mark". The frozen hypothesis H1 tests creation days with this confound in view; the capture
now keeps every version of a revised file.

**F5. The issuer revises holdings files the next day.** The files for the 10/05, 10/06 and 10/12 closes carry
server timestamps of 15:42-20:03 UTC the following day, after first publication around 23:00 UTC. The capture
re-checks recent days and keeps every version.

Could not test: the content of those revisions (the capture never saw the first versions); the seed price (no
document states it).

### S1: what the daily marks imply (2026-10-14; `scripts/implied_vols.py`, `scripts/ratio_band.py`)

Data: Roundhill holdings marks 2026-09-30..10-13. Spot: Roundhill's reference close for 10/13; TradingView
AMEX:SPY closes for earlier days (they match Roundhill on 10/13 and track SPYM in Roundhill's holdings). US Treasury
par curves (same day; Columbus Day 10/12 uses 10/09). SPY dividends to 2026-09-18 from a Yahoo chart file
(secondary source; State Street check pending), projected at 5%/yr growth (ASSUMED). Model: Black-76 on the
escrowed-dividend forward, ACT/365. Audited (A1).

**F6. The marks imply a volatility of about 16.5-16.9% and a leverage of about 5.6.** Base case (Treasury curve,
no spread, discrete dividends):

| Quantity (per XX share unless stated) | Range over the 10 closes |
|---|---|
| Implied vol | 16.51% - 16.93% |
| ... financing spread +25 / +50 bp | -0.39 / -0.79 points (the largest input uncertainty) |
| ... dividend growth 0% / 10% (base 5%) | -0.15 / +0.16 points |
| ... continuous dividend yield instead of discrete | -0.01 to -0.03 points |
| Delta per option | 0.379 - 0.408 SPY shares (each XX share ≈ 0.18-0.20 SPY shares) |
| Elasticity ("leverage") | 5.50 - 5.77 (% NAV move per 1% SPY move) |
| Vega | 9.2 - 10.1% of NAV per volatility point |
| Rho | 14.5 - 15.4% of NAV per percentage point of rates |
| Theta (spot, vol and curve held) | -0.13% of NAV per calendar day: about half time decay, half carry; -0.05% if SPY drifts along its forward (see F8) |
| Risk-neutral P(SPY above 995.01 at expiry) | 0.27 - 0.29 in the simple lognormal model (a market price, not a forecast); listed call spreads give about 0.39 (F10) |
| Cash break-even for the option bought at the mark | SPY 1,044.7 - 1,052.3 at expiry |

The two creation days (10/06, 10/12) show the highest implied vols (16.93%, 16.85%); the 10/01 redemption day
does not (16.51%). Ordinary days move 0.1-0.25 points, so these numbers fit H1 without supporting it.

**F7. The S&P "finish line" sits near 10,000 (central ~9,995), and its range depends on the history used.** The
7-session median S&P/SPY ratio nearest Jan 10 rose from 10.0008 (2011) to 10.0369 (2026), with small dips in 2016
and 2026. Effective trigger at 2030-01-10 (995.01 × ratio):

| Sample start | Range of 4-year changes applied to the 2026 ratio | Fitted line at 2030 |
|---|---|---|
| 1994 | 9,955 - 10,040 | 9,993 |
| 2005 / 2008 | 9,968 - 10,002 | 9,994 / 9,998 |
| 2011 / 2015 | 9,989 - 10,001 | 10,002 / 10,001 |

Overlapping spans produce these ranges, so they describe history; they do not form confidence intervals. An
independent October-anchored check in audit A1 gave 9,997.5. Pricing runs on SPY, so these figures leave it
unchanged; a 20-point S&P difference equals about 1.5% of option value. The planning estimate "~9,980-9,995" ran
too low. Data caveat: the local library stores SPY closes from a BATS export, not the official close; medians damp
the noise.

**F8. "Time decay" has two parts, and its size depends on the dividend convention.** With discrete dividends,
holding SPY's price flat for a day costs about 0.0721 per option on 2026-10-13 (-0.13% of NAV). About half is time
decay (0.0368: the option's time value shrinks with the forward price held fixed) and half is carry (0.0353: with
SPY flat, the forward price falls by a day of interest net of dividends). If SPY instead drifts along its forward,
as the option price assumes on average, a day costs about -0.053% of NAV. With a continuous dividend yield the
flat-SPY cost is about 0.0627; SPY's daily dividend accrual explains the 0.0094 gap. The conventions differ in
timing, not in total. In the discrete model SPY's price drops by the dividend on the ex-date, and the option, which
never receives it, keeps its value. The continuous model spreads the dividend across the quarter. Over a quarter
both lose about the same (flat spot for 91 days: -5.78 vs -5.70 per option, audit A1). Hypothesis H5 tests which
convention the vendor uses (first test: SPY ex-date 2026-12-18).

*v0.1.1 correction (audit B10a).* v0.1.0 stepped one day across an ex-date without taking the dividend off SPY's
price, so on the eve of an ex-date its theta showed a false gain. Projected with the 2026-10-15 inputs, the report
for 2026-12-17 would have shown +1.24% of NAV per day; the corrected value is -0.14% (the day before: -0.13%).
v0.1.1 also adds a dividend line to the daily attribution on ex-dates, so SPY's price drop does not read as a loss
the option did not suffer.

**F9. The pricing vendor discounts at about 0.5 points above Treasuries, at least short-dated.** XBOX's SPY box
spread (expiry 2026-12-30), valued as of each file's effective date, implies a steady 4.62-4.65% (continuous),
against a same-term Treasury zero rate of 4.12-4.17%: a gap of +47 to +50 bp. This one short-dated instrument does
NOT show the rate at 3 years. If a similar gap applied to XX's option, the implied vol would run about 0.75-0.8
points below the base case (the +50 bp column). Corrected after audit A1: the first version used the close date
(false Friday dips) and compared a continuous rate with bond-equivalent yields. Since v0.1.1 the model table flags
weaker box evidence: `box_leg_decimals` (2 decimals on the legs suggests traded prices, not the vendor's 4-decimal
model) and `box_flow_day` (XBOX's shares changed). On 2026-10-15 both flags fire and the rate reads 4.48%, so that
day says little about the vendor's rate (audit B10a).

**F10. The listed S&P 500 options price more modest wins than the simple model; at 5× and 10× the two cannot be
told apart.** Exploratory, one snapshot: delayed Cboe bid/ask quotes for the S&P 500 Dec-2029 calls (strikes
8,000-15,000), read through TradingView early on 2026-10-14 (New York time) and kept local. The read fell after the
2026-10-13 close but inside Cboe's overnight session, so the quotes may not be the regular-session closing quotes.
Method (`definedtarget.analytics.call_spread_prob` and `prob_above_from_spreads`, tested on synthetic quotes): a
call spread (one call minus a higher-strike call) divided by the strike gap and the discount factor gives the
market's average risk-neutral chance of finishing above a level between the two strikes; the code interpolates the
log of that chance between spread centers and extends the last trend beyond the last center (14,500). The simple
model is the report's lognormal at XX's own implied vol (16.61%), to the listed expiry. A buyer at NAV ($26.51):

| XX ends at | S&P needed | Price growth per year | Simple model | Listed options |
|---|---|---|---|---|
| Options pay anything | 9,995 | 7.9% | 28.5% | 39% |
| Break even (1×) | 10,563 | 9.8% | 22.6% | 29% |
| 2× | 11,136 | 11.6% | 17.6% | 21% |
| 3× | 11,708 | 13.3% | 13.6% | 15% |
| 5× | 12,854 | 16.6% | 7.9% | 7% (3-11% across the bid/ask quotes) |
| 10× | 15,719 | 24.1% | 1.8% | about 1% (0-3%; extrapolated beyond the last spread) |

The volatility skew drives the gap in the first rows: lower strikes trade at higher implied vol (near 10,000 the
listed skew runs -0.165 vol per unit of log-strike), and the skew contributes about 10 of the 10.5 points at the
strike. Only the first rows are robust: the 9,000-11,000 spread bounds the first row at 36-42% even at the worst
bid/ask prices, while the 5× and 10× rows overlap the simple model within their quote ranges. Other limits: one day;
the S&P forward comes from SPY's forward times the S&P/SPY ratio (this affects the implied vols and the simple
model, not the call-spread odds); the listed calls settle on 2029-12-21, three weeks before XX's option (the report's
own figure to 2030-01-10 is 29.1%, not 28.5%). Risk-neutral odds are prices, not forecasts. In implied-vol terms
XX's mark (16.61%) sits at the listed 10,000 call's bid (16.65%), below its mid (17.26%). Hypotheses H13, H17 and H19
test these points daily from 2026-11-01. Audit A2 re-derived every number in this table with its own code.

**F11. Over one two-week span, XX's extra day-to-day leverage did not build up.** Exploratory, 10 closes; weak
evidence. Day to day, NAV moved 1.15-1.25 times the model elasticity of about 5.6 (1.26 against the same day's
elasticity, 1.25 against the previous close's). A volatility slide fits it: the vendor's implied vol rose about
0.13-0.18 points per 1% SPY move (a one-variable slope; R² 0.20 on 9 changes, 0.62 on the 3 clean ones), which adds
1.23-1.31 times. But from 2026-09-30 to 2026-10-13, SPY rose 2.02% while implied vol rose only 0.09 points; a slide
that lasted would have added 0.37. One span cannot separate a fading slide from noise: the gap of 0.28 points is
smaller than the noise one would expect over nine days of independent changes (about 0.5 points). Hypotheses H8
(day to day) and H14 (10-session spans) test both from 2026-10-14.

**F12. When the S&P falls, the 3-year volatility at XX's strike rises only about 1 point per 10% fall: a fifth of
what one-year volatility does.** Measured 2026-10-17 for the browser simulator's "vol follows the S&P" default.
The slide is the change in implied volatility (points) for a 10% fall, from a regression on the S&P's log return.

| Series | Span | Window | Slide per -10% (standard error) | R² |
|---|---|---|---|---|
| VIX1Y (Cboe one-year index) | 2007-2026 | 1 / 5 / 21 / 63 days | +4.6 / +4.3 / +4.8 / +4.3 (0.05-0.40) | 0.55-0.60 |
| Listed S&P 500 Dec-2029 10,000 call, implied vol of each day's last trade | 2024-06 to 2026-09, 252 traded days | consecutive trades | +1.1 (0.13) | 0.22 |
| ... days with 5 or more contracts | same | consecutive trades | +1.1 (0.15) | 0.30 |
| ... last trade of each week | same | weekly | +0.8 (0.17) | 0.17 |
| ... last trade of each month | same | monthly | -0.3 (0.33) | 0.02 |
| VIX1Y on the same traded days | same | consecutive trades | +5.0 (0.21) | 0.69 |

The listed call has the same strike as XX's option (S&P 10,000 ≈ SPY 995.01) and an expiry three weeks earlier, so
its implied vol is the closest public proxy for XX's: in late September 2026 it traded at 17.0-17.3%, against the
vendor's 16.4-16.9% marks for XX. Its slide is about 0.2 times VIX1Y's on the same days; 0.2 × VIX1Y's 19-year +4.6
gives a similar +0.9. In the April 2025 sell-off the S&P fell 13.6% (2025-03-07 to 04-08) and this vol rose 2.3
points (14.0% to 16.3%), then fell back to 13.5% on the next day's rebound. So the response is real but small and it
fades: at monthly spacing it is indistinguishable from zero (26 months; wide error).

What it means: XX's "volatility cushion" in a crash is small, and the slide you assume decides a crash scenario.
On the 2026-10-13 example (simulator model), an instant 10% fall takes NAV down 45.8% with no vol change, 38.4% with
the measured +1.0 point, and only 10.0% if XX's vol jumped like VIX1Y (+4.6). The simulator uses +1.0 point per
-10% as its default slide, with the switch and slider to try others (VIX1Y's +4.6 is the upper reference). Caveats: thin trading
(often one contract a day), each day's last trade is not at the index close, rates and the dividend yield are
assumed (Treasury zero curve; SPY's trailing 12-month yield), and F11 saw the vendor's marks move the other way
over ten closes. Data stay local (TradingView quotes and Cboe's VIX1Y file); only these statistics are published.
Code: `scripts/measure_vol_slide.py` (the VIX1Y rows, from your own closes and Cboe's free file).
More caveats (audit B10b): each implied vol comes from a trade made before the close it is paired with, so the
S&P returns carry noise and the regression slope is pulled toward zero: +1.1 is more likely too low than too high.
The April 2025 episode alone ran about +1.6 to +1.7 points per 10% fall. A fixed strike also slides further out of the money
as the S&P falls, and the skew (F10) lowers its vol by about 1.7 points per 10%, which helps explain why this slide
is so much smaller than at-the-money measures. The simulator applies the slide symmetrically, so the rally side is
an extrapolation (F11 saw the vendor's marks move the other way over ten closes). The monthly estimate cannot tell
+1 from zero either way.


**F13. Over stretches as long as XX's life, the S&P has cleared XX's break-even about a third to a half of the
time.** Measured 2026-10-17 for the simulator's odds rows (`scripts/history_windows.py`, Shiller's monthly data).
Every 39-month window (3¼ years, one starting each month), annualized:

| Start | Windows (separate) | Total return: median / top 25% / top 10% / top 5% / best | Price only: median / top 10% |
|---|---|---|---|
| 1871 | 1,831 (~46) | 10.0% / 16.1% / 22.5% / 26.6% / 40.2% | 5.6% / 17.5% |
| 1950 | 883 (~22) | 11.7% / 16.5% / 21.5% / 26.0% / 33.5% | 8.8% / 17.9% |
| 1988 | 427 (~10) | 12.4% / 16.4% / 21.4% / 27.0% / 33.5% | 10.2% / 19.4% |

The best window began in mid-1926 and ended just before the 1929 crash. A check on the daily S&P 500 total-return
index (1988-2026) gives 12.5% / 16.6% / 22.1% / 27.0% / 34.4%, close to Shiller's monthly figures.

Share of windows whose price growth cleared XX's bars (2026-10-13 example): reaching the strike (7.9%/yr) 41% since
1871, 56% since 1950, 69% since 1988; breaking even at NAV (9.8%/yr) 33%, 43%, 52%; the 16% row's median path
(13.3%/yr) 20%, 23%, 27%. The option price implies a 23% chance of profit, below the historical shares since 1950:
stocks have paid a reward for their risk, and option prices do not include it.

Caveats: the windows overlap (about 46 separate ones since 1871, 22 since 1950, 10 since 1988), so these shares are
rough. They ignore today's starting valuation; high valuations have been followed by lower returns, and the planned
history study conditions on it (CAPE). Shiller's price is a monthly average, which smooths the windows a little.
The simulator's odds rows run to a 24% mean total return (its median path is about history's top 10%), plus a custom
row that flags inputs beyond history's top 5% and its best.

**F14. A gaming GPU prices XX's option by Monte Carlo about 25 times faster than all 24 CPU threads, but only on
big batches.** Measured 2026-10-17 (`scripts/b17a_gpu_first_test.py`; i7-13700K, numba 0.68 on 24 threads; NVIDIA
RTX 3080 10 GB, CuPy 14.2 with CUDA 13.2). One lognormal end price per path; the README's 2026-10-13 example (exact
Black-76 value 54.677436). Median of 3 runs up to 100 million paths, one run at 1 billion:

| Engine | 10 thousand | 1 million | 100 million | 1 billion | Speed at the largest size |
|---|---|---|---|---|---|
| numpy, 1 core | 0.000 s | 0.025 s | 2.83 s | (not run) | 35 million paths/s |
| numba, 24 threads | 0.000 s | 0.003 s | 0.203 s | 2.006 s | 498 million paths/s |
| CuPy, 64-bit | 0.001 s | 0.002 s | 0.037 s | 0.340 s | 2.9 billion paths/s |
| CuPy, 32-bit | 0.001 s | 0.001 s | 0.010 s | 0.080 s | 12.5 billion paths/s |

All 19 estimates sat within 3 standard errors of the exact value (largest gap 1.73 SE). At 1 billion paths the
standard error is 0.004 on a price of 54.68 (0.007%).

What it teaches, for the v0.3 engine:
- Small jobs belong on the CPU. At 10 thousand paths the GPU did 8 million paths/s against numpy's 85 million:
  starting a GPU job and copying results back costs more than the work.
- 64-bit math cost only about 4x here, not the 64x the card's arithmetic rate suggests: this job spends its time
  moving random numbers through the card's memory, not on arithmetic. A kernel that makes its random numbers
  inside (the planned counter-based Philox) avoids that traffic, so v0.3 sets its targets on that design and
  measures the 32-bit vs 64-bit gap again there.
- Sums of millions of 32-bit numbers were accumulated in 64-bit; the 32-bit results agree with the exact value as
  closely as the 64-bit ones.

Caveats: one simple payoff and one machine; wall-clock times on a desktop that was also running other programs.
The engines use different random streams (numpy PCG64, numba per-thread streams, cuRAND), so they agree only within
Monte Carlo error, not path by path.

**F15. Post-1950 S&P 500 history's shape lowers XX's model value by 13-24% against the simple model at the same
daily volatility (history since 1926, with the 1930s: +6% to -14%), and one era's history understated the next
era's drawdowns.** Measured 2026-10-17
(`scripts/bootstrap_study.py`, montecarlo engine v0.3 steps 1-5; S&P 500 price closes 1950-01-03 to 2026-10-09;
Ken French's daily US market total return 1926-07-01 to 2026-08-31; 1 million paths per row; the script fixed the
study's outputs before the first run, and one diagnostic column, "value, E = F", came after it).

Out of sample (build 1950-1999, test 2000-2026, 814-session windows, about 8 separate ones):

| Generator built from 1950-1999 | Test windows below its 10th / above its 90th percentile: return | ... : maximum drawdown |
|---|---|---|
| lognormal | 19.3% / 1.5% | 10.9% / 38.8% |
| bootstrap, mean blocks 21 / 126 / 504 sessions | 18.7-19.6% / 1.3% | 10.8-10.9% / 36.5-37.1% |

Ideal: 10% / 10%. The build period ran at 13.6% annualized volatility and the test period at 19.2%: generators built
from one era's history, at that era's volatility, put 2000-2026's drawdowns above their 90th percentile in 37-39% of
windows. A history bootstrap is a shape, not a forecast; its volatility has to be an input. The block length did
not fix this (it moved the median drawdown percentile from 0.66 to 0.53).

Shape alone (each history reset to the same daily mean and spread as the lognormal: XX's 2026-10-13 example, vol
16.6%, drift 3.87% a year, strike 1.279 x spot, 814 real sessions):

| Generator | P(finish above strike) | P(touch strike at a close) | S_T/S_0 10th / 50th / 90th | Value per $1 of spot, E[S_T] = F |
|---|---|---|---|---|
| lognormal (Black-76's model) | 29.0% | 49.3% | 0.740 / 1.085 / 1.590 | 0.0701 |
| S&P 500 1950-2026, blocks 21 / 126 / 504 | 28.5% / 28.9% / 27.3% | 46.5% / 45.0% / 42.8% | 0.75 / 1.10-1.11 / 1.51-1.55 | 0.0610 / 0.0575 / 0.0530 |
| US market 1926-2026, blocks 21 / 126 / 504 | 30.3% / 31.9% / 32.2% | 49.2-49.4% | 0.71-0.72 / 1.09-1.14 / 1.56-1.61 | 0.0743 / 0.0705 / 0.0604 |

Standard errors: 0.05 points on the probabilities, about 0.0002 on the values. The chance of finishing above the
strike moves little (27-32% against 29%); the value moves more, because a far out-of-the-money call lives in the upper
tail, and post-1950 S&P history at the same daily volatility has a thinner 3-year upper tail (90th percentile
1.51-1.55 against 1.59): its daily swings partly undo each other over years. History since 1926, with the 1930s,
has the opposite tilt at short blocks.

Caveats: the daily spread is matched, not the 3-year one; the option market's 16.6% is a 3-year implied volatility,
so this is not a mispricing claim, it is a statement about how daily risk adds up over years in each history.
Few separate 3-year spans exist in any history (about 23 since 1950, 31 since 1926), so the block length (a choice,
reported at three values) moves the answer as much as the history does. Price returns, demeaned: dividends do not
change the shape. A regime model and a joint volatility path are later work (v0.3.x).

**F16. The engine's GPU path kernels run XX's 814-session paths 2.4x (lognormal) to 16x (bootstrap) faster than 24
CPU threads, and held up through a 5-minute full-load stress test with no errors.** Measured 2026-10-17
(`scripts/gpu_stress_test.py`, `tests/test_mc_gpu.py`; RTX 3080 10 GB, CuPy 14.2 with CUDA 13.2; i7-13700K):

| Kernel (1 million paths x 814 real sessions, 64-bit) | 24 CPU threads | RTX 3080 | GPU / CPU |
|---|---|---|---|
| lognormal (with barrier and Brownian-bridge touch) | 0.57 billion path-steps/s | 1.36 billion | 2.4x |
| bootstrap (S&P-shaped history, mean block 126) | 1.26 billion | 19.7 billion | 16x |

Stress test: 301 s at 100% load, alternating the two kernels on new paths: 229 batches, 916 million paths, 0.75
trillion path-steps. Temperature 45-66 C (stop set at 83 C), core clock 1,935-1,965 MHz, power up to 190 W, fan up to
72%. Speed in the last minute was 2-4% below the first (lognormal 1.36 to 1.33, bootstrap 19.65 to 18.91), with the
core clock 15-30 MHz lower: a small warm-up effect, no throttling. A CPU recomputation of 2,000 paths from every batch
found 0 mismatches (bootstrap bit for bit; lognormal within 2.2e-16). Over 460 million lognormal paths, P(touch +28%
before expiry) = 0.502710 +/- 0.000023 against the exact 0.502716.

What it teaches: the bootstrap is mostly integer work (Philox) and table lookups, where a gaming GPU excels;
the lognormal turns every random number into a normal with about 30 64-bit operations, and this card does 64-bit
arithmetic at 1/64 of its 32-bit rate, so that kernel is limited by arithmetic and draws only about half the card's
power. A 32-bit lognormal kernel is the next speed step, with its accuracy measured against the 64-bit one. GPU and
CPU values agree to the last digit except about 1 in 40,000 tail normals, where NVIDIA's log rounds differently.

**F17. Path by path, XX doubles at some point in about half of all futures but ends up in about a quarter, and the
"vol follows the index" rule pulls its expected mid-life mark about 5% below today's price.** Measured 2026-10-17
(`definedtarget/xxpaths.py`, `scripts/xx_paths_study.py`; outputs fixed in the script before the run; 1 million
lognormal paths per row; XX's 2026-10-13 example, 815 real sessions; 16.6% realized and implied vol; model start NAV
26.49 against the published 26.51). The real-world drifts are assumptions, not forecasts.

| SPY drift | P(end above start) | P(end below 10% of start) | P(NAV reached 2x / 5x / 10x at some session) | P(reached 2x, ended below it) | Mean largest drawdown | P(XX beat SPY) |
|---|---|---|---|---|---|---|
| risk-neutral (mean 3.87%/yr) | 23.1% | 71.5% | 49.3% / 16.7% / 3.7% | 31.2% | 93.9% | 21.1% |
| median 7%/yr price growth | 39.3% | 54.1% | 63.5% / 29.4% / 9.0% | 30.8% | 89.2% | 36.8% |
| median 10%/yr | 51.2% | 42.2% | 72.3% / 39.7% / 14.7% | 28.1% | 85.3% | 48.5% |

With the slide rule at 1 point per -10% (F12), the end-of-life numbers do not change (the payoff ignores vol) and the
chance of touching 2x along the way falls 2.7-3.8 points (45.5% / 60.2% / 69.6%), because vol falls as SPY rises. The
risk-neutral profit chance (23.1%) matches the 23% that F13 read from the option price by a different route.

The martingale gap (the bare option: no fee, no cash; risk-neutral drift; 1 million paths, seed 2026, printed by
`scripts/xx_paths_study.py` since 2026-10-18): under constant vol the discounted mark averages to today's price, as it
must (+0.03%, +0.07%, +0.23%, +0.12% at sessions 100, 400, 700 and expiry; all within 1.1 standard errors). With the
slide rule it does not: -2.33% (SE 0.05%), -5.13% (0.13%), -2.29% (0.21%), then +0.12% at expiry. A vendor
whose marks followed this rule would show XX's mid-life NAVs below what today's option price implies, on average.
Either the rule is not how the market moves long-dated vol (F12 measured it on two years of one listed option), or
today's price embeds a dynamic the rule leaves out (the skew moving with the index). This is the project's
first open mystery; the v0.3.x vol model and the real-path test against VIX1Y are where it gets resolved.

Caveats: lognormal paths at one vol; fee taken by selling options, cash flat (sim.py's model, not the full fund
engine); no FLEX trading costs, closure or roll scenarios; touch probabilities count every session, which is not a
trading result (a sale at 2x pays the bid, after costs, on a day you must recognize).

**F18. On real S&P paths since 2007, how XX's vol is assumed to move changes its drawdowns and peaks by several
points: real VIX1Y moves 3.5 times as much as the slide rule, cushions the crashes, and trims the rallies.** Measured
2026-10-17 (`scripts/slide_rule_test.py`; outputs fixed in the script before the run; S&P 500 closes and Cboe VIX1Y,
2007-01-03 to 2026-10-09; 198 synthetic XXs started every 21 sessions, each with 815 real sessions of life, about 6
separate spans; XX's shape, cash and fee; rate and dividend yield held constant; all three markings start at
0.77 x VIX1Y on the start day, 0.77 being the design review's implied-vol / VIX1Y multiplier, an assumption).
Corrected 2026-10-18 after an independent external audit of v0.3.0: the first run gave each synthetic XX half of
XX's cash weight (0.45% instead of 0.94% at the start); the table below is the corrected run. The correction moved
drawdowns by 0.3-0.4 points, peaks by 0.01-0.03x and one touched-2x share by 0.5 points; the gaps and the
measured slide did not change:

| Marking | Largest NAV drawdown: median / 90th pct | Longest underwater (sessions): median / 90th | Highest NAV / start: median / 90th | Touched 2x |
|---|---|---|---|---|
| A constant vol | 91.2% / 99.4% | 460 / 752 | 2.67 / 6.46 | 69.2% |
| B slide rule, 1 point per -10% | 91.3% / 99.4% | 464 / 765 | 2.52 / 6.46 | 65.2% |
| C real VIX1Y x 0.77 | 83.7% / 99.3% | 464 / 757 | 2.51 / 6.47 | 70.2% |

Start by start, the largest drawdown under B and C differs by a median 3.3 points (90th percentile 13.0), and the
peak by 0.04x (0.37x); A against C: 2.7 (14.7) points and 0.07x (0.28x). The slide rule is no closer to the
VIX1Y-marked life than constant vol is. Measured on these data, 0.77 x VIX1Y rises 3.55 points per -10% of the S&P
(daily changes; 3.27 over 63 sessions), against the rule's 1.00, consistent with F12's 4.6 points for VIX1Y itself.

What it says about the F17 mystery: in the long rallies vol fell as the S&P rose, and both moving-vol markings peak
lower than constant vol (median 2.51-2.52x against 2.67x): the mechanism behind F17's mid-life gap (a cut to vol
when the option is most sensitive to it) appears on real paths too. In the crashes vol spiked and cushioned the
mark (median drawdown 84% against 91%). Which vol is right for XX's own option is still open: F12 found the listed
3-year strike vol moving about 1 point per -10%, far less than VIX1Y; the frozen hypotheses H4 and H10 test the
vendor's marks directly.

Consequence for the engine's outputs: XX's path statistics (drawdowns, peaks, touches) carry a model-risk band from
the vol rule of several points; report them under at least two vol rules until H4/H10 or the v0.3.x vol model
settles it. Caveats: 2007-2023 starts include strong rallies (69% touched 2x), so these are not odds; rates were
held constant (they were near zero in 2009-15 and 2020-21); the 0.77 multiplier is assumed, not fitted.

**F19. The slide-rule effect belongs to the option, not to XX: XX, the bare SPY call inside it and the listed S&P
10,000 call of December 2029 lose the same effective leverage on up and down days and show the same mid-life
value gap.** Measured 2026-10-17 (`scripts/effective_leverage_study.py`, `xxpaths.effective_leverage`; outputs fixed
before the run; one disclosed change after it: days count only while the previous mark is at least 25% of the start,
because near-worthless bare calls have unbounded elasticity and dominated the first run; 400,000 risk-neutral paths
per row, 16.6% vol, the real calendar):

| Instrument | Slide | Effective leverage up / down days | Mid-life gap, session 400 (+/- 1 SE) |
|---|---|---|---|
| XX (cash, fee) | 0 / 1 / 3.5 points per -10% | 1.009 / 0.990; 0.895 / 0.878; 0.625 / 0.608 | -1.80%; -6.88%; -15.73% (0.2%) |
| XX's option alone | same | 1.010 / 0.990; 0.895 / 0.878; 0.626 / 0.607 | -0.19%; -5.39%; -14.47% |
| listed S&P 10,000 call, Dec 2029 | same | 1.010 / 0.989; 0.895 / 0.877; 0.623 / 0.604 | -0.19%; -5.55%; -14.97% |

Effective leverage = the realized move per 1% S&P move divided by the elasticity known at the previous close, pooled
over paths and days; 1.0 = moved as the elasticity said (curvature alone gives 1.01 / 0.99). XX's -1.80% at constant
vol is its fee and its flat cash, not the slide. With vol moving against the index, all three move about 11-12% less
than their elasticity says on both up and down days at the rule's slide, and about 38-40% less at a VIX1Y-sized slide
(F18).

What it means: the listed call is traded, so under the option price's own (risk-neutral) view its expected discounted
mark cannot fall 5% by mid-life. If its strike vol truly followed the rule, today's price would already be lower than
the constant-vol Black-76 value. The test: compare the listed call's real quotes since 2024 (the F12 data; H17 from
2026-11-01) with both readings, in the v0.3.x vol study.

**F20. The VIX family alone cannot pin down the long-run vol that a 3-year option depends on; the listed 3-year
strikes can.** Measured 2026-10-21 (`scripts/vol_identification.py`; design fixed before the run, two reporting
changes after it, both disclosed in the script; synthetic quotes only). 200 noisy copies of a 34-quote layout like
this project's data (the VIX family at 9 days to 1 year; listed calls at 6,500-10,000 for Dec-2027, Dec-2028 and
Dec-2029; XX's strikes 11,000-15,000 for Dec-2029), each fitted with the Heston model from one generic start:

| Truth (long-run vol / vol at XX's strike) | Quotes fitted | Long-run vol, 5-95% | Vol at XX's strike (10,000, 3.17 years), 5-95% | Fits converged |
|---|---|---|---|---|
| Gatheral (2006) S&P 500 fit (18.81 / 13.56) | VIX family alone (a) | 17.53 to 25.68 | 12.52 to 14.23 | 200 |
| | listed options alone | 17.90 to 20.04 | 13.34 to 13.76 | 200 |
| | both | 18.24 to 19.53 | 13.46 to 13.64 | 200 |
| Andersen (2008) Case III (30.00 / 23.55) | VIX family alone (a) | 0.00 to 63,183 | 16.25 to 29.99 (b) | 169 |
| | listed options alone | 28.55 to 32.45 | 23.38 to 23.73 | 200 |
| | both | 29.62 to 30.45 | 23.38 to 23.72 | 200 |

(a) Vol of vol and correlation fixed at the truth, because no variance-swap rate depends on them; this flatters
these rows. (b) 12 of the converged fits gave no vol at that strike.

Why: a variance-swap rate depends only on today's variance, the long-run variance and the speed of mean reversion,
and the VIX family stops at one year, so everything beyond is an extrapolation along one curve. When the term
structure is flat (Case III starts at its long-run level) the speed is not identified at all, and the long-run level
drifts with it. Adding the listed Dec-2027 to Dec-2029 strikes pins the vol at XX's strike to within 0.1-0.2 points
(5-95%), given the model. So the open question for v0.3.1 is model error (can Heston fit the listed strikes and
their dynamics at all?), not parameter noise. Where the parameters are identified, the fit's own standard errors
match the spread of the estimates (speed of mean reversion, Gatheral truth, both quote sets: spread 0.200, reported
0.212). Caveats: the noise levels (0.10 vol points for the indices; 0.10, 0.25 and 0.50 for the three expiries) are
assumptions near half a bid-ask spread, not measurements; the noise is independent and normal; the model is right by
construction.

**F21. The COS pricing method's standard reference needs two corrections for long-dated Heston options.** Found
2026-10-21 while building `definedtarget/volmodel` (tests in `tests/test_vol_heston.py` and
`tests/test_vol_pricers.py`). Fang and Oosterlee (2008), the paper that introduced the COS method:
1. Its Table 11 prints the variance of the Heston log-price with the term theta (6 e^{-kappa T} - 7); the correct
   term is theta (4 e^{-kappa T} - 5). The printed form disagrees with the numerical derivatives of the
   characteristic function (by up to 223% in our test cases), has a term of order T that no variance can have at
   short maturities, and turns negative, which matches the paper's own remark that it "may become negative". The
   corrected form equals Andersen's (2008, Table A.1) independent formula to rounding. The paper uses this value only
   to size its integration range, so its prices are unaffected.
2. Its integration range, the mean plus or minus 12 standard deviations, is too narrow for Heston's exponential tails
   at long maturities: with the correct variance it prices the December 2029 10,000 call 0.38 too low under
   Andersen's Case I (a published stress case), and 4e-5 too low at the paper's own one-year test. The package sizes
   the range instead with Chernoff's bound from the model's moment generating function, so that each tail holds at
   most 1e-15 of the probability; the COS price and an independent Lewis-Lipton integral (Andersen 2008,
   Proposition 3) then agree to about 1e-14 of the forward on every case tested.

Also: at the paper's one-year Heston test, three methods here (COS, Gauss-Legendre and scipy's adaptive quadrature
of the integral) agree with each other to 1e-14 and sit 1.6e-8 below the paper's printed reference value, which
came from a Carr-Madan run on a fixed grid; at ten years they match the paper's reference to 3.5e-10.

**F22. In the Heston model the marked option of XX does not drift, and the error of the simulation is far below a
vol point.** The script `scripts/qe_bias_study.py` measured this on 2026-10-23. Its design was fixed before the run.
It uses 16,777,216 synthetic paths for each case, the parameters of Gatheral (2006, Table 3.2), and the listed
December 2029 10,000 call. The two pricers of `volmodel.pricers` give the exact price.

The first table shows the bias of each scheme, the Monte Carlo price minus the exact price, in vol points. The
standard error is 0.0037 vol points in each case.

| Time step | Steps | QE-M | QE | Euler |
|---|---|---|---|---|
| 1 session | 795 | +0.0006 | +0.0006 | +0.0014 |
| 5 sessions | 159 | -0.0096 | -0.0091 | -0.0017 |
| 21 sessions | 38 | -0.0080 | +0.0005 | +0.0647 |
| 63 sessions | 13 | -0.0265 | +0.0407 | +0.3455 |

The second table shows the martingale identity. The price table marks the call along daily QE-M paths. The gap is
the expected value of the mark divided by the price now, minus 1.

| Checkpoint (years) | Gap | Standard error |
|---|---|---|
| 0.51 | -0.017% | 0.016% |
| 1.50 | -0.034% | 0.032% |
| 2.50 | -0.052% | 0.047% |

What it means:
- On daily steps the QE-M scheme has no measurable bias. Even on steps of 63 sessions, QE and QE-M stay within
  0.05 vol points. The Euler scheme reaches 0.35 vol points on those steps.
- So the simulation adds no error that matters next to the uncertainty of the vol itself (F20).
- The marked option keeps its value in expectation. Each gap is within 1.1 standard errors, and each standard
  error is at most 0.05% of the price. This meets the accuracy target of v0.3.1.
- The slide rule gave a gap of -5.55% for the same option at mid-life (F19). So the drift of F17 comes from the
  slide rule, and a model with a random variance removes it.

Limits: here the model is right by construction, so the test says nothing yet about the real market. The three
checkpoints use the same paths, so their errors are correlated.

**F23. Under the fixed holdout the one-factor Heston model misses the accuracy target, but it describes the weekly
moves of the 3-year strike vol best.** The script `scripts/vol_holdout_study.py` measured this on 2026-10-23. The
protocol is in the script, and its commit came before any test-window number.

Stage 1 fits the VIX family (VIX, VIX3M, VIX6M and VIX1Y) from 2009-09-18 to 2019-12-31 (2,587 days). One speed
cannot fit it. The fit puts kappa at 0.147 a year and the long-run vol at 54.9%. The block bootstrap gives no upper
end for that vol: 64 of 200 refits put it above 100%. The RMSE in the fit window is 1.52 vol points for VIX and 1.41
for VIX1Y.

Out of sample (2020-01-01 to 2026-10-16), each day's VIX alone predicts the longer indices:

| Index | Model RMSE | Flat (= VIX) | VIX plus the mean spread |
|---|---|---|---|
| VIX3M | 2.27 | 2.91 | 2.03 |
| VIX6M | 3.40 | 4.90 | 3.36 |
| VIX1Y | 3.98 | 6.39 | 4.82 |

Stage 2 fits the smile to 157 listed trades of the December 2029 calls (2024-06-14 to 2025-06-30). Variant B, the
primary, takes the long-run level from the trades: long-run vol 70.4%, eta 1.14, rho -0.85, in-sample RMSE 0.42
vol points. Variant A keeps the level of stage 1.

The accuracy target uses 54 days of the 10,000 strike, from 2025-07-01 to 2026-09-30:

| Prediction | RMSE (vol points) | Mean error (vol points) |
|---|---|---|
| Model B (primary) | 1.39 | -1.20 |
| Model A | 1.32 | -1.12 |
| Constant 16.6% | 1.35 | +0.68 |
| Sticky strike plus the change of VIX1Y | 1.82 | -1.01 |
| 0.77 x VIX1Y | 2.10 | +1.59 |
| Slide rule (1.0 point per 10% fall) | 3.62 | -3.12 |

The verdict is MISS. Model B is inside the 1.5-point target, but the constant 16.6% is better. Most of the error of
B is a bias of -1.20 points. The constant comes from the marks of October 2026, so it knows the level of the test
window in advance. The protocol named this benchmark, so the verdict stands.

The mystery test compares weekly changes of the traded vol with the change that each reading predicts (test window,
54 pairs):

| Reading | RMSE of the change error | Slope of traded on predicted (SE) |
|---|---|---|
| Fitted dynamics (B) | 0.27 | 0.71 (0.18) |
| Sticky strike (no change) | 0.30 | - |
| Slide rule | 0.34 | 0.19 (0.24) |
| Change of VIX1Y | 0.72 | 0.11 (0.05) |

The belief probe regresses ln(iv / VIX1Y) on ln(VIX1Y): the slope is -0.67 (SE 0.10) on 124 days, and -0.61 (SE 0.10)
at weekly spacing.

What it means:
- The 3-year vol at the strike of XX moves about 0.1 points for each point of VIX1Y. In relative terms it moves
  about a third as much as VIX1Y. A fixed multiple of VIX1Y, such as the 0.77 rule, overstates its moves.
- The fitted dynamics explain the weekly moves best. This agrees with frozen prediction H20: at 3 years the fixed
  strike vol moves little when the index moves.
- The level is the problem. One-factor Heston cannot fit the VIX family, and its long-run level has no upper bound.
  So the model predicts the level of the 3-year vol with a bias. The two-factor model (v0.3.1, step 5) must fix this.

Limits: 54 test days of thin trading; the last trade is not at the close; 25 of the 54 weekly pairs span more than
one week (on one-week pairs the dynamics give 0.25 and a slope of 0.79); the VIX family drops strikes at two zero
bids; F12 saw these trades before the protocol.
