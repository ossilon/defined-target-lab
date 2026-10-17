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
