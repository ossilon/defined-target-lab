# Pre-registered predictions (forward test)

Generated from `forward/predictions.jsonl` by `definedtarget.ledger.render()`; do not edit by hand.
Chain check: **chain intact** (26 entries). README.md, "Check the freeze", shows how to verify that the entries existed before the data that tests them.

## Protocol P1 (frozen 2026-10-14)

- Test data: only closes from 2026-10-14 onward. The 10 closes from 2026-09-30 to 2026-10-13 shaped the thresholds below (scripts/exploratory_stats.py) and count toward no test.
- File versions: each test uses the latest version of every issuer file available on the review date (issuer corrections count); H1 also examines earlier versions.
- Base-case implied vol: Black-76 with the US Treasury par curve bootstrapped to zero rates (0 bp spread), discrete SPY dividends projected at 5% a year growth, ACT/365, and the issuer's reference SPY close (a TradingView close only where the issuer file is missing), as implemented in definedtarget on 2026-10-14 (ASSUMPTIONS A2-A6).
- Clean day: a close that is neither a flow day (shares outstanding changed), nor the day after one, nor an SPY ex-dividend date or the session before one (those days belong to H5). Daily changes use consecutive available closes; the tests drop missing closes and never fill them.
- Listed quotes (H9, H13, H17, H19): the delayed closing bid and ask captured after each close. Implied vols use the base-case curve and dividends; the S&P 500 forward equals SPY's forward to the listed expiry times that close's S&P/SPY reference ratio. A close without a captured quote counts as missing.
- Regressions: ordinary least squares; realized beta through the origin; all other regressions with an intercept.
- Void events: if the fund closes, liquidates, changes its option line (a roll) or reverse-splits before a review date, every affected open hypothesis receives the verdict 'void', naming the event.
- Verdicts: a script computes each verdict from its pass rule; no judgment overrides it. A verdict reads 'inconclusive' only where a rule sets a minimum sample that the data did not reach.
- Minimum samples: if a test with a fixed review date lacks its minimum sample on that date, its review moves to the close that completes the sample, at most three months later; only then does it read 'inconclusive'.
- Multiple testing: about 20 tests; at a nominal 10% error rate each, expect about 2 chance failures. A failure marks a lead to investigate, not proof on its own.

## Protocol P2 (frozen 2026-10-15)

- Scope: P2 clarifies P1 and the hypotheses so that a script can compute every verdict without judgment. It changes no claim, test window start, pass band or minimum sample. Written 2026-10-15 after audit A2, before the capture of any close from the test period.
- Verdict words: pass, fail, inconclusive (a stated minimum sample not reached by the final review date) or void (a P1 void event). H7a receives one verdict per checkpoint and an overall verdict at its last checkpoint: pass if all three checkpoints pass, fail if any fails, otherwise inconclusive.
- Windows: 'from A' includes the close of A; 'to B' ends with the last close before B. Each review uses the data captured by its review date.
- Minimum-sample extension (P1): it applies to the tests with a fixed review date and a stated minimum (H4, H8, H10, H12, H13, H14, H15, H16, H17, H18, H19); the data window grows with the extended review date. H1 and H9 keep their own rule (review after the 15th event, else 'inconclusive' on 2027-10-14) and never extend.
- Multiple testing: P1's '10% error rate' describes chance failures loosely. Only H1 and H9 use p-values; the band tests carry no formal error rate, and no verdict receives a correction.
- File versions: verdicts use the latest saved version of each issuer file. For H1, 'also examines earlier versions' means the verdict reports, beside its result, the same statistic from the first saved version of each file; only the latest-version result decides.
- Model values that scale a daily change (elasticity, vega) come from the previous close, known before the move. H8's 'mean model elasticity over the same days' is the mean of those previous-close values.
- Prices: SPY and S&P 500 changes use the issuer's reference closes (daily_instrument_pricing.csv), else official closes from data/external/closes.csv; a close with neither counts as missing (P1 rule 4).
- Ex-dividend dates for clean days: SPY's actual ex-dates in data/external/spy_dividends.csv (State Street's published distributions; Yahoo where State Street's are unavailable). The excluded closes are each ex-date and the session before it.
- H5: if SPY's December 2026 ex-date is not 2026-12-18, the window is the two sessions ending on the actual ex-date. The dividend is SPY's declared amount for that ex-date. A contract held by more than one fund (same strike and expiry) counts once.
- H7a: the 2026 baseline ratio (10.03693) stands as frozen. Later Januaries use the issuer's reference closes where captured, else official closes; the 7-session median uses only sessions with both closes.
- H12: if XBOX's box expires (2026-12-30) or leaves the file, a 4-leg SPY box in any Roundhill fund's holdings counts; closes without any box are absent, not failures.
- H14: the verdict reports the slope's standard error beside the result, for information.
- H16: the market price is the 'Market Price' column of the issuer's xx_daily.csv, which the fund's factsheet defines as the primary exchange's official closing price.
- Listed quotes (H9, H13, H17, H19): a quote counts only if read between 16:15 and 20:15 New York time on its close date (after the regular session, before Cboe's overnight session); a quote read at another time counts as missing. The S&P forward includes SPY's dividend that goes ex on the morning of the listed expiry (2029-12-21). In H19, XX's risk-neutral probability is the report's N(d2) to XX's own expiry (2030-01-10); the exploratory 28.5% used the listed expiry.
- Verdict script: scripts/verdicts.py will compute every verdict from these rules, with tests on synthetic data, and will be published before the first review date (2026-12-21).
- Disclosure (pre-freeze data; audit A2 found it after the freeze): on the 10 pre-freeze closes, several in-sample values sat outside their frozen bands: the two-regressor H4/H10 test gives a VIX1Y coefficient of -0.83 and an SPY coefficient of +0.077 (the bands came from one-regressor slopes); H16's ratio is 1.15-1.18; H18's same-day-curve coefficient is -0.013. H19's band can also break from a large S&P 500 move alone, with no change in skew. None of this changes a band.

## Protocol FC1 (frozen 2026-10-17)

- Rule 1. Origins and targets. The origin close t is the last NYSE session of each month, from 2026-10-30 to 2029-12-31 (39 origins). The target close T is the last NYSE session of the next month; for the last origin (2029-12-31), T is 2030-01-10. Every origin counts; none may be skipped or replaced.
- Rule 2. The quantity forecast is XX's growth R = NAV(T) / NAV(t), from the issuer's published NAVs.
- Rule 3. The inputs come only from close t: the row for t in the project's model table (SPY reference close, the mark's base-case implied vol `iv_disc_0bp`, the zero rate to expiry, the dividend forward), the fund state for t (options per share, cash per share), the 0.99% fee, and Cboe's VIX close for t.
- Rule 4. Four models, computed every month with `definedtarget.forward.forecast` (which calls `definedtarget.xxpaths.simulate`): SPY follows a lognormal path at the risk-neutral drift (rate minus dividend yield) on the real NYSE calendar; XX is repriced at T with the mark's implied vol moved by the slide rule. The model's growth is the model NAV at T over the model NAV at t, so the gap between the model's and the issuer's NAV at t cancels. M1 (primary): SPY path vol = the mark's implied vol; slide 1 vol point per -10% of SPY (the simulator's rule, FINDINGS F12). M2: path vol = the mark's implied vol; slide 0 (constant vol). M3: path vol = the mark's implied vol; slide 3.5 points per -10% (the slide measured on real VIX1Y paths, F18). M4: path vol = VIX / 100 (the market's 30-day vol); slide as M1. It differs from M1 in one input only.
- Rule 5. One million paths per model, seed = the origin's year x 100 + month (2026-10 -> 202610), stream 0. The saved forecast holds each model's growth at its 0.1%, 0.2%, ..., 99.9% points (999 values, 12 significant digits) and SPY's lognormal terms (drift, path vol, years from t to T).
- Rule 6. Scoring, from the saved forecast only: XX's PIT = the share of the forecast below R, by linear interpolation between the saved points (0.0005 below the first, 0.9995 above the last). SPY's PIT = the lognormal probability that SPY's close at T is below the one observed, under the model's drift and path vol (closed form). The SPY closes are the model table's reference closes for t and T. PITs are rounded to 4 decimals.
- Rule 7. Recording. By the end of the second NYSE session after t (New York time), save the forecast privately (it holds issuer-derived numbers; written once, never overwritten) and append a forecast entry to `forward/forecast_log.jsonl`, hash chained like the predictions ledger: origin, target, seed, paths, the models, the SHA-256 of the saved forecast and of each code file that computes it, and the UTC date it was recorded (the date, not the time). By the end of the second session after T, append the outcome entry: each model's two PITs, after checking that the saved forecast still matches its logged SHA-256. A late entry is still recorded and marked late: the forecast is fixed by these rules and data dated t or earlier, so lateness cannot change it, and no month is ever dropped. The daily routine notes a due entry and STOPs on a late one.
- Rule 8. Void months (the outcome entry records "void" and the reason; the forecast entry stays): XX splits, reverse splits, pays a distribution, rolls, announces closure, or changes its option between t and T; or NAV(t) or NAV(T) is never published. A creation or redemption is not a void.
- Rule 9. What gets reported (no verdict): after every 12 outcome entries and at the end, per model and for both PITs: the count, the share inside the middle 50% band (0.25-0.75) and the middle 90% band (0.05-0.95), each with a 95% Wilson interval, the mean PIT, and a 10-bin histogram. Public files hold PITs and these summaries only, never NAVs, returns or the saved forecasts.
- Rule 10. A later protocol may add models (for example bootstrap-driven paths or the v0.3.1 vol model) or analyses of the saved forecasts (scores, baselines, regime splits) from its own freeze date onward; it never changes M1-M4, this protocol's reports or a recorded entry.
- Known biases, stated before any data: the risk-neutral drift pushes PITs slightly toward 1 (a 4% a year premium moves XX's one-month median by about 2% against a one-month sd of about 28%); M4 is expected to be a little too wide (VIX usually exceeds the volatility that follows); M1-M3 use a 3-year vol for a one-month move, so they should be too wide in calm months and too narrow in stressed ones.

## Protocol FC2 (frozen 2026-10-23)

- Rule 1. Scope. FC2 adds two models, M5 and M6, to the monthly forecast check under FC1 rule 10. They run for every FC1 origin from 2026-10-30 onward. FC2 never changes M1-M4, the FC1 rules, the FC1 reports or a recorded entry. FC1 rules 1, 2, 5, 7 and 8 apply to M5 and M6 without change, and one run saves M1-M6 in one record.
- Rule 2. FC2 never stops FC1. If the inputs of M5 or M6 are missing, or a model fails, the record keeps M1-M4. It marks each FC2 model that failed as void, with the reason. That month is then void for that model.
- Rule 3. Inputs. All inputs come from close t: the inputs of FC1 rule 3, and the Cboe closes of VIX, VIX3M, VIX6M and VIX1Y for t. Each close comes from the newest Cboe file that holds t and that the project saved by the deadline of the entry.
- Rule 4. The model. M5 and M6 use the two-factor Heston model of definedtarget.volmodel (FINDINGS F24; docs/MODELS.md, section 1.11), with one eta and one rho for both factors. The function termstructure.states_two gives the two factor variances v1 and v2 of close t from its four index closes. It uses tenors of 30, 93, 184 and 366 days over 365, and an index noise of 0.5 vol points.
- Rule 5. The parameters of M5, fixed here. M5 uses the parameters that passed the holdout test of F24. That fit used index days from 2009-09-18 to 2019-12-31 and trades from 2024-06-14 to 2025-06-30. M5: kappa1 = 5.09994244246, theta1 = 0.0262268545928, kappa2 = 1e-06, c2 = 0.0173301546638, theta2 = 19477.9086347, eta = 1.31369997526, rho = -0.780185481629.
- Rule 6. The parameters of M6, fixed here. M6 uses the same fit on all data before this freeze. That fit used index days from 2009-09-18 to 2026-10-16 and trades from 2024-06-14 to 2026-10-20. M6: kappa1 = 5.07978559489, theta1 = 0.0303613259579, kappa2 = 1e-06, c2 = 0.0125599861037, theta2 = 73763.8148299, eta = 2.47779211628, rho = -0.831301163622. No test has judged the fit of M6 on data that it did not see.
- Rule 7. The paths. SPY and both variances move together from t to T under the pricing measure, on the real NYSE calendar. Their drift is the FC1 rate minus the dividend yield. The function volmodel.stochvol.simulate_two makes the paths by the QE scheme with the martingale correction, with the seed and the number of paths of FC1. Stream 0 drives SPY, so SPY has the draws of M1, and streams 1 and 2 drive the two factors.
- Rule 8. The value of XX. The model NAV of XX is its options per share times the model price of one option, plus cash, with the FC1 fee. The COS pricer gives that price at t, and a two-factor price table at the remaining maturity gives it at T (the defaults of pricetable.build_two). At the last origin, T is the expiry, and the option pays its intrinsic value. The growth is the model NAV at T divided by the model NAV at t.
- Rule 9. The saved forecast. For each of M5 and M6, the record keeps the growth of XX and the growth of SPY (S_T / S_t) at the 999 FC1 levels. It also keeps the variances of close t, the four index closes and the parameters.
- Rule 10. Scoring. The PIT of XX follows FC1 rule 6. The PIT of SPY is the share of the saved SPY growth below the SPY growth that occurs, by the interpolation of FC1 rule 6. SPY is not lognormal under M5 and M6, so FC2 does not use the closed form of FC1.
- Rule 11. Reports. After every 12 outcome entries and at the end, FC2 appends a report entry. It gives the statistics of FC1 rule 9 for M5 and M6 next to M1-M4. The FC1 reports stay M1-M4. Neither M5 nor M6 replaces the other because of the results: the reports show both to the end.
- Known bias 1, stated before any data: the risk-neutral drift, as in FC1.
- Known bias 2, stated before any data: the model ran low in the holdout test of F24. Its 3-year vol at the strike of XX was 0.96 points too low. The growth ratio removes the error in the level of the NAV. It does not remove the error in its sensitivity.
- Known bias 3, stated before any data: the smile needs an eta of 1.31 (M5) and 2.48 (M6), against 0.73 from the indices. If the indices are right, the months of M5 and M6 are too wide, and those of M6 more so.
- Known bias 4, stated before any data: the parameters are not stable. The fit on more data almost doubles eta and almost quadruples the slow pricing drift (0.0195 to 0.0738). Neither set is the truth.

## Hypotheses

| ID | Frozen (UTC) | Claim | Review on | Status |
|---|---|---|---|---|
| H2 | 2026-10-14 | Options per share fall about 1% a year, through whole-contract sales that fund the fee. | 2027-10-14 | open |
| H3 | 2026-10-14 | The cash buffer stays near 1% of net assets. | 2027-04-14 | open |
| H4 | 2026-10-14 | The vendor passes part of the change in one-year index volatility into XX's mark. | 2027-01-14 | open |
| H5 | 2026-10-14 | The vendor treats SPY's dividends as discrete payments. | 2026-12-21 | open |
| H6 | 2026-10-14 | XX trades close to its NAV. | 2027-04-14 | open |
| H7 | 2026-10-14 | XX's effective S&P 500 finish line falls between 9,968 and 10,002. | 2030-01-10 | open |
| H7a | 2026-10-14 | Each year until expiry, the January S&P/SPY ratio drifts as it did from 2011 to 2026. | 2027-01-20, 2028-01-20, 2029-01-19 | open |
| H8 | 2026-10-14 | Day to day, XX's real leverage exceeds the Black-Scholes number: NAV moves more per 1% of SPY than the model elasticity says. | 2027-01-14 | open |
| H10 | 2026-10-14 | When SPY rises, the vendor raises XX's implied vol: the mark slides along the volatility skew toward at-the-money levels. | 2027-01-14 | open |
| H11 | 2026-10-14 | XX's implied vol stays near 0.77 times one-year index volatility, the band the history study uses. | 2027-04-14 | open |
| H12 | 2026-10-14 | The pricing vendor discounts at a steady spread above Treasuries. | 2027-04-14 | open |
| H13 | 2026-10-14 | The vendor marks XX's option at the low end of the listed S&P 500 option market: near the bid, below the mid. | 2027-04-14 | open |
| H14 | 2026-10-14 | XX's extra day-to-day leverage (H8) fades within two weeks: over 10-session spans, NAV moves about as much per 1% of SPY as the model elasticity says. | 2027-04-14 | open |
| H15 | 2026-10-14 | XX gets no cushion on the market's bad days: when SPY falls 1% or more, NAV falls at least as much as the model elasticity says. | 2027-10-14 | open |
| H16 | 2026-10-14 | The exchange price carries the leverage that NAV shows. | 2027-04-14 | open |
| H17 | 2026-10-14 | The listed market does not share the vendor's slide: the listed Dec-2029 10,000 call's implied vol holds steady or falls when the S&P 500 rises, so XX's extra day-to-day leverage comes from how the vendor marks it. | 2027-04-14 | open |
| H18 | 2026-10-14 | The vendor passes daily interest-rate moves into XX's mark, so NAV carries the rate bet the model shows (about +0.15% of NAV per basis point of 3-year rates, with SPY and vol unchanged). | 2027-04-14 | open |
| H19 | 2026-10-14 | The listed market gives better odds of finishing above 10,000 than the report's simple model: the volatility skew moves probability toward modest wins. | 2027-04-14 | open |
| H1 | 2026-10-14 | Creation days inflate XX's mark, and it falls back the next day, so existing holders share the creation cost. | after the 15th creation event from 2026-10-14, or 2027-10-14 with 'inconclusive' if fewer | open |
| H9 | 2026-10-14 | Heavy buying of XX raises the price of the calls it buys, relative to the listed market. | after the 15th creation event with chain data, or 2027-10-14 with 'inconclusive' if fewer | open |
| H20 | 2026-10-21 | At three years, the listed S&P 500 options move sticky-strike: when the index moves, the at-the-money-forward vol of the Dec-2029 expiry changes by about one times its skew (a skew-stickiness ratio near 1, as stochastic-volatility models predict), not two or more (as local-volatility models predict). | 2027-04-15 | open |
| H21 | 2026-10-21 | Between one and three years, the listed S&P 500 at-the-money skew falls with maturity no slower than the square-root law and no faster than one over maturity. | 2027-04-15 | open |

### H2: Options per share fall about 1% a year, through whole-contract sales that fund the fee.

- **Test:** Contracts per share from the issuer's holdings, 2026-10-14 to 2027-10-14. On clean days contracts per share must not rise. On flow days, any change must come from rounding the contracts bought or sold to whole contracts (at most 1 contract per creation unit).
- **Pass if:** No unexplained rise, and the total fall from 0.0048 (the 2026-10-13 value) to the last close before 2027-10-14 lies between 0.5% and 1.5%.
- **If it fails:** A fall near 0% with cash draining: the fee consumes the cash floor (H3 fails too). A fall above 1.5%: costs beyond the 0.99% fee. Any rise: the fund buys extra options.
- **Before the freeze (shaped the band; tests nothing):** One contract equals 0.34% of assets on 2026-10-13, so the fee equals about 2.9 contracts a year; no change in the first 10 closes.
- **Review on:** 2027-10-14

### H3: The cash buffer stays near 1% of net assets.

- **Test:** Cash (money market + Cash & Other) / NetAssets at each close, 2026-10-14 to 2027-04-14.
- **Pass if:** Between 0.5% and 1.5% on at least 90% of closes.
- **If it fails:** Below: the fund spends its floor on fees. Above: it holds more cash than it needs, which lowers the upside per dollar.
- **Before the freeze (shaped the band; tests nothing):** 0.90%-1.02% on the first 10 closes; one fee sale of one contract would add about 0.34%.
- **Review on:** 2027-04-14

### H4: The vendor passes part of the change in one-year index volatility into XX's mark.

- **Test:** OLS on clean days, 2026-10-14 to 2027-01-14 (at least 40 days): daily change in base-case implied vol (points) on the daily change in Cboe VIX1Y (points) and the daily SPY % change, with an intercept.
- **Pass if:** The VIX1Y coefficient lies between 0.1 and 0.9; inconclusive below 40 clean days.
- **If it fails:** Below 0.1: the vendor's long-dated vol ignores market vol (stale or smoothed marks). Above 0.9: XX's 3-year vol moves as much as one-year vol, unusual for a long-dated option.
- **Before the freeze (shaped the band; tests nothing):** Only 3 clean days so far; the one-regressor slope reads -1.0 because SPY's moves dominate, which is why the test controls for them.
- **Review on:** 2027-01-14

### H5: The vendor treats SPY's dividends as discrete payments.

- **Test:** For every deep in-the-money SPY call in the issuer's holdings file (strike below $100): the change from the 2026-12-16 close to the 2026-12-18 close (SPY's ex-date) minus SPY's change over the same two days, divided by SPY's declared dividend. The two-day window covers either valuation-date convention.
- **Pass if:** The median ratio lies between 0.6 and 1.4 (discrete pricing predicts about 1.0; a continuous yield predicts about 0). If the file holds fewer than 3 such calls, the test moves to the next SPY ex-date.
- **If it fails:** Near 0: the vendor uses a continuous dividend yield, so XX's mark drops by about delta x dividend on each ex-date, and the report's theta convention (FINDINGS F8) should switch.
- **Before the freeze (shaped the band; tests nothing):** Daily noise of these marks (change minus SPY's change): about $0.30 against a dividend of about $1.9, a 6-to-1 signal.
- **Review on:** 2026-12-21

### H6: XX trades close to its NAV.

- **Test:** The issuer's closing market price / NAV - 1, 2026-10-14 to 2027-04-14.
- **Pass if:** Median absolute premium/discount at most 0.5%, and absolute value below 1% on at least 85% of closes.
- **If it fails:** Buyers and sellers pay a hidden cost on top of the bid/ask spread.
- **Before the freeze (shaped the band; tests nothing):** Median 0.35%; within 1% on 9 of 10 closes.
- **Review on:** 2027-04-14

### H7: XX's effective S&P 500 finish line falls between 9,968 and 10,002.

- **Test:** 995.01 x (S&P 500 close / SPY close) on 2030-01-10, using the issuer's reference closes.
- **Pass if:** The result lies between 9,968 and 10,002 (FINDINGS F7, sample from 2005).
- **If it fails:** SPY's price drifted against the S&P 500 more than at any time since 2005; the strike's real index level moved.
- **Before the freeze (shaped the band; tests nothing):** Central estimate about 9,995; 2011-2026 4-year changes give 9,989-10,001.
- **Review on:** 2030-01-10

### H7a: Each year until expiry, the January S&P/SPY ratio drifts as it did from 2011 to 2026.

- **Test:** The 7-session median S&P/SPY ratio nearest January 10 in 2027, 2028 and 2029 (same method as FINDINGS F7), against the previous January (2026: 10.03693).
- **Pass if:** Each yearly change lies between -0.02% and +0.08% (the 2011-2026 range of -0.012% to +0.072%, rounded out). One verdict per checkpoint; the hypothesis passes if all three pass.
- **If it fails:** The calm drift regime since 2011 broke; the H7 band and the report's finish-line ratio need updating.
- **Before the freeze (shaped the band; tests nothing):** 1-year changes from 2011: -0.012% to +0.072%, median +0.012%; from 2005 the range widens to -0.25% to +0.14%.
- **Review on:** 2027-01-20, 2028-01-20, 2029-01-19

### H8: Day to day, XX's real leverage exceeds the Black-Scholes number: NAV moves more per 1% of SPY than the model elasticity says.

- **Test:** Clean days, 2026-10-14 to 2027-01-14 (at least 40 days): OLS through the origin of the daily NAV % change on the daily SPY % change, divided by the mean model elasticity over the same days.
- **Pass if:** The ratio lies between 1.05 and 1.60; inconclusive below 40 clean days.
- **If it fails:** 0.70-1.05: the vendor's marks keep a fixed-strike vol and the model's leverage holds. Below 0.70: vol falls when SPY rises, as in the listed index-option market, and XX's real leverage sits below the model's.
- **Before the freeze (shaped the band; tests nothing):** 1.26 on all 9 daily changes; 1.15 on the 3 clean days.
- **Review on:** 2027-01-14

### H10: When SPY rises, the vendor raises XX's implied vol: the mark slides along the volatility skew toward at-the-money levels.

- **Test:** The H4 regression: the coefficient on the daily SPY % change (implied-vol points per 1% SPY move).
- **Pass if:** Between +0.05 and +0.35; inconclusive below 40 clean days.
- **If it fails:** Near 0: fixed-strike marking. Negative: index-market behavior, with vol falling on rallies; H8 then fails as well.
- **Before the freeze (shaped the band; tests nothing):** +0.13 to +0.18 points per 1% (all days and clean days).
- **Review on:** 2027-01-14

### H11: XX's implied vol stays near 0.77 times one-year index volatility, the band the history study uses.

- **Test:** Base-case implied vol / Cboe VIX1Y close at each close, 2026-10-14 to 2027-04-14.
- **Pass if:** Between 0.72 and 0.82 on at least 80% of closes.
- **If it fails:** The backtest's volatility proxy (0.77 +/- 0.05 x one-year volatility) needs a wider band or a moneyness adjustment (see H10).
- **Before the freeze (shaped the band; tests nothing):** 0.753-0.782 on the first 10 closes.
- **Review on:** 2027-04-14

### H12: The pricing vendor discounts at a steady spread above Treasuries.

- **Test:** On closes when the issuer's file holds a 4-leg SPY box: the box-implied rate (valued at the file's effective date) minus the Treasury zero rate for the same term, 2026-10-14 to 2027-04-14.
- **Pass if:** Between +25 and +75 bp on at least 90% of such closes (at least 40); otherwise inconclusive.
- **If it fails:** The financing spread inside XX's implied vol moves around; the rate sensitivity in the report (ASSUMPTIONS A4) needs a wider range.
- **Before the freeze (shaped the band; tests nothing):** +47 to +50 bp on the first 10 closes.
- **Review on:** 2027-04-14

### H13: The vendor marks XX's option at the low end of the listed S&P 500 option market: near the bid, below the mid.

- **Test:** On closes with a captured quote for the listed SPX Dec-2029 10000 call, 2026-11-01 to 2027-04-14: XX's base-case implied vol against the implied vols of that call's bid, mid-quote and ask (conventions in P1).
- **Pass if:** XX's implied vol lies between 0.5 points below the bid's implied vol and the mid-quote's implied vol on at least 80% of closes (at least 60 closes); otherwise inconclusive.
- **If it fails:** Above the mid: NAV values the options above where the listed market would buy them. More than 0.5 points below the bid: NAV values them below what the market would pay, so buyers at NAV get a discount, and cash creations, which buy at market prices, cost existing holders.
- **Before the freeze (shaped the band; tests nothing):** Delayed quotes of the 2026-10-13 close: XX 16.61% against the listed call's bid / mid / ask of 16.65% / 17.26% / 17.87%. The conventions (S&P dividends from SPY's, the three-week expiry gap) can move the listed vols by a few tenths of a point, so 'inside the bid-ask range' would have been a coin flip; the band allows for that. Requires adding a daily chain capture.
- **Review on:** 2027-04-14

### H14: XX's extra day-to-day leverage (H8) fades within two weeks: over 10-session spans, NAV moves about as much per 1% of SPY as the model elasticity says.

- **Test:** Consecutive non-overlapping windows of 10 sessions from the first clean close on or after 2026-10-14 to 2027-04-14. Each window starts on a clean close; if its 10th session is not a clean close, the window ends at the next clean close. OLS, with an intercept, of each window's NAV % change on (the model elasticity at the window's first close x the window's SPY % change). The slope is the two-week leverage ratio.
- **Pass if:** The slope lies between 0.85 and 1.10; inconclusive below 8 windows.
- **If it fails:** Above 1.10: the vendor's slide persists, so holders keep the extra leverage, up and down. Below 0.85: over weeks the vendor lowers XX's vol in rallies, as index option markets usually do, so holders get less leverage than the model says.
- **Before the freeze (shaped the band; tests nothing):** Day to day, NAV moved 1.15-1.25 times the elasticity, and the measured slide accounts for it (1 + vega x slide / elasticity = 1.23-1.31). From 2026-09-30 to 2026-10-13, SPY rose 2.02% while implied vol rose 0.09 points, against 0.37 for a slide that lasts.
- **Review on:** 2027-04-14

### H15: XX gets no cushion on the market's bad days: when SPY falls 1% or more, NAV falls at least as much as the model elasticity says.

- **Test:** Clean days from 2026-10-14 to 2027-10-14 with an SPY change of -1.0% or lower. For each day: NAV % change / (model elasticity at the previous close x SPY % change). The median across those days.
- **Pass if:** The median lies between 1.00 and 1.60; inconclusive below 8 such days.
- **If it fails:** Below 1.00: the vendor lifts XX's vol when the market drops, which softens losses (a cushion the model leaves out). Above 1.60: losses run far beyond the leverage figure.
- **Before the freeze (shaped the band; tests nothing):** No close in the first 10 fell 1% or more (largest fall -0.42%). The day-to-day slide (vol falls as SPY falls, H10) predicts about 1.2.
- **Review on:** 2027-10-14

### H16: The exchange price carries the leverage that NAV shows.

- **Test:** Clean days from 2026-10-14 to 2027-04-14 (at least 40): OLS through the origin of the daily % change in the issuer's market price (the primary exchange's official closing price) on the daily SPY % change, divided by the same coefficient for NAV.
- **Pass if:** The ratio lies between 0.90 and 1.10; inconclusive below 40 clean days.
- **If it fails:** Above 1.10: the premium rises when SPY rises and falls when SPY falls; buyers chase rallies (or the vendor's marks lag), so momentum traders pay on the way in and on the way out. Below 0.90: market makers price XX with less leverage than the vendor's NAV, so the extra NAV leverage does not reach people who trade the shares.
- **Before the freeze (shaped the band; tests nothing):** 1.15 on all 9 daily changes and 1.18 on the 3 clean days; one close (2026-10-09: NAV +3.1%, market price +4.7%) drives most of it.
- **Review on:** 2027-04-14

### H17: The listed market does not share the vendor's slide: the listed Dec-2029 10,000 call's implied vol holds steady or falls when the S&P 500 rises, so XX's extra day-to-day leverage comes from how the vendor marks it.

- **Test:** Closes with a captured quote, 2026-11-01 to 2027-04-14 (at least 60): OLS, with an intercept, of the daily change in the listed SPX Dec-2029 10000 call's mid-quote implied vol (points) on the daily S&P 500 % change (conventions in P1). The verdict also reports XX's H10 coefficient on the same closes.
- **Pass if:** The coefficient lies between -0.20 and +0.05 points per 1%; inconclusive below 60 closes.
- **If it fails:** Above +0.05: the listed market slides like the vendor (a market that keeps vol fixed by moneyness would show about +0.165), so XX's extra leverage is real market behavior. Below -0.20: listed vol falls fast in rallies, and the gap to the vendor widens.
- **Before the freeze (shaped the band; tests nothing):** Delayed quotes of the 2026-10-13 close: the listed skew near 10,000 was -0.165 vol per unit of log-strike, close to what XX's measured slide implies (+0.13 to +0.18 points per 1%). Published studies find that index option markets lower fixed-strike vol in rallies (Bergomi 2016; Hull and White 2017; not verified here), which predicts about -0.08.
- **Review on:** 2027-04-14

### H18: The vendor passes daily interest-rate moves into XX's mark, so NAV carries the rate bet the model shows (about +0.15% of NAV per basis point of 3-year rates, with SPY and vol unchanged).

- **Test:** Clean days from 2026-10-14 to 2027-04-14 (at least 40): OLS, with an intercept, of the daily change in base-case implied vol (points) on the daily SPY % change, the daily change in Cboe VIX1Y (points) and the daily change in the Treasury zero rate to the option's expiry (basis points). The rate coefficient.
- **Pass if:** The rate coefficient lies between -0.008 and +0.008 implied-vol points per basis point; inconclusive below 40 clean days.
- **If it fails:** Below -0.008: the vendor holds its rate fixed or updates it rarely (a rate that never moves gives -0.0156); NAV ignores rate moves the model says it carries, our implied vol absorbs them, and the report's rate line overstates daily rate risk. Above +0.008: the vendor's rate moves more than Treasuries.
- **Before the freeze (shaped the band; tests nothing):** Nine daily changes cannot tell: the coefficient reads -0.013 with each day's curve, +0.008 with the previous session's curve and +0.003 with a fixed curve, and the implied-vol series looks equally smooth under all three. Daily rate changes: sd 4.7 bp.
- **Review on:** 2027-04-14

### H19: The listed market gives better odds of finishing above 10,000 than the report's simple model: the volatility skew moves probability toward modest wins.

- **Test:** Closes with captured quotes, 2026-11-01 to 2027-04-14 (at least 60): the listed call-spread probability of finishing above 10,000, (mid(K1) - mid(K2)) / ((K2 - K1) x discount factor), with K1 = 9,600 and K2 = 10,400 (the nearest listed strikes if those lapse), minus XX's base-case risk-neutral probability of finishing above the strike on the same close.
- **Pass if:** The gap lies between +5 and +15 percentage points on at least 80% of closes; inconclusive below 60 closes.
- **If it fails:** Below +5: the skew near 10,000 flattened, and the simple model's odds suffice. Above +15: the skew steepened, and the simple model understates the odds further.
- **Before the freeze (shaped the band; tests nothing):** 2026-10-13: 39.0% from call spreads against 28.5% from the simple model at the listed expiry, a gap of +10.5 points. The market's odds exceed the simple model's up to about a 3x outcome and fall below them beyond (10x: about 1% against 1.8%, extrapolated past the last listed strike).
- **Review on:** 2027-04-14

### H1: Creation days inflate XX's mark, and it falls back the next day, so existing holders share the creation cost.

- **Test:** The change in base-case implied vol on the day after each creation day (shares outstanding rose), against the same change on clean days, from 2026-10-14.
- **Pass if:** The mean day-after-creation change sits more than 0.10 points below the clean-day mean, one-sided Welch t-test p < 0.05.
- **If it fails:** Creation costs do not reach existing holders through the mark (the issuer's creation charges may cover them).
- **Before the freeze (shaped the band; tests nothing):** Two events before the freeze: -0.14 and -0.24 points, against a clean-day sd of 0.10.
- **Review on:** after the 15th creation event from 2026-10-14, or 2027-10-14 with 'inconclusive' if fewer

### H9: Heavy buying of XX raises the price of the calls it buys, relative to the listed market.

- **Test:** XX's base-case implied vol minus the listed SPX Dec-2029 10000 call's mid-quote implied vol, on creation days against clean days, from 2026-11-01.
- **Pass if:** The creation-day gap exceeds the clean-day gap by more than 0.10 points, one-sided Welch t-test p < 0.05.
- **If it fails:** XX's flows stay too small to move the price of long-dated calls.
- **Before the freeze (shaped the band; tests nothing):** No data yet; flows remain small (assets about $1.6 million).
- **Review on:** after the 15th creation event with chain data, or 2027-10-14 with 'inconclusive' if fewer

### H20: At three years, the listed S&P 500 options move sticky-strike: when the index moves, the at-the-money-forward vol of the Dec-2029 expiry changes by about one times its skew (a skew-stickiness ratio near 1, as stochastic-volatility models predict), not two or more (as local-volatility models predict).

- **Test:** Closes with counted Dec-2029 quotes, 2026-11-02 to 2027-04-14 (at least 60). OLS, with an intercept, of the daily change in the Dec-2029 ATMF vol on the daily change in ln(S&P 500 close); the skew-stickiness ratio is the slope divided by the mean ATMF skew over the same closes. Reported with its standard error.
- **Pass if:** The ratio lies between 0.5 and 1.5; inconclusive below 60 closes.
- **If it fails:** Above 1.5, the 3-year vols move more with the index than stochastic-volatility dynamics allow (local-volatility-like); the v0.3.1 model needs that dynamic and the simulator's slide is closer to right. Below 0.5, the smile moves with the index (sticky delta): the fixed-strike vol of XX's option moves more than the model assumes.
- **Before the freeze (shaped the band; tests nothing):** Bergomi, *Stochastic Volatility Modeling* (2016), ch. 2, and "Smile Dynamics IV" (Risk, 2009): sticky strike is a ratio of 1, sticky delta 0, local volatility 2 or more; sticky strike appears as the long-maturity limit of stochastic-volatility models. F12's fixed-strike slide (+1.1 points per -10%, against +4.6 for one-year index vol) is consistent with a ratio near 1 but does not measure it.
- **Review on:** 2027-04-15

### H21: Between one and three years, the listed S&P 500 at-the-money skew falls with maturity no slower than the square-root law and no faster than one over maturity.

- **Test:** Closes with counted quotes for all three expiries Dec-2027, Dec-2028 and Dec-2029, 2026-11-02 to 2027-04-14 (at least 40). On each close, OLS of ln|S_T| on ln(T) across the three expiries (T in years to each expiry); the decay exponent is minus the slope. The median exponent across closes, with its bootstrap interval.
- **Pass if:** The median exponent lies between 0.4 and 1.1; inconclusive below 40 closes.
- **If it fails:** Below 0.4, long-dated skew decays more slowly than typical equity smiles: a two-factor or rough-volatility model is needed even at these maturities. Above 1.1, faster than any standard model predicts: check the data before concluding anything.
- **Before the freeze (shaped the band; tests nothing):** Gatheral, *The Volatility Surface* (2006), ch. 3: in Heston the long-dated variance skew falls like 1/T (an exponent of 1 for the vol skew only at maturities long against the mean-reversion time; between 1 and 3 years Heston sits in transition, below 1). Bergomi (2016), ch. 2: typical equity smiles decay like T^-1/2. The band covers both; the measured exponent then decides how much the two-factor model is needed. Requires a new daily capture of the Dec-2027 and Dec-2028 strips (approved 2026-10-21), started after this freeze.
- **Review on:** 2027-04-15
