# Changelog

## Unreleased (v0.3 in progress)

The `definedtarget.montecarlo` interface is provisional until v0.3.3: names and arguments may change.

### Changed after the independent pre-release audit
The audit found no defect in the engine's results, the XX layer or the forecast check's code, and reproduced every
published table value in FINDINGS F15, F17, F18 and F19. Fixed from its findings:
- The README's speed line credited the SPY-path kernels' times to "XX paths"; the XX layer, which reprices the fund
  every session, runs on the CPU only, at about 10 s per million paths.
- Run manifests fingerprint only the code a run loaded (plus the script), so a new, unrelated file no longer makes a
  published manifest fail `same_output`; the published manifests were regenerated.
- Recorded sources for quoted numbers: `xx_paths_study.py` prints F17's martingale gaps (updated to its 1-million-path
  output), `bootstrap_study.py` prints F15's two volatilities, and `scripts/billion_paths_demo.py` (seed 777) produces
  the billion-path figures (updated). F15's headline now names post-1950 S&P history (history since 1926 gives +6% to
  -14%); F17's touch change reads 2.7-3.8 points.
- The engine refuses NaN or infinite vol, drift and times, a constant history in `equal_moments`, and fractional
  seeds or path counts, instead of returning NaN or truncating.
- `causal.check` also tries a future with no rows and one with twice the rows, so a rule that reads the series
  length is caught.
- The forecast check: a PIT exactly at the first or last saved point interpolates, as FC1 rule 6 says (it returned
  0.0005 / 0.9995); the saved forecast also records the library versions and the SPY close's source; its test
  judges the forecasts with an independent scipy formula.
- Public code no longer cites private planning files.

### Added
- **The monthly forecast check** (protocol FC1, frozen 2026-10-17 UTC before its first origin, 2026-10-30):
  at each month's last close the engine forecasts XX's growth to the next month's last close under four models
  (M1 the simulator's slide rule, M2 no slide, M3 the slide measured on VIX1Y paths, M4 VIX as the month's path
  vol), saves the forecast privately and logs its SHA-256 in `forward/forecast_log.jsonl`; a month later it logs
  each model's PIT for XX and for SPY. Log only, no verdict; 39 months to expiry. `definedtarget/forward/forecast.py`,
  `scripts/forecast_check.py`; the daily routine notes a due entry and stops on a late one. Tests judge every model
  against an exact answer (XX's NAV at the target depends only on SPY's level there): worst gap 0.0019 at 200,000
  paths.
- **No look-ahead** (`montecarlo.causal`, tests/test_lookahead.py): `causal.check(rule, series)` replaces the
  future after each cut point with six different futures (reversed, shuffled, flat, a crash, a spike, noise) and
  requires every earlier output to stay identical, bit for bit; series up to 1,000 rows are cut at every row. It
  passes causal rules and catches a centered average, tomorrow's value, a full-sample z-score, a one-row peek at
  the end, a peek on month-ends only (at exactly those 28 cuts) and a rule that peeks only when a crash is coming.
  It proves that a rule does not read the future; it does not check execution timing (a decision at a close may
  only earn later returns), parameters fitted on the whole history, or data revised or published late. Every path
  engine (lognormal and bootstrap, CPU and GPU, and the XX layer) reports at session k exactly what
  a run stopped at k reports, bit for bit; the bootstrap's row choices up to step k do not depend on the path's
  length; and FC1's inputs for an origin ignore every later row. Later strategy rules (v0.3.2) must pass the same
  checker.
- **Run manifests** (`montecarlo.manifest`, standard library only): every study script (bootstrap, XX paths, slide
  rule, effective leverage, and the two GPU speed tests) writes `results/manifests/<study>_<date>_<output>.json`
  with its arguments, seeds, settings, the SHA-256 and size of each input file, the SHA-256 of every code file, the
  library versions and the SHA-256 of everything it printed. A seeded study run again on the same inputs and code
  must print the same output (`manifest.same_output`); the four seeded studies reproduce FINDINGS F15, F17, F18 and
  F19, and the repository publishes their manifests. File names only, the date only, no folders or machine names.
- The identifier scan now catches a Windows home path written with doubled backslashes (as JSON writes it), accepts
  the forecast log and run manifests (it still scans their contents), and accepts "analyses" (the American plural
  of "analysis"). `slide_rule_test.py` prints the VIX1Y file's fetch date, not its time.
- **`definedtarget.montecarlo`**, the start of a Monte Carlo engine that imports nothing else from the package, so
  other projects can use it unchanged. Step 1: counter-based random numbers (Philox4x32-10) addressed by (seed, path,
  draw, stream), 52-bit uniforms that are never 0 or 1, and Wichura's AS241 inverse normal. Results do not depend on
  the number of threads, the chunk size or the device: the optional GPU twin (`montecarlo.gpu`, needs CuPy) returns
  the same 32-bit words and uniforms bit for bit, and the same normals except about 1 in 40,000 tail draws, whose
  last digit can differ because NVIDIA's log rounds differently (405 of 16.28 million on the test machine).
- Tests against Random123's published known answers (7 and 10 rounds), scipy's `ndtri` (to 1e-14 relative from
  p = 1e-300), thread and chunk invariance, and loose statistical checks.
- `scripts/b17a_gpu_first_test.py` and FINDINGS F14: a first GPU speed test.
- **Honest error bars** (`montecarlo.stats`, step 2): mergeable moments (chunks, threads and devices combine
  exactly), Wilson intervals for probabilities, distribution-free percentile intervals. A coverage test runs 1,000
  independent runs against exact lognormal answers: the 95% intervals contained the truth 94.0% (price), 94.2%
  (P(S_T > K)) and 95.4-95.8% (percentiles) of the time, inside the 93.2-96.8% band fixed in advance. For a rare
  event with about 2 hits per run, the textbook (Wald) interval covered only 87.7%; Wilson's covered 94.8%.
- **Lognormal paths** (`montecarlo.paths`, step 3) on any time grid, including the real trading calendar
  (`times_from_dates`). Each path reports, in one causal pass: the end level, the highest close, the drawdown at the
  end, the maximum drawdown, the first close at a barrier, the exact chance that the continuous path touched the
  barrier between closes (Brownian bridge), and levels at checkpoints. Exact formulas judge it, all within 3 SE:
  the end distribution on 814 unequal calendar steps, the martingale property at 11 checkpoints, Spitzer's identity
  for the expected highest close and (by time reversal) the drawdown at the end, and the reflection principle for a
  continuous touch (+28% before expiry: exact 0.54079). Two approximations sit within allowances fixed in advance:
  Broadie-Glasserman-Kou for touches checked at closes, and Magdon-Ismail's zero-drift expected maximum drawdown
  (extrapolated 0.25090 against 0.25066).
- **Fixed-memory results** (`montecarlo.stream`, step 4): results from any number of paths, chunk by chunk.
  Percentiles come from a 65,536-bin histogram plus one replay of the same paths (the random numbers are addressed,
  so a replay repeats them exactly); the replay keeps only the values in the needed bin and returns the same order
  statistics as a full sort, bit for bit (tested on 3 million values; it kept 7 to 1,248 of them). `run_to_precision`
  runs until a target standard error is met or a path limit is reached, and reports which. On the test machine's CPU
  (`scripts/billion_paths_demo.py`, seed 777): the option value to SE 0.004 took 1.10 billion paths in 29 s (54.67960,
  SE 0.00383, against Black-76 54.67744), and the exact median of 1 billion end prices took 57 s while keeping
  191,183 values (843.67031 against the exact 843.66528).
- **History bootstrap** (`montecarlo.bootstrap`, step 5): the circular stationary bootstrap (Politis and Romano,
  1994) builds paths from blocks of real history with random lengths of a chosen mean; several columns travel
  together; `equal_moments` gives history any chosen mean and spread, so a bootstrap and a lognormal with the same
  inputs differ only in shape. Tests on synthetic history with fat tails and strong serial correlation check: every row
  equally likely at every step, geometric block starts, the exact mean, and the exact variance of a path's sum
  (errors of 1.09, -0.92 and -0.12 standard errors for mean blocks 1, 5 and 40).
- **History study** (`scripts/bootstrap_study.py`, step 6; FINDINGS F15) on the S&P 500 since 1950 and the US
  market since 1926: generators built from 1950-1999 understated 2000-2026's drawdowns (37-39% of windows above their
  90th percentile; that era ran at 19.2% volatility against 13.6%), and at the same daily volatility, post-1950 S&P
  history makes XX's option worth 13-24% less than the lognormal model, mostly through a thinner 3-year upper tail.
- **GPU path kernels** (`montecarlo.gpu.lognormal`, `.bootstrap`): the same inputs, measures and operations as the CPU
  kernels; the bootstrap matches the CPU bit for bit, the lognormal to about 1e-16. On the test machine they run
  2.4x and 16x faster than 24 CPU threads. `scripts/gpu_stress_test.py` runs them at full load for a set time with
  temperature logging, a temperature stop and CPU spot checks (FINDINGS F16: 5 minutes, 0.75 trillion path-steps,
  66 C peak, 0 mismatches).
- **XX along every path** (`definedtarget/xxpaths.py`, the XX layer on top of the engine): the fund repriced each
  session with the simulator's model (Black-76, fee, cash, the slide rule), with NAV drawdowns, time underwater,
  first touches of 2x / 5x / 10x and XX against SPY. Tests judge it against sim.share_value, the payoff, a
  plain-Python recomputation of whole paths, and the martingale identity. `scripts/xx_paths_study.py` and FINDINGS F17: at the
  risk-neutral drift XX touches 2x in 49% of futures but ends above its start in 23%; the slide rule pulls the
  expected mid-life mark about 5% below today's price, an open question for the vol model.
- **Slide-rule test on real history** (`scripts/slide_rule_test.py`, FINDINGS F18): 198 synthetic XXs on real S&P
  paths since 2007, marked with constant vol, the slide rule and real VIX1Y. VIX1Y moves 3.5 times as much as the rule;
  the vol assumption moves XX's drawdowns and peaks by several points, so path statistics carry a model-risk band.
- **Effective leverage** (`xxpaths.effective_leverage`): the fund's realized move per 1% ETF move on up days and on
  down days, against its elasticity at the previous close. `scripts/effective_leverage_study.py` and FINDINGS F19:
  with vol moving against the index, XX, the call inside it and the listed S&P 10,000 call lose the same leverage on
  both sides and show the same mid-life value gap, so the effect is the option's, not the fund's.

## v0.2.3 (2026-10-17)

Fixes from an independent external audit of v0.2.2, checked by a second independent audit before release. The
external audit reproduced each defect with synthetic data; its probes now run as regression tests, except one
documented reading (price sources) and one design point left for a later release (each test loads every input). No
frozen claim, band, window or minimum sample changed.

### Fixed
- **H1 and H9** count only creation events before 2027-10-14 and never review after that date (P2: they never
  extend). A void event before the deadline now gives "void" instead of "inconclusive".
- **H5** reads "no data" when the dividend file has no December 2026 ex-date. It no longer moves on to a later
  ex-date; only fewer than 3 qualifying calls moves the test.
- **H7** reads one fixed cutoff, the end of its review date (2030-01-10, UTC), so its result cannot change with the
  run date. Open point: the issuer publishes its reference close the next day, so as P2 stands only an official
  close added by the end of 2030-01-10 can decide H7, while H7's test names the issuer's close. A clarification is
  planned for protocol P3.
- **H19** applies the listed-quote rule to its call-spread legs: a quote whose implied vol does not solve (for example
  a price above the no-arbitrage bound, or a zero bid) counts as missing, and so does a spread probability outside
  0-1. It moves to other strikes only when 9,600 or 10,400 is no longer quoted at all.
- **Hand-maintained inputs** (`closes.csv`, `tv_closes.csv`, `spy_dividends.csv`, `void_events.csv`) are saved as
  dated copies on each capture run, and a review reads the copy that existed by its cutoff. A later edit no longer
  changes an earlier review. H5's December dividend must therefore be imported by 2026-12-21 23:59:59 UTC; from
  2026-12-01 the status line says so.
- **Recording** waits while the data behind a fund-data verdict hold an unexplained accounting-identity failure, or
  while a void event dated before the review is known but was missing from the data captured by then (P1 voids by
  the event's date).
- **Regressions** with a regressor that never moves raise an error instead of returning a meaningless coefficient.
- **Captures** never overwrite raw bytes: copies saved in the same second each get their own file. The cloud
  import refuses a manifest path outside the data folder and keeps every earlier copy on a repeated name clash.
- **Simulator** (Python and browser): the value percentiles of a sold call now reach its falling tail beyond twice
  the strike. Example: a sold 100 call with 200 cash, S_T mean 180, vol 20%, one year: the 10th percentile is 72.02,
  not 100. New golden cases check this against an independent formula.
- **Packaging**: scipy is a declared dependency; openpyxl (reading saved Cboe FLEX reports) is the optional `flex`
  extra, and its tests skip without it.
- **Golden-file test**: compares numbers within each case's tolerance, so a fresh run on another platform passes.

### Documented
- H2: the frozen pass rule fails only an unexplained rise. A flow-day fall beyond rounding is judged by the
  total-fall band and is now reported for information (`flow_day_shortfalls`).
- Price sources: SPY and S&P 500 changes never use TradingView closes (P2 rule 7); the base-case implied vol may use
  one where the issuer file is missing (P1 rule 2).

## v0.2.2 (2026-10-17)

### Added
- **The verdict script** (`definedtarget/verdicts.py`, `scripts/verdicts.py`), promised by protocol P2 before the
  first review date (2026-12-21). It turns every frozen test and pass band into code: H1-H19 and H7a's checkpoints,
  P1's clean days, windows, minimum-sample extension (up to three months) and void events, P2's price, file-version
  and listed-quote rules. Before a review date it shows provisional numbers; on and after it, the verdict word. It
  appends a verdict to the ledger only when asked (`--record`), and refuses anything not final.
- **Data captured by the review date** (P2): `parse.captured_by()` makes every loader see only the files downloaded
  by a cutoff, so each review reads exactly what existed on its date.
- Tests on synthetic data with known answers: pass and fail for every band, pending, extension, inconclusive, void,
  the 15-event tests (H1, H9), H5's ex-date ratio, H7a's checkpoints, and which listed readings count under P2 and
  under a future P3.

### Changed
- VIX1Y (used by H4, H11, H18) is downloaded on every run instead of weekly.
- The listed-quote parser drops a zero bid as one-sided (it already dropped a zero ask).

### Fixed (independent audit B10c, before release)
- `--record` refuses an `--as-of` date after today, and records a verdict only after the day whose captured data
  decided it has ended (UTC). Each recorded verdict now names its data cutoff.
- H5 shows "no data" when an input is missing (dividend amount, reference close, holdings file). It never moves on
  to a later ex-date for that reason; only "fewer than 3 contracts" does.
- P2's price rule: TradingView closes (`tv_closes.csv`) no longer pass the source filter.
- H19 treats a one-day gap in the 9,600 or 10,400 quote as a missing close. It switches strikes only when a strike
  has lapsed.
- A zero bid, or a quote whose implied vol does not solve inside the solver's limits, counts as a missing close
  (H9, H13, H17); the stats report how many.
- Minimum-sample extension: a close on the last allowed day can complete the sample (verdict the next session), and
  a void event during the extension gives "void".
- H1 and H9 review only once the data captured by the review date hold 15 events. Fewer than 2 clean days, or a
  p-value that cannot be computed, gives "no data" instead of a crash or a verdict.
- H2 checks for rises on clean days only, as the frozen text says (the day after a creation may restore contracts).
- Void detection finds an exact 1-for-2 reverse split, ignores a one-day file glitch (a roll needs the new option
  line on two consecutive closes), and requires a source for every hand-listed event. Fund events no longer void
  H7a, which reads index closes only, as H7 does.
- P3 (when frozen) applies only to readings taken after its ledger entry. A crash in one review shows "error" for
  that hypothesis only.
- The conventions the frozen text leaves implicit are written in the module header.

## v0.2.1 (2026-10-17)

### Added
- **Link previews**: the simulator page carries preview tags, so a shared link shows a card (title, description
  and `docs/img/simulator-card.png`, drawn by `scripts/make_card.py` from the engine's own grid for the README example).
- **README** links to the live simulator. The identifier scan allows the simulator's own GitHub Pages address in the
  README and the page, and nowhere else (tests in `tests/test_license_guard.py`).
- Clearer page title for search results and shared links.

## v0.2.0 (2026-10-17)

The browser simulator: a no-install page that shows what one share of a defined-target call ETF is worth at any
index level on any date up to its target date.

### Added
- **Simulator model** (`definedtarget/sim.py`): value, leverage, one day's cost and odds for any fund made of option
  legs plus cash (calls or puts, bought or sold), with volatility that can follow the index. Pinned to
  `analytics.py` (odds, break-evens, values) and `pricing.implied_vol` for XX to 1e-9, and checked against a million
  simulated outcomes for puts, sold calls and strangles.
- **Browser engine** (`web/engine.js`, no libraries) and **self-test** (`web/selftest.html`): 780 answers written by
  the Python engine (`scripts/make_golden.py` -> `web/golden.js`, 101 KB) must all match in the browser; it opens
  from a double-clicked file too, and a test fails if the answers go stale.
- **The simulator page** (`web/index.html`, `web/app.js`): a date x S&P heatmap of one share's P/L, value or
  leverage, with the odds of finishing above each level; a summary (max loss, break-even, growth needed, leverage,
  one day's cost split into decay and carry); volatility shift and crash/crush presets; volatility that follows the
  S&P; the odds table with median-path growth. Inputs can arrive after a `#` in the link (never sent to a server).
  No libraries, no requests, no storage.
- **More of the simulator**: an ASCII graph mode; presets for a listed S&P 500 Dec-2029 10,000 call (per contract)
  and for any fund pasted as a short spec (calls or puts, bought or sold); the ETF and S&P fields fill each other;
  odds rows up to a 24% mean total return plus a custom row checked against history (FINDINGS F13,
  `scripts/history_windows.py`); a link at the end of the daily report opens the simulator with that close's inputs.
  Volatility is solved on the rising side of the value curve, so call spreads work. Scripts load with a version tag,
  so browsers never mix an old engine with a new page. A spec names its index (or the page infers it from SPY, QQQ,
  DIA, IWM and refuses a contradiction), can carry its own example prices, and comes with SPY call-spread, QQQ/NDX
  50,000 and DIA/DJIA 75,000 examples; for an index other than the S&P 500 the page relabels itself and greys out the
  S&P-only evidence (F10, F12, F13). An impossible price gets an
  explanation and a one-click fair price; SOLVE VOL and EXAMPLE confirm what they did.
- **docs/learn/05**: how to read the simulator.
- **Audit B10b** (independent model, full review of v0.2): its fixes are in this release: exact odds for funds with
  puts or sold calls, a link with a stray % no longer stops the page, index names containing "S&P" label correctly,
  hardened text escaping, comma checks on numbers, a pure-ASCII page, and broader self-test coverage.
- **Volatility slide measured** (FINDINGS F12, `scripts/measure_vol_slide.py`): the 3-year vol at XX's strike rises
  about 1 point per 10% S&P fall, a fifth of VIX1Y's 4.6.

## v0.1.1 (2026-10-17)

Fixes from audit B10a (an independent review of the pricing core, which it reproduced to 1e-7), plus new captures.

### Fixed
- **Theta across an SPY ex-dividend date.** The one-day step now takes a dividend going ex overnight off SPY's
  price. Before, the report for the eve of an ex-date would have shown a false gain (+1.24% of NAV for 2026-12-17,
  projected; corrected: -0.14%). The daily attribution gains a dividend line on ex-dates. (FINDINGS F8)
- **Odds rows relabeled.** "Expected total return" rows now read "mean total return", and two new columns show the
  median path's yearly growth (price only and total return). On 2026-10-13 the 10% row's median path grows 7.4% a
  year in price, below the 7.9% the strike needs. README, `docs/learn/02` and the calculator say so.

### Changed
- **Theta splits into time decay and carry**, with a second line for one day's cost if SPY rises along its forward
  price (2026-10-13: -0.13% of NAV = -0.07% decay - 0.06% carry; along the forward -0.05%).
- **Attribution in % of NAV, with a total row.** The "share of the move" column is gone: on flat days it divided
  by a number near zero.
- **Box-rate evidence flags** in the model table: `box_leg_decimals` and `box_flow_day`. (FINDINGS F9)

### Added
- **Listed-option capture** (`forward/listed.py`, `scripts/capture_listed.py`): files each TradingView reply for the
  listed S&P 500 Dec-2029 calls, checks protocol P2's read window across daylight saving and holidays, keeps
  out-of-window reads flagged, and treats one-sided quotes as missing for that strike.
- **Cboe FLEX execution reports** (`forward/cboe_flex.py`, `scripts/capture_flex.py`): downloads each trading day's
  report once and extracts the fund's own option trades, the raw material for measuring what creations cost.
- **Cloud capture import** (`forward/cloud_import.py`, `scripts/import_captures.py`): copies files from a private
  capture repository, refuses any file whose fingerprint no longer matches, and never overwrites a local copy. File
  versions now sort by download time.

## v0.1.0 (2026-10-16)

The first release covers one fund, Roundhill XX (S&P 500® Target 10,000 2030 ETF).

### Added
- **Pricing core** (`definedtarget`): Black-76 on an escrowed discrete-dividend forward, the US Treasury par curve
  bootstrapped to zero rates, Greeks per fund share, and an implied-volatility solver (Newton with a bisection
  bracket). numba prices 1 million options in 0.033 s on one CPU core. Audit A1 reproduced the core independently.
- **Fund spec** (`definedtarget/funds/xx.json`): every field carries a value, a status (VERIFIED, MEASURED,
  PROBABLE, ASSUMED or OPEN) and a source.
- **Daily tracker and forward test**: downloads the issuer's public files, keeps every version unchanged with a
  SHA-256 manifest, rebuilds NAV from the holdings to the cent, stops on any broken identity or stale input, and
  writes a one-page report (implied vol, leverage, Greeks, break-evens, odds, a daily move split into causes).
- **Calculator**: `py -m definedtarget calc` computes break-evens, a payoff table and odds without downloaded data.
- **Pre-registered predictions**: 20 testable claims about XX's behavior and their protocol, frozen on 2026-10-14
  in a hash-chained ledger (`forward/HYPOTHESES.md`). Only verdicts get appended.
- **Findings** F1-F11, including the odds that listed S&P 500 option prices imply (F10).
- **Explainers**: `docs/learn/` 01-04.
- **License guard**: a test fails if any publishable file holds a data file, an image, a daily series or a personal
  identifier; the scanner matches `.gitignore` the way git does.
- **Freeze proof**: an OpenTimestamps proof of the frozen ledger, `.gitattributes` that keeps its bytes, a test that
  pins them, and "Check the freeze" steps in the README. Protocol P2 clarifies verdict rules without changing any band.
- **Call-spread odds**: `analytics.call_spread_prob` and `prob_above_from_spreads`, tested on synthetic quotes.
- **Your own closes**: the tracker reads `data/external/closes.csv`, prices every close it can and lists the rest.
- **Citation**: `CITATION.cff` and a "How to cite" section in the README.
- Audit A2 (independent model) reviewed the release; FINDINGS S3 lists its fixes.
