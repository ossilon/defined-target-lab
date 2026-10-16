# Changelog

## Unreleased (v0.2.0: browser simulator)

### Added
- **Simulator model** (`definedtarget/sim.py`): value, leverage, one day's cost and odds for any fund made of option
  legs plus cash (calls or puts, bought or sold), with volatility that can follow the index. Pinned to
  `analytics.py` and `valuation.py` for XX to 1e-9.
- **Browser engine** (`web/engine.js`, no libraries) and **self-test** (`web/selftest.html`): 716 answers written by
  the Python engine (`scripts/make_golden.py` -> `web/golden.json`) must all match in the browser; a test fails if
  the answers go stale.
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
