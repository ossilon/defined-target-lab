# Changelog

## v0.1.0 (unreleased)

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

### Not included yet
- Browser simulator, leverage tracker and predictor, paper portfolios (planned for v0.2).
