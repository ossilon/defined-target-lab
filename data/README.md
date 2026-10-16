# Data folder (git-ignored except this file)

Nothing in this folder enters the repository, and nobody publishes it. Each user builds a private copy with the
scripts.

## Filled by `py -I scripts\capture.py`

| Folder | Source | License / terms | Refresh |
|---|---|---|---|
| `raw/roundhill/<UTC date>/` | Roundhill's public fund files: daily NAV history, daily NAV snapshot, full holdings by date, SPY/SPX reference closes, 30-day bid/ask spread, fund documents | Roundhill site terms: download and use for informational purposes; reproduction, redistribution or storage in a database requires written permission. Keep the files private. | every run (documents weekly) |
| `raw/cboe/<UTC date>/` | Cboe daily history CSVs: VIX (from 1990), VIX3M (2009-09), VIX6M (2008), VIX1Y (2007) | Cboe terms | weekly |
| `raw/treasury/<UTC date>/` | US Treasury daily par yield curve, one file per year (2024 onward) | Public domain (US government) | current year every run; past years once |
| `raw/manifest.jsonl` | One line per fetch: URL, UTC time, status, server Last-Modified, SHA-256, saved path, result | — | append-only |

`py -I scripts\build_tables.py` rebuilds these tables from the raw files on every run:

| File | Contents |
|---|---|
| `tables/xx_holdings.csv` | Every XX holdings line, every saved version |
| `tables/xx_state.csv` | One row per close: contracts, mark, cash, NAV checks, flow days, premium |
| `tables/vendor_marks.csv` | Every SPY/IVV/VOO/SPYM option mark across all Roundhill funds |

## Filled by `py -I scripts\capture_flex.py` (Cboe FLEX execution reports)

| Folder / file | Source | Terms | Refresh |
|---|---|---|---|
| `raw/cboe_flex/<UTC date>/cone_options_rpt_flex_executions_late_<YYYYMMDD>.xlsx` | Cboe's daily "U.S. Options Flex Trade Reports" (one spreadsheet per trading day, every FLEX execution on that exchange) | Cboe site terms: one copy for personal, non-commercial use. Keep private; publish derived statistics only | each run fetches any missing trading day since 2026-09-30 |
| `tables/xx_flex_fills.csv` | XX's own option rows (`4SPY`, strike 995.01, expiry 2030-01-10, call): report date, trade time, size, price | private | rebuilt each run |

## Two files you add yourself (`external/`)

The issuer's files do not include SPY's dividends or SPY's past closes, so you supply them once:

| File | Columns | Where to get it | Without it |
|---|---|---|---|
| `external/spy_dividends.csv` | `ex_date,amount,source` (one row per SPY dividend) | `py -I scripts\import_spy_dividends.py --csv <file>` from State Street's published SPY distribution history (columns `ex_date,amount`) | Pricing stops: the forward needs the dividends. A history older than 100 days also stops it |
| `external/closes.csv` | `date,symbol,close,source` (symbol `SPY` or `SPX`; an exchange prefix such as `AMEX:SPY` also works) | Any source you may use. **Use the official closing price**: other feeds can differ (one differed by 0.9% on a volatile day), and a different close shifts the implied vol | The tracker prices only the days it captured the issuer's reference close itself (that file shows only the latest day), skips the rest and lists them in the daily note |

The issuer's own reference closes win over `closes.csv` wherever both exist.

## Optional: prices from your own TradingView account (display-only license; never publish)

A local price library tops up `CBOE:XX`, `AMEX:SPY`, `SP:SPX` and the listed S&P 500 Dec-2029 option chain
(`OPRA:SPX291221C10000.0` and its neighbors). The code uses them for local cross-checks and publishes only
aggregates. `external/listed_spx_dec2029_<close date>.csv` holds a bid/ask snapshot of those calls; the exploratory
script reads the newest one (FINDINGS F10).

The daily listed-option capture (`py -I scripts\capture_listed.py`, run by a scheduled task) keeps each TradingView
reply unchanged in `raw/tradingview_chain/<close date>/<read time UTC>.json` and appends one row per call to
`tables/listed_chain.csv` (close date, read time, inside protocol P2's 16:15-20:15 New York window or not, strike,
bid, ask, S&P and SPY closes). Readings outside the window stay flagged and never count.

## Why the scripts keep raw files unchanged

The code rebuilds every table from the raw files. So a parsing fix can run again over the full history, and anyone
can check our tables against the issuer's own files. The SHA-256 in the manifest proves that nobody edited a file.
