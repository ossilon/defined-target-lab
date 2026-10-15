# XX fund facts (every fact with its source, date and status)

Status words: **VERIFIED** (read in a filing or a captured issuer file, cited), **MEASURED** (computed by tested code
from captured files), **PROBABLE** (strong evidence; no primary document confirms it yet), **OPEN** (unknown).
Nothing here forecasts. Raw files cited as `raw/...` sit in the git-ignored `data/` folder (see data/README.md).

Last updated 2026-10-14.

## The fund

| Fact | Value | Status | Source |
|---|---|---|---|
| Name | Roundhill S&P 500® Target 10,000 2030 ETF | VERIFIED | Prospectus 2026-09-29, cover (`raw/roundhill/2026-10-14/XX_Prospectus.pdf`) |
| Ticker / exchange | XX, Cboe BZX (TradingView `CBOE:XX`) | VERIFIED | Prospectus cover; TradingView bars from 2026-10-01 |
| CUSIP / ISIN | 77926Y823 / US77926Y8232 | VERIFIED | Fund page 2026-10-14; `dailynav.csv` |
| Launch | 2026-10-01 (first close with holdings: 2026-09-30) | VERIFIED | Prospectus p.2 ("October 1, 2026 commencement"); `holdings_09302026.csv` |
| Fee | 0.99% a year of average daily net assets, unitary; the fund pays trading costs, taxes, interest and extraordinary costs on top | VERIFIED | Prospectus p.1 and p.35; SAI p.17 |
| Objective | Gains if the S&P 500 exceeds 10,000 on 2030-01-10, measured by the Underlying ETF(s) against the Underlying ETF Target | VERIFIED | Prospectus p.1-2 |
| Defined Target Period | 2026-10-01 to 2030-01-10 | VERIFIED | Prospectus p.2 |
| Underlying ETF universe | SPY, IVV, VOO, SPYM (one or more; each with its own strike) | VERIFIED | Prospectus p.8, p.22 |
| After expiry | The fund rolls into new FLEX calls for a new target and date (rule undisclosed). If the calls expire worthless: a reverse split, with "approximately 1% to 10% of its net assets remaining" | VERIFIED | Prospectus p.9 |
| Creations | In cash; creation unit of 10,000 shares | VERIFIED (cash: prospectus p.17); MEASURED (unit size: shares moved 10,000 → 40,000 → 60,000 as `CreationUnits` moved 1 → 4 → 6) | holdings files 10/05, 10/06, 10/12 |
| Holdings published | Every business day on the website; quarterly on N-PORT | VERIFIED | SAI p.26 |
| NAV time | NYSE close, normally 4:00 pm ET; Roundhill sets fair values as valuation designee | VERIFIED | Prospectus p.40; SAI p.36 |
| Valuation vendor / FLEX pricing method | Undisclosed | OPEN | The SAI is silent (p.36) |
| Closure risk | The Board may close the fund at any time; the exchange may delist it below 50 holders after 12 months | VERIFIED | SAI p.1-2 |

## The option it holds

| Fact | Value | Status | Source |
|---|---|---|---|
| Identifier | `4SPY  300110C00995010`, "SPY 01/10/2030 995.01 C" | VERIFIED | every holdings file |
| Underlying / type / strike / expiry | SPY, call, 995.01, 2030-01-10 (a Thursday; an NYSE session) | VERIFIED (terms); MEASURED (session) | holdings files; `tests/test_calendar.py` |
| Exercise / settlement | European, cash-settled | VERIFIED | Prospectus p.7 |
| Settlement price | SPY's official closing price on 2030-01-10; cash paid the next business day | PROBABLE | OCC FLEX symbology ("4" prefix = European, cash-settled), per NYSE "Cash Settled FLEX ETF Options" and OCC filing 34-94910; no Roundhill document confirms it yet |
| Strike vs index target | 10,000 / 995.01 = 10.0501 implied S&P/SPY ratio | MEASURED | arithmetic |
| Observed S&P/SPY ratio | 7,805.4538 / 778.01 = 10.0325 at the 2026-10-13 close | MEASURED | `raw/roundhill/2026-10-14/daily_instrument_pricing.csv` (Roundhill's own reference closes) |
| Effective S&P trigger | **Central ~9,995; about 9,955-10,040 depending on how much history the estimate uses** (from 1994: 9,955-10,040; from 2005 or 2008: 9,968-10,002; from 2011: 9,989-10,001). The January S&P/SPY ratio rose from 10.0008 (2011) to 10.0369 (2026), with small dips in 2016 and 2026 (FINDINGS F7) | MEASURED (`scripts/ratio_band.py`, local SP:SPX + AMEX:SPY library, 2026-10-14; audited) | The library's SPY rows come from a BATS export, not the official close; 7-session medians damp this |
| SPY's last ex-dividend date before expiry | 2029-12-21 (third Friday, Dec 2029) | PROBABLE (SPY's usual schedule) | `definedtarget.calendar`; SPY ex-dates 2025-12-19, 2026-03-20, 2026-06-18 (Juneteenth shift), 2026-09-18 |

## Measured from the first 10 closes (2026-09-30 to 2026-10-13)

Source: `py -I scripts\build_tables.py` → `data/tables/xx_state.csv`, built from the holdings and `xx_daily.csv` files.

| Fact | Value | Status |
|---|---|---|
| Contracts per share | 48/10,001 on the seed day, then exactly 0.0048 every day (calls on 0.48 SPY shares per XX share), through a 1-share redemption and creations of +30,000 (10/06) and +20,000 (10/12) shares | MEASURED |
| Shares outstanding | 10,001 → 10,000 (10/01) → 40,000 (10/06) → 60,000 (10/12) | MEASURED |
| NAV | 24.20 (9/30) … 27.73 (10/06) … 26.51 (10/13) | VERIFIED (`xx_daily.csv`) |
| Seed value | The factsheet's "AUM $0.25MM, 10,000 shares" (10/01) suggests a ~$25 seed; first close 24.20 | PROBABLE (no document states the seed price) |
| Cash buffer | 0.90-1.02% of net assets (money market FGXXX + "Cash & Other") | MEASURED |
| NetAssets ÷ shares vs published NAV | Rounds to the published NAV every day (within half a cent) | MEASURED |
| Listed lines vs NetAssets | Differ by −$81 to +$185 (≤0.5 cents/share) after the seed day; seed day +$60.37 (0.6 cents/share), a documented exception | MEASURED |
| Option mark decimals | 2 decimals on 9/30 (seed), 10/06 and 10/12 (both creation days); 4 decimals otherwise, or 3 where a revised file dropped a trailing zero (10/05: 54.924). Confounded: Roundhill rewrote the 10/05, 10/06 and 10/12 files, and each rewrite strips trailing zeros from every field (FINDINGS F4) | MEASURED |
| Premium/discount (close vs NAV) | −0.83% to +1.28% | MEASURED |
| 30-day median bid/ask spread | 1.18% (Solactive feed `level` 1.1821) | VERIFIED (`bidask_xx.json`, 2026-10-14) |
| AUM at 10/13 | $1,590,876 (holdings) / $1,590,877.93 (daily NAV file) | VERIFIED |

## When Roundhill updates its files (server Last-Modified, UTC)

| File | Typical update | Notes |
|---|---|---|
| `xx_daily.csv`, `dailynav.csv`, holdings for close D | 23:02-00:42 UTC on day D (7-8:40 pm New York) | |
| `daily_instrument_pricing.csv` (SPY, SPX reference closes) | ~02:03 UTC on D+1 | latest day only |
| Holdings revisions | Roundhill rewrote the files for 10/05, 10/06 and 10/12 at 15:42-20:03 UTC on D+1 | the capture re-checks the newest 3 days every run |

Schedule a daily capture **Tuesday to Saturday, a few hours after the reference-close file appears** (after about
02:00 UTC on D+1), so it runs after every regular update in both US summer and winter time. Each capture also
re-checks the newest 3 holdings days to pick up next-day revisions.

## Other Roundhill funds in the same holdings file (market evidence)

| Fund | Holding | Use |
|---|---|---|
| XBOX | A SPY FLEX box spread (long 1.06 C / short 10000.06 C / short 1.06 P / long 10000.06 P, 2026-12-30) | Its marks imply the pricing vendor's discount rate: 4.62-4.65%, about 0.5 points above Treasuries at that term (FINDINGS F9) |
| TPAY | Deep in-the-money SPY FLEX calls (33.67 C 2027-03-12, 33.18 C 2027-04-16) | Imply the vendor's SPY forward |
| XDIV | SPYM shares (price 91.55 on 10/13) | Give a reference price for another ETF in XX's universe |

## Sister and competing funds (not yet launched or not yet modeled)

| Fund | Status | Source |
|---|---|---|
| Roundhill Dow 75,000 Target 2030; Innovation-100 50,000 Target 2030 | Filed Jan 2026 | The Daily Upside, 2026-02-04 |
| "QQ" (QQQM-based target fund) | In the same SEC 497 filing as XX | SEC EDGAR 0001976517 |
| Defiance "Convexity" / "Ultra Convexity" ETFs | Filed 2026-09-30 | 247 Wall St, 2026-10-01 |

## Open questions (answer from filings or by asking Roundhill)

1. How did Roundhill set 995.01 (which ratio, which date)?
2. Which pricing vendor marks the FLEX option, and how?
3. What are the creation and redemption fees, and the "maximum additional charge" for cash creations?
4. What roll rule governs the next Defined Target Period?
5. Why "1% to 10%" cash if the calls expire worthless, when the fund holds ~1%?
6. Does the fund pay its fee from cash, then by selling whole contracts? (Watch contracts per share; hypotheses H2 and H3.)
