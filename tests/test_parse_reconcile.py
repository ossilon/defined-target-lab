"""Parsing and the accounting identities: synthetic files first, then the real captured files if present."""
from datetime import date

import pandas as pd

from _harness import check, raises

from definedtarget.forward import parse, reconcile

check("decimals: '49.920000' carries 2", parse.decimals("49.920000") == 2)
check("decimals: '49.677800' carries 4", parse.decimals("49.677800") == 4)
check("decimals: '288' carries 0", parse.decimals("288") == 0)
occ = parse.parse_occ("4SPY  300110C00995010")
check("OCC id parsed: SPY call, strike 995.01, expiry 2030-01-10, FLEX prefix 4",
      occ == {"root": "4SPY", "underlying": "SPY", "flex_prefix": "4", "expiry": date(2030, 1, 10),
              "cp": "C", "strike": 995.01}, occ)
check("money-market ticker is not an option", parse.parse_occ("FGXXX") is None)

HEADER = ",".join(parse.HOLDINGS_COLUMNS)
# Synthetic holdings (made-up round numbers in the issuer's file layout; no real issuer values)
SYN = (HEADER + "\n"
       "10/14/2026,XX,4SPY  300110C00995010,4SPY  300110C00995010,SPY 01/10/2030 995.01 C,200,50.000000,"
       "1000000.00,99.01%,1010000.00,40000,4,\n"
       "10/14/2026,XX,FGXXX,31846V336,Money Market Fund,10100.00,100,10100.00,1.00%,1010000.00,40000,4,Y\n"
       "10/14/2026,XX,Cash&Other,Cash&Other,Cash & Other,-100.00,1,-100.00,-0.01%,1010000.00,40000,4,Y\n"
       "10/14/2026,ZZZ,AAPL,037833100,Apple Inc,10,250,2500,100%,2500,100,1,\n")
rows = parse.holdings_rows(SYN, date(2026, 10, 13), "v1", {"XX"})
check("only the requested fund's rows are kept (3 XX lines)", len(rows) == 3, len(rows))
check("percent weights become fractions", abs(rows[0]["weight"] - 0.9901) < 1e-12)
raises("a changed holdings header is refused", lambda: parse.holdings_rows("Date,Fund\n", date(2026, 10, 13), "v1"),
       "schema changed")

h = pd.DataFrame(rows)
daily = pd.DataFrame({"as_of": [date(2026, 10, 13)], "nav": [25.25], "market_price": [25.30]})
st = reconcile.fund_state(h, daily)
r = st.iloc[0]
check("contracts per share = 200 / 40,000 = 0.005", abs(r.contracts_per_share - 0.005) < 1e-15)
check("cash adds money market and Cash & Other (10,000)", abs(r.cash - 10000.0) < 1e-6, r.cash)
check("NetAssets / shares (25.25) matches the published NAV", bool(r.check_nav))
check("date rule: file for close 10/13 is dated 10/14", bool(r.check_date_rule))
check("market value = contracts x 100 x mark", bool(r.check_mv))
check("listed lines within half a cent per share of NetAssets", bool(r.check_lines), r.nav_lines - r.nav_net_assets)
bad = h.copy()
bad["effective_date"] = date(2026, 10, 13)
check("a file dated the same day as its close fails the date rule",
      not bool(reconcile.fund_state(bad, daily).iloc[0].check_date_rule))

# Checks on the real captured files live in tests/realdata_local.py (git-ignored: they hold issuer values).
