"""definedtarget.listedvols: trade dates, forwards and implied vols of the listed Dec-2029 calls' daily bars.

Synthetic data only (no TradingView data). Judges (fixed 2026-10-23, before the first run):
1. Trade dates: a bar stamped 20:15 New York belongs to the next NYSE session, in summer (00:15 UTC) and in winter
   (01:15 UTC), over a weekend and over a holiday.
2. A price made by Black-76 at a known vol, with the same forward, returns that vol (1e-10).
3. Both bar-file formats load; the 10,000 strike's Python-literal file is read as data, never run.
4. Bars below the volume floor, or without an S&P close on their trade date, are left out.
"""
import json
import math
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from _harness import check

from definedtarget import curves, listedvols, pricing

NY = ZoneInfo("America/New_York")


def stamp(y, m, d):
    return int(datetime(y, m, d, 20, 15, tzinfo=NY).timestamp())


# 1. Trade dates.
cases = [(stamp(2024, 6, 13), date(2024, 6, 14), "summer, Thursday evening -> Friday"),
         (stamp(2025, 1, 16), date(2025, 1, 17), "winter, Thursday evening -> Friday"),
         (stamp(2025, 1, 17), date(2025, 1, 21), "Friday evening -> Tuesday (Martin Luther King Day on Monday)"),
         (stamp(2024, 12, 24), date(2024, 12, 26), "Christmas Eve evening -> December 26")]
for t, want, why in cases:
    check(f"trade date: {why}", listedvols.trade_date(t) == want, (listedvols.trade_date(t), want))
check("the first 10,000-strike bar (1718324100) is the 2024-06-14 session", listedvols.trade_date(1718324100)
      == date(2024, 6, 14))

# 2. Round trip through Black-76 with the module's own forward.
curve = curves.Curve(date(2025, 3, 3), {0.25: 0.043, 1.0: 0.041, 2.0: 0.040, 5.0: 0.041})
history = [(date(2024, 3, 15), 1.59), (date(2024, 6, 21), 1.76), (date(2024, 9, 20), 1.75), (date(2024, 12, 20), 1.97)]
F, df, T = listedvols.forward(5950.0, date(2025, 3, 3), curve, history)
q = sum(a for _, a in history) / (5950.0 / listedvols.RATIO)
check("the forward is S e^(-qT) / DF with q from SPY's last four dividends at S / RATIO",
      abs(F - 5950.0 * math.exp(-q * T) / df) < 1e-9 and abs(T - (date(2029, 12, 21) - date(2025, 3, 3)).days / 365) < 1e-15)
price = pricing.call_price(F, 10000.0, df, T, 0.1437)
check("a Black-76 price at 14.37% returns 14.37% (1e-10)",
      abs(pricing.implied_vol(price, F, 10000.0, df, T) - 0.1437) < 1e-10)

# 3 and 4. Loading and filtering.
with tempfile.TemporaryDirectory() as tmp:
    ext = Path(tmp) / "external"
    (ext / "listed_bars").mkdir(parents=True)
    t1, t2 = stamp(2025, 3, 2), stamp(2025, 3, 3)         # trade dates 2025-03-03 and 2025-03-04
    (ext / "listed_bars" / "SPX291221C12000_daily.json").write_text(
        json.dumps({"t": [t1, t2], "c": [40.0, 41.0], "v": [10, 2]}), encoding="utf-8")
    (ext / listedvols.TEN_K_FILE).write_text(f"# comment\nt = [{t1}]\nc = [{price:.6f}]\nv = [7]\n", encoding="utf-8")
    bars = listedvols.load_bars(tmp)
    check("both bar formats load (JSON for 12,000; Python literals for 10,000)",
          sorted(bars) == [10000.0, 12000.0] and bars[12000.0][1] == (t2, 41.0, 2) and bars[10000.0][0][2] == 7, bars)
    curves_by_date = {date(2025, 3, 3): curve.par_points, date(2025, 3, 4): curve.par_points}
    rows = listedvols.trade_vols(bars, {date(2025, 3, 3): 5950.0}, curves_by_date, history, min_volume=5)
    check("bars below the volume floor or without an S&P close on their trade date are left out",
          [(r["date"], r["strike"]) for r in rows] == [(date(2025, 3, 3), 10000.0), (date(2025, 3, 3), 12000.0)], rows)
    check("the 10,000 bar's vol comes back at 14.37% (1e-6, the price was written with 6 decimals)",
          abs(rows[0]["iv"] - 0.1437) < 1e-6, rows[0]["iv"])
