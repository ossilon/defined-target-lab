"""S&P/SPY ratio and the effective trigger, on synthetic prices with a known answer."""
from datetime import date, timedelta

import pandas as pd

from _harness import check, raises

from definedtarget import ratio

# Synthetic: ratio exactly 10.0 + 0.003 per year, every business day 2011-2026
days = pd.bdate_range("2011-01-01", "2026-12-31")
yrs = (days - pd.Timestamp("2011-01-01")).days / 365.25
spy = pd.Series(500.0, index=days)
spx = spy * (10.0 + 0.003 * yrs)
jan = ratio.january_ratios(spx, spy)
check("one January ratio per year 2011-2026", sorted(jan) == list(range(2011, 2027)), sorted(jan))
check("January ratio follows the known line (10 + 0.003 t)",
      all(abs(jan[y] - (10.0 + 0.003 * (y - 2011 + 9 / 365.25))) < 0.0003 for y in jan))
b = ratio.trigger_band(jan, strike=995.01, expiry_year=2030)
check("horizon is 2030 - 2026 = 4 years", b["horizon_years"] == 4)
check("trend recovered as 0.003 per year", abs(b["trend_per_year"] - 0.003) < 1e-4, b["trend_per_year"])
fitted_2030 = 10.0 + 0.003 * (2030 - 2011 + 9 / 365.25)              # the known line, evaluated at Jan 2030
check("trend trigger = 995.01 x the fitted line's value in 2030",
      abs(b["trigger_trend"] - 995.01 * fitted_2030) < 0.02, (b["trigger_trend"], 995.01 * fitted_2030))
check("band reports the sample's first year", b["first_year"] == 2011)
check("no-drift trigger = 995.01 x latest ratio", abs(b["trigger_no_drift"] - 995.01 * b["latest_ratio"]) < 1e-9)
check("rising ratio: smallest-drift trigger is above the no-drift trigger", b["trigger_min_drift"] > b["trigger_no_drift"])
spike = spy.copy()
spike.loc[pd.Timestamp("2020-01-10")] = 450.0                        # one bad SPY close on the anchor day
check("a single bad close on the anchor day is damped by the 7-session median",
      abs(ratio.january_ratios(spx, spike)[2020] - jan[2020]) < 0.001)
raises("expiry year not after the data is refused", lambda: ratio.trigger_band(jan, expiry_year=2026), "not before")
