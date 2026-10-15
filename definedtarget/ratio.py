"""Where is the real finish line? The S&P 500 level at which SPY reaches the strike.

Plain words
- The option pays off on SPY's price, not on the S&P 500. SPY is close to S&P/10, but not exactly: SPY holds the
  dividends it has received in cash until its quarterly payout, it charges a small fee, and the two closing prices
  come from slightly different processes. So the ratio S&P / SPY drifts a little and wobbles day to day.
- XX's strike 995.01 corresponds to a ratio of 10,000 / 995.01 = 10.0501. If on 2030-01-10 the ratio is R, the
  S&P level at which the option starts paying is 995.01 x R. A ratio below 10.0501 means the effective trigger
  is a little BELOW 10,000 (good for holders).
- On 2030-01-10, SPY will be about 3 weeks past its December ex-dividend date. So the fair comparison is the ratio
  on the session nearest January 10 in past years (same point in SPY's dividend cycle).

Method (ASSUMED, ASSUMPTIONS.md A8): for each year, the median ratio over the 7 sessions centered on the session
nearest January 10 (the median damps one-day closing-price noise). The projection to the expiry January uses
the latest January ratio plus the smallest and largest change seen over every span of the same length
(latest January -> expiry January, e.g. 2026 -> 2030 = 4 years), and a straight-line fit evaluated at the expiry
year. The spans overlap, so their min/max DESCRIBES history; it is not a confidence interval. The band depends on
how far back the sample starts (`first_year`), so callers should show several start years (audit A1).
"""
from datetime import date

import numpy as np
import pandas as pd


def january_ratios(spx, spy, first_year=2011, window=3, anchor_day=10):
    """{year: median S&P/SPY ratio over the sessions within `window` of the session nearest Jan `anchor_day`}.

    spx, spy: pandas Series of closes indexed by date (datetime.date or Timestamp).
    """
    both = pd.concat([pd.Series(spx, dtype=float), pd.Series(spy, dtype=float)], axis=1, join="inner").dropna()
    both.index = pd.to_datetime(both.index)
    ratio = both.iloc[:, 0] / both.iloc[:, 1]
    out = {}
    for year in sorted(set(ratio.index.year)):
        if year < first_year:
            continue
        target = pd.Timestamp(year, 1, anchor_day)
        pos = int(np.argmin(np.abs((ratio.index - target).days)))
        if abs((ratio.index[pos] - target).days) > 5:
            continue
        lo, hi = max(0, pos - window), min(len(ratio), pos + window + 1)
        out[year] = float(ratio.iloc[lo:hi].median())
    return out


def trigger_band(jan, strike=995.01, expiry_year=2030):
    """Effective S&P trigger at the expiry January.

    horizon = expiry_year - latest January in the data. Returns the latest ratio, the observed range of ratio
    changes over spans of that length, and the trigger with no drift, with the smallest and with the largest
    observed drift, plus a straight-line trend extrapolation.
    """
    years = sorted(jan)
    latest_year = years[-1]
    horizon = expiry_year - latest_year
    if horizon < 1:
        raise ValueError("latest January is not before the expiry year")
    changes = [jan[y] / jan[y - horizon] - 1.0 for y in years if (y - horizon) in jan]
    if not changes:
        raise ValueError(f"need January ratios at least {horizon} years apart")
    latest = jan[latest_year]
    slope, intercept = (float(x) for x in np.polyfit(years, [jan[y] for y in years], 1))
    lo, hi = min(changes), max(changes)
    return {
        "first_year": years[0], "latest_year": latest_year, "horizon_years": horizon, "latest_ratio": latest,
        "implied_by_strike": 10000.0 / strike, "n_spans": len(changes),
        "change_min": lo, "change_max": hi, "trend_per_year": slope,
        "trigger_no_drift": strike * latest,
        "trigger_min_drift": strike * latest * (1 + lo),
        "trigger_max_drift": strike * latest * (1 + hi),
        "trigger_trend": strike * (intercept + slope * expiry_year),     # the fitted line's value at expiry
    }
