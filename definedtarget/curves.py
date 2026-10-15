"""Interest-rate curve: turn the US Treasury's daily par yields into discount factors.

Plain words
- A *discount factor* DF(T) is today's price of $1 paid T years from now. Everything in option pricing that
  involves "money later" uses it: the option's payoff at expiry, and each SPY dividend.
- The Treasury publishes *par yields*: the coupon rate at which a new Treasury note of that maturity would sell
  for exactly 100. A par yield is not the same as the rate for a single payment at T (a *zero rate*), because a
  note pays coupons along the way. "Bootstrapping" recovers the zero rates from the par yields, one half-year
  at a time, so that every par note prices back to exactly 100.
- We quote zero rates *continuously compounded* (DF = exp(-z T)), the convention option formulas use.
- Treasury rates are not exactly the rate option markets use; options typically imply a slightly higher rate.
  So every price carries a *financing spread* s (ASSUMED 0 / +25 / +50 bp, ASSUMPTIONS.md A4), and the curve
  reports DF(T) = exp(-(z(T) + s) T).

Conventions (ASSUMED, standard): the code treats maturities up to 1 year (bills) as zero rates quoted on a
semiannual bond-equivalent basis; beyond 1 year, par notes pay semiannual coupons. The code interpolates par
yields linearly in maturity onto a half-year grid before bootstrapping, and interpolates zero rates linearly in
maturity. Time in years = calendar days / 365 (ACT/365).
"""
import csv
import io
import math
from datetime import datetime

import numpy as np

TENORS = {"1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25, "4 Mo": 4 / 12, "6 Mo": 0.5,
          "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0,
          "30 Yr": 30.0}


def parse_treasury_csv(text):
    """{date: {tenor_years: par_yield_decimal}} from one Treasury par-yield-curve CSV (blanks skipped)."""
    out = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        d = datetime.strptime(row["Date"], "%m/%d/%Y").date()
        pts = {}
        for name, years in TENORS.items():
            val = (row.get(name) or "").strip()
            if val:
                pts[years] = float(val) / 100.0
        out[d] = pts
    return out


def load_treasury(root=None):
    """All saved Treasury par curves (latest saved copy of each year's file), merged by date."""
    from .forward import fetch, parse   # local import: curves.py stays usable without the forward package
    latest = {}
    for rec in parse.saved_versions(root, "par_yield_curve_"):
        latest[rec["name"]] = rec
    curves = {}
    for name in sorted(latest):
        curves.update(parse_treasury_csv(parse.read_text(root, latest[name])))
    if not curves:
        raise FileNotFoundError("no Treasury curve files yet - run scripts/capture.py")
    return dict(sorted(curves.items()))


def bootstrap_zero(par_points, max_years=30.0):
    """Continuous zero rates on a half-year grid from par yields {years: par_yield}.

    Returns (grid_years, zero_rates_continuous). Bills (<= 1 year): zero rate = par yield (semiannual basis).
    Notes (> 1 year): DF_n = (1 - c/2 * sum(DF_1..DF_{n-1})) / (1 + c/2), c = par yield at that maturity.
    """
    xs = np.array(sorted(par_points))
    ys = np.array([par_points[x] for x in xs])
    grid = np.arange(0.5, max_years + 1e-9, 0.5)
    par = np.interp(grid, xs, ys)                          # flat beyond the ends
    dfs = np.empty(len(grid))
    for i, (t, c) in enumerate(zip(grid, par)):
        if t <= 1.0 + 1e-12:
            dfs[i] = (1.0 + c / 2.0) ** (-2.0 * t)
        else:
            dfs[i] = (1.0 - c / 2.0 * dfs[:i].sum()) / (1.0 + c / 2.0)
    zero = -np.log(dfs) / grid
    # short end below 6 months: use the bill points directly (semiannual basis -> continuous)
    short = [(x, 2.0 * math.log(1.0 + par_points[x] / 2.0)) for x in xs if x < 0.5]
    gx = np.concatenate([[x for x, _ in short], grid])
    gz = np.concatenate([[z for _, z in short], zero])
    return gx, gz


class Curve:
    """Discount curve for one date: zero rates interpolated linearly in maturity, plus a parallel spread."""

    def __init__(self, as_of, par_points, spread=0.0, source=""):
        self.as_of = as_of
        self.par_points = dict(par_points)
        self.spread = float(spread)
        self.source = source
        self.grid, self.zero_rates = bootstrap_zero(self.par_points)

    def zero(self, t):
        """Continuous zero rate (decimal) for maturity t years, including the spread."""
        return float(np.interp(t, self.grid, self.zero_rates)) + self.spread

    def df(self, t):
        """Discount factor for t years (t <= 0 gives 1)."""
        return 1.0 if t <= 0 else math.exp(-self.zero(t) * t)

    def shifted(self, bump):
        """Same curve with the spread moved by `bump` (decimal); the rho calculation uses it."""
        c = Curve.__new__(Curve)
        c.__dict__.update(self.__dict__)
        c.spread = self.spread + bump
        return c


MAX_CURVE_AGE_DAYS = 5     # calendar days; covers weekends and a bond-market holiday, refuses a stale file


def curve_on(curves, as_of, spread=0.0, max_age_days=MAX_CURVE_AGE_DAYS):
    """Curve for a close date: that date's row if published, else the latest earlier one (flagged in .source).

    Uses the same-day curve (published after the close) because the code uses it to MARK a close, not to decide
    anything; decision rules must lag it. Refuses a curve older than `max_age_days` (audit A1: a stale curve
    must stop the run, not pass silently).
    """
    days = [d for d in curves if d <= as_of]
    if not days:
        raise ValueError(f"no Treasury curve on or before {as_of}")
    d = max(days)
    if (as_of - d).days > max_age_days:
        raise ValueError(f"stale Treasury curve: latest is {d}, {(as_of - d).days} days before {as_of}")
    src = f"Treasury par curve {d.isoformat()}" + ("" if d == as_of else f" (latest before {as_of.isoformat()})")
    return Curve(d, curves[d], spread=spread, source=src)


def year_fraction(start, end):
    """ACT/365 year fraction between two dates."""
    return (end - start).days / 365.0


def box_implied_rate(k_low, k_high, box_value, as_of, expiry):
    """Continuous rate implied by a box spread: the box pays (k_high - k_low) at expiry for sure.

    The code applies it to XBOX's SPY box in Roundhill's holdings to cross-check the vendor's discounting.
    """
    t = year_fraction(as_of, expiry)
    return -math.log(box_value / (k_high - k_low)) / t
