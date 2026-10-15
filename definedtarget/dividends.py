"""SPY's future dividends, projected as discrete cash payments.

Plain words
- A call option on SPY does NOT receive SPY's dividends. Each time SPY goes "ex-dividend" its price drops by
  about the dividend, which hurts a call holder. So the option's fair value depends on the dividends expected
  before expiry. Between now and 2030-01-10 there are about 13 quarterly SPY dividends.
- SPY's ex-dividend date is normally the third Friday of March, June, September and December (e.g. 2025-12-19,
  2026-03-20, 2026-09-18). When that Friday is a market holiday it moves to the session before (2026-06-19 is
  Juneteenth, so the ex-date was 2026-06-18). Payment is near the end of the following month (ASSUMED: last NYSE
  session of the month after the ex-date).
- Amounts are unknown in advance. ASSUMED projection: each future dividend = (sum of the last 4) / 4, grown at
  `growth` per year from the last ex-date. Default growth 5%/yr, with 0% and 10% as sensitivities
  (ASSUMPTIONS.md A3). The pricing result is not very sensitive to this (about 0.1 vol point, design review).

Two ways to feed dividends to the pricing formula:
- discrete (default): forward F = (S - PV of dividends before expiry) / DF(T), each dividend discounted to its
  pay date ("escrowed dividends").
- continuous yield (sensitivity): q = trailing-4 dividends / S, forward F = S * exp(-q T) / DF(T).
"""
import csv
import io
import math
from dataclasses import dataclass
from datetime import date, timedelta

from . import calendar as cal

DEFAULT_GROWTH = 0.05


@dataclass(frozen=True)
class Dividend:
    ex_date: date
    pay_date: date
    amount: float
    projected: bool


def third_friday(year, month):
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(days=14)


def ex_date_for(year, month):
    """SPY's usual ex-date: third Friday of the month, moved to the previous session if that is a holiday."""
    d = third_friday(year, month)
    return d if cal.is_session(d) else cal.previous_session(d)


def pay_date_for(ex):
    """ASSUMED pay date: last NYSE session of the month after the ex-date's month."""
    y, m = (ex.year + (ex.month == 12), ex.month % 12 + 1)
    last = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return last if cal.is_session(last) else cal.previous_session(last)


def read_history_csv(text):
    """History from a CSV with columns ex_date (YYYY-MM-DD), amount. Returns a sorted list of (date, amount)."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rows.append((date.fromisoformat(r["ex_date"].strip()), float(r["amount"])))
    return sorted(rows)


def project(history, after, until, growth=DEFAULT_GROWTH):
    """Dividends with ex-date in (after, until], projected from the last 4 known payments.

    history: sorted list of (ex_date, amount) known on the valuation date (the function uses only ex-dates
    <= after, so the projection never peeks at a later announcement).
    """
    known = [(d, a) for d, a in history if d <= after]
    if len(known) < 4:
        raise ValueError("need at least 4 past SPY dividends to project")
    last_ex = known[-1][0]
    base = sum(a for _, a in known[-4:]) / 4.0
    out = []
    for year in range(after.year, until.year + 1):
        for month in (3, 6, 9, 12):
            ex = ex_date_for(year, month)
            if after < ex <= until and ex > last_ex:
                yrs = (ex - last_ex).days / 365.0
                out.append(Dividend(ex, pay_date_for(ex), base * (1.0 + growth) ** yrs, True))
    return out


MAX_DIVIDEND_GAP_DAYS = 100     # SPY pays quarterly (~91 days); a longer gap means the history file is stale


def check_fresh(history, as_of, max_gap_days=MAX_DIVIDEND_GAP_DAYS):
    """Refuse a dividend history whose last ex-date is more than a quarter (plus slack) before as_of."""
    known = [d for d, _ in history if d <= as_of]
    if not known:
        raise ValueError("no SPY dividends on or before the valuation date")
    gap = (as_of - known[-1]).days
    if gap > max_gap_days:
        raise ValueError(f"stale SPY dividend history: last ex-date {known[-1]} is {gap} days before {as_of} "
                         "- re-run scripts/import_spy_dividends.py")
    return known[-1]


def trailing_yield(history, as_of, spot):
    """Continuous dividend yield from the last 4 payments: q = -ln(1 - D4/S) ~ D4/S (we use D4/S)."""
    known = [a for d, a in history if d <= as_of]
    if len(known) < 4:
        raise ValueError("need at least 4 past SPY dividends")
    return sum(known[-4:]) / spot


def pv_dividends(divs, as_of, curve):
    """Present value at as_of of each projected dividend, discounted to its PAY date."""
    return sum(d.amount * curve.df((d.pay_date - as_of).days / 365.0) for d in divs)


def forward_discrete(spot, as_of, expiry, curve, divs):
    """Escrowed-dividend forward price of SPY for expiry, and the discount factor to expiry."""
    t = (expiry - as_of).days / 365.0
    df_t = curve.df(t)
    return (spot - pv_dividends(divs, as_of, curve)) / df_t, df_t


def forward_continuous(spot, as_of, expiry, curve, q):
    """Continuous-yield forward: F = S exp(-q T) / DF(T)."""
    t = (expiry - as_of).days / 365.0
    df_t = curve.df(t)
    return spot * math.exp(-q * t) / df_t, df_t
