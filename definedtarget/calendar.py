"""NYSE trading sessions, written out from the exchange's holiday rules (no extra packages, no downloads).

Full-day closures (NYSE rule set in force since 2022): New Year's Day, Martin Luther King Jr. Day (3rd Monday of
January), Washington's Birthday (3rd Monday of February), Good Friday, Memorial Day (last Monday of May),
Juneteenth (June 19, from 2022), Independence Day, Labor Day (1st Monday of September), Thanksgiving (4th
Thursday of November), Christmas.

Weekend rule: the NYSE observes a Saturday holiday on the Friday before, a Sunday one on the Monday after.
Exception: when New Year's Day falls on a Saturday the NYSE does NOT close on the Friday before (that Friday is
Dec 31, the last session of the year).

Not modeled: one-off closures (national days of mourning, e.g. 2018-12-05, 2025-01-09) and early (1 pm)
closes. When the NYSE announces a one-off closure, add it to EXTRA_CLOSURES by hand.
"""
from datetime import date, timedelta
from functools import lru_cache

EXTRA_CLOSURES = {date(2018, 12, 5), date(2025, 1, 9)}     # announced one-off closures (add new ones here)


def easter(year):
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year, month, weekday, n):
    """The n-th given weekday (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(year):
    """Set of NYSE full-day holidays in a calendar year."""
    out = set()
    ny = date(year, 1, 1)
    if ny.weekday() != 5:                       # Saturday New Year's Day: no Friday closure
        out.add(_observed(ny))                  # Sunday New Year's Day -> closed Monday Jan 2
    out.add(_nth_weekday(year, 1, 0, 3))        # MLK Day
    out.add(_nth_weekday(year, 2, 0, 3))        # Washington's Birthday
    out.add(easter(year) - timedelta(days=2))   # Good Friday
    out.add(_nth_weekday(year, 5, 0, -1))       # Memorial Day
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))   # Juneteenth
    out.add(_observed(date(year, 7, 4)))        # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))        # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))       # Thanksgiving
    out.add(_observed(date(year, 12, 25)))      # Christmas
    return frozenset(out)


def is_session(d):
    """True if the NYSE is open (full or early close) on date d."""
    return d.weekday() < 5 and d not in holidays(d.year) and d not in EXTRA_CLOSURES


def next_session(d):
    """The first NYSE session strictly after d."""
    d += timedelta(days=1)
    while not is_session(d):
        d += timedelta(days=1)
    return d


def previous_session(d):
    """The last NYSE session strictly before d."""
    d -= timedelta(days=1)
    while not is_session(d):
        d -= timedelta(days=1)
    return d


def sessions(start, end):
    """All NYSE sessions from start to end inclusive."""
    out, d = [], start
    while d <= end:
        if is_session(d):
            out.append(d)
        d += timedelta(days=1)
    return out
