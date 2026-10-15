"""NYSE calendar against the exchange's published holiday dates (nyse.com holiday schedule, read 2026-10-14)."""
from datetime import date

from _harness import check

from definedtarget import calendar as cal

check("Easter Sunday 2024 is Mar 31", cal.easter(2024) == date(2024, 3, 31))
check("Easter Sunday 2026 is Apr 5", cal.easter(2026) == date(2026, 4, 5))
check("Easter Sunday 2030 is Apr 21", cal.easter(2030) == date(2030, 4, 21))

nyse_2026 = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
             date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}
check("2026 NYSE holidays match the published list", cal.holidays(2026) == nyse_2026,
      sorted(cal.holidays(2026) ^ nyse_2026))
nyse_2027 = {date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
             date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24)}
check("2027 NYSE holidays match the published list", cal.holidays(2027) == nyse_2027,
      sorted(cal.holidays(2027) ^ nyse_2027))

check("Saturday New Year's Day (2022): Friday Dec 31, 2021 is still a session", cal.is_session(date(2021, 12, 31)))
check("Saturday New Year's Day (2028): Friday Dec 31, 2027 is still a session", cal.is_session(date(2027, 12, 31)))
check("Columbus Day is a trading day (2026-10-12)", cal.is_session(date(2026, 10, 12)))
check("XX's target date 2030-01-10 (a Thursday) is a session", cal.is_session(date(2030, 1, 10)))
check("2029-12-21 (SPY's last quarterly ex-date before expiry) is a session", cal.is_session(date(2029, 12, 21)))
check("next session after Fri 2026-10-09 is Mon 2026-10-12", cal.next_session(date(2026, 10, 9)) == date(2026, 10, 12))
check("next session after Wed 2026-11-25 skips Thanksgiving", cal.next_session(date(2026, 11, 25)) == date(2026, 11, 27))
check("previous session before Mon 2026-09-07 (Labor Day) is Fri 09-04",
      cal.previous_session(date(2026, 9, 8)) == date(2026, 9, 4))
check("one-off closure 2025-01-09 (national day of mourning) is not a session", not cal.is_session(date(2025, 1, 9)))
n = len(cal.sessions(date(2026, 10, 14), date(2030, 1, 10)))
check("sessions from 2026-10-14 to 2030-01-10 inclusive: 815 (design review count)", n == 815, n)
