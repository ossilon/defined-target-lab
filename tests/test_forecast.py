"""The monthly forecast check (definedtarget/forward/forecast.py, protocol FC1) on made-up inputs (no data needed).

Judges: the calendar of origins and targets (rule 1); the short time grid gives the same numbers as the full one to
expiry; an exact answer for every model: XX's NAV at the target depends only on SPY's level there, so the share of
the forecast below a growth g equals the lognormal probability that SPY ends below the level whose NAV gives g;
the SPY PIT's closed form; fingerprints; the log's chain, refusals and deadlines; the summary.
"""
import json
import math
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from _harness import check, raises

from definedtarget import calendar as nyse, ledger, pricing, sim
from definedtarget.forward import forecast as fc

# ------------------------------------------------------------------ rule 1: origins, targets, seeds, deadlines
o = fc.origins()
check("39 origins, 2026-10-30 to 2029-12-31", len(o) == 39 and o[0][0] == date(2026, 10, 30)
      and o[-1][0] == date(2029, 12, 31))
check("first target 2026-11-30; last target is expiry 2030-01-10", o[0][1] == date(2026, 11, 30)
      and o[-1][1] == date(2030, 1, 10))
check("every origin and target is the last NYSE session of its month (or expiry)",
      all(nyse.is_session(t) and nyse.next_session(t).month != t.month for t, _ in o)
      and all(nyse.is_session(T) for _, T in o)
      and all(T == fc.EXPIRY or nyse.next_session(T).month != T.month for _, T in o))
check("each target is the next origin (months chain without gaps)", all(o[i][1] == o[i + 1][0] for i in range(38)))
raises("a close that is not an origin is refused", lambda: fc.target_for(date(2026, 10, 29)), "not an FC1 origin")
check("seed = year x 100 + month", fc.seed_for(date(2026, 10, 30)) == 202610)
check("deadline: two sessions after Fri 2026-10-30 is Tue 2026-11-03", fc.deadline(date(2026, 10, 30)) == date(2026, 11, 3))
check("on time until the end of the deadline day in New York (2026-11-04 04:59 UTC = 23:59 EST)",
      not fc.is_late(date(2026, 10, 30), datetime(2026, 11, 4, 4, 59, tzinfo=timezone.utc))
      and fc.is_late(date(2026, 10, 30), datetime(2026, 11, 4, 5, 0, tzinfo=timezone.utc)))

# ------------------------------------------------------------------ made-up inputs (the README's 2026-10-13 example)
INP = fc.Inputs(date(2026, 10, 30), 778.01, 0.166, 0.0486498, 882.2603, 3.2, 995.01, 0.48, 0.2474, 22.0)
T1 = date(2026, 11, 30)
N = 200_000

for name in ("M1", "M4"):
    a = fc.simulate_growth(INP, T1, name, N)
    b = fc.simulate_growth(INP, T1, name, N, full=True)
    check(f"{name}: the short grid gives the full calendar's numbers bit for bit", np.array_equal(a[0], b[0]) and a[1:] == b[1:])

check("M4's path vol is VIX / 100, the others the mark's implied vol",
      fc.model_vols(INP)["M4"][0] == 0.22 and all(fc.model_vols(INP)[m][0] == 0.166 for m in ("M1", "M2", "M3")))

record = fc.make_forecast(INP, T1, N)


def nav_at(S, name, tau1):
    """XX's model NAV at the target with SPY at S (the closed form the simulator reprices with)."""
    _pv, iv, slide = fc.model_vols(INP)[name]
    horizon = (fc.EXPIRY - INP.origin).days / 365.0
    q = INP.rate - math.log(INP.forward / INP.spot) / INP.tau
    m = sim.Market(INP.rate, q, horizon)
    fund = sim.Fund((sim.Leg("call", 1, INP.strike, INP.options_per_share),), cash=INP.cash_per_share, fee=fc.FEE)
    return sim.share_value(fund, S, tau1, sim.vol_at(iv, 0.0, slide, S, INP.spot), m), \
        sim.share_value(fund, INP.spot, 0.0, max(iv, sim.VOL_FLOOR), m)


worst = 0.0
# The exact lognormal answer, built here from the inputs with scipy's normal CDF (independent of forecast.spy_pit):
# ln(S_T / S_t) ~ N((r - q - vol^2 / 2) tau1, vol^2 tau1), tau1 = ACT/365 years from the origin to the target.
Q = INP.rate - math.log(INP.forward / INP.spot) / INP.tau
TAU1 = (T1 - INP.origin).days / 365.0
for name, m in record["models"].items():
    vol = INP.vix / 100.0 if name == "M4" else INP.implied_vol
    check(f"{name}: the saved SPY terms are the inputs' drift, path vol and years",
          abs(m["spy_mu"] - (INP.rate - Q)) < 1e-15 and m["spy_vol"] == vol and abs(m["years"] - TAU1) < 1e-15)
    for g in (0.8, 0.95, 1.0, 1.05, 1.25):
        lo, hi = INP.spot * 0.5, INP.spot * 1.5             # solve NAV(S*) / NAV(t) = g by bisection
        n0 = nav_at(INP.spot, name, TAU1)[1]
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if nav_at(mid, name, TAU1)[0] / n0 < g else (lo, mid)
        s_star = 0.5 * (lo + hi)
        exact = float(ndtr((math.log(s_star / INP.spot) - (INP.rate - Q - 0.5 * vol * vol) * TAU1) / (vol * math.sqrt(TAU1))))
        worst = max(worst, abs(fc.pit_from_levels(m["growth_levels"], g) - exact))
        if g == 1.05:
            check(f"{name}: forecast.spy_pit equals the independent formula", abs(fc.spy_pit(m, INP.spot, s_star) - exact) < 1e-12)
check(f"every model's forecast matches the exact answer at 5 growth levels (worst gap {worst:.4f}, allowance 0.004 "
      "= 4 SE at 200,000 paths + interpolation)", worst < 0.004)

check("SPY PIT is 0.5 at SPY's lognormal median", all(abs(fc.spy_pit(m, 100.0, 100.0 * math.exp(
    (m["spy_mu"] - 0.5 * m["spy_vol"] ** 2) * m["years"])) - 0.5) < 1e-12 for m in record["models"].values()))
check("PIT strictly outside the saved points is 0.0005 / 0.9995",
      fc.pit_from_levels([1.0, 2.0], 0.5) == 0.0005 and fc.pit_from_levels([1.0, 2.0], 3.0) == 0.9995)
lv = list(np.linspace(1.0, 2.0, 999))
check("PIT exactly at the first / last saved point interpolates (0.001 / 0.999), as rule 6 says",
      fc.pit_from_levels(lv, 1.0) == 0.001 and abs(fc.pit_from_levels(lv, 2.0) - 0.999) < 1e-15)
check("the saved forecast records the library versions and the SPY close's source",
      set(record["versions"]) == {"python", "numpy", "numba", "system"} and "spot_source" in record["inputs"])
check("the record round-trips through its canonical bytes with the same fingerprint",
      fc.fingerprint(json.loads(fc.canonical(record))) == fc.fingerprint(record))
check("999 saved points per model, increasing", all(len(m["growth_levels"]) == 999 and
      np.all(np.diff(m["growth_levels"]) >= 0) for m in record["models"].values()))

# ------------------------------------------------------------------ the log
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "log.jsonl"
    on_time = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)
    e = fc.append_log(fc.forecast_entry(record, code={"x": "0"}), log, now=on_time)
    check("forecast entry recorded on time, chain intact", e["late"] is False and ledger.verify(log) == (True, "chain intact"))
    check("the log keeps the date, never the clock time", e["recorded_utc_date"] == "2026-11-02"
          and "recorded_utc" not in e and "15:00" not in log.read_text(encoding="utf-8"))
    raises("the same forecast twice is refused", lambda: fc.append_log(fc.forecast_entry(record, code={}), log),
           "already in the log")
    other = dict(record, origin="2026-11-30", target="2026-12-31")
    raises("an outcome without its forecast is refused", lambda: fc.append_log(fc.outcome_entry(other, {}), log),
           "no forecast")
    bad = dict(record, origin="2026-10-29")
    raises("a non-origin is refused", lambda: fc.append_log(fc.forecast_entry(bad, code={}), log), "not an FC1 origin")
    raises("dry runs never write to the public log",
           lambda: fc.append_log(fc.forecast_entry(bad, code={}), fc.LOG_PATH, any_close=True), "never write")
    pits = fc.score(record, 26.0, 27.0, 778.01, 790.0)
    late = datetime(2026, 12, 9, 15, 0, tzinfo=timezone.utc)
    e2 = fc.append_log(fc.outcome_entry(record, pits), log, now=late)
    check("an outcome after its deadline is still recorded, marked late", e2["late"] is True and e2["pit"] == pits)
    check("log chain intact after two entries", ledger.verify(log) == (True, "chain intact"))
    raw = log.read_text(encoding="utf-8").replace('"late": false', '"late": true')
    log.write_text(raw, encoding="utf-8", newline="")
    check("an edited entry breaks the chain", ledger.verify(log)[0] is False)
    check("the log file is pure LF", b"\r\n" not in log.read_bytes())
    empty = Path(tmp) / "empty.jsonl"
    check("due: nothing before the first origin's close has passed", fc.due(date(2026, 10, 30), empty) == [])
    check("due: the first forecast is due (not late) on 2026-11-03",
          fc.due(date(2026, 11, 3), empty) == [("forecast", date(2026, 10, 30), date(2026, 10, 30), date(2026, 11, 3), False)])
    check("due: and late on 2026-11-04", fc.due(date(2026, 11, 4), empty)[0][4] is True)

entries = [{"kind": "outcome", "pit": {"M1": {"xx": p, "spy": 0.5}}} for p in (0.1, 0.3, 0.5, 0.7, 0.97)] + \
          [{"kind": "outcome", "void": "split"}]
s = fc.summary(entries)
r = s[("M1", "xx")]
check("summary: n 5 (the void skipped), mean 0.514, 3 of 5 inside 25-75%, 4 of 5 inside 5-95%",
      r["n"] == 5 and abs(r["mean"] - 0.514) < 1e-12 and r["in50"][0] == 0.6 and r["in90"][0] == 0.8
      and sum(r["histogram"]) == 5)
check("the Wilson interval brackets the share", r["in50"][1] < 0.6 < r["in50"][2])
