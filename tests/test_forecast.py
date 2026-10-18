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
check("on time until the END OF THE SESSION on the deadline day (rule 7; 16:00 New York = 21:00 UTC in November)",
      not fc.is_late(date(2026, 10, 30), datetime(2026, 11, 3, 20, 59, tzinfo=timezone.utc))
      and fc.is_late(date(2026, 10, 30), datetime(2026, 11, 3, 21, 1, tzinfo=timezone.utc)))
check("early closes (13:00): the day after Thanksgiving and Christmas Eve 2026; July 3 2026 is a holiday, not one",
      fc.is_early_close(date(2026, 11, 27)) and fc.is_early_close(date(2026, 12, 24))
      and not fc.is_early_close(date(2026, 7, 3)) and not fc.is_early_close(date(2026, 11, 30))
      and fc.close_time(date(2026, 11, 27)).hour == 13 and fc.close_time(date(2026, 11, 30)).hour == 16)
early = sorted({fc.deadline(c) for t, T in o for c in (t, T) if fc.is_early_close(fc.deadline(c))})
check("one FC1 deadline is an early close: 2029-07-03 (the day before Independence Day) ends at 13:00 New York",
      early == [date(2029, 7, 3)] and fc.close_time(date(2029, 7, 3)).hour == 13
      and fc.is_late(date(2029, 6, 29), datetime(2029, 7, 3, 17, 1, tzinfo=timezone.utc))
      and not fc.is_late(date(2029, 6, 29), datetime(2029, 7, 3, 16, 59, tzinfo=timezone.utc)), early)

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

# ------------------------------------------------------------------ fixes from the external audit of v0.3.0
nan = float("nan")
# D01: a missing or invalid number is refused, never scored or written
for label, args in (("a blank (NaN) target NAV", (26.0, nan, 778.01, 790.0)), ("a zero NAV", (26.0, 0.0, 778.01, 790.0)),
                    ("a negative SPY close", (26.0, 27.0, 778.01, -1.0))):
    raises(f"scoring refuses {label}", lambda a=args: fc.score(record, *a), "missing or invalid")
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "log.jsonl"
    fc.append_log(fc.forecast_entry(record, code={}), log, now=datetime(2026, 11, 2, 15, tzinfo=timezone.utc))
    raises("the log refuses to write a NaN (allow_nan=False)", lambda: fc.append_log(
        fc.outcome_entry(record, {"M1": {"xx": nan, "spy": 0.5}}), log,
        now=datetime(2026, 12, 2, 15, tzinfo=timezone.utc)), "json compliant")
    check("... and nothing was written, so the month can still be recorded", len(fc.read_log(log)) == 1)
    # D03: no entry before its session has ended
    raises("a forecast entry before the origin's session ends is refused (2026-10-30 15:00 New York)", lambda:
           fc.append_log(fc.forecast_entry(dict(record, origin="2026-10-30"), code={}), Path(tmp) / "x.jsonl",
                         now=datetime(2026, 10, 30, 19, tzinfo=timezone.utc)), "has not ended")
    # D19: a report after every 12 outcome entries, once, holding PITs and summaries only
    rlog = Path(tmp) / "r.jsonl"
    for t_, T_ in fc.origins()[:12]:
        rec_ = dict(record, origin=t_.isoformat(), target=T_.isoformat())
        fc.append_log(fc.forecast_entry(rec_, code={}), rlog, now=datetime(2030, 2, 1, tzinfo=timezone.utc))
        fc.append_log(fc.outcome_entry(rec_, {m: {"xx": 0.5, "spy": 0.5} for m in fc.MODELS}), rlog,
                      now=datetime(2030, 2, 1, tzinfo=timezone.utc))
    check("due lists rule 9's report after 12 outcome entries", ("report", 12, None, None, False) in fc.due(date(2030, 2, 1), rlog))
    rep = fc.append_log(fc.report_entry(fc.read_log(rlog)), rlog, now=datetime(2030, 2, 1, tzinfo=timezone.utc))
    check("the report entry holds the summary of all 12, and no NAV or return", rep["outcomes"] == 12
          and rep["summary"]["M1/xx"]["n"] == 12 and "nav" not in json.dumps(rep).lower())
    check("... and then no report is due", not any(k == "report" for k, *_ in fc.due(date(2030, 2, 1), rlog)))
    raises("the same report twice is refused", lambda: fc.append_log(fc.report_entry(fc.read_log(rlog)), rlog), "once")

# D01, D02, D04 through the command itself, on a made-up data folder
import importlib.util  # noqa: E402
import os  # noqa: E402

spec = importlib.util.spec_from_file_location("forecast_check_cli", Path(fc.REPO) / "scripts" / "forecast_check.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def data_folder(root, nav_T="27", split=False):
    (root / "tables").mkdir(parents=True, exist_ok=True)
    tau = (fc.EXPIRY - date(2026, 10, 30)).days / 365
    (root / "tables" / "xx_model_daily.csv").write_text(
        "as_of,spot,iv_disc_0bp,zero_rate_T,forward,tau,options_per_share\n"
        f"2026-10-30,778.01,0.166,0.0486,882.0,{tau},0.48\n2026-11-30,790.0,0.166,0.0486,890.0,3.1,0.48\n",
        encoding="utf-8")
    later = "2026-11-30,995.01,2030-01-10,0.4948,{},0.0096,50000\n" if split else "2026-11-30,995.01,2030-01-10,0.2474,{},0.0048,100000\n"
    (root / "tables" / "xx_state.csv").write_text(
        "as_of,strike,expiry,cash_per_share,nav_published,contracts_per_share,shares_outstanding\n"
        "2026-10-30,995.01,2030-01-10,0.2474,26,0.0048,100000\n" + later.format(nav_T), encoding="utf-8")


def args(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{"origin": "2026-10-30", "void": None, "no_events": False, "dry_run": True,
                              "target": None, "paths": fc.PATHS,
                              "now": datetime(2026, 12, 2, 15, tzinfo=timezone.utc), **kw})


old_env = os.environ.get("DEFINEDTARGET_DATA")
def q(fn, *a):
    """Run a command without its printout, so only the checks print."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


try:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rec0 = dict(record, origin="2026-10-30", target="2026-11-30")
        for case, nav_T, split in (("blank", "", False), ("split", "54", True), ("normal", "27", False)):
            root = tmp / case
            data_folder(root / "data", nav_T, split)
            os.environ["DEFINEDTARGET_DATA"] = str(root / "data")
            store = root / "store"
            store.mkdir()
            (store / "FC1_2026-10-30.json").write_bytes(fc.canonical(rec0))
            log = root / "log.jsonl"
            fc.append_log(fc.forecast_entry(rec0, code={}), log, now=datetime(2026, 11, 2, 15, tzinfo=timezone.utc))
            if True:
                if case == "blank":
                    raises("D01: the command refuses a blank target NAV (nothing recorded)",
                           lambda: q(cli.cmd_outcome, args(no_events=True), log, store), "missing or invalid")
                    check("... and the month can still be recorded later", len(fc.read_log(log)) == 1)
                elif case == "split":
                    check("D02: a reverse split between the closes blocks scoring, even with --no-events",
                          q(cli.cmd_outcome, args(no_events=True), log, store) == 2 and len(fc.read_log(log)) == 1)
                    check("... and found the event in the data", any("reverse split" in e for _, e in
                                                                    fc.events_between(date(2026, 10, 30), date(2026, 11, 30))[0]))
                    rc = q(cli.cmd_outcome, args(void="1-for-2 reverse split (made-up notice)"), log, store)
                    last = fc.read_log(log)[-1]
                    check("... and --void records the void with its reason and no PIT", rc == 0 and "void" in last
                          and "pit" not in last)
                else:
                    check("D02: without --no-events (the operator's review) nothing is scored",
                          q(cli.cmd_outcome, args(), log, store) == 2 and len(fc.read_log(log)) == 1)
                    rc = q(cli.cmd_outcome, args(no_events=True), log, store)
                    last = fc.read_log(log)[-1]
                    check("... with it, the PITs are recorded with the review", rc == 0 and "pit" in last
                          and "operator confirmed" in last["events_review"])
        # D04: a saved forecast whose log write failed is logged on the next run, from its saved bytes
        root = tmp / "orphan"
        data_folder(root / "data")
        os.environ["DEFINEDTARGET_DATA"] = str(root / "data")
        store, log = root / "store", root / "log.jsonl"
        store.mkdir()
        (store / "FC1_2026-10-30.json").write_bytes(fc.canonical(rec0))
        rc = q(cli.cmd_forecast, args(), log, store)
        rc2 = q(cli.cmd_forecast, args(), log, store)
        check("D04: an orphaned saved forecast is logged once from its saved bytes, then refused",
              rc == 0 and rc2 == 2 and fc.read_log(log)[0]["forecast_sha256"] == fc.fingerprint(rec0))

        # Re-check of v0.3.0.1: guards that must not block an ordinary month, and voids that must not slip through
        def state(root, rows):
            (root / "tables").mkdir(parents=True, exist_ok=True)
            (root / "tables" / "xx_state.csv").write_text(
                "as_of,strike,expiry,cash_per_share,nav_published,contracts_per_share,shares_outstanding\n" +
                "".join(f"{d},{k},2030-01-10,0.2474,26,{c},100000\n" for d, k, c in rows), encoding="utf-8")
            os.environ["DEFINEDTARGET_DATA"] = str(root)

        state(tmp / "glitch", [("2026-10-30", 995.01, 0.0048), ("2026-11-10", 995.01, 0.0096),
                               ("2026-11-11", 995.01, 0.0048), ("2026-11-30", 995.01, 0.0048)])
        ev, notes = fc.events_between(date(2026, 10, 30), date(2026, 11, 30))
        check("a one-session doubling that moves back (a duplicated option line) is a note, not a void event",
              ev == [] and len(notes) == 1 and "glitch" in notes[0], (ev, notes))
        state(tmp / "persist", [("2026-10-30", 995.01, 0.0048), ("2026-11-10", 995.01, 0.0096),
                                ("2026-11-11", 995.01, 0.0096), ("2026-11-30", 995.01, 0.0096)])
        check("a doubling that stays is a void event", len(fc.events_between(date(2026, 10, 30), date(2026, 11, 30))[0]) == 1)
        state(tmp / "rollT", [("2026-10-30", 995.01, 0.0048), ("2026-11-30", 1100.0, 0.0048),
                              ("2026-12-01", 1100.0, 0.0048)])
        check("a roll first seen on the target close T is found (the next close confirms it)",
              any("option line changed" in e for _, e in fc.events_between(date(2026, 10, 30), date(2026, 11, 30))[0]))
        D_ = (date(2026, 10, 30), date(2026, 11, 30))
        state(tmp / "S3", [("2026-10-30", 995.01, 0.0048), ("2026-11-27", 995.01, 0.0024),
                           ("2026-11-30", 995.01, 0.0048), ("2026-12-01", 995.01, 0.0024)])
        check("a real halving on T-1 followed by one bad file on T is still a void event (glitch not confirmed)",
              len(fc.events_between(*D_)[0]) >= 1, fc.events_between(*D_))
        state(tmp / "S4", [("2026-10-30", 995.01, 0.0048), ("2026-11-30", 995.01, 0.0024),
                           ("2026-12-01", 995.01, 0.0048)])
        check("a real halving ON T followed by one bad file on T+1 is still a void event",
              len(fc.events_between(*D_)[0]) >= 1, fc.events_between(*D_))
        state(tmp / "rollT_only", [("2026-10-30", 995.01, 0.0048), ("2026-11-30", 1100.0, 0.0048)])
        check("a roll seen only on T, before the next file, is an unconfirmed event, not silence",
              any("unconfirmed" in e for _, e in fc.events_between(*D_)[0]), fc.events_between(*D_))
        state(tmp / "glitch_t", [("2026-10-29", 995.01, 0.0048), ("2026-10-30", 995.01, 0.0096),
                                 ("2026-11-02", 995.01, 0.0048), ("2026-11-03", 995.01, 0.0048),
                                 ("2026-11-30", 995.01, 0.0048), ("2026-12-01", 995.01, 0.0048)])
        check("a confirmed glitch on the origin day itself is a note, not a false void the next day",
              fc.events_between(*D_)[0] == [] and len(fc.events_between(*D_)[1]) == 1, fc.events_between(*D_))
        gl = tmp / "glitch_cli"
        data_folder(gl / "data")
        (gl / "data" / "tables" / "xx_state.csv").write_text(
            "as_of,strike,expiry,cash_per_share,nav_published,contracts_per_share,shares_outstanding\n"
            "2026-10-30,995.01,2030-01-10,0.2474,26,0.0048,100000\n2026-11-10,995.01,2030-01-10,0.2474,26,0.0096,100000\n"
            "2026-11-11,995.01,2030-01-10,0.2474,26,0.0048,100000\n2026-11-12,995.01,2030-01-10,0.2474,26,0.0048,100000\n"
            "2026-11-30,995.01,2030-01-10,0.2474,27,0.0048,100000\n", encoding="utf-8")
        os.environ["DEFINEDTARGET_DATA"] = str(gl / "data")
        (gl / "store").mkdir()
        (gl / "store" / "FC1_2026-10-30.json").write_bytes(fc.canonical(rec0))
        fc.append_log(fc.forecast_entry(rec0, code={}), gl / "log.jsonl", now=datetime(2026, 11, 2, 15, tzinfo=timezone.utc))
        import contextlib as _cl, io as _io
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            cli.cmd_outcome(args(), gl / "log.jsonl", gl / "store")
        check("the data's notes are printed before the operator is asked to confirm", "glitch" in _buf.getvalue())
        root = tmp / "lost"
        data_folder(root / "data")
        os.environ["DEFINEDTARGET_DATA"] = str(root / "data")
        store, log = root / "store", root / "log.jsonl"
        store.mkdir()
        fc.append_log(fc.forecast_entry(rec0, code={}), log, now=datetime(2026, 11, 2, 15, tzinfo=timezone.utc))
        check("a logged forecast whose private file is lost is never recomputed (the command stops at once)",
              q(cli.cmd_forecast, args(), log, store) == 2 and not (store / "FC1_2026-10-30.json").exists())
        check("... and outcome says the file is missing instead of crashing",
              q(cli.cmd_outcome, args(no_events=True), log, store) == 2)
finally:
    if old_env is None:
        os.environ.pop("DEFINEDTARGET_DATA", None)
    else:
        os.environ["DEFINEDTARGET_DATA"] = old_env
check("D06: the forecast entry fingerprints the loaded input code too (parse, fetch, marks)",
      {"definedtarget/forward/parse.py", "definedtarget/forward/fetch.py"} <= set(fc.code_fingerprints()))

# Re-check of v0.3.0.1: extreme but real months are recorded, not refused
crash = fc.score(record, 26.0, 2.6, 778.0, 778.0 * 0.83)          # SPY -17% in a month (October 2008: -16.9%)
check("a -17% SPY month is scored: M1-M3's SPY PITs round to 0.0 (a valid rule 6 value), not refused",
      all(crash[m]["spy"] == 0.0 for m in ("M1", "M2", "M3")) and all(0.0 <= p[k] <= 1.0 for p in crash.values()
                                                                    for k in ("xx", "spy")), crash)
check("due marks an entry late after the deadline session's end, when given the time",
      fc.due(date(2026, 11, 3), Path(tempfile.gettempdir()) / "no_such_fc1_log.jsonl",
             now=datetime(2026, 11, 3, 21, 1, tzinfo=timezone.utc))[0][4] is True)
check("the saved forecast carries the code that computed it, and its log entry names that code",
      "code_sha256" in record and fc.forecast_entry(record)["code_sha256"] == record["code_sha256"])
