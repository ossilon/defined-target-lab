#!/usr/bin/env python3
"""The monthly forecast check (protocol FC1; definedtarget/forward/forecast.py).  From target_funds/:

    py -I scripts\\forecast_check.py due                      what is due or missing today (reports included)
    py -I scripts\\forecast_check.py forecast --origin 2026-10-30
    py -I scripts\\forecast_check.py outcome --origin 2026-10-30 --no-events      (or --void "event and source")
    py -I scripts\\forecast_check.py report                   rule 9's report, when `due` lists it
    py -I scripts\\forecast_check.py summary

forecast: computes the four models' forecasts from the origin close's data, saves the record privately in
data/forecasts/ (it holds issuer-derived numbers; written once, never overwritten; back it up: the public
log is useless without it) and appends its SHA-256 to the
public log forward/forecast_log.jsonl. If the log write failed after the save, running it again logs the saved file
(never a new forecast).  outcome: reads the SAVED forecast (its fingerprint must match the log), checks the month
for void events (rule 8: the data's own checks, plus the operator's confirmation: --no-events after reviewing the
issuer's notices, or --void "..."), then appends each model's PITs, or the void and its reason.  report: appends
rule 9's summary to the log after every 12 outcome entries and after the last.  summary: prints that table.

--dry-run --log PATH --store DIR [--target YYYY-MM-DD]: the same steps on any close, into a scratch log and folder,
for testing; dry runs never write to the public log.
Research, not investment advice.
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget.forward import fetch  # noqa: E402
from definedtarget.forward import forecast as fc  # noqa: E402


def _now(a):
    """The time of this run (tests pass a fixed one as a.now)."""
    return getattr(a, "now", None) or datetime.now(timezone.utc)


def store_path(store, origin):
    return Path(store) / f"FC1_{origin}.json"


def cmd_forecast(a, log, store):
    t = date.fromisoformat(a.origin)
    T = date.fromisoformat(a.target) if a.dry_run and a.target else fc.target_for(t)
    path = store_path(store, t)
    ok, msg = fc.ledger.verify(log)                      # check the log before any work (audit of v0.3.0, D04)
    if not ok:
        print(f"the forecast log is damaged ({msg}); nothing computed")
        return 2
    logged = any(e["kind"] == "forecast" and e.get("origin") == t.isoformat() for e in fc.read_log(log))
    if path.exists():
        if logged:
            print(f"{path.name} already exists and is logged: a forecast is written once (never recomputed over)")
            return 2
        # Saved but never logged (the log write failed): log the SAVED bytes once; never recompute (D04)
        raw = path.read_bytes()
        record = json.loads(raw)
        if fc.canonical(record) != raw:
            print(f"{path.name} is not a forecast this tool wrote (bytes are not canonical); nothing recorded")
            return 2
        entry = fc.append_log(fc.forecast_entry(record), log, now=_now(a), any_close=a.dry_run)
        print(f"recovered: logged the saved {path.name} ({entry['row_hash'][:12]}; late: {entry['late']})")
        return 0
    if logged:   # the log has it but the private file is gone: a new computation could differ (re-check of v0.3.0.1)
        print(f"the log already holds the forecast for {t}, but {path} is missing: restore it from your backup of "
              "data/forecasts/ (never recompute a logged forecast); nothing done")
        return 2
    if fc.is_late(t, _now(a)):
        print("note: past the deadline; the entry will be marked late (it still counts)")
    record = fc.make_forecast(fc.load_inputs(t), T, a.paths, provenance=fc.input_provenance(t))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:                         # exclusive create: never overwrite a saved forecast
        fh.write(fc.canonical(record))
    entry = fc.append_log(fc.forecast_entry(record), log, now=_now(a), any_close=a.dry_run)
    print(f"forecast {t} -> {T}: saved {path.name}; log entry {entry['row_hash'][:12]} (late: {entry['late']})")
    for name, m in record["models"].items():
        lv = m["growth_levels"]
        print(f"  {name}: path vol {m['spy_vol']:.4f}; growth p5 / p50 / p95 = {lv[49]:.4f} / {lv[499]:.4f} / "
              f"{lv[949]:.4f}" + ("  (private: not for publishing)" if not a.dry_run else ""))
    return 0


def cmd_outcome(a, log, store):
    t = date.fromisoformat(a.origin)
    path = store_path(store, t)
    if not path.exists():
        print(f"{path} is missing: restore the saved forecast from your backup of data/forecasts/; nothing recorded")
        return 2
    record = json.loads(path.read_bytes())
    logged = [e for e in fc.read_log(log) if e["kind"] == "forecast" and e["origin"] == t.isoformat()]
    if not logged:
        print(f"no forecast for {t} in {log}")
        return 2
    if fc.fingerprint(record) != logged[0]["forecast_sha256"]:
        print(f"{path.name} does not match its logged fingerprint: the saved forecast changed; nothing recorded")
        return 2
    T = date.fromisoformat(record["target"])
    # FC1 rule 8: a month is void after a split, reverse split, distribution, roll, closure notice or option change.
    # The data shows some of these; the operator confirms the rest (audit of v0.3.0, D02).
    try:
        found, notes = fc.events_between(t, T)
    except ValueError as err:          # e.g. a void_events.csv row without its source
        print(f"cannot check the month for void events: {err}; nothing recorded")
        return 2
    if a.void:
        entry = fc.outcome_entry(record, void=a.void, review="void, recorded by the operator")
    elif found:
        print(f"void events between {t} and {T}: " + "; ".join(f"{d} {e}" for d, e in found))
        print('this month is void (FC1 rule 8): rerun with --void "<the event and its source>"; nothing recorded')
        return 2
    elif not a.no_events:
        print(f"review the issuer's notices for {t} to {T} first: a split, reverse split, distribution, roll, "
              "closure notice or option change makes the month void (FC1 rule 8). Then rerun with --no-events "
              '(none happened) or --void "<event and source>"; nothing recorded')
        return 2
    else:
        nav_t, spy_t = fc.closes(t)
        nav_T, spy_T = fc.closes(T)
        entry = fc.outcome_entry(record, fc.score(record, nav_t, nav_T, spy_t, spy_T),
                                 review="none: the data showed no void event and the operator confirmed none"
                                 + ("; noted: " + "; ".join(notes) if notes else ""))
    entry = fc.append_log(entry, log, now=_now(a), any_close=a.dry_run)
    print(f"outcome {t} -> {T}: " + (f"void ({a.void})" if a.void else json.dumps(entry["pit"])) +
          f"; late: {entry['late']}")
    return 0


def cmd_summary(_a, log, _store):
    s = fc.summary(fc.read_log(log))
    if not s:
        print("no scored months yet")
        return 0
    print("| model | PIT of | n | mean (0.5 if honest) | inside 25-75% (50%) | inside 5-95% (90%) | histogram, 10 bins |")
    print("|---|---|---|---|---|---|---|")
    for (name, kind), r in s.items():
        b50, b90 = r["in50"], r["in90"]
        print(f"| {name} | {kind} | {r['n']} | {r['mean']:.3f} | {b50[0]:.0%} ({b50[1]:.0%}-{b50[2]:.0%}) | "
              f"{b90[0]:.0%} ({b90[1]:.0%}-{b90[2]:.0%}) | {r['histogram']} |")
    return 0


def cmd_report(a, log, _store):
    """Rule 9: append the summary of every scored outcome so far, when `due` lists a report."""
    entries = fc.read_log(log)
    if not any(k == "report" for k, *_ in fc.due(datetime.now(timezone.utc).astimezone(fc.NEW_YORK).date(), log)):
        print("no report is due (one is due after every 12 outcome entries and after the last)")
        return 2
    entry = fc.append_log(fc.report_entry(entries), log, now=_now(a), any_close=a.dry_run)
    print(f"report after {entry['outcomes']} outcome entries ({entry['voids']} void) appended: {entry['row_hash'][:12]}")
    return cmd_summary(a, log, _store)


def cmd_due(_a, log, store):
    today = datetime.now(timezone.utc).astimezone(fc.NEW_YORK).date()
    items = fc.due(today, log, now=_now(_a))
    for kind, t, close, dl, late in items:
        if kind == "report":
            print(f"report due after {t} outcome entries: py -I scripts\\forecast_check.py report")
        else:
            print(f"{kind} for origin {t} (close {close}): {'LATE' if late else 'due'}, deadline end of {dl} "
                  f"({fc.close_time(dl):%H:%M} New York)")
    if not items:
        print("nothing due")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("due", "forecast", "outcome", "report", "summary"))
    ap.add_argument("--origin")
    ap.add_argument("--void")
    ap.add_argument("--no-events", action="store_true",
                    help="outcome: you reviewed the issuer's notices and no void event (rule 8) happened")
    ap.add_argument("--paths", type=int, default=fc.PATHS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--store")
    ap.add_argument("--target")
    a = ap.parse_args(argv)
    if a.dry_run and not (a.log and a.store):
        ap.error("--dry-run needs --log and --store (a scratch log and folder)")
    if not a.dry_run and (a.log or a.store or a.target or a.paths != fc.PATHS):
        ap.error("--log, --store, --target and --paths are for dry runs only (FC1 fixes them)")
    log = Path(a.log) if a.dry_run else fc.LOG_PATH
    store = Path(a.store) if a.dry_run else fetch.data_dir() / "forecasts"
    if a.cmd in ("forecast", "outcome") and not a.origin:
        ap.error("--origin is required")
    if a.void and a.no_events:
        ap.error("--void and --no-events contradict each other")
    return {"due": cmd_due, "forecast": cmd_forecast, "outcome": cmd_outcome, "report": cmd_report,
            "summary": cmd_summary}[a.cmd](a, log, store)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
