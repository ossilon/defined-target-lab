#!/usr/bin/env python3
"""The monthly forecast check (protocol FC1; definedtarget/forward/forecast.py).  From target_funds/:

    py -I scripts\\forecast_check.py due                      what is due or missing today
    py -I scripts\\forecast_check.py forecast --origin 2026-10-30
    py -I scripts\\forecast_check.py outcome --origin 2026-10-30 [--void "reason"]
    py -I scripts\\forecast_check.py summary

forecast: computes the four models' forecasts from the origin close's data, saves the record privately in
data/forecasts/ (it holds issuer-derived numbers; written once, never overwritten) and appends its SHA-256 to the
public log forward/forecast_log.jsonl.  outcome: reads the SAVED forecast (its fingerprint must match the log), the
issuer's NAVs and SPY's closes at both ends, and appends each model's PITs (or a void with its reason, rule 7).
summary: rule 8's table (PITs only).

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


def store_path(store, origin):
    return Path(store) / f"FC1_{origin}.json"


def cmd_forecast(a, log, store):
    t = date.fromisoformat(a.origin)
    T = date.fromisoformat(a.target) if a.dry_run and a.target else fc.target_for(t)
    path = store_path(store, t)
    if path.exists():
        print(f"{path.name} already exists: a forecast is written once (never recomputed over)")
        return 2
    record = fc.make_forecast(fc.load_inputs(t), T, a.paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:                         # exclusive create: never overwrite a saved forecast
        fh.write(fc.canonical(record))
    entry = fc.append_log(fc.forecast_entry(record), log, any_close=a.dry_run)
    print(f"forecast {t} -> {T}: saved {path.name}; log entry {entry['row_hash'][:12]} (late: {entry['late']})")
    for name, m in record["models"].items():
        lv = m["growth_levels"]
        print(f"  {name}: path vol {m['spy_vol']:.4f}; growth p5 / p50 / p95 = {lv[49]:.4f} / {lv[499]:.4f} / "
              f"{lv[949]:.4f}" + ("  (private: not for publishing)" if not a.dry_run else ""))
    return 0


def cmd_outcome(a, log, store):
    t = date.fromisoformat(a.origin)
    path = store_path(store, t)
    record = json.loads(path.read_bytes())
    logged = [e for e in fc.read_log(log) if e["kind"] == "forecast" and e["origin"] == t.isoformat()]
    if not logged:
        print(f"no forecast for {t} in {log}")
        return 2
    if fc.fingerprint(record) != logged[0]["forecast_sha256"]:
        print(f"{path.name} does not match its logged fingerprint: the saved forecast changed; nothing recorded")
        return 2
    T = date.fromisoformat(record["target"])
    if a.void:
        entry = fc.outcome_entry(record, void=a.void)
    else:
        nav_t, spy_t = fc.closes(t)
        nav_T, spy_T = fc.closes(T)
        entry = fc.outcome_entry(record, fc.score(record, nav_t, nav_T, spy_t, spy_T))
    entry = fc.append_log(entry, log, any_close=a.dry_run)
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


def cmd_due(_a, log, store):
    today = datetime.now(timezone.utc).astimezone(fc.NEW_YORK).date()
    items = fc.due(today, log)
    for kind, t, close, dl, late in items:
        print(f"{kind} for origin {t} (close {close}): {'LATE' if late else 'due'}, deadline {dl}")
    if not items:
        print("nothing due")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("due", "forecast", "outcome", "summary"))
    ap.add_argument("--origin")
    ap.add_argument("--void")
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
    return {"due": cmd_due, "forecast": cmd_forecast, "outcome": cmd_outcome, "summary": cmd_summary}[a.cmd](a, log, store)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
