"""The daily forward-test routine (scripts/daily.py, the CLI and the scheduled task run it).

1. Capture the issuer's files and the public series.
2. Rebuild the tables and check the accounting identities.
3. Price the option marks and write the daily report (<report dir>/<close date>_report.md and .html) plus a short
   note (<report dir>/<close date>.md). Default report dir: <repo>/forward/daily (git-ignored: issuer-derived).
4. Return STOP messages for anything that needs a human: a latest-day file failed, an identity broke, the
   holdings schema changed, contracts per share moved without a flow, the option line changed (a roll), a fund
   document changed (supplement, new target period), or the engine refused a stale or missing input.
"""
from datetime import datetime, timezone
from pathlib import Path

from .. import curves, marks, report, spec
from . import external, fetch, parse, reconcile, roundhill

REPO = Path(__file__).resolve().parents[2]
EXPECTED_OPTION = (995.01, "2030-01-10")


def run(root=None, report_dir=None, now=None, capture=True):
    """Run the routine. Returns (note_lines, stops)."""
    now = now or datetime.now(timezone.utc)
    root = fetch.data_dir(root)
    report_dir = Path(report_dir) if report_dir else REPO / "forward" / "daily"
    stops, notes, records = [], [], []

    if capture:
        records = roundhill.capture_all(root=root, now=now) + external.capture_all(root=root, now=now)
        records += parse.snapshot_external(root=root, now=now)   # dated copies of the hand-maintained inputs (P2)
    for r in records:
        bad = r["result"] in ("network_error", "invalid", "invalid_repeat", "http_error")
        if bad and r["source"] == roundhill.SOURCE and not r["name"].startswith("holdings_"):
            stops.append(f"latest-day file failed: {r['name']} ({r.get('error') or r.get('status')})")
        elif bad and not r["name"].startswith("holdings_"):
            notes.append(f"public series did not refresh: {r['name']}")
        if r["result"] == "saved" and r["name"].endswith(".pdf"):
            earlier = [x for x in fetch.read_manifest(root)
                       if x["url"] == r["url"] and x["result"] == "saved" and x["fetched_utc"] < r["fetched_utc"]]
            if earlier:
                stops.append(f"fund document changed: {r['name']} (the capture saved the new version; read it)")
        if r["result"] == "saved" and r["name"].startswith("holdings_") and "__" in r.get("path", ""):
            notes.append(f"the issuer revised {r['name']}; the capture kept both versions")

    try:
        xx = parse.load_holdings(root, accounts=("XX",))
        state = reconcile.fund_state(xx, parse.load_fund_daily(root, "XX"))
    except (ValueError, FileNotFoundError) as err:          # schema change, nothing captured yet, ...
        stops.append(f"parse refused: {err}")
        state = None

    note = []
    if state is not None and not state.empty:
        out = root / "tables"
        out.mkdir(parents=True, exist_ok=True)
        xx.to_csv(out / "xx_holdings.csv", index=False)
        state.to_csv(out / "xx_state.csv", index=False)
        unexplained, _ = reconcile.failures(state)
        stops += [f"identity failed: {d} {c}" for d, c in unexplained]
        last, prev = state.iloc[-1], state.iloc[-2] if len(state) > 1 else None
        if (round(last.strike, 2), str(last.expiry)) != EXPECTED_OPTION:
            stops.append(f"option line changed: strike {last.strike} expiry {last.expiry} (roll or new period?)")
        if prev is not None and abs(last.contracts_per_share - prev.contracts_per_share) > 1e-9 and not last.flow_day:
            stops.append(f"contracts per share moved without a flow: {prev.contracts_per_share:.7f} -> "
                         f"{last.contracts_per_share:.7f} (fee sale? see H2)")
        # Pricing and the daily report. The engine refuses any stale or missing input (curve > 5 days old,
        # dividends > 100 days old, a mark with no implied vol); each refusal becomes a STOP here. An earlier close
        # with no SPY price is skipped and listed in a note (a new user has prices only from their first capture);
        # the LATEST close without one is a STOP, because its report cannot be written.
        model_line = "- model: not run"
        report_dir.mkdir(parents=True, exist_ok=True)
        try:
            vm = parse.load_holdings(root, accounts=None, all_versions=False)
            vm = vm[vm["is_option"] & (vm["opt_underlying"] == "SPY")]
            model = marks.model_daily(state, curves.load_treasury(root), marks.load_dividends(root),
                                      marks.load_spots(root), vm, skip_missing_spot=True)
            skipped = model.attrs["skipped_no_spot"]
            if last.as_of in skipped:
                raise ValueError(f"no SPY close for the latest close {last.as_of} (the issuer's reference file "
                                 "did not arrive; add it to data/external/closes.csv)")
            if skipped:
                notes.append(f"priced {len(model)} of {len(state)} closes; no SPY close for {len(skipped)} earlier "
                             f"close(s) ({skipped[0]} to {skipped[-1]}): add official closes to "
                             "data/external/closes.csv to price them (data/README.md)")
            model.to_csv(out / "xx_model_daily.csv", index=False)
            m = model.iloc[-1]
            fund_spec = spec.load("XX")
            # Write the latest close's report every run (it may change if the issuer revises a file), and any
            # earlier close that has no report yet: this backfills launch-to-date and catches up after missed days.
            for day in model["as_of"]:
                md_path = report_dir / f"{day}_report.md"
                if day != last.as_of and md_path.exists():
                    continue
                md_text = report.build(state, model, fund_spec, as_of=day)
                md_path.write_text(md_text, encoding="utf-8")
                (report_dir / f"{day}_report.html").write_text(
                    report.to_html(md_text, f"XX daily report {day}"), encoding="utf-8")
            model_line = (f"- model: implied vol {m.iv_disc_0bp:.2%}; leverage {m.xx_elasticity:.1f}x; "
                          f"vega {m.xx_vega_pct_nav_per_pt:.1%} of NAV per vol point; risk-neutral odds above strike "
                          f"{m.prob_itm_rn:.0%}; report {report_dir.name}/{last.as_of}_report.html")
        except (ValueError, FileNotFoundError) as err:
            stops.append(f"pricing refused: {err}")
        # The monthly forecast check (FC1): an entry is due once its close has passed (a note), and a STOP once it
        # is past its deadline, so a month cannot slip by unnoticed. Recording stays a manual command.
        try:
            from . import forecast
            for kind, t, close, dl, late in forecast.due(now.astimezone(forecast.NEW_YORK).date()):
                msg = (f"FC1 {kind} for origin {t} (close {close}) {'is past its deadline ' if late else 'is due by '}"
                       f"{dl}: py -I scripts\\forecast_check.py {kind} --origin {t}")
                (stops if late else notes).append(msg)
        except (ValueError, FileNotFoundError) as err:
            stops.append(f"forecast check refused: {err}")
        note = [
            f"# XX daily note, close {last.as_of}",
            "",
            f"- NAV {last.nav_published:.2f}; market {last.market_price:.2f}; premium {last.premium:+.2%}",
            f"- shares {last.shares_outstanding:,.0f} ({'flow day' if last.flow_day else 'no flow'}); "
            f"contracts {last.contracts:,.0f}; per share {last.contracts_per_share:.7f}",
            f"- option mark {last.mark_text} ({last.mark_decimals} decimals); cash {last.cash_weight:.2%} of net assets",
            f"- identities: {'all hold' if not unexplained else 'FAILED'}; lines gap ${last.lines_gap_usd:,.2f}",
            model_line,
            "- captured: " + (", ".join(f"{r['name']}={r['result']}" for r in records) or "skipped"),
        ] + [f"- note: {n}" for n in notes] + [f"- STOP: {s}" for s in stops]
        (report_dir / f"{last.as_of}.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    return note, stops
