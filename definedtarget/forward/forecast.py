"""The monthly forecast check (protocol FC1): the engine's one-month forecast of XX, scored when the month is over.

Plain words
- At the last NYSE session of each month (the origin, t) the engine forecasts XX's growth to the last session of the
  next month (the target, T): NAV(T) / NAV(t), as a full distribution, under four models (MODELS).
- The forecast is fixed by the frozen rules and data dated t or earlier, so computing it a day late changes nothing.
  It is saved privately (it holds issuer-derived numbers) and its SHA-256 fingerprint goes into the public,
  hash-chained log `forward/forecast_log.jsonl` at once.
- When the month is over, the PIT (probability integral transform) is read off the SAVED forecast: the share of the
  forecast below what happened. Honest forecasts give PITs spread evenly between 0 and 1. The SPY PIT does the same
  for SPY's own month under each model's path vol (a closed form), which tells a wrong vol input (both PITs off) from
  a wrong repricing of XX (only XX's PIT off).
- Log only, no verdict. Public files hold PITs and summaries, never NAVs or returns.

Models (all: SPY lognormal at the risk-neutral drift on the real NYSE calendar; XX repriced with sim.py's model):
  M1 path vol = the mark's implied vol, slide 1 vol point per -10% of SPY (the simulator's rule, FINDINGS F12)
  M2 path vol = the mark's implied vol, slide 0
  M3 path vol = the mark's implied vol, slide 3.5 points per -10% (the slide measured on real VIX1Y paths, F18)
  M4 path vol = that close's VIX / 100 (the market's 30-day vol), slide 1 point per -10% (one change from M1)
"""
import calendar as _cal
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .. import calendar as nyse
from .. import ledger, pricing, sim
from .. import xxpaths as xp
from ..montecarlo import paths as lp

REPO = Path(__file__).resolve().parents[2]
LOG_PATH = REPO / "forward" / "forecast_log.jsonl"
NEW_YORK = ZoneInfo("America/New_York")
FIRST_ORIGIN = date(2026, 10, 30)
LAST_ORIGIN = date(2029, 12, 31)
EXPIRY = date(2030, 1, 10)
FEE = 0.0099
PATHS = 1_000_000
LEVELS = [k / 1000 for k in range(1, 1000)]       # the saved forecast: its 0.1% ... 99.9% points
MODELS = {"M1": ("mark", 0.01), "M2": ("mark", 0.0), "M3": ("mark", 0.035), "M4": ("vix", 0.01)}
DEADLINE_SESSIONS = 2                             # entries are due by the end of the 2nd NYSE session after the close
CLOSE_HOUR, EARLY_CLOSE_HOUR = 16, 13             # NYSE session end, New York time (early closes at 13:00)
REPORT_EVERY = 12                                 # rule 9: a report after every 12 outcome entries and at the end
OPTIONS_JUMP = 0.05                               # options per share moving more than 5% between closes: an event
# Code files whose fingerprints go into each forecast entry; every definedtarget module loaded by the run is added
# too (audit of v0.3.0, D06: the input-preparation code as well as the computation).
CODE_FILES = ("definedtarget/forward/forecast.py", "definedtarget/xxpaths.py", "definedtarget/sim.py",
              "definedtarget/pricing.py", "definedtarget/calendar.py", "definedtarget/montecarlo/paths.py",
              "definedtarget/montecarlo/rng.py")


@dataclass(frozen=True)
class Inputs:
    """Everything a forecast uses, all dated at the origin close."""
    origin: date
    spot: float               # SPY reference close
    implied_vol: float        # the mark's base-case implied vol (iv_disc_0bp)
    rate: float               # zero rate to expiry (continuous)
    forward: float            # SPY forward to expiry (discrete dividends)
    tau: float                # years to expiry (ACT/365)
    strike: float
    options_per_share: float  # one-share options per XX share
    cash_per_share: float
    vix: float                # VIX close (index points), for M4
    spot_source: str = ""     # where the model table's SPY close came from (the issuer's reference file, or a fallback)


def last_session(year, month):
    d = date(year, month, _cal.monthrange(year, month)[1])
    while not nyse.is_session(d):
        d = date.fromordinal(d.toordinal() - 1)
    return d


def origins():
    """[(t, T)]: every origin close and its target close (rule 1)."""
    out, y, m = [], FIRST_ORIGIN.year, FIRST_ORIGIN.month
    while (y, m) <= (LAST_ORIGIN.year, LAST_ORIGIN.month):
        t = last_session(y, m)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((t, EXPIRY if t == LAST_ORIGIN else last_session(ny, nm)))
        y, m = ny, nm
    return out


def target_for(t):
    for o, T in origins():
        if o == t:
            return T
    raise ValueError(f"{t} is not an FC1 origin (the last NYSE session of a month, {FIRST_ORIGIN} to {LAST_ORIGIN})")


def seed_for(t):
    return t.year * 100 + t.month


def deadline(close):
    """The session by whose end an entry about `close` is on time (the 2nd NYSE session after it)."""
    d = close
    for _ in range(DEADLINE_SESSIONS):
        d = nyse.next_session(d)
    return d


def is_early_close(d):
    """NYSE early-close days (13:00), rules as of 2026: the day after Thanksgiving, Christmas Eve and July 3, when
    each is a session."""
    if not nyse.is_session(d):
        return False
    thanksgiving = date(d.year, 11, 22 + (3 - date(d.year, 11, 22).weekday()) % 7)   # 4th Thursday of November
    return d in (date.fromordinal(thanksgiving.toordinal() + 1), date(d.year, 12, 24), date(d.year, 7, 3))


def close_time(d):
    """The end of session d, New York time (FC1 rule 7: entries are due by the END of the session, 16:00 or 13:00)."""
    hour = EARLY_CLOSE_HOUR if is_early_close(d) else CLOSE_HOUR
    return datetime(d.year, d.month, d.day, hour, tzinfo=NEW_YORK)


def is_late(close, now_utc):
    """After the end of the deadline session (audit of v0.3.0, D03: the session's end, not midnight)."""
    return now_utc.astimezone(NEW_YORK) > close_time(deadline(close))


def _grid(t, T, full=False):
    """(times, step of T): the sessions from t to T, then one step to expiry (the simulator needs the target date,
    and NAV at T depends only on the path up to T). The engine's draws are addressed by (seed, path, step), so steps
    up to T are the same numbers as on the full calendar to expiry (full=True; tests/test_forecast.py checks it)."""
    sessions = nyse.sessions(t, EXPIRY if full else T)
    if sessions[-1] != EXPIRY:
        sessions.append(EXPIRY)
    return lp.times_from_dates(sessions), sessions.index(T)


def model_vols(inp):
    """{model: (path vol, implied vol, slide)}."""
    return {name: (inp.vix / 100.0 if src == "vix" else inp.implied_vol, inp.implied_vol, slide)
            for name, (src, slide) in MODELS.items()}


def simulate_growth(inp, T, name, npaths=PATHS, full=False):
    """Model growth NAV(T) / NAV(t) on each path (model NAVs at both ends, so the model's gap from the issuer's NAV
    at t cancels), the risk-neutral drift and the years from t to T."""
    times, k = _grid(inp.origin, T, full)
    horizon = float(times[-1])
    q = inp.rate - math.log(inp.forward / inp.spot) / inp.tau
    market = sim.Market(inp.rate, q, horizon)
    fund = sim.Fund((sim.Leg("call", 1, inp.strike, inp.options_per_share),), cash=inp.cash_per_share, fee=FEE)
    path_vol, iv, slide = model_vols(inp)[name]
    o = xp.simulate(seed_for(inp.origin), npaths, fund, market, inp.spot, times, path_vol, iv, slide=slide,
                    checkpoints=(k,))
    return o["nav_at_checkpoint"][:, 0] / o["nav0"], inp.rate - q, float(times[k])


def make_forecast(inp, T, npaths=PATHS, provenance=None):
    """The forecast record (saved privately): inputs, and per model the growth at LEVELS and SPY's lognormal terms;
    provenance: input_provenance() (which saved files the inputs came from), when known."""
    models = {}
    for name in MODELS:
        g, mu, tau1 = simulate_growth(inp, T, name, npaths)
        path_vol = model_vols(inp)[name][0]
        models[name] = {"growth_levels": [float(f"{v:.12g}") for v in np.quantile(g, LEVELS)],
                        "spy_mu": mu, "spy_vol": path_vol, "years": tau1}
    rec = {"protocol": "FC1", "origin": inp.origin.isoformat(), "target": T.isoformat(), "paths": npaths,
           "seed": seed_for(inp.origin), "levels": "0.001 to 0.999 in steps of 0.001",
           "inputs": {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(inp).items()},
           "models": models, "versions": _versions()}
    if provenance:
        rec["provenance"] = provenance
    return rec


def _versions():
    """Library versions behind a forecast: re-running it elsewhere reproduces the fingerprint only with the same
    numerics (numpy, numba and the platform's maths library)."""
    import platform

    import numba
    return {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
            "system": platform.system()}


def canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def fingerprint(record):
    return hashlib.sha256(canonical(record)).hexdigest()


def code_fingerprints(root=REPO):
    """{file: SHA-256} for CODE_FILES plus every definedtarget module and script the process has loaded (the input
    preparation as well as the computation; audit of v0.3.0, D06)."""
    from ..montecarlo import manifest
    root = Path(root).resolve()
    files = {f for f in CODE_FILES}
    for p in manifest.loaded_files(root):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("definedtarget/", "scripts/")):
            files.add(rel)
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in sorted(files)}


def _finite_positive(**values):
    """Refuse a missing, non-finite or non-positive number (audit of v0.3.0, D01: a blank NAV used to be scored as
    NaN and recorded for good)."""
    bad = [f"{k}={v!r}" for k, v in values.items()
           if not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(v) or v <= 0]
    if bad:
        raise ValueError("missing or invalid number(s): " + ", ".join(bad) + ". If the issuer has not published it "
                         "yet, wait; if it will never be published, record the month as void (FC1 rule 8)")


def pit_from_levels(levels, x):
    """Share of the saved forecast below x, by linear interpolation between its 0.1% points; below the first point
    0.0005 and above the last 0.9995 (FC1 rule 6: strictly below / above; at a point itself, interpolation)."""
    lv = np.asarray(levels, dtype=float)
    if x < lv[0]:
        return 0.0005
    if x > lv[-1]:
        return 0.9995
    return float(np.interp(x, lv, np.asarray(LEVELS)))


def spy_pit(m, spy_t, spy_T):
    """Closed form: P(SPY's lognormal month below what happened) under the model's drift and path vol."""
    sd = m["spy_vol"] * math.sqrt(m["years"])
    z = (math.log(spy_T / spy_t) - (m["spy_mu"] - 0.5 * m["spy_vol"] ** 2) * m["years"]) / sd
    return pricing.norm_cdf(z)


def score(forecast, nav_t, nav_T, spy_t, spy_T):
    """{model: {"xx": PIT, "spy": PIT}} from the saved forecast and the issuer's NAVs and SPY reference closes.
    Refuses missing or invalid numbers rather than scoring them."""
    _finite_positive(nav_t=nav_t, nav_T=nav_T, spy_t=spy_t, spy_T=spy_T)
    growth = nav_T / nav_t
    out = {name: {"xx": round(pit_from_levels(m["growth_levels"], growth), 4),
                  "spy": round(spy_pit(m, spy_t, spy_T), 4)} for name, m in forecast["models"].items()}
    if not all(0.0 < p[k] < 1.0 for p in out.values() for k in ("xx", "spy")):
        raise ValueError(f"a PIT outside (0, 1): {out}; nothing recorded")
    return out


# ------------------------------------------------------------------ inputs from the data folder

def load_inputs(t, root=None, fetch_vix=True):
    """Inputs for origin t from the project's tables (model row and fund state for t) and Cboe's VIX history
    (fetched again if the saved copy predates t). Raises if any is missing; never fills a gap from another date."""
    import pandas as pd

    from . import fetch
    root = fetch.data_dir(root)
    m = pd.read_csv(root / "tables" / "xx_model_daily.csv")
    s = pd.read_csv(root / "tables" / "xx_state.csv")
    m, s = m[m["as_of"] == t.isoformat()], s[s["as_of"] == t.isoformat()]
    if m.empty or s.empty:
        raise ValueError(f"no model row or fund state for {t}: run the daily routine for that close first")
    m, s = m.iloc[0], s.iloc[0]
    vix, _ = _vix_closes(t, root, fetch_vix)
    inp = Inputs(t, float(m["spot"]), float(m["iv_disc_0bp"]), float(m["zero_rate_T"]), float(m["forward"]),
                 float(m["tau"]), float(s["strike"]), float(m["options_per_share"]), float(s["cash_per_share"]),
                 vix, str(m.get("spot_source", "")))
    # Refuse invalid inputs before any forecast is saved (audit of v0.3.0, D15)
    _finite_positive(spot=inp.spot, implied_vol=inp.implied_vol, forward=inp.forward, tau=inp.tau,
                     strike=inp.strike, options_per_share=inp.options_per_share, vix=inp.vix)
    if not (math.isfinite(inp.rate) and math.isfinite(inp.cash_per_share) and inp.cash_per_share >= 0):
        raise ValueError(f"invalid rate or cash per share for {t}: {inp.rate!r}, {inp.cash_per_share!r}")
    return inp


def _vix_closes(t, root, fetch_vix=True):
    """(VIX close for t, the manifest record of the file it came from). Prefers the newest file captured by the
    entry's deadline, so a late run reads what an on-time run would have read (audit of v0.3.0, D05); falls back to
    the newest file, and fetches a fresh copy once if none has t (the external capture refreshes VIX weekly)."""
    import io

    import pandas as pd

    from . import external, fetch, parse

    def from_file(recs):
        for rec in reversed(recs):
            v = pd.read_csv(io.StringIO(parse.read_text(root, rec)))
            closes_ = dict(zip(pd.to_datetime(v["DATE"], format="%m/%d/%Y").dt.date, v["CLOSE"].astype(float)))
            if t in closes_:
                return closes_[t], rec
        return None, None

    cutoff = close_time(deadline(t)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with parse.captured_by(cutoff):
        value, rec = from_file(parse.saved_versions(root, "VIX_History.csv"))
    if value is None:
        value, rec = from_file(parse.saved_versions(root, "VIX_History.csv"))
    if value is None and fetch_vix:
        fetch.capture(external.CBOE.format("VIX"), "VIX_History.csv", "cboe", b"DATE,", root=root)
        value, rec = from_file(parse.saved_versions(root, "VIX_History.csv"))
    if value is None:
        raise ValueError(f"no saved VIX history has a close for {t}")
    return value, rec


def input_provenance(t, root=None):
    """Which saved files a forecast's inputs came from: the issuer's holdings file for t (newest saved version) and
    the VIX history, by name, download date and SHA-256 (audit of v0.3.0, D05: a later revision of either becomes
    visible). Saved in the private forecast."""
    from . import fetch, parse, roundhill
    root = fetch.data_dir(root)
    out = {}
    recs = parse.saved_versions(root, roundhill.holdings_name(t))
    if recs:
        out["holdings"] = {"name": recs[-1]["name"], "fetched_utc_date": recs[-1]["fetched_utc"][:10],
                           "sha256": recs[-1].get("sha256", "")}
    _, rec = _vix_closes(t, root, fetch_vix=False)
    out["vix"] = {"name": rec["name"], "fetched_utc_date": rec["fetched_utc"][:10], "sha256": rec.get("sha256", "")}
    return out


def closes(t, root=None):
    """(published NAV, SPY reference close) for close t, from the project's tables."""
    import pandas as pd

    from . import fetch
    root = fetch.data_dir(root)
    m = pd.read_csv(root / "tables" / "xx_model_daily.csv")
    s = pd.read_csv(root / "tables" / "xx_state.csv")
    m, s = m[m["as_of"] == t.isoformat()], s[s["as_of"] == t.isoformat()]
    if m.empty or s.empty:
        raise ValueError(f"no NAV or SPY close for {t} in the tables")
    nav, spot = float(s.iloc[0]["nav_published"]), float(m.iloc[0]["spot"])
    _finite_positive(**{f"NAV on {t}": nav, f"SPY close on {t}": spot})
    return nav, spot


def events_between(t, T, root=None):
    """[(date, event)] after t up to T that make a month void (FC1 rule 8) and the data can show: a roll or option
    change and a reverse split (the hypotheses' own void test), options per share moving more than 5% between two
    closes (a split or a payout in kind), and every event listed in data/external/void_events.csv. A distribution or a
    closure notice may not show in the data at all: the outcome command also asks the operator to confirm the review
    (audit of v0.3.0, D02)."""
    import pandas as pd

    from .. import verdicts
    from . import fetch
    root = fetch.data_dir(root)
    s = pd.read_csv(root / "tables" / "xx_state.csv")
    s["as_of"] = pd.to_datetime(s["as_of"]).dt.date
    s = s[(s["as_of"] >= t) & (s["as_of"] <= T)].sort_values("as_of").reset_index(drop=True)
    out = [(d, e) for d, e in verdicts.find_voids(s, root) if t < d <= T]
    if "contracts_per_share" in s.columns:
        c = s["contracts_per_share"].astype(float).to_numpy()
        for i in range(1, len(s)):
            if c[i - 1] > 0 and abs(c[i] / c[i - 1] - 1.0) > OPTIONS_JUMP:
                out.append((s.loc[i, "as_of"], f"options per share moved {c[i - 1]:.7f} -> {c[i]:.7f}"))
    return sorted(set(out))


# ------------------------------------------------------------------ the public log (hash chained, like the ledger)

def read_log(path=LOG_PATH):
    return ledger.read(path)


def append_log(entry, path=LOG_PATH, now=None, any_close=False):
    """Append one entry ("forecast", "outcome" or "report"); refuses a broken chain, a duplicate, an outcome without
    its forecast, an entry before its close has ended, a number that is not finite, or (unless any_close, for dry runs
    into a scratch log) an origin outside rule 1. Line endings: LF (a new file) or the file's own."""
    now = now or datetime.now(timezone.utc)
    ok, msg = ledger.verify(path)
    if not ok:
        raise ValueError(f"refusing to append: {msg}")
    if entry.get("kind") not in ("forecast", "outcome", "report"):
        raise ValueError("kind must be 'forecast', 'outcome' or 'report'")
    if any_close and Path(path).resolve() == LOG_PATH.resolve():
        raise ValueError("dry runs (any_close) never write to the public forecast log")
    existing = read_log(path)
    body = dict(entry)
    if entry["kind"] == "report":
        n = sum(e["kind"] == "outcome" for e in existing)
        if entry["outcomes"] != n or any(e["kind"] == "report" and e["outcomes"] == n for e in existing):
            raise ValueError(f"a report must cover all {n} outcome entries so far, once")
        body["recorded_utc_date"] = now.strftime("%Y-%m-%d")
    else:
        origin = date.fromisoformat(entry["origin"])
        if not any_close:
            target_for(origin)
        if any(e["kind"] == entry["kind"] and e.get("origin") == entry["origin"] for e in existing):
            raise ValueError(f"the {entry['kind']} for {entry['origin']} is already in the log (never re-record)")
        if entry["kind"] == "outcome" and not any(e["kind"] == "forecast" and e.get("origin") == entry["origin"]
                                                  for e in existing):
            raise ValueError(f"no forecast for {entry['origin']} in the log")
        close = origin if entry["kind"] == "forecast" else date.fromisoformat(entry["target"])
        if now.astimezone(NEW_YORK) < close_time(close):        # audit of v0.3.0, D03: nothing before its close ends
            raise ValueError(f"the session of {close} has not ended yet: an entry about it cannot be recorded")
        # The date only, never the clock time: the log records the author's activity every month, and public files
        # carry no times of it (project rule). "late" is decided from the full time before it is dropped.
        body["recorded_utc_date"] = now.strftime("%Y-%m-%d")
        body["late"] = is_late(close, now)
    prev = existing[-1]["row_hash"] if existing else ledger.GENESIS
    full = {**body, "prev_hash": prev, "row_hash": ledger._hash(prev, body)}
    line = json.dumps(full, ensure_ascii=False, sort_keys=True, allow_nan=False)   # D01: never write NaN
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eol = "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write(line + eol)
    return full


def due(today, path=LOG_PATH):
    """[(kind, origin, close, deadline, late)] for every entry whose close has passed and is not in the log yet, and
    ("report", number of outcomes, None, None, False) when rule 9's report is due: after every 12 outcome entries
    (voids included) and after the last one (audit of v0.3.0, D19)."""
    entries = read_log(path)
    have = {(e["kind"], e.get("origin")) for e in entries}
    out = []
    for t, T in origins():
        for kind, close in (("forecast", t), ("outcome", T)):
            if close < today and (kind, t.isoformat()) not in have:
                out.append((kind, t, close, deadline(close), today > deadline(close)))
    n = sum(e["kind"] == "outcome" for e in entries)
    reported = {e["outcomes"] for e in entries if e["kind"] == "report"}
    if n and (n % REPORT_EVERY == 0 or n == len(origins())) and n not in reported:
        out.append(("report", n, None, None, False))
    return out


def forecast_entry(record, code=None):
    return {"kind": "forecast", "protocol": "FC1", "origin": record["origin"], "target": record["target"],
            "seed": record["seed"], "paths": record["paths"], "models": sorted(record["models"]),
            "forecast_sha256": fingerprint(record), "code_sha256": code or code_fingerprints()}


def outcome_entry(record, pits=None, void=None, review=None):
    """review: how the month's void events were checked (FC1 rule 8), recorded with the outcome."""
    e = {"kind": "outcome", "protocol": "FC1", "origin": record["origin"], "target": record["target"],
         "forecast_sha256": fingerprint(record)}
    if void:
        e["void"] = void
    else:
        e["pit"] = pits
    if review:
        e["events_review"] = review
    return e


def report_entry(entries):
    """Rule 9's report as a public log entry: the summary of every scored outcome so far (PITs only, no NAVs)."""
    n = sum(e["kind"] == "outcome" for e in entries)
    out = {}
    for (name, kind), r in summary(entries).items():
        out[f"{name}/{kind}"] = {"n": r["n"], "mean": round(r["mean"], 4),
                                 "in50": [round(v, 4) for v in r["in50"]], "in90": [round(v, 4) for v in r["in90"]],
                                 "histogram": r["histogram"]}
    return {"kind": "report", "protocol": "FC1", "outcomes": n,
            "voids": sum(e["kind"] == "outcome" and "void" in e for e in entries), "summary": out}


def summary(entries):
    """FC1 rule 9: per model and per PIT kind, the count, the share inside the middle 50% and 90% bands (95% Wilson
    intervals), the mean PIT and a 10-bin histogram."""
    from ..montecarlo import stats
    out = {}
    scored = [e for e in entries if e["kind"] == "outcome" and "pit" in e]
    for name in MODELS:
        for kind in ("xx", "spy"):
            p = np.array([e["pit"][name][kind] for e in scored if name in e["pit"]])
            if p.size == 0:
                continue
            in50 = int(((p >= 0.25) & (p <= 0.75)).sum())
            in90 = int(((p >= 0.05) & (p <= 0.95)).sum())
            out[(name, kind)] = {"n": int(p.size), "mean": float(p.mean()),
                                 "in50": stats.wilson(in50, p.size), "in90": stats.wilson(in90, p.size),
                                 "histogram": np.histogram(p, bins=10, range=(0.0, 1.0))[0].tolist()}
    return out
