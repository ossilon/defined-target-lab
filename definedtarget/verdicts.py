"""Verdicts for the pre-registered hypotheses (forward/predictions.jsonl), computed mechanically from the frozen rules.

Plain words
- Each hypothesis H1-H19 (and H7a's yearly checkpoints) has a test, a pass band and a review date, frozen on
  2026-10-14; protocols P1 and P2 add the shared rules (clean days, windows, minimum samples, void events, which
  listed quotes count). This module turns each rule into code. No judgment overrides a result (P1).
- Before its review date a hypothesis is "pending" and the module shows a provisional number on the data so far
  (it is not a verdict). On or after the review date it gives the verdict word: pass, fail, inconclusive (a stated
  minimum sample not reached even after the allowed extension) or void (a P1 void event).
- P2: each review uses the data captured by its review date. A review on date R reads only files downloaded by the
  end of R (UTC), through parse.captured_by(); its windows end with the last close before R.
- Conventions written down here because the frozen text leaves them implicit (each is the plain reading):
  * a daily change belongs to the close where it ends; it counts in a window if that close does, and it is clean
    if that close is clean (P1's clean-day rule);
  * "daily change" = between consecutive closes that both have every value the test needs (P1: drop missing closes,
    never fill them);
  * an extended review (P1 minimum-sample rule) moves to the session after the close that completes the sample, so
    the window includes that close, and reads the data captured by then. "At most three months later" bounds the
    completing close: it must fall on or before the last session on or before review + 3 months. The final review
    is the session after that last close; if the sample is still short there, the verdict is inconclusive;
  * H1 and H9 review on the session after the 15th event, and only once the data captured by that session holds 15
    events (a late capture moves the review to the next session);
  * under P2 several in-window readings of one close may exist: the first counts (nearest the close). Under P3 the
    latest qualifying reading counts; P3 applies to readings taken after its ledger entry was recorded;
  * a listed quote counts only if it is two-sided (bid > 0, ask >= bid) and its implied vol solves inside the
    solver's limits; otherwise that close counts as missing for the test (P1), and the stats report how many;
  * H19 uses the 9,600 and 10,400 calls; it moves to the nearest quoted strikes below and above 10,000 only when a
    strike has lapsed (absent from that close and every later one, with at least one later close to show it); any
    other gap is a missing close;
  * H2: contracts per share may not rise on clean days; on flow days a rise must stay within one contract per
    creation unit (rounding); other days (the day after a flow day, ex-dates) are unconstrained, as the text says;
  * H1's "day after" is the next close with an implied vol (normally the next session); H14's "10th session" is
    the session 10 sessions after the window's start; H17's n counts closes (the regression uses n - 1 changes);
  * void events: "affected" means the hypothesis reads fund data (NAV, holdings, marks, market price). H7 and H7a
    read only S&P 500 and SPY closes, which no fund event changes, so no fund event voids them;
  * a daily change that spans a missing close is clean if its end close is clean (the plain reading above).
- scripts/verdicts.py prints every status and, only when asked, appends a verdict to the ledger: never before the
  day whose captured data decided it has ended (UTC).
"""
import calendar as _cal
import io
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import calendar as nyse

TEST_START = date(2026, 10, 14)            # P1: only closes from 2026-10-14 onward
LISTED_START = date(2026, 11, 1)           # H9, H13, H17, H19 windows
XX_STRIKE, XX_EXPIRY = 995.01, date(2030, 1, 10)
LISTED_EXPIRY = date(2029, 12, 21)
H2_BASE = 0.0048                           # contracts per share on 2026-10-13
H7A_BASELINE = 10.03693                    # 2026 January ratio, frozen (P2)
CREATION_UNIT = 10_000
WINDOW_REVIEWS = {"H3": date(2027, 4, 14), "H4": date(2027, 1, 14), "H6": date(2027, 4, 14), "H8": date(2027, 1, 14),
                  "H10": date(2027, 1, 14), "H11": date(2027, 4, 14), "H12": date(2027, 4, 14),
                  "H13": date(2027, 4, 14), "H14": date(2027, 4, 14), "H15": date(2027, 10, 14),
                  "H16": date(2027, 4, 14), "H17": date(2027, 4, 14), "H18": date(2027, 4, 14),
                  "H19": date(2027, 4, 14), "H2": date(2027, 10, 14)}
MIN_SAMPLE = {"H4": 40, "H8": 40, "H10": 40, "H12": 40, "H13": 60, "H14": 8, "H15": 8, "H16": 40, "H17": 60,
              "H18": 40, "H19": 60}       # P2: these extend by up to three months
EVENT_FINAL = date(2027, 10, 14)           # H1 and H9: 'inconclusive' then if fewer than 15 events
H7A_CHECKPOINTS = {2027: date(2027, 1, 20), 2028: date(2028, 1, 20), 2029: date(2029, 1, 19)}
H5_FIRST_REVIEW = date(2026, 12, 21)
ORDER = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H7a", "H8", "H9", "H10", "H11", "H12", "H13", "H14", "H15", "H16",
         "H17", "H18", "H19"]


@dataclass
class Inputs:
    """Everything a verdict reads. Built from the data folder by load(); tests build it from synthetic tables.

    state: one row per close (reconcile.fund_state): as_of, shares_outstanding, contracts, contracts_per_share,
           cash_weight, nav_published, market_price, flow_day, day_after_flow, strike, expiry
    model: one row per priced close (marks.model_daily): as_of, iv_disc_0bp, xx_elasticity, zero_rate_T,
           vendor_box_rate, treasury_rate_box_term, prob_itm_rn
    prices: {close date: (spy, spx)} from the issuer's reference closes, else data/external/closes.csv (P2)
    vix1y: {date: close}; dividends: sorted [(ex_date, amount)]
    calls: deep in-the-money SPY call marks from the whole holdings file: as_of, account, strike, expiry, price
    listed: one row per counted listed reading and strike: close_date, strike, bid, ask (already filtered by the
            listed-quote rule in force, P2 or P3)
    curve_on / forward_fn: functions for the listed conventions (base-case curve and dividends)
    voids: [(date, event)] void events found in the data or listed in data/external/void_events.csv
    """
    state: pd.DataFrame
    model: pd.DataFrame
    prices: dict
    vix1y: dict = field(default_factory=dict)
    dividends: list = field(default_factory=list)
    calls: pd.DataFrame = None
    listed: pd.DataFrame = None
    listed_forward: object = None          # function(close) -> (F, DF, tau) for the listed expiry, in SPY terms
    voids: list = field(default_factory=list)
    cutoff: str = None


@dataclass
class Result:
    id: str
    status: str                            # pending, pass, fail, inconclusive, void, no data
    review: str
    n: int = 0
    stats: dict = field(default_factory=dict)
    window: str = ""
    note: str = ""
    decided: date = None                   # the day whose captured data decided a final word (its cutoff, P2)

    def line(self):
        s = ", ".join(f"{k} {_fmt(v)}" for k, v in self.stats.items())
        return f"{self.id:<4} {self.status:<13} review {self.review:<24} n={self.n:<4} {s}" + (f"  [{self.note}]" if self.note else "")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# ---------------------------------------------------------------- shared rules

def add_months(d, months):
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, _cal.monthrange(y, m)[1]))


def excluded_ex(dividends):
    """SPY ex-dates and the session before each (P1: those closes belong to H5, not to clean days)."""
    out = set()
    for ex, _ in dividends:
        out.add(ex)
        out.add(nyse.previous_session(ex))
    return out


def frame(inp):
    """One row per close with every series a test may need (NaN where a value is missing)."""
    st = inp.state.set_index("as_of")
    md = inp.model.set_index("as_of") if len(inp.model) else pd.DataFrame()
    days = sorted(set(st.index) | set(md.index))
    ex = excluded_ex(inp.dividends)
    rows = []
    for d in days:
        s = st.loc[d] if d in st.index else None
        m = md.loc[d] if d in md.index else None
        spy, spx = inp.prices.get(d, (np.nan, np.nan))
        g = (lambda r, k: float(r[k]) if r is not None and k in r and pd.notna(r[k]) else np.nan)
        flow = bool(s["flow_day"]) if s is not None else False
        after = bool(s["day_after_flow"]) if s is not None else False
        rows.append({"as_of": d, "nav": g(s, "nav_published"), "mkt": g(s, "market_price"),
                     "cash_weight": g(s, "cash_weight"), "shares": g(s, "shares_outstanding"),
                     "contracts": g(s, "contracts"), "cps": g(s, "contracts_per_share"),
                     "iv_pts": g(m, "iv_disc_0bp") * 100, "E": g(m, "xx_elasticity"),
                     "rate_bp": g(m, "zero_rate_T") * 1e4, "box": g(m, "vendor_box_rate"),
                     "tbox": g(m, "treasury_rate_box_term"), "prob": g(m, "prob_itm_rn"),
                     "spy": spy, "spx": spx, "vix1y": float(inp.vix1y.get(d, np.nan)),
                     "flow": flow, "after": after, "clean": not flow and not after and d not in ex})
    return pd.DataFrame(rows)


def changes(f, cols, start, end, clean_only=True):
    """Daily changes over consecutive closes that have every column in `cols`; ends in [start, end)."""
    g = f.dropna(subset=cols).reset_index(drop=True)
    prev = g.shift(1)
    out = pd.DataFrame({"as_of": g["as_of"], "clean": g["clean"]})
    for c in cols:
        out[c] = g[c]
        out[c + "_prev"] = prev[c]
    out = out.iloc[1:]
    keep = (out["as_of"] >= start) & (out["as_of"] < end)
    if clean_only:
        keep &= out["clean"]
    return out[keep].reset_index(drop=True)


def pct(a, b):
    return (a / b - 1.0) * 100.0                       # % change, in percent


def ols(y, X, intercept=True):
    """Coefficients (intercept first when included) and their standard errors."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X]) if intercept else \
        np.column_stack([np.asarray(x, float) for x in X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    return coef, np.sqrt(np.diag(cov))


def welch_one_sided(a, b, alternative):
    """One-sided Welch t-test of mean(a) vs mean(b): p-value for 'less' (a < b) or 'greater' (a > b)."""
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)


# ---------------------------------------------------------------- one function per hypothesis
# Each takes (inputs, window end) and returns (n, verdict or None if not computable, stats dict).

def t_h2(inp, end):
    f = frame(inp)
    g = f[(f["as_of"] >= TEST_START) & (f["as_of"] < end)].dropna(subset=["cps", "contracts", "shares"])
    full = f.dropna(subset=["cps", "contracts", "shares"]).reset_index(drop=True)
    rises = []
    for i in range(1, len(full)):
        r, p = full.iloc[i], full.iloc[i - 1]
        if not (TEST_START <= r["as_of"] < end):
            continue
        if r["clean"]:
            if r["cps"] > p["cps"] + 1e-12:
                rises.append(str(r["as_of"]))
        elif r["flow"]:
            expected = p["cps"] * r["shares"]
            units = max(1.0, math.ceil(abs(r["shares"] - p["shares"]) / CREATION_UNIT))
            if r["contracts"] > expected + units + 1e-9:
                rises.append(str(r["as_of"]))
    if g.empty:
        return 0, None, {}
    fall = 1.0 - g["cps"].iloc[-1] / H2_BASE
    ok = not rises and 0.005 <= fall <= 0.015
    return len(g), ok, {"total_fall": fall, "unexplained_rises": len(rises), "last_close": str(g["as_of"].iloc[-1])}


def _share_in(values, lo, hi):
    v = np.asarray(values, float)
    return float(((v >= lo) & (v <= hi)).mean()) if len(v) else float("nan")


def t_h3(inp, end):
    f = frame(inp)
    g = f[(f["as_of"] >= TEST_START) & (f["as_of"] < end)].dropna(subset=["cash_weight"])
    if g.empty:
        return 0, None, {}
    share = _share_in(g["cash_weight"], 0.005, 0.015)
    return len(g), share >= 0.90, {"share_in_band": share}


def _h4_regression(inp, end, extra_rate=False):
    cols = ["iv_pts", "spy", "vix1y"] + (["rate_bp"] if extra_rate else [])
    c = changes(frame(inp), cols, TEST_START, end)
    if len(c) < 4:
        return len(c), None
    X = [pct(c["spy"], c["spy_prev"]), c["vix1y"] - c["vix1y_prev"]]
    if extra_rate:
        X.append(c["rate_bp"] - c["rate_bp_prev"])
    coef, se = ols(c["iv_pts"] - c["iv_pts_prev"], X)
    return len(c), (coef, se)


def t_h4(inp, end):
    n, r = _h4_regression(inp, end)
    if r is None:
        return n, None, {}
    b = r[0][2]
    return n, 0.1 <= b <= 0.9, {"vix1y_coef": b}


def t_h10(inp, end):
    n, r = _h4_regression(inp, end)
    if r is None:
        return n, None, {}
    b = r[0][1]
    return n, 0.05 <= b <= 0.35, {"spy_coef": b}


def t_h18(inp, end):
    n, r = _h4_regression(inp, end, extra_rate=True)
    if r is None:
        return n, None, {}
    b = r[0][3]
    return n, -0.008 <= b <= 0.008, {"rate_coef": b}


def t_h6(inp, end):
    f = frame(inp)
    g = f[(f["as_of"] >= TEST_START) & (f["as_of"] < end)].dropna(subset=["mkt", "nav"])
    if g.empty:
        return 0, None, {}
    p = (g["mkt"] / g["nav"] - 1.0).abs()
    med, share = float(p.median()), float((p < 0.01).mean())
    return len(g), med <= 0.005 and share >= 0.85, {"median_abs": med, "share_below_1pct": share}


def t_h8(inp, end):
    c = changes(frame(inp), ["nav", "spy", "E"], TEST_START, end)
    if len(c) < 2:
        return len(c), None, {}
    (beta,), _ = ols(pct(c["nav"], c["nav_prev"]), [pct(c["spy"], c["spy_prev"])], intercept=False)
    ratio = beta / c["E_prev"].mean()
    return len(c), 1.05 <= ratio <= 1.60, {"ratio": ratio, "beta": beta}


def t_h11(inp, end):
    f = frame(inp)
    g = f[(f["as_of"] >= TEST_START) & (f["as_of"] < end)].dropna(subset=["iv_pts", "vix1y"])
    if g.empty:
        return 0, None, {}
    share = _share_in(g["iv_pts"] / g["vix1y"], 0.72, 0.82)
    return len(g), share >= 0.80, {"share_in_band": share}


def t_h12(inp, end):
    f = frame(inp)
    g = f[(f["as_of"] >= TEST_START) & (f["as_of"] < end)].dropna(subset=["box", "tbox"])
    if g.empty:
        return 0, None, {}
    share = _share_in((g["box"] - g["tbox"]) * 1e4, 25, 75)
    return len(g), share >= 0.90, {"share_in_band": share}


def t_h14(inp, end):
    f = frame(inp).dropna(subset=["nav", "spy", "E"]).set_index("as_of")
    have = set(f.index)
    clean = {d for d in have if f.loc[d, "clean"]}
    starts = sorted(d for d in clean if d >= TEST_START)
    if not starts:
        return 0, None, {}
    windows, s = [], starts[0]
    while True:
        tenth = nyse.sessions(s, add_months(s, 2))[10] if len(nyse.sessions(s, add_months(s, 2))) > 10 else None
        if tenth is None:
            break
        e = tenth if tenth in clean else next((d for d in sorted(clean) if d > tenth), None)
        if e is None or e >= end:
            break
        a, b = f.loc[s], f.loc[e]
        windows.append((pct(b["nav"], a["nav"]), a["E"] * pct(b["spy"], a["spy"])))
        s = e
    if len(windows) < 3:
        return len(windows), None, {}
    y, x = zip(*windows)
    coef, se = ols(y, [x])
    return len(windows), 0.85 <= coef[1] <= 1.10, {"slope": coef[1], "slope_se": se[1]}


def t_h15(inp, end):
    c = changes(frame(inp), ["nav", "spy", "E"], TEST_START, end)
    r_spy = pct(c["spy"], c["spy_prev"])
    bad = c[r_spy <= -1.0]
    if bad.empty:
        return 0, None, {}
    ratio = pct(bad["nav"], bad["nav_prev"]) / (bad["E_prev"] * pct(bad["spy"], bad["spy_prev"]))
    med = float(np.median(ratio))
    return len(bad), 1.00 <= med <= 1.60, {"median_ratio": med}


def t_h16(inp, end):
    c = changes(frame(inp), ["nav", "mkt", "spy"], TEST_START, end)
    if len(c) < 2:
        return len(c), None, {}
    x = [pct(c["spy"], c["spy_prev"])]
    (b_mkt,), _ = ols(pct(c["mkt"], c["mkt_prev"]), x, intercept=False)
    (b_nav,), _ = ols(pct(c["nav"], c["nav_prev"]), x, intercept=False)
    ratio = b_mkt / b_nav
    return len(c), 0.90 <= ratio <= 1.10, {"ratio": ratio}


# ---- listed-market tests (H9, H13, H17, H19)

def two_sided(bid, ask):
    """A usable listed quote (P1: a close without one counts as missing): bid > 0 and ask >= bid."""
    return np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid


def listed_vols(inp, strikes=(10000,)):
    """{close: (ivs, quotes, DF, bad)} for counted listed readings, P1/P2 conventions.

    ivs: {strike: (bid_iv, mid_iv, ask_iv) in points} for the strikes asked for, only when the quote is two-sided and
    all three vols solve strictly inside the solver's limits; quotes: {strike: (bid, ask)}, two-sided quotes only;
    bad: the asked-for strikes whose quote was present but unusable (they count as missing for that close)."""
    from . import pricing
    out = {}
    if inp.listed is None or inp.listed.empty or inp.listed_forward is None:
        return out
    for close, g in inp.listed.groupby("close_date"):
        if close not in inp.prices or any(np.isnan(inp.prices[close])):
            continue
        spy, spx = inp.prices[close]
        ratio = spx / spy
        F, DF, tau = inp.listed_forward(close)
        raw = {int(r.strike): (float(r.bid), float(r.ask)) for r in g.itertuples()}
        q = {k: v for k, v in raw.items() if two_sided(*v)}
        row, bad = {}, set()
        for k in strikes:
            if k not in raw:
                continue
            if k not in q:
                bad.add(k)
                continue
            b, a = q[k]
            ivs = [pricing.implied_vol(x / ratio, F, k / ratio, DF, tau) for x in (b, (b + a) / 2, a)]
            if all(np.isfinite(s) and pricing.VOL_LO * 10 < s < pricing.VOL_HI * 0.99 for s in ivs):
                row[k] = tuple(s * 100 for s in ivs)
            else:
                bad.add(k)
        out[close] = (row, q, DF, bad)
    return out


def _unusable(lv, start, end, strike=10000):
    return sum(1 for d, v in lv.items() if start <= d < end and strike in v[3])


def t_h13(inp, end):
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp)
    rows = [(d, v[0][10000]) for d, v in lv.items() if LISTED_START <= d < end and 10000 in v[0]
            and d in f.index and not np.isnan(f.loc[d, "iv_pts"])]
    if not rows:
        return 0, None, {}
    ok = [f.loc[d, "iv_pts"] >= b - 0.5 and f.loc[d, "iv_pts"] <= m for d, (b, m, a) in rows]
    share = float(np.mean(ok))
    return len(rows), share >= 0.80, {"share_in_band": share, "unusable_quotes": _unusable(lv, LISTED_START, end)}


def t_h17(inp, end):
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp)
    days = sorted(d for d, v in lv.items() if LISTED_START <= d < end and 10000 in v[0] and d in f.index
                  and not np.isnan(f.loc[d, "spx"]))
    if len(days) < 4:
        return len(days), None, {}
    y = [lv[b][0][10000][1] - lv[a][0][10000][1] for a, b in zip(days, days[1:])]
    x = [pct(f.loc[b, "spx"], f.loc[a, "spx"]) for a, b in zip(days, days[1:])]
    coef, se = ols(y, [x])
    stats = {"listed_coef": coef[1], "unusable_quotes": _unusable(lv, LISTED_START, end)}
    xx =[(f.loc[b, "iv_pts"] - f.loc[a, "iv_pts"], pct(f.loc[b, "spy"], f.loc[a, "spy"]),
           f.loc[b, "vix1y"] - f.loc[a, "vix1y"]) for a, b in zip(days, days[1:])]
    xx = [r for r in xx if not any(np.isnan(r))]
    if len(xx) >= 4:
        dy, ds, dv = zip(*xx)
        stats["xx_h10_coef_same_closes"] = ols(dy, [ds, dv])[0][1]
    return len(days), -0.20 <= coef[1] <= 0.05, stats


def t_h19(inp, end):
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp, strikes=())
    quoted_after = {}                                  # strike -> last close with a usable quote
    for d, v in lv.items():
        for k in v[1]:
            quoted_after[k] = max(quoted_after.get(k, d), d)

    newest = max(lv) if lv else None

    def pick(k, q, d, fallback):
        if k in q:
            return k
        if quoted_after.get(k, date.min) > d or d == newest:   # quoted again later, or no later close to tell:
            return None                                          # a gap, not a lapse -> missing close
        return fallback

    gaps, missing = [], 0
    for d, (row, q, DF, bad) in lv.items():
        if not (LISTED_START <= d < end) or d not in f.index or np.isnan(f.loc[d, "prob"]):
            continue
        below = [k for k in q if k < 10000]
        above = [k for k in q if k > 10000]
        k1 = pick(9600, q, d, max(below) if below else None)
        k2 = pick(10400, q, d, min(above) if above else None)
        if k1 is None or k2 is None:
            missing += 1
            continue
        mid = {k: sum(q[k]) / 2 for k in (k1, k2)}
        p = (mid[k1] - mid[k2]) / ((k2 - k1) * DF)
        gaps.append(100 * (p - f.loc[d, "prob"]))
    if not gaps:
        return 0, None, {}
    share = _share_in(gaps, 5, 15)
    return len(gaps), share >= 0.80, {"share_in_band": share, "median_gap_pts": float(np.median(gaps)),
                                      "missing_closes": missing}


# ---- event-based (H1, H9), H5, H7, H7a

def creation_days(f, start):
    g = f.dropna(subset=["shares"]).reset_index(drop=True)
    return [g["as_of"][i] for i in range(1, len(g)) if g["shares"][i] > g["shares"][i - 1] and g["as_of"][i] >= start]


def h1_stat(inp):
    f = frame(inp)
    g = f.dropna(subset=["iv_pts"]).reset_index(drop=True)
    pos = {d: i for i, d in enumerate(g["as_of"])}
    events = []
    for c in creation_days(f, TEST_START):
        i = pos.get(c)
        if i is not None and i + 1 < len(g):
            events.append((g["as_of"][i + 1], g["iv_pts"][i + 1] - g["iv_pts"][i]))
    clean = changes(f, ["iv_pts"], TEST_START, date.max)
    clean_d = (clean["iv_pts"] - clean["iv_pts_prev"]).to_numpy()
    return events, clean_d


def h9_stat(inp):
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp)
    gap = {d: f.loc[d, "iv_pts"] - v[0][10000][1] for d, v in lv.items()
           if d >= LISTED_START and 10000 in v[0] and d in f.index and not np.isnan(f.loc[d, "iv_pts"])}
    full = frame(inp)
    creations = [d for d in creation_days(full, LISTED_START) if d in gap]
    clean = [gap[d] for d in gap if f.loc[d, "clean"]]
    return [(d, gap[d]) for d in creations], np.asarray(clean, float)


def h5_compute(inp, ex):
    """(n contracts, median ratio, reason) for the two sessions ending on SPY ex-date `ex`. `reason` names a missing
    input (the dividend amount, a reference close, a holdings file); it is None when the inputs exist, and then n
    may still be below 3 (too few deep in-the-money calls held across both closes)."""
    d0 = nyse.previous_session(nyse.previous_session(ex))
    amount = dict(inp.dividends).get(ex)
    if amount is None or not np.isfinite(amount):
        return 0, float("nan"), f"no dividend amount for {ex}"
    for d in (d0, ex):
        if d not in inp.prices or not np.isfinite(inp.prices[d][0]):
            return 0, float("nan"), f"no SPY reference close for {d}"
        if d not in set(inp.state["as_of"]):
            return 0, float("nan"), f"no holdings file for {d}"
    if inp.calls is None:
        return 0, float("nan"), None
    s0, s1 = inp.prices[d0][0], inp.prices[ex][0]
    c = inp.calls
    a = c[c["as_of"] == d0].groupby(["strike", "expiry"])["price"].first()
    b = c[c["as_of"] == ex].groupby(["strike", "expiry"])["price"].first()
    both = a.index.intersection(b.index)
    if len(both) == 0:
        return 0, float("nan"), None
    ratios = [((b[k] - a[k]) - (s1 - s0)) / amount for k in both]
    return len(both), float(np.median(ratios)), None


def h7a_ratio(inp, year):
    """Median S&P/SPY ratio over the 7 sessions centered on the session nearest Jan 10 (sessions with both)."""
    jan10 = date(year, 1, 10)
    near = min(nyse.sessions(jan10 - timedelta(days=7), jan10 + timedelta(days=7)), key=lambda d: abs((d - jan10).days))
    before = [nyse.previous_session(near)]
    for _ in range(2):
        before.insert(0, nyse.previous_session(before[0]))
    after = [nyse.next_session(near)]
    for _ in range(2):
        after.append(nyse.next_session(after[-1]))
    vals = [inp.prices[d][1] / inp.prices[d][0] for d in before + [near] + after
            if d in inp.prices and not any(np.isnan(inp.prices[d]))]
    return (float(np.median(vals)) if vals else float("nan")), len(vals), after[-1]


# ---------------------------------------------------------------- reviews

WINDOW_TESTS = {"H2": t_h2, "H3": t_h3, "H4": t_h4, "H6": t_h6, "H8": t_h8, "H10": t_h10, "H11": t_h11,
                "H12": t_h12, "H13": t_h13, "H14": t_h14, "H15": t_h15, "H16": t_h16, "H17": t_h17, "H18": t_h18,
                "H19": t_h19}


def cutoff_for(d):
    """Files downloaded by the end of day d (UTC) count for a review on d (P2)."""
    return f"{d.isoformat()}T23:59:59Z"


def void_before(inp, review):
    hits = [(d, e) for d, e in inp.voids if d < review]
    return hits[0] if hits else None


def review_window_test(hid, today, provider):
    """Fixed-review tests, with P1's minimum-sample extension (P2's list) of up to three months."""
    test, review = WINDOW_TESTS[hid], WINDOW_REVIEWS[hid]
    rv = review.isoformat()
    if today < review:
        inp = provider(cutoff_for(today))
        n, ok, stats = test(inp, today + timedelta(days=1))
        return Result(hid, "pending", rv, n, stats, f"{TEST_START} to {today} (provisional)")
    inp = provider(cutoff_for(review))
    v = void_before(inp, review)
    if v:
        return Result(hid, "void", rv, note=f"{v[1]} on {v[0]}", decided=review)
    n, ok, stats = test(inp, review)
    need = MIN_SAMPLE.get(hid)
    if need is None or n >= need:
        if ok is None:
            return Result(hid, "no data", rv, n, stats, note="nothing to compute in the window")
        return Result(hid, "pass" if ok else "fail", rv, n, stats, f"to {review}", decided=review)
    last = add_months(review, 3)
    last_close = last if nyse.is_session(last) else nyse.previous_session(last)   # latest completing close allowed
    final = nyse.next_session(last_close)                                          # the extension's last review
    probe = provider(cutoff_for(min(today, final)))     # find the completing close cheaply on the latest data ...
    for d in sorted(set(frame(probe)["as_of"])):
        if d < review or d > last_close:
            continue
        new_review = nyse.next_session(d)
        if new_review > today:
            break
        if test(probe, new_review)[0] < need:
            continue
        inp2 = provider(cutoff_for(new_review))          # ... then confirm on the data captured by that date
        n2, ok2, stats2 = test(inp2, new_review)
        if n2 >= need:
            v = void_before(inp2, new_review)
            if v:
                return Result(hid, "void", new_review.isoformat(), note=f"{v[1]} on {v[0]}", decided=new_review)
            return Result(hid, "pass" if ok2 else "fail", f"{new_review} (extended)", n2, stats2, f"to {new_review}",
                          decided=new_review)
    if today >= final:
        inp3 = provider(cutoff_for(final))
        v = void_before(inp3, final)
        if v:
            return Result(hid, "void", final.isoformat(), note=f"{v[1]} on {v[0]}", decided=final)
        n3, ok3, stats3 = test(inp3, final)
        if n3 >= need and ok3 is not None:              # the completing close arrived late but by the final review
            return Result(hid, "pass" if ok3 else "fail", f"{final} (extended)", n3, stats3, f"to {final}",
                          decided=final)
        return Result(hid, "inconclusive", f"{final} (extended)", n3, stats3, note=f"fewer than {need} by {last_close}",
                      decided=final)
    return Result(hid, "pending", f"{rv} (extending)", n, stats,
                  note=f"{n} of {need} so far; extension to the close of {last_close}")


def review_events(hid, today, provider, provider_first=None):
    """H1 and H9: review after the 15th creation event, else 'inconclusive' on 2027-10-14. For H1, P2 also asks for
    the same statistic from the first saved version of each holdings file (`provider_first`); it informs, the latest
    version decides."""
    stat = h1_stat if hid == "H1" else h9_stat
    inp = provider(cutoff_for(today))
    events, clean = stat(inp)
    if len(events) < 15:
        if today >= EVENT_FINAL:
            return Result(hid, "inconclusive", str(EVENT_FINAL), len(events), note="fewer than 15 creation events",
                          decided=EVENT_FINAL)
        return Result(hid, "pending", "after the 15th creation event", len(events), _event_stats(hid, events, clean),
                      note=f"{len(events)} of 15 events so far")
    review = nyse.next_session(events[14][0])
    if today < review:
        return Result(hid, "pending", str(review), len(events), note="15th event complete; review next session")
    while True:                                         # the review needs 15 events in the data captured by then
        inp = provider(cutoff_for(review))
        v = void_before(inp, review)
        if v:
            return Result(hid, "void", str(review), note=f"{v[1]} on {v[0]}", decided=review)
        events = [e for e in stat(inp)[0] if e[0] < review]
        if len(events) >= 15:
            break
        review = nyse.next_session(review)
        if review > today:
            return Result(hid, "pending", str(review), len(events),
                          note="15th event not yet in the data captured by the review; review moves to the next session")
    events = events[:15]
    fifteenth = events[14][0]
    clean = clean_until(inp, hid, fifteenth)
    stats = _event_stats(hid, events, clean)
    if not stats or len(clean) < 2 or not np.isfinite(stats["p_value"]):
        return Result(hid, "no data", str(review), len(events), stats, note="fewer than 2 clean comparison days")
    ok = (stats["difference"] < -0.10 and stats["p_value"] < 0.05) if hid == "H1" else \
         (stats["difference"] > 0.10 and stats["p_value"] < 0.05)
    if hid == "H1" and provider_first is not None:
        inp1 = provider_first(cutoff_for(review))
        ev1 = h1_stat(inp1)[0]
        first = _event_stats(hid, [e for e in ev1 if e[0] <= fifteenth], clean_until(inp1, hid, fifteenth))
        stats.update({f"first_version_{k}": v for k, v in first.items() if k != "clean_n"})
    return Result(hid, "pass" if ok else "fail", str(review), len(events), stats, decided=review)


def clean_until(inp, hid, last_day):
    """Clean-day comparison values up to the review's last event day."""
    if hid == "H1":
        c = changes(frame(inp), ["iv_pts"], TEST_START, last_day + timedelta(days=1))
        return (c["iv_pts"] - c["iv_pts_prev"]).to_numpy()
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp)
    return np.asarray([f.loc[d, "iv_pts"] - v[0][10000][1] for d, v in lv.items()
                       if LISTED_START <= d <= last_day and 10000 in v[0] and d in f.index and f.loc[d, "clean"]
                       and not np.isnan(f.loc[d, "iv_pts"])], float)


def _event_stats(hid, events, clean):
    if not events or len(clean) == 0:
        return {}
    a = np.asarray([e[1] for e in events], float)
    diff = float(a.mean() - clean.mean())
    p = welch_one_sided(a, clean, "less" if hid == "H1" else "greater")
    return {"difference": diff, "p_value": p, "clean_n": int(len(clean))}


def review_h5(today, provider):
    inp = provider(cutoff_for(today))
    exs = [ex for ex, _ in inp.dividends if ex >= date(2026, 12, 1)]
    if not exs:
        return Result("H5", "pending", str(H5_FIRST_REVIEW), note="no December 2026 ex-date in the dividend file yet")
    for ex in exs:
        review = max(H5_FIRST_REVIEW, nyse.next_session(ex)) if ex == exs[0] else nyse.next_session(ex)
        if today < review:
            return Result("H5", "pending", str(review), note=f"ex-date {ex}")
        inp = provider(cutoff_for(review))
        v = void_before(inp, review)
        if v:
            return Result("H5", "void", str(review), note=f"{v[1]} on {v[0]}", decided=review)
        n, med, why = h5_compute(inp, ex)
        if why:                                         # missing input: never final, never moves to the next ex-date
            return Result("H5", "no data", str(review), note=f"ex-date {ex}: {why} in the data captured by {review}")
        if n >= 3:
            return Result("H5", "pass" if 0.6 <= med <= 1.4 else "fail", str(review), n, {"median_ratio": med},
                          f"ex-date {ex}", decided=review)
    return Result("H5", "pending", "next SPY ex-date", note="fewer than 3 deep in-the-money calls so far")


def review_h7(today, provider):
    if today <= XX_EXPIRY:
        return Result("H7", "pending", str(XX_EXPIRY))
    # The expiry closes are read from the data captured in the week after the expiry (no void: index data only).
    review = nyse.next_session(XX_EXPIRY)
    used = max(review, min(today, review + timedelta(days=7)))
    inp = provider(cutoff_for(used))
    spy, spx = inp.prices.get(XX_EXPIRY, (np.nan, np.nan))
    if np.isnan(spy) or np.isnan(spx):
        return Result("H7", "pending", str(XX_EXPIRY), note="no reference closes for 2030-01-10 yet")
    trig = XX_STRIKE * spx / spy
    return Result("H7", "pass" if 9968 <= trig <= 10002 else "fail", str(XX_EXPIRY), 1, {"trigger": trig},
                  decided=used)


def review_h7a(today, provider):
    checks, prev = [], H7A_BASELINE                    # no void: H7a reads index closes only (header)
    for year, review in H7A_CHECKPOINTS.items():
        if today < review:
            checks.append((year, None))
            break
        inp = provider(cutoff_for(review))
        r, n, _ = h7a_ratio(inp, year)
        if n == 0:
            checks.append((year, "no data"))
            prev = float("nan")
            continue
        change = r / prev - 1.0 if not np.isnan(prev) else float("nan")
        checks.append((year, "pass" if -0.0002 <= change <= 0.0008 else "fail" if not np.isnan(change) else "no data"))
        prev = r
    words = [c[1] for c in checks]
    stats = {str(y): (w or "pending") for y, w in checks}
    if None in words:
        return Result("H7a", "pending", "2027-01-20, 2028-01-20, 2029-01-19", stats=stats)
    overall = "fail" if "fail" in words else "pass" if all(w == "pass" for w in words) else "inconclusive"
    return Result("H7a", overall, "2029-01-19", len(words), stats, decided=H7A_CHECKPOINTS[2029])


def one_result(hid, today, provider, provider_first=None):
    if hid in WINDOW_TESTS:
        return review_window_test(hid, today, provider)
    if hid in ("H1", "H9"):
        return review_events(hid, today, provider, provider_first)
    if hid == "H5":
        return review_h5(today, provider)
    if hid == "H7":
        return review_h7(today, provider)
    return review_h7a(today, provider)


def all_results(today, provider, provider_first=None):
    """Every hypothesis's status. A crash in one (bad data) shows as 'error' for that one only; never final."""
    out = []
    for hid in ORDER:
        try:
            out.append(one_result(hid, today, provider, provider_first))
        except Exception as e:                          # noqa: BLE001 - one bad input must not hide the others
            out.append(Result(hid, "error", "-", note=f"{type(e).__name__}: {e}"))
    return out


# ---------------------------------------------------------------- recording (scripts/verdicts.py --record)

FINAL = {"pass", "fail", "inconclusive", "void"}


def recorded(entries):
    """{(id, checkpoint or None)} already in the ledger."""
    return {(e["id"], (e.get("evidence") or {}).get("checkpoint")) for e in entries if e.get("kind") == "verdict"}


def to_record(results, entries, now_utc):
    """(ready, waiting): final words not yet in the ledger. A word is ready only after the day whose captured data
    decided it has ended (its cutoff is in the past at `now_utc`, an ISO 'YYYY-MM-DDTHH:MM:SSZ' string); before
    that, a later capture could still change it, so it waits."""
    done, ready, waiting = recorded(entries), [], []
    for r in results:
        items = []
        if r.id == "H7a":
            for year, word in r.stats.items():
                if word in ("pass", "fail"):
                    items.append((year, word, H7A_CHECKPOINTS[int(year)]))
            if r.status in FINAL:
                items.append(("overall", r.status, r.decided))
        elif r.status in FINAL:
            items.append((None, r.status, r.decided))
        for cp, word, decided in items:
            if (r.id, cp) in done:
                continue
            if decided is None or cutoff_for(decided) >= now_utc:
                waiting.append((r, cp, word, decided))
            else:
                ready.append((r, cp, word))
    return ready, waiting


# ---------------------------------------------------------------- loading real data

def load(root=None, cutoff=None, holdings="latest"):
    """Inputs from the data folder, using only files downloaded by `cutoff` (P2). holdings='first' rebuilds the
    state and model from the first saved version of each holdings file (P2: H1 reports that too)."""
    from . import curves, marks, valuation
    from .forward import fetch, listed as lst, parse, reconcile
    root = fetch.data_dir(root)
    with parse.captured_by(cutoff):
        xx = parse.load_holdings(root, accounts=("XX",), all_versions=(holdings == "first"))
        if holdings == "first" and not xx.empty:
            first = xx.groupby("as_of")["version"].transform("min")
            xx = xx[xx["version"] == first].copy()
            xx["latest_version"] = True
        daily = parse.load_fund_daily(root, "XX")
        state = reconcile.fund_state(xx, daily)
        every = parse.load_holdings(root, accounts=None, all_versions=False)
        opts = every[every["is_option"] & (every["opt_underlying"] == "SPY")] if not every.empty else every
        spots = marks.load_spots(root)
        curve_table, divs = curves.load_treasury(root), marks.load_dividends(root)
        model = marks.model_daily(state, curve_table, divs, spots, opts, skip_missing_spot=True)
        vix = {}
        recs = parse.saved_versions(root, "VIX1Y_History.csv")
        if recs:
            v = pd.read_csv(io.StringIO(parse.read_text(root, recs[-1])))
            vix = dict(zip(pd.to_datetime(v["DATE"], format="%m/%d/%Y").dt.date, v["CLOSE"].astype(float)))
    prices = price_rule(spots)
    deep = opts[(opts["opt_cp"] == "C") & (opts["opt_strike"] < 100)] if not opts.empty else opts
    calls = pd.DataFrame({"as_of": deep["as_of"], "account": deep["account"], "strike": deep["opt_strike"],
                          "expiry": deep["opt_expiry"], "price": deep["price"]}) if len(deep) else None
    listed_rows = counted_listed(root, cutoff)

    def listed_forward(close):
        cv = curves.curve_on(curve_table, close, spread=0.0)
        inp = valuation.Inputs(close, prices[close][0], XX_STRIKE, LISTED_EXPIRY, cv, divs)
        F, DF, tau, _ = valuation.forward(inp)
        return F, DF, tau

    return Inputs(state=state, model=model, prices=prices, vix1y=vix, dividends=divs, calls=calls,
                  listed=listed_rows, listed_forward=listed_forward, voids=find_voids(state, root), cutoff=cutoff)


def price_rule(spots):
    """P2's price rule: the issuer's reference closes, else a close the user added in data/external/closes.csv;
    never a TradingView close (tv_closes.csv counts as missing). spots: {date: (spy, spx, source label)}."""
    return {d: (spy, spx) for d, (spy, spx, src) in spots.items()
            if src.startswith("Roundhill") or src.startswith("user-supplied close")}


def counted_listed(root, cutoff, ledger_path=None):
    """Listed readings that count under the rule in force: P2's read window, or P3's market-time window for readings
    taken after a P3 protocol entry was recorded in the ledger. One reading per close (P2: the first in the window;
    P3: the latest); a P3 reading replaces a P2 reading of the same close."""
    from . import ledger
    from .forward import listed as lst
    from datetime import datetime, timezone
    entries = ledger.read(ledger_path) if ledger_path else ledger.read()
    p3_from = min((e["recorded_utc"] for e in entries if e.get("id") == "P3" and e.get("kind") == "protocol"),
                  default=None)
    raw = root / "raw" / "tradingview_chain"
    chosen = {}
    for path in sorted(raw.glob("*/*.json")) if raw.exists() else []:
        try:
            read = datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        stamp = read.strftime("%Y-%m-%dT%H:%M:%SZ")
        if cutoff and stamp > cutoff:
            continue
        text = path.read_text(encoding="utf-8")
        p3 = p3_from is not None and stamp > p3_from
        if p3:
            delay = lst.feed_delay_seconds(text)
            if delay is None:
                continue
            close, _, ok = lst.p3_market_time(read, delay)
        else:
            close, ok = lst.close_for(read)
        if not ok:
            continue
        old = chosen.get(close)
        if old is None or (p3 and (not old[2] or read > old[0])) or (not p3 and not old[2] and read < old[0]):
            chosen[close] = (read, text, p3)
    rows = []
    for close, (read, text, _) in chosen.items():
        quotes, _ = lst.parse(text)
        for sym, (bid, ask) in quotes.items():
            if sym in lst.REFERENCE:
                continue
            rows.append({"close_date": close, "strike": int(float(sym.split("C")[-1])), "bid": bid, "ask": ask})
    return pd.DataFrame(rows, columns=["close_date", "strike", "bid", "ask"])


def _option_line(r):
    """(strike, expiry) of the close's single option line, or None (no line, or several: strike shows as NaN)."""
    k, e = r["strike"], r["expiry"]
    if k is None or e is None or pd.isna(k) or pd.isna(e):
        return None
    return round(float(k), 2), str(e)


def find_voids(state, root=None, events_csv=None):
    """P1 void events: an option-line change (a roll), a reverse split, or an event listed by hand in
    data/external/void_events.csv (columns date, event, source; source names the filing or notice, required).

    A roll needs one new option line, with a real strike, on two consecutive closes: a one-day file glitch (two lines,
    or a blank) is not a roll. A reverse split: shares at most half and NAV at least 1.8x the previous close."""
    out = []
    if state is not None and not state.empty:
        s = state.sort_values("as_of").reset_index(drop=True)
        lines = [_option_line(s.iloc[i]) for i in range(len(s))]
        for i in range(len(s) - 1):
            if lines[i] is not None and lines[i] != (XX_STRIKE, str(XX_EXPIRY)) and lines[i + 1] == lines[i]:
                out.append((s.iloc[i]["as_of"], f"option line changed to {lines[i][0]} {lines[i][1]}"))
                break
        for i in range(1, len(s)):
            a, b = s.iloc[i - 1], s.iloc[i]
            if b["shares_outstanding"] <= 0.5 * a["shares_outstanding"] * (1 + 1e-9) and \
                    b["nav_published"] >= 1.8 * a["nav_published"]:
                out.append((b["as_of"], "reverse split"))
    path = events_csv or (root / "external" / "void_events.csv" if root is not None else None)
    if path is not None and path.exists():
        ev = pd.read_csv(path, dtype=str).fillna("")
        if "source" not in ev.columns or (ev["source"].str.strip() == "").any():
            raise ValueError(f"{path.name}: every void event needs a source (the filing or notice that announced it)")
        out += [(date.fromisoformat(d.strip()), f"{e} ({src})") for d, e, src in zip(ev["date"], ev["event"],
                                                                                     ev["source"])]
    return sorted(out)
