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
    the window includes that close, and reads the data captured by then;
  * under P2 several in-window readings of one close may exist: the first counts (nearest the close).
- scripts/verdicts.py prints every status and, only when asked, appends a verdict to the ledger.
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
        if not r["flow"]:
            if r["cps"] > p["cps"] + 1e-12:
                rises.append(str(r["as_of"]))
        else:
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

def listed_vols(inp, strikes=(10000,)):
    """{close: {strike: (bid_iv, mid_iv, ask_iv) in points}} for counted listed readings, P1/P2 conventions."""
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
        q = {int(r.strike): (float(r.bid), float(r.ask)) for r in g.itertuples()}
        row = {}
        for k in strikes:
            if k in q:
                b, a = q[k]
                row[k] = tuple(pricing.implied_vol(x / ratio, F, k / ratio, DF, tau) * 100 for x in (b, (b + a) / 2, a))
        out[close] = (row, q, DF)
    return out


def t_h13(inp, end):
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp)
    rows = [(d, v[0][10000]) for d, v in lv.items() if LISTED_START <= d < end and 10000 in v[0]
            and d in f.index and not np.isnan(f.loc[d, "iv_pts"])]
    if not rows:
        return 0, None, {}
    ok = [f.loc[d, "iv_pts"] >= b - 0.5 and f.loc[d, "iv_pts"] <= m for d, (b, m, a) in rows]
    share = float(np.mean(ok))
    return len(rows), share >= 0.80, {"share_in_band": share}


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
    stats = {"listed_coef": coef[1]}
    xx = [(f.loc[b, "iv_pts"] - f.loc[a, "iv_pts"], pct(f.loc[b, "spy"], f.loc[a, "spy"]),
           f.loc[b, "vix1y"] - f.loc[a, "vix1y"]) for a, b in zip(days, days[1:])]
    xx = [r for r in xx if not any(np.isnan(r))]
    if len(xx) >= 4:
        dy, ds, dv = zip(*xx)
        stats["xx_h10_coef_same_closes"] = ols(dy, [ds, dv])[0][1]
    return len(days), -0.20 <= coef[1] <= 0.05, stats


def t_h19(inp, end):
    f = frame(inp).set_index("as_of")
    lv = listed_vols(inp, strikes=())
    gaps = []
    for d, (row, q, DF) in lv.items():
        if not (LISTED_START <= d < end) or d not in f.index or np.isnan(f.loc[d, "prob"]):
            continue
        below = [k for k in q if k < 10000]
        above = [k for k in q if k > 10000]
        k1 = 9600 if 9600 in q else (max(below) if below else None)
        k2 = 10400 if 10400 in q else (min(above) if above else None)
        if k1 is None or k2 is None:
            continue
        mid = {k: sum(q[k]) / 2 for k in (k1, k2)}
        p = (mid[k1] - mid[k2]) / ((k2 - k1) * DF)
        gaps.append(100 * (p - f.loc[d, "prob"]))
    if not gaps:
        return 0, None, {}
    share = _share_in(gaps, 5, 15)
    return len(gaps), share >= 0.80, {"share_in_band": share, "median_gap_pts": float(np.median(gaps))}


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
    """(n contracts, median ratio) for the two sessions ending on SPY ex-date `ex`."""
    d0 = nyse.previous_session(nyse.previous_session(ex))
    amount = dict(inp.dividends).get(ex)
    if amount is None or inp.calls is None or d0 not in inp.prices or ex not in inp.prices:
        return 0, float("nan")
    s0, s1 = inp.prices[d0][0], inp.prices[ex][0]
    c = inp.calls
    a = c[c["as_of"] == d0].groupby(["strike", "expiry"])["price"].first()
    b = c[c["as_of"] == ex].groupby(["strike", "expiry"])["price"].first()
    both = a.index.intersection(b.index)
    if len(both) == 0 or np.isnan(s0) or np.isnan(s1):
        return 0, float("nan")
    ratios = [((b[k] - a[k]) - (s1 - s0)) / amount for k in both]
    return len(both), float(np.median(ratios))


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
        return Result(hid, "void", rv, note=f"{v[1]} on {v[0]}")
    n, ok, stats = test(inp, review)
    need = MIN_SAMPLE.get(hid)
    if need is None or n >= need:
        if ok is None:
            return Result(hid, "no data", rv, n, stats, note="nothing to compute in the window")
        return Result(hid, "pass" if ok else "fail", rv, n, stats, f"to {review}")
    last = add_months(review, 3)
    probe = provider(cutoff_for(min(today, last)))      # find the completing close cheaply on the latest data ...
    for d in sorted(set(frame(probe)["as_of"])):
        if d < review or d > last:
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
                return Result(hid, "void", new_review.isoformat(), note=f"{v[1]} on {v[0]}")
            return Result(hid, "pass" if ok2 else "fail", f"{new_review} (extended)", n2, stats2, f"to {new_review}")
    if today >= last:
        inp3 = provider(cutoff_for(last))
        n3, _, stats3 = test(inp3, last)
        return Result(hid, "inconclusive", f"{last} (extended)", n3, stats3, note=f"fewer than {need} by {last}")
    return Result(hid, "pending", f"{rv} (extending)", n, stats, note=f"{n} of {need} so far; extension to {last}")


def review_events(hid, today, provider, provider_first=None):
    """H1 and H9: review after the 15th creation event, else 'inconclusive' on 2027-10-14. For H1, P2 also asks for
    the same statistic from the first saved version of each holdings file (`provider_first`); it informs, the latest
    version decides."""
    inp = provider(cutoff_for(today))
    events, clean = (h1_stat if hid == "H1" else h9_stat)(inp)
    if len(events) < 15:
        if today >= EVENT_FINAL:
            return Result(hid, "inconclusive", str(EVENT_FINAL), len(events), note="fewer than 15 creation events")
        return Result(hid, "pending", "after the 15th creation event", len(events), _event_stats(hid, events, clean),
                      note=f"{len(events)} of 15 events so far")
    fifteenth = events[14][0]
    review = nyse.next_session(fifteenth)
    if today < review:
        return Result(hid, "pending", str(review), len(events), note="15th event complete; review next session")
    inp = provider(cutoff_for(review))
    v = void_before(inp, review)
    if v:
        return Result(hid, "void", str(review), note=f"{v[1]} on {v[0]}")
    events, clean = (h1_stat if hid == "H1" else h9_stat)(inp)
    events = [e for e in events if e[0] <= fifteenth]
    clean = clean_until(inp, hid, fifteenth)
    stats = _event_stats(hid, events, clean)
    ok = (stats["difference"] < -0.10 and stats["p_value"] < 0.05) if hid == "H1" else \
         (stats["difference"] > 0.10 and stats["p_value"] < 0.05)
    if hid == "H1" and provider_first is not None:
        ev1, cl1 = h1_stat(provider_first(cutoff_for(review)))
        first = _event_stats(hid, [e for e in ev1 if e[0] <= fifteenth], cl1)
        stats.update({f"first_version_{k}": v for k, v in first.items() if k != "clean_n"})
    return Result(hid, "pass" if ok else "fail", str(review), len(events), stats)


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
            return Result("H5", "void", str(review), note=f"{v[1]} on {v[0]}")
        n, med = h5_compute(inp, ex)
        if n >= 3:
            return Result("H5", "pass" if 0.6 <= med <= 1.4 else "fail", str(review), n, {"median_ratio": med},
                          f"ex-date {ex}")
    return Result("H5", "pending", "next SPY ex-date", note="fewer than 3 deep in-the-money calls so far")


def review_h7(today, provider):
    if today <= XX_EXPIRY:
        return Result("H7", "pending", str(XX_EXPIRY))
    review = nyse.next_session(XX_EXPIRY)
    inp = provider(cutoff_for(max(review, min(today, review + timedelta(days=7)))))
    spy, spx = inp.prices.get(XX_EXPIRY, (np.nan, np.nan))
    if np.isnan(spy) or np.isnan(spx):
        return Result("H7", "pending", str(XX_EXPIRY), note="no reference closes for 2030-01-10 yet")
    trig = XX_STRIKE * spx / spy
    return Result("H7", "pass" if 9968 <= trig <= 10002 else "fail", str(XX_EXPIRY), 1, {"trigger": trig})


def review_h7a(today, provider):
    checks, prev = [], H7A_BASELINE
    for year, review in H7A_CHECKPOINTS.items():
        if today < review:
            checks.append((year, None))
            break
        inp = provider(cutoff_for(review))
        v = void_before(inp, review)
        if v:
            return Result("H7a", "void", str(review), note=f"{v[1]} on {v[0]}")
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
    return Result("H7a", overall, "2029-01-19", len(words), stats)


def all_results(today, provider, provider_first=None):
    out = []
    for hid in ORDER:
        if hid in WINDOW_TESTS:
            out.append(review_window_test(hid, today, provider))
        elif hid in ("H1", "H9"):
            out.append(review_events(hid, today, provider, provider_first))
        elif hid == "H5":
            out.append(review_h5(today, provider))
        elif hid == "H7":
            out.append(review_h7(today, provider))
        elif hid == "H7a":
            out.append(review_h7a(today, provider))
    return out


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
    prices = {d: (spy, spx) for d, (spy, spx, src) in spots.items()
              if src.startswith("Roundhill") or "closes.csv" in src}
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


def counted_listed(root, cutoff, ledger_path=None):
    """Listed readings that count under the rule in force: P2's read window, or P3's market-time window once a P3
    protocol entry exists in the ledger. One reading per close (P2: the first in the window; P3: the latest)."""
    from . import ledger
    from .forward import listed as lst
    from datetime import datetime, timezone
    entries = ledger.read(ledger_path) if ledger_path else ledger.read()
    p3 = any(e.get("id") == "P3" and e.get("kind") == "protocol" for e in entries)
    raw = root / "raw" / "tradingview_chain"
    chosen = {}
    for path in sorted(raw.glob("*/*.json")) if raw.exists() else []:
        try:
            read = datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if cutoff and read.strftime("%Y-%m-%dT%H:%M:%SZ") > cutoff:
            continue
        text = path.read_text(encoding="utf-8")
        if p3:
            delay = lst.feed_delay_seconds(text)
            if delay is None:
                continue
            close, _, ok = lst.p3_market_time(read, delay)
        else:
            close, ok = lst.close_for(read)
        if not ok:
            continue
        if close not in chosen or (p3 and read > chosen[close][0]) or (not p3 and read < chosen[close][0]):
            chosen[close] = (read, text)
    rows = []
    for close, (read, text) in chosen.items():
        quotes, _ = lst.parse(text)
        for sym, (bid, ask) in quotes.items():
            if sym in lst.REFERENCE:
                continue
            rows.append({"close_date": close, "strike": int(float(sym.split("C")[-1])), "bid": bid, "ask": ask})
    return pd.DataFrame(rows, columns=["close_date", "strike", "bid", "ask"])


def find_voids(state, root):
    """P1 void events: an option-line change (a roll), a reverse split, or an event listed by hand in
    data/external/void_events.csv (columns date, event: a closure or liquidation notice)."""
    out = []
    if state is not None and not state.empty:
        s = state.sort_values("as_of").reset_index(drop=True)
        for i in range(len(s)):
            r = s.iloc[i]
            if (round(float(r["strike"]), 2), str(r["expiry"])) != (XX_STRIKE, str(XX_EXPIRY)):
                out.append((r["as_of"], f"option line changed to {r['strike']} {r['expiry']}"))
                break
        for i in range(1, len(s)):
            a, b = s.iloc[i - 1], s.iloc[i]
            if b["shares_outstanding"] < 0.5 * a["shares_outstanding"] and b["nav_published"] > 1.8 * a["nav_published"]:
                out.append((b["as_of"], "reverse split"))
    path = root / "external" / "void_events.csv"
    if path.exists():
        ev = pd.read_csv(path)
        out += [(date.fromisoformat(str(d)), str(e)) for d, e in zip(ev["date"], ev["event"])]
    return sorted(out)
