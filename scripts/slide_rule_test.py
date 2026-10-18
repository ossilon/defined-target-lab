#!/usr/bin/env python3
"""Does the "vol follows the index" slide rule change XX's life on REAL paths? (v0.3 design review)

    py -I scripts\\slide_rule_test.py            From target_funds/. Inputs: data/external/spx_daily_library.csv
                                                 (S&P 500 closes) and the captured Cboe VIX1Y history.

Fixed before the first run (2026-10-17):
- A synthetic XX starts at every 21st session from 2007-01-03 that has 815 sessions of real S&P 500 history ahead
  (XX's life from 2026-10-13 to 2030-01-10). Same shape as XX: one call per share-unit at 1.27892 x the start
  level, cash 0.2474 / 778.01 per $1 of start level, 0.99% fee paid by selling options, rate 4.86498% and dividend
  yield 1.03% held constant (to isolate vol), year fractions ACT/365 from the real session dates.
- Each is marked every session three ways, all starting at the same implied vol v0 = 0.77 x VIX1Y on the start day
  (0.77: the implied-vol / VIX1Y multiplier from the design review of 2026-10-14, an assumption):
  (A) constant v0; (B) the slide rule, v0 + 1 point per -10% of the S&P since the start (sim.vol_at, F12);
  (C) 0.77 x the real VIX1Y each day.
- Reported across starts (about 200, overlapping; about 5-6 separate spans): median and 90th percentile of the
  largest NAV drawdown, the longest spell below a previous high (sessions), the highest NAV / start, the share of
  starts that touched 2x, for A, B and C; the median and 90th percentile of |B - C| and |A - C| in largest drawdown
  (percentage points) and in highest NAV / start; and the slide of 0.77 x VIX1Y measured on the real data (points of
  implied vol per -10% of the S&P, from a regression of daily changes on daily log returns, and from 63-session
  changes). End values are identical in A, B and C (the payoff ignores vol) and are not reported.
Prints aggregates only; research, not investment advice.
"""
import io
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from definedtarget import pricing, sim  # noqa: E402
from definedtarget.forward import fetch, parse  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402

LIFE, EVERY, M_MULT, SLIDE = 815, 21, 0.77, 0.01
K, CASH, FEE, R, Q = 1.27892, 0.2474 / 778.01, 0.0099, 0.0486498, 0.0103


def load():
    spx = pd.read_csv(REPO / "data" / "external" / "spx_daily_library.csv", parse_dates=["date"])
    rec = parse.saved_versions(None, "VIX1Y_History.csv")[-1]
    v = pd.read_csv(io.StringIO(parse.read_text(None, rec)))
    v["date"] = pd.to_datetime(v["DATE"], format="%m/%d/%Y")
    df = spx.merge(v[["date", "CLOSE"]].rename(columns={"CLOSE": "vix1y"}), on="date").sort_values("date")
    # the fetch DATE only: public text carries no clock times of the author's activity
    return df.reset_index(drop=True), rec["fetched_utc"][:10], fetch.data_dir(None) / rec["path"]


def navs(S, t, vols):
    """NAV per $1 of start level along one path (vectorized Black-76; payoff at the end)."""
    T = t[-1]
    tau = T - t
    out = np.empty(S.size)
    live = tau > 0
    F = S[live] * np.exp((R - Q) * tau[live])
    DF = np.exp(-R * tau[live])
    out[live] = pricing.call_price_array(F, np.full(F.size, K), DF, tau[live], vols[live])
    out[~live] = np.maximum(S[~live] - K, 0.0)
    return np.exp(-FEE * t) * out + CASH


def life_stats(nav):
    top = np.maximum.accumulate(nav)
    mdd = float((1 - nav / top).max())
    under, spell = 0, 0
    for i in range(1, nav.size):
        spell = 0 if nav[i] > top[i - 1] else spell + 1
        under = max(under, spell)
    return mdd, under, float(nav.max() / nav[0]), bool((nav >= 2 * nav[0]).any())


def main():
    spx_path = REPO / "data" / "external" / "spx_daily_library.csv"
    df, fetched, vix_path = load()
    with manifest.run("slide_rule_test", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/slide_rule_test.py",),
                      inputs={"spx": spx_path, "vix1y": vix_path},
                      settings={"life": LIFE, "every": EVERY, "m": M_MULT, "slide": SLIDE}) as m:
        report(df, fetched)
    print(f"run manifest: results/manifests/{m['path'].name}")


def report(df, fetched):
    S_all, V_all = df["close"].to_numpy(float), df["vix1y"].to_numpy(float) / 100.0
    dates = df["date"].to_numpy()
    print(f"S&P 500 and VIX1Y on common sessions: {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()} "
          f"({len(df):,} sessions; VIX1Y file fetched {fetched})")
    starts = range(0, len(df) - LIFE, EVERY)
    res = {k: [] for k in "ABC"}
    for s in starts:
        S = S_all[s:s + LIFE + 1] / S_all[s]
        t = (df["date"].iloc[s:s + LIFE + 1] - df["date"].iloc[s]).dt.days.to_numpy(float) / 365.0
        v0 = M_MULT * V_all[s]
        vols = {"A": np.full(S.size, v0),
                "B": np.maximum(v0 + SLIDE * np.log(S) / math.log(0.9), sim.VOL_FLOOR),
                "C": M_MULT * V_all[s:s + LIFE + 1]}
        for k, v in vols.items():
            res[k].append(life_stats(navs(S, t, v)))
    arr = {k: np.array(v, dtype=float) for k, v in res.items()}
    n = len(arr["A"])
    print(f"{n} synthetic XXs (every {EVERY} sessions, {LIFE}-session lives, about {len(df) / LIFE:.1f} separate spans)\n")
    print("| marking | largest drawdown: median / p90 | longest underwater (sessions): median / p90 | highest NAV / start: "
          "median / p90 | share that touched 2x |")
    print("|---|---|---|---|---|")
    names = {"A": "A constant vol", "B": "B slide rule (1 pt per -10%)", "C": "C real VIX1Y x 0.77"}
    for k in "ABC":
        a = arr[k]
        print(f"| {names[k]} | {np.median(a[:, 0]):.1%} / {np.quantile(a[:, 0], 0.9):.1%} | {np.median(a[:, 1]):.0f} / "
              f"{np.quantile(a[:, 1], 0.9):.0f} | {np.median(a[:, 2]):.2f} / {np.quantile(a[:, 2], 0.9):.2f} | "
              f"{a[:, 3].mean():.1%} |")
    print("\n| gap | largest drawdown, points: median / p90 | highest NAV / start: median / p90 |")
    print("|---|---|---|")
    for x, y in (("B", "C"), ("A", "C")):
        d1 = np.abs(arr[x][:, 0] - arr[y][:, 0]) * 100
        d2 = np.abs(arr[x][:, 2] - arr[y][:, 2])
        print(f"| {x} vs {y} | {np.median(d1):.1f} / {np.quantile(d1, 0.9):.1f} | {np.median(d2):.2f} / "
              f"{np.quantile(d2, 0.9):.2f} |")
    iv = M_MULT * V_all
    lr = np.diff(np.log(S_all))
    beta = np.polyfit(lr, np.diff(iv), 1)[0]
    lr63 = np.log(S_all[63:] / S_all[:-63])
    beta63 = np.polyfit(lr63, iv[63:] - iv[:-63], 1)[0]
    to_pts = lambda b: 100 * b * math.log(0.9)  # noqa: E731  (points of implied vol per -10% of the S&P)
    print(f"\nmeasured slide of 0.77 x VIX1Y, points per -10% of the S&P: daily {to_pts(beta):.2f}, 63-session "
          f"{to_pts(beta63):.2f} (the rule: 1.00; VIX1Y itself: divide by 0.77)")


if __name__ == "__main__":
    sys.exit(main())
