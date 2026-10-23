#!/usr/bin/env python3
"""v0.3.1 step 4: the Heston model fitted to the VIX family and the listed December 2029 calls under the fixed holdout,
and judged out of sample.

    py -I scripts\\vol_holdout_study.py <S&P 500 daily closes CSV> [--fit-only] [--boot N]      From target_funds/.

PROTOCOL (fixed 2026-10-23 and committed before any test-window number was computed; the holdout and the accuracy
target were fixed 2026-10-21 in the private v0.3.1 brief):

Data. Cboe's VIX (30 days), VIX3M (93), VIX6M (184) and VIX1Y (366 calendar days; Cboe's methodology for the term
indices), daily closes, the latest copies in data/raw/cboe. VIX9D is left out: a one-factor Heston model cannot fit
the very short end (Gatheral 2006, ch. 3). The listed S&P 500 calls expiring 2029-12-21 (15 strikes, 8,000 to 15,000),
each trading day's last trade (TradingView bars, local only), turned into implied vols by definedtarget.listedvols
(F12's conventions). The S&P 500 closes come from the CSV given on the command line.

Stage 1, the term structure. Fit window 2009-09-18 to 2019-12-31 (days with all four indices). volmodel.termstructure
fits one kappa and one theta and each day's variance, with every index weighted equally (sd 0.5 vol points). A
moving-block bootstrap (blocks of 63 days, 200 resamples, seed 20261023) gives the band of theta and kappa.
Out of sample, 2020-01-01 to the last day: each day's variance from that day's VIX alone (kappa and theta fixed)
predicts VIX3M, VIX6M and VIX1Y. Reported: RMSE in vol points per index, against two benchmarks: flat (each index
equals that day's VIX) and the fit window's mean spread (VIX plus the mean of index minus VIX over the fit window).
No pass mark for this part.

Stage 2, the smile. Fit window: trade dates 2024-06-14 to 2025-06-30, every strike's last trade on days with at least 5
contracts. Each trade day's variance comes from that day's four indices (stage 1's kappa and theta). Residuals in vol
units, sd 1 vol point. Two variants:
- A (the VIX anchor): theta and kappa from stage 1; eta and rho fitted.
- B (the trade anchor, PRIMARY; the brief's "long-run variance from the 1-year tenor plus the 3-year trades"):
  kappa from stage 1; the pricing theta, eta and rho fitted. Each day's variance still comes from stage 1's theta.

The accuracy target (PRIMARY result). Test window: trade dates 2025-07-01 to the last trade, the 10,000 strike on days
with at least 5 contracts. Each day's prediction: variant B's model at that day's variance, forward and maturity.
PASS if its RMSE is at most 1.5 vol points AND below each benchmark's RMSE:
1. constant 16.6%;
2. 0.77 x VIX1Y;
3. sticky strike with a level shift: the fit window's mean traded vol (10,000 strike, 5+ contracts) plus the change
   of VIX1Y from its mean on those days, one point for one point;
4. the slide rule: the last fit-window traded vol (10,000 strike, 5+ contracts) plus 1.0 point per 10% fall of the
   S&P 500 from that day's close (the simulator's default slide, FINDINGS F12).
A miss is a result. Variant A is reported beside B; it does not change the verdict.

The decisive mystery test (depth pass rank 7). The 10,000 strike's last trade of each ISO week (any volume). For each
pair of consecutive weekly observations, the change in traded vol against the change each reading predicts: the slide
rule (1.0 point per -10%), the level shift (the change of VIX1Y), sticky strike (no change) and the dynamics (the
change of variant B's model vol). Reported per reading: RMSE of the error in the change, and the OLS slope of the
traded change on the predicted change with its standard error (a reading that describes the moves has a slope near 1).
PRIMARY: pairs with both weeks in the test window; the full history beside it. Weekly spacing limits the timing noise
of single trades (the errors-in-variables issue of F12).

The belief probe (depth pass rank 8). All 10,000-strike trade days with at least 5 contracts: OLS of ln(iv / VIX1Y) on
ln(VIX1Y), slope with its standard error; and the same at weekly spacing. A slope of 0 means the 3-year strike vol is a
fixed multiple of VIX1Y; -1 means it does not move with VIX1Y at all. Interpreted, never tuned.

Already seen before this protocol (disclosed): F12 (2026-10-17) used the 10,000 strike's trades from 2024-06 to 2026-09,
including the test window, for its slide of +1.1 points per -10% and the 2025 sell-off example; F18 used VIX1Y since
2007. The constant 16.6% and the 0.77 multiple come from XX's October 2026 marks (test-window information; they are the
benchmarks the brief named). Variant B's parameters see no test-window data.

--fit-only prints stages 1 and 2 on their fit windows and nothing from the test windows. Aggregates only; the data
stay local; research, not investment advice.
"""
import argparse
import math
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from definedtarget import curves, listedvols, marks  # noqa: E402
from definedtarget.forward import parse  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.volmodel import calibrate, pricers, termstructure as ts  # noqa: E402
from definedtarget.volmodel.heston import Heston  # noqa: E402

SEED = 20261023
INDICES = ("VIX", "VIX3M", "VIX6M", "VIX1Y")
TENORS = np.array([30, 93, 184, 366]) / 365.0
SD_INDEX = 0.005
FIT_VIX, TEST_VIX_START = (date(2009, 9, 18), date(2019, 12, 31)), date(2020, 1, 1)
FIT_TRADES, TEST_TRADES_START = (date(2024, 6, 14), date(2025, 6, 30)), date(2025, 7, 1)
MIN_CONTRACTS = 5


def load_index(name):
    rec = parse.saved_versions(None, f"{name}_History.csv")[-1]
    out = {}
    for line in parse.read_text(None, rec).splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[-1].strip():
            m, d, y = (int(x) for x in parts[0].split("/"))
            out[date(y, m, d)] = float(parts[-1]) / 100.0          # the close (last column), as a decimal vol
    return out


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(math.sqrt(np.mean(e * e)))


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    b = (xm * ym).sum() / (xm * xm).sum()
    res = ym - b * xm
    se = math.sqrt((res * res).sum() / (x.size - 2) / (xm * xm).sum())
    return b, se


def model_iv(model, v, F, K, T):
    m = Heston(max(v, 0.0), model.theta, model.kappa, model.eta, model.rho)
    return float(pricers.implied_vols(m, F, np.array([K]), T)[0])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("closes")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)

    idx = {n: load_index(n) for n in INDICES}
    days = sorted(set.intersection(*(set(idx[n]) for n in INDICES)))
    vols = np.array([[idx[n][d] for n in INDICES] for d in days])
    sd = np.full(4, SD_INDEX)
    fit_mask = np.array([FIT_VIX[0] <= d <= FIT_VIX[1] for d in days])
    tf = ts.fit(TENORS, vols[fit_mask], sd)
    boot, conv = ts.bootstrap(TENORS, vols[fit_mask], sd, block=63, reps=a.boot, seed=SEED, return_converged=True)
    print("Index files end: " + ", ".join(f"{n} {max(idx[n])}" for n in INDICES) + f"; common end {days[-1]}")
    print(f"Stage 1, the term structure: {fit_mask.sum()} days from {FIT_VIX[0]} to {FIT_VIX[1]}")
    print(f"  kappa {tf.kappa:.4f} (5-95% {np.percentile(boot[:, 0], 5):.4f} to {np.percentile(boot[:, 0], 95):.4f}); "
          f"theta {tf.theta:.5f}, long-run vol {100 * math.sqrt(tf.theta):.2f}% (5-95% "
          f"{100 * math.sqrt(np.percentile(boot[:, 1], 5)):.2f}% to {100 * math.sqrt(np.percentile(boot[:, 1], 95)):.2f}%)")
    # added after the code review (2026-10-23): how many refits converged, and how many ran off
    runaway = int((np.sqrt(boot[:, 1]) > 1.0).sum())
    print(f"  bootstrap: {int(conv.sum())} of {len(conv)} refits converged; {runaway} put the long-run vol above 100% "
          f"(the upper end is not bounded); over converged refits only, long-run vol 5-95% "
          f"{100 * math.sqrt(np.percentile(boot[conv, 1], 5)):.2f}% to "
          f"{100 * math.sqrt(np.percentile(boot[conv, 1], 95)):.2f}%")
    print("  in-sample RMSE (vol points): " + ", ".join(f"{n} {100 * r:.2f}" for n, r in zip(INDICES, tf.rmse)))
    k1, th1 = tf.kappa, tf.theta
    print(f"  model 3-year variance-swap vol at the long-run level: {100 * math.sqrt(th1):.2f}%; the 3-year rate at the "
          f"fit window's median state: {100 * float(ts.predict(k1, th1, np.median(tf.v), 3.0)):.2f}%")

    closes = listedvols.read_closes_csv(a.closes)
    rows = listedvols.trade_vols(listedvols.load_bars(), closes, curves.load_treasury(), marks.load_dividends(),
                                 min_volume=1)
    for r in rows:
        iv_idx = [idx[n].get(r["date"]) for n in INDICES]
        r["v"] = float(ts.states(k1, th1, TENORS, np.array([[np.nan if x is None else x for x in iv_idx]]), sd)[0])
        r["vix1y"] = idx["VIX1Y"].get(r["date"])
    rows = [r for r in rows if np.isfinite(r["iv"]) and np.isfinite(r["v"]) and r["vix1y"] is not None]
    fit_rows = [r for r in rows if FIT_TRADES[0] <= r["date"] <= FIT_TRADES[1] and r["volume"] >= MIN_CONTRACTS]
    by_day = {}
    for r in fit_rows:
        by_day.setdefault(r["date"], []).append(r)
    dlist = [calibrate.Day(v=rs[0]["v"], F=rs[0]["F"], T=rs[0]["T"], K=np.array([r["strike"] for r in rs]),
                           iv=np.array([r["iv"] for r in rs])) for _, rs in sorted(by_day.items())]
    start = Heston(0.03, th1, k1, 0.5, -0.7)
    fa = calibrate.fit_heston_days(dlist, start, fixed=("v0", "kappa", "theta"))
    fb = calibrate.fit_heston_days(dlist, start, fixed=("v0", "kappa"))
    print(f"\nStage 2, the smile: {len(fit_rows)} trades on {len(dlist)} days from {FIT_TRADES[0]} to {FIT_TRADES[1]} "
          f"({MIN_CONTRACTS}+ contracts, all strikes)")
    for label, f in (("A (VIX anchor)", fa), ("B (trade anchor)", fb)):
        m = f.model
        print(f"  {label}: theta {m.theta:.5f} (long-run vol {100 * math.sqrt(m.theta):.2f}%), eta {m.eta:.3f}, "
              f"rho {m.rho:.3f}; SE theta {f.se['theta']:.5f}, eta {f.se['eta']:.3f}, rho {f.se['rho']:.3f}; "
              f"in-sample RMSE {100 * f.rmse:.2f} vol points; converged {f.converged}")
    if a.fit_only:
        print("\n--fit-only: no test-window numbers computed.")
        return 0

    # The term structure out of sample.
    test_mask = np.array([d >= TEST_VIX_START for d in days])
    a30, b30 = ts.coefficients(k1, th1, TENORS[:1])
    v_test = np.maximum((vols[test_mask, 0] ** 2 - a30[0]) / b30[0], 0.0)
    spread = (vols[fit_mask, 1:] - vols[fit_mask, :1]).mean(axis=0)
    print(f"\nThe term structure out of sample: {test_mask.sum()} days from {TEST_VIX_START} to {days[-1]} (each day's "
          "variance from VIX alone)")
    print("| index | model RMSE | flat (= VIX) | VIX + fit-window spread |\n|---|---|---|---|")
    for j, n in enumerate(INDICES[1:], start=1):
        pred = ts.predict(k1, th1, v_test, TENORS[j])
        act = vols[test_mask, j]
        print(f"| {n} | {100 * rmse(pred - act):.2f} | {100 * rmse(vols[test_mask, 0] - act):.2f} | "
              f"{100 * rmse(vols[test_mask, 0] + spread[j - 1] - act):.2f} |")

    # The accuracy target.
    ten = [r for r in rows if r["strike"] == 10000.0]
    fit10 = [r for r in ten if FIT_TRADES[0] <= r["date"] <= FIT_TRADES[1] and r["volume"] >= MIN_CONTRACTS]
    test10 = [r for r in ten if r["date"] >= TEST_TRADES_START and r["volume"] >= MIN_CONTRACTS]
    iv_bar = float(np.mean([r["iv"] for r in fit10]))
    vix_bar = float(np.mean([r["vix1y"] for r in fit10]))
    ref = fit10[-1]
    preds = {"model B (primary)": [model_iv(fb.model, r["v"], r["F"], 10000.0, r["T"]) for r in test10],
             "model A": [model_iv(fa.model, r["v"], r["F"], 10000.0, r["T"]) for r in test10],
             "constant 16.6%": [0.166 for _ in test10],
             "0.77 x VIX1Y": [0.77 * r["vix1y"] for r in test10],
             "sticky strike + level shift": [iv_bar + r["vix1y"] - vix_bar for r in test10],
             "slide rule (1.0 per -10%)": [ref["iv"] + 0.01 * math.log(r["S"] / ref["S"]) / math.log(0.9)
                                           for r in test10]}
    actual = np.array([r["iv"] for r in test10])
    print(f"\nThe accuracy target: {len(test10)} days of the 10,000 strike from {TEST_TRADES_START} to "
          f"{test10[-1]['date']} ({MIN_CONTRACTS}+ contracts)")
    print("| prediction | RMSE (vol points) | mean error (vol points) |\n|---|---|---|")
    scores = {}
    for name, p in preds.items():
        err = np.array(p) - actual
        scores[name] = rmse(err)
        print(f"| {name} | {100 * scores[name]:.2f} | {100 * err.mean():+.2f} |")
    bench = [k for k in preds if k not in ("model B (primary)", "model A")]
    ok = scores["model B (primary)"] <= 0.015 and all(scores["model B (primary)"] < scores[k] for k in bench)
    print(f"VERDICT: {'PASS' if ok else 'MISS'} (target: RMSE at most 1.50 and below every benchmark)")

    # The decisive mystery test, weekly.
    weekly = {}
    for r in ten:
        weekly[r["date"].isocalendar()[:2]] = r                    # last trade of each ISO week (sorted by date)
    wk = [weekly[k] for k in sorted(weekly)]
    for r in wk:
        r["model"] = model_iv(fb.model, r["v"], r["F"], 10000.0, r["T"])
    print("\nThe mystery test: weekly changes of the 10,000 strike's traded vol against each reading")
    print("| pairs | reading | RMSE of the change error | slope of traded on predicted (SE) |\n|---|---|---|---|")
    test_pairs = [(p, q) for p, q in zip(wk, wk[1:]) if p["date"] >= TEST_TRADES_START]
    # added after the code review (2026-10-23): some pairs span weeks without a trade; the frozen rows keep them,
    # and the extra row uses one-week pairs only
    one_week = [(p, q) for p, q in test_pairs if (q["date"] - p["date"]).days <= 7]
    print(f"(test-window pairs spanning more than one week: {len(test_pairs) - len(one_week)} of {len(test_pairs)})")
    for label, pairs in (("test window", test_pairs), ("full history", list(zip(wk, wk[1:]))),
                         ("test window, one-week pairs only", one_week)):
        d_tr = np.array([q["iv"] - p["iv"] for p, q in pairs])
        readings = {"slide rule": [0.01 * math.log(q["S"] / p["S"]) / math.log(0.9) for p, q in pairs],
                    "level shift": [q["vix1y"] - p["vix1y"] for p, q in pairs],
                    "sticky strike": [0.0 for _ in pairs],
                    "dynamics (B)": [q["model"] - p["model"] for p, q in pairs]}
        for name, dp in readings.items():
            dp = np.array(dp)
            slope = "-" if name == "sticky strike" else "{:.2f} ({:.2f})".format(*ols(dp, d_tr))
            print(f"| {label} ({len(pairs)}) | {name} | {100 * rmse(d_tr - dp):.2f} | {slope} |")

    # The belief probe.
    belief = [r for r in ten if r["volume"] >= MIN_CONTRACTS]
    x = np.log([r["vix1y"] for r in belief])
    y = np.log([r["iv"] / r["vix1y"] for r in belief])
    xw = np.log([r["vix1y"] for r in wk])
    yw = np.log([r["iv"] / r["vix1y"] for r in wk])
    print("\nThe belief probe: ln(iv / VIX1Y) on ln(VIX1Y)")
    print("  daily ({} days): slope {:.2f} (SE {:.2f}); weekly ({} weeks): slope {:.2f} (SE {:.2f})".format(
        len(belief), *ols(x, y), len(wk), *ols(xw, yw)))
    return 0


if __name__ == "__main__":
    with manifest.run("vol_holdout_study", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/vol_holdout_study.py",), seeds=[SEED]) as run:
        rc = main(sys.argv[1:])
    print(f"\nrun manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
