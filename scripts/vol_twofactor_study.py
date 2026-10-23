#!/usr/bin/env python3
"""v0.3.1 step 5: the two-factor Heston model under the fixed holdout, judged out of sample against step 4's one-factor
model and the same benchmarks.

    py -I scripts\\vol_twofactor_study.py <S&P 500 daily closes CSV> [--fit-only] [--boot N]      From target_funds/.

PROTOCOL (fixed 2026-10-23 and committed before any test-window number of the two-factor model was computed; the
holdout and the accuracy target were fixed 2026-10-21 in the private v0.3.1 brief):

One change from step 4 (scripts/vol_holdout_study.py, FINDINGS F23): the variance model has two factors instead of
one. The data, the windows, the noise sds, the trades, the benchmarks and the pass rule are step 4's.

The model. heston.DoubleHeston with one eta and one rho for both factors: the 3-year trades cannot tell a fast factor's
eta and rho from a slow one's, and with equal speeds this model is step 4's one-factor model exactly.

Stage 1, the term structure. The fit window and days are step 4's (2009-09-18 to 2019-12-31, all four indices, sd 0.5
vol points). volmodel.termstructure.fit_two fits kappa1 > kappa2 (at least KAPPA2_MIN = 1e-6 a year), theta1 and the
slow drift c2 = kappa2 theta2, with each day's two variances at 0 or above. Three starts, (kappa1, theta1, kappa2,
c2) = (5, 0.02, 0.5, 0.01), (2, 0.01, 0.1, 0.003) and (15, 0.04, 1, 0.03); the lowest cost is the fit, and the three
are printed. Profiles: the cost with theta1 held at 0.015, 0.02, 0.03 and 0.04, and with kappa2 held at 0.05, 0.1,
0.2 and 0.5 (how much fit a mean-reverting slow factor gives up). A moving-block bootstrap (63-day blocks, 200
resamples, seed 20261023, first start) gives the bands of the four, and counts the converged refits and those at the
boundary (kappa2 below 1e-3). Printed: the share of fit-window days with v1 = 0 and with v2 = 0 (the constraint at
work), and each factor's eta from the time series (termstructure.vol_of_variance, weekly, with the daily value as a
check on index noise).
Out of sample, 2020-01-01 to the last common day, two checks without a pass mark:
(i) each day's states from VIX and VIX3M only predict VIX6M and VIX1Y, against the one-factor model (step 4's kappa
    and theta, its state from the same two indices), flat (= VIX3M) and VIX3M plus the fit window's mean spread;
(ii) each day's states from all four indices: the RMSE of the model by index, two factors against one.

Stage 2, the smile. Step 4's trades (2024-06-14 to 2025-06-30, every strike, 5+ contracts), sd 1 vol point, each day's
v1 and v2 from its four indices. Three variants:
- A2 (the VIX anchor): kappa1, theta1, kappa2 and theta2 (= c2 / kappa2) from stage 1; eta and rho fitted.
- B2 (the trade anchor, PRIMARY): kappa1, theta1 and kappa2 from stage 1; theta2, eta and rho fitted. The slow
  factor's drift beyond the longest index is the one anchor the indices give least well, so the 3-year trades set
  it. With kappa2 fixed, fitting theta2 is fitting the slow factor's pricing drift kappa2 theta2, which is printed.
- C2 (diffusion-consistent, a diagnostic): as B2, but eta fixed at the time series' pooled weekly value; theta2 and
  rho fitted. In a diffusion model eta is the same under both measures, so a large gap between B2's eta and the
  time series' is a misfit of the model, not of the data.
Step 4's one-factor variants A and B are refitted here by the same code (a reproduction check: F23 printed RMSE 1.32
and 1.39 vol points).

The accuracy target (PRIMARY result). Step 4's test days (the 10,000 strike, 5+ contracts, from 2025-07-01) and its
rule: PASS if B2's RMSE is at most 1.5 vol points AND below each of step 4's four benchmarks (constant 16.6%; 0.77 x
VIX1Y; sticky strike with a level shift; the slide rule). A miss is a result. Beside the verdict: A2, C2, the
one-factor A and B, and the paired comparison of B2 with the one-factor B (the mean of the difference of squared
errors, with an independent-days standard error; the days are serially correlated, so it is a rough guide).
Sensitivity, which does not change the verdict: stage 1 refitted with theta1 held at its bootstrap 5% and 95% values,
then B2 refitted on the new states.

The mystery test and the belief probe. Step 4's weekly design, with the dynamics of both B (one factor) and B2 as
readings. The belief probe adds the model-implied slope: ln(model iv / VIX1Y) on ln(VIX1Y) on the probe's days, for B
and B2, beside the traded slope (F23: -0.67).

Already seen before this protocol (disclosed): everything step 4 lists, and step 4's own test-window results (F23):
the one-factor model's bias of -1.20 points, the constant's lower RMSE, the weekly-move results and the belief slope.
The two-factor model was planned before step 4 (the private brief, 2026-10-21: "two-factor second"). The shared eta
and rho and B2's free level were chosen at build time, after F23, for identification. Two drafts of this protocol
were run with --fit-only (fit windows only, 2026-10-23); both left each day's variances free in sign and took theta1
from a time-series split. The first parameterized the slow factor by its long-run level, which ran off to 461,000
with kappa2 near 1e-7 (the drift the same, 0.0115, from all starts); the second used the drift and showed the
variances negative on 1,868 of 2,587 fit days and no theta1 that keeps every day's variances at 0 or above, a B2
eta of 2.38 against a time-series eta of 0.81, and C2 at an RMSE of 2.27 with rho at -1. This version fits the
model with its constraint (both variances at 0 or above), which identifies theta1, and keeps the time-series eta as
a diagnostic. No two-factor number from a test window was computed before this protocol was committed.

--fit-only prints the fit-window numbers only. Aggregates only; the data stay local; research, not investment advice.
"""
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import vol_holdout_study as h4  # noqa: E402
from definedtarget import curves, listedvols, marks  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.volmodel import calibrate, pricers, termstructure as ts  # noqa: E402
from definedtarget.volmodel.heston import DoubleHeston, Heston  # noqa: E402

SEED = 20261023
STARTS = ((5.0, 0.02, 0.5, 0.01), (2.0, 0.01, 0.1, 0.003), (15.0, 0.04, 1.0, 0.03))
PROFILE_THETA1 = (0.015, 0.02, 0.03, 0.04)
PROFILE_KAPPA2 = (0.05, 0.1, 0.2, 0.5)
INDICES, TENORS, SD_INDEX, MIN_CONTRACTS = h4.INDICES, h4.TENORS, h4.SD_INDEX, h4.MIN_CONTRACTS
FIT_VIX, TEST_VIX_START = h4.FIT_VIX, h4.TEST_VIX_START
FIT_TRADES, TEST_TRADES_START = h4.FIT_TRADES, h4.TEST_TRADES_START


def model_iv2(model, v1, v2, F, K, T):
    m = DoubleHeston(Heston(v1, model.f1.theta, model.f1.kappa, model.f1.eta, model.f1.rho),
                     Heston(v2, model.f2.theta, model.f2.kappa, model.f2.eta, model.f2.rho))
    return float(pricers.implied_vols(m, F, np.array([K]), T)[0])


def fit_stage2(rows, k1, th1, k2, theta2, key, free=("theta2", "eta", "rho"), eta=0.5):
    """Stage 2 for the two-factor model on the fit-window trades; key: the row fields holding (v1, v2)."""
    by_day = {}
    for r in rows:
        by_day.setdefault(r["date"], []).append(r)
    days = [calibrate.Day2(v1=rs[0][key[0]], v2=rs[0][key[1]], F=rs[0]["F"], T=rs[0]["T"],
                           K=np.array([r["strike"] for r in rs]), iv=np.array([r["iv"] for r in rs]))
            for _, rs in sorted(by_day.items())]
    start = DoubleHeston(Heston(0.01, th1, k1, eta, -0.7), Heston(0.02, theta2, k2, eta, -0.7))
    return calibrate.fit_double_days(days, start, free=free)


def describe(f):
    return (f"kappa1 {f.kappa1:.4f} (half-life {365 * math.log(2) / f.kappa1:.0f} days), theta1 {f.theta1:.5f} (fast "
            f"long-run vol {100 * math.sqrt(f.theta1):.2f}%), kappa2 {f.kappa2:.2e}, slow drift c2 {f.c2:.5f} a year")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("closes")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)

    idx = {n: h4.load_index(n) for n in INDICES}
    days = sorted(set.intersection(*(set(idx[n]) for n in INDICES)))
    vols = np.array([[idx[n][d] for n in INDICES] for d in days])
    sd = np.full(4, SD_INDEX)
    fit_mask = np.array([FIT_VIX[0] <= d <= FIT_VIX[1] for d in days])
    vf = vols[fit_mask]
    print("Index files end: " + ", ".join(f"{n} {max(idx[n])}" for n in INDICES) + f"; common end {days[-1]}")

    # Stage 1: one factor (step 4's fit, for the comparisons) and two factors.
    tf = ts.fit(TENORS, vf, sd)
    k1f, th1f = tf.kappa, tf.theta
    fits = [ts.fit_two(TENORS, vf, sd, start=s) for s in STARTS]
    f2 = min(fits, key=lambda f: f.cost)
    P = f2.params
    print(f"Stage 1, the term structure: {fit_mask.sum()} days from {FIT_VIX[0]} to {FIT_VIX[1]}")
    print(f"  one factor (step 4): kappa {k1f:.4f}, long-run vol {100 * math.sqrt(th1f):.2f}%; cost {tf.cost:.1f}; "
          "in-sample RMSE " + ", ".join(f"{n} {100 * r:.2f}" for n, r in zip(INDICES, tf.rmse)))
    for s, f in zip(STARTS, fits):
        print(f"  start {s}: {describe(f)}; cost {f.cost:.1f}{'' if f.converged else ' (not converged)'}")
    print(f"  two factors: cost {f2.cost:.1f}; in-sample RMSE "
          + ", ".join(f"{n} {100 * r:.2f}" for n, r in zip(INDICES, f2.rmse)))
    prof = [(f"theta1 {x}", ts.fit_two(TENORS, vf, sd, start=(P[0], x, P[2] if P[2] > 1e-3 else 0.01, P[3]),
                                         fixed=("theta1",))) for x in PROFILE_THETA1]
    prof += [(f"kappa2 {x}", ts.fit_two(TENORS, vf, sd, start=(P[0], P[1], x, P[3]), fixed=("kappa2",)))
             for x in PROFILE_KAPPA2]
    print("  profiles (cost; the other three refitted):")
    for label, f in prof:
        print(f"    {label} held: cost {f.cost:.1f}; {describe(f)}; slow long-run vol {100 * math.sqrt(f.theta2):.1f}%")
    boot, conv = ts.bootstrap_two(TENORS, vf, sd, block=63, reps=a.boot, seed=SEED, start=STARTS[0])
    q = lambda col, p: float(np.percentile(boot[:, col], p))  # noqa: E731
    print(f"  bootstrap ({int(conv.sum())} of {len(conv)} refits converged; {int((boot[:, 2] < 1e-3).sum())} at the "
          f"boundary, kappa2 below 1e-3), 5-95%: kappa1 {q(0, 5):.3f} to {q(0, 95):.3f}; theta1 {q(1, 5):.5f} to "
          f"{q(1, 95):.5f}; kappa2 {q(2, 5):.2e} to {q(2, 95):.2e}; c2 {q(3, 5):.5f} to {q(3, 95):.5f}")
    print(f"  fit-window days with v1 = 0: {int((f2.v[:, 0] == 0).sum())}; with v2 = 0: {int((f2.v[:, 1] == 0).sum())} "
          f"(of {f2.v.shape[0]})")
    t_fit = np.array([(d - FIT_VIX[0]).days / 365.0 for d, m in zip(days, fit_mask) if m])
    for label, step in (("weekly", 5), ("daily (noise check)", 1)):
        vv = ts.vol_of_variance(*P, f2.v, t_fit, step=step)
        print(f"  eta from the time series, {label} ({vv['n']} innovations): fast {vv['eta1']:.3f}, slow "
              f"{vv['eta2']:.3f}, pooled {vv['eta']:.3f}; innovation correlation {vv['corr']:+.2f}")
    eta_ts = ts.vol_of_variance(*P, f2.v, t_fit, step=5)["eta"]
    print(f"  3-year variance-swap vol at the fit window's median state: two factors "
          f"{100 * float(ts.predict_two(*P, np.median(f2.v, axis=0), 3.0)):.2f}%, one factor "
          f"{100 * float(ts.predict(k1f, th1f, np.median(tf.v), 3.0)):.2f}%")

    # Trades, with both models' states.
    closes = listedvols.read_closes_csv(a.closes)
    rows = listedvols.trade_vols(listedvols.load_bars(), closes, curves.load_treasury(), marks.load_dividends(),
                                 min_volume=1)
    for r in rows:
        iv_idx = np.array([[np.nan if idx[n].get(r["date"]) is None else idx[n][r["date"]] for n in INDICES]])
        r["v"] = float(ts.states(k1f, th1f, TENORS, iv_idx, sd)[0])
        r["v1"], r["v2"] = (float(x) for x in ts.states_two(*P, TENORS, iv_idx, sd)[0])
        r["iv_idx"] = iv_idx
        r["vix1y"] = idx["VIX1Y"].get(r["date"])
    rows = [r for r in rows if np.isfinite(r["iv"]) and np.isfinite(r["v"]) and np.isfinite(r["v1"])
            and r["vix1y"] is not None]
    fit_rows = [r for r in rows if FIT_TRADES[0] <= r["date"] <= FIT_TRADES[1] and r["volume"] >= MIN_CONTRACTS]
    trade_days = {r["date"]: r for r in rows}
    print(f"  trade days with v1 = 0: {sum(r['v1'] == 0 for r in trade_days.values())}; with v2 = 0: "
          f"{sum(r['v2'] == 0 for r in trade_days.values())} (of {len(trade_days)})")

    # Stage 2: one factor (step 4, refitted) and two factors.
    by_day = {}
    for r in fit_rows:
        by_day.setdefault(r["date"], []).append(r)
    dlist = [calibrate.Day(v=rs[0]["v"], F=rs[0]["F"], T=rs[0]["T"], K=np.array([r["strike"] for r in rs]),
                           iv=np.array([r["iv"] for r in rs])) for _, rs in sorted(by_day.items())]
    start = Heston(0.03, th1f, k1f, 0.5, -0.7)
    fa = calibrate.fit_heston_days(dlist, start, fixed=("v0", "kappa", "theta"))
    fb = calibrate.fit_heston_days(dlist, start, fixed=("v0", "kappa"))
    k1, th1, k2 = P[0], P[1], P[2]
    fa2 = fit_stage2(fit_rows, k1, th1, k2, f2.theta2, ("v1", "v2"), free=("eta", "rho"))
    fb2 = fit_stage2(fit_rows, k1, th1, k2, f2.theta2, ("v1", "v2"))
    fc2 = fit_stage2(fit_rows, k1, th1, k2, f2.theta2, ("v1", "v2"), free=("theta2", "rho"), eta=eta_ts)
    print(f"\nStage 2, the smile: {len(fit_rows)} trades on {len(dlist)} days from {FIT_TRADES[0]} to {FIT_TRADES[1]}")
    for label, f in (("one factor A", fa), ("one factor B", fb)):
        print(f"  {label}: long-run vol {100 * math.sqrt(f.model.theta):.2f}%, eta {f.model.eta:.3f}, rho "
              f"{f.model.rho:.3f}; in-sample RMSE {100 * f.rmse:.2f}")
    for label, f in (("A2 (VIX anchor)", fa2), ("B2 (trade anchor)", fb2), ("C2 (time-series eta)", fc2)):
        m = f.model
        print(f"  {label}: slow pricing drift kappa2 theta2 {m.f2.kappa * m.f2.theta:.5f} a year (SE "
              f"{m.f2.kappa * f.se['theta2']:.5f}), eta {m.f1.eta:.3f} (SE {f.se['eta']:.3f}), rho {m.f1.rho:.3f} "
              f"(SE {f.se['rho']:.3f}); in-sample RMSE {100 * f.rmse:.2f} vol points; converged {f.converged}")
    if a.fit_only:
        print("\n--fit-only: no test-window numbers computed.")
        return 0

    # (i) and (ii): the term structure out of sample.
    test_mask = np.array([d >= TEST_VIX_START for d in days])
    vt = vols[test_mask]
    short = vt.copy()
    short[:, 2:] = np.nan
    v_short2 = ts.states_two(*P, TENORS, short, sd)
    v_short1 = ts.states(k1f, th1f, TENORS, short, sd)
    spread = (vf[:, 2:] - vf[:, 1:2]).mean(axis=0)
    print(f"\nThe term structure out of sample: {test_mask.sum()} days from {TEST_VIX_START} to {days[-1]}")
    print("(i) states from VIX and VIX3M only\n| index | two factors | one factor | flat (= VIX3M) | VIX3M + fit-window "
          "spread |\n|---|---|---|---|---|")
    for j in (2, 3):
        p2 = ts.predict_two(*P, v_short2, TENORS[j])
        p1 = ts.predict(k1f, th1f, v_short1, TENORS[j])
        act = vt[:, j]
        print(f"| {INDICES[j]} | {100 * h4.rmse(p2 - act):.2f} | {100 * h4.rmse(p1 - act):.2f} | "
              f"{100 * h4.rmse(vt[:, 1] - act):.2f} | {100 * h4.rmse(vt[:, 1] + spread[j - 2] - act):.2f} |")
    v_all2 = ts.states_two(*P, TENORS, vt, sd)
    v_all1 = ts.states(k1f, th1f, TENORS, vt, sd)
    e2 = ts.predict_two(*P, v_all2[:, None, :], TENORS[None, :]) - vt
    e1 = ts.predict(k1f, th1f, v_all1[:, None], TENORS[None, :]) - vt
    print("(ii) states from all four indices, RMSE by index (vol points): two factors "
          + ", ".join(f"{n} {100 * h4.rmse(e2[:, j]):.2f}" for j, n in enumerate(INDICES)) + "; one factor "
          + ", ".join(f"{n} {100 * h4.rmse(e1[:, j]):.2f}" for j, n in enumerate(INDICES)))

    # The accuracy target.
    ten = [r for r in rows if r["strike"] == 10000.0]
    fit10 = [r for r in ten if FIT_TRADES[0] <= r["date"] <= FIT_TRADES[1] and r["volume"] >= MIN_CONTRACTS]
    test10 = [r for r in ten if r["date"] >= TEST_TRADES_START and r["volume"] >= MIN_CONTRACTS]
    iv_bar = float(np.mean([r["iv"] for r in fit10]))
    vix_bar = float(np.mean([r["vix1y"] for r in fit10]))
    ref = fit10[-1]
    two = lambda f, k=("v1", "v2"): [model_iv2(f.model, r[k[0]], r[k[1]], r["F"], 10000.0, r["T"])  # noqa: E731
                                     for r in test10]
    preds = {"two factors B2 (primary)": two(fb2),
             "two factors A2": two(fa2),
             "two factors C2 (time-series eta)": two(fc2),
             "one factor B (step 4)": [h4.model_iv(fb.model, r["v"], r["F"], 10000.0, r["T"]) for r in test10],
             "one factor A (step 4)": [h4.model_iv(fa.model, r["v"], r["F"], 10000.0, r["T"]) for r in test10],
             "constant 16.6%": [0.166 for _ in test10],
             "0.77 x VIX1Y": [0.77 * r["vix1y"] for r in test10],
             "sticky strike + level shift": [iv_bar + r["vix1y"] - vix_bar for r in test10],
             "slide rule (1.0 per -10%)": [ref["iv"] + 0.01 * math.log(r["S"] / ref["S"]) / math.log(0.9)
                                           for r in test10]}
    actual = np.array([r["iv"] for r in test10])
    print(f"\nThe accuracy target: {len(test10)} days of the 10,000 strike from {TEST_TRADES_START} to "
          f"{test10[-1]['date']} ({MIN_CONTRACTS}+ contracts)")
    print("| prediction | RMSE (vol points) | mean error (vol points) |\n|---|---|---|")
    scores, errs = {}, {}
    for name, p in preds.items():
        errs[name] = np.array(p) - actual
        scores[name] = h4.rmse(errs[name])
        print(f"| {name} | {100 * scores[name]:.2f} | {100 * errs[name].mean():+.2f} |")
    primary = "two factors B2 (primary)"
    bench = ["constant 16.6%", "0.77 x VIX1Y", "sticky strike + level shift", "slide rule (1.0 per -10%)"]
    ok = scores[primary] <= 0.015 and all(scores[primary] < scores[k] for k in bench)
    print(f"VERDICT: {'PASS' if ok else 'MISS'} (target: RMSE at most 1.50 and below every benchmark)")
    print(f"reproduction of F23: one factor B {100 * scores['one factor B (step 4)']:.2f} (F23: 1.39), A "
          f"{100 * scores['one factor A (step 4)']:.2f} (F23: 1.32)")
    d = (100 * errs[primary]) ** 2 - (100 * errs["one factor B (step 4)"]) ** 2
    d_se = d.std(ddof=1) / math.sqrt(d.size)
    print(f"paired, B2 against one factor B: mean difference of squared errors {d.mean():+.3f} (SE {d_se:.3f}, days "
          f"treated as independent); B2 closer on "
          f"{int((np.abs(errs[primary]) < np.abs(errs['one factor B (step 4)'])).sum())} of {d.size} days")

    # Sensitivity of B2 to theta1 (does not change the verdict).
    print("Sensitivity of B2 to theta1 (stage 1 refitted with theta1 held, then B2; RMSE, mean error):")
    for label, t1 in (("theta1 at its bootstrap 5%", q(1, 5)), ("theta1 at its bootstrap 95%", q(1, 95))):
        fs1 = ts.fit_two(TENORS, vf, sd, start=(P[0], t1, P[2] if P[2] > 1e-3 else 0.01, P[3]), fixed=("theta1",))
        for r in rows:
            r["s1"], r["s2"] = (float(x) for x in ts.states_two(*fs1.params, TENORS, r["iv_idx"], sd)[0])
        fs = fit_stage2(fit_rows, fs1.kappa1, fs1.theta1, fs1.kappa2, fs1.theta2, ("s1", "s2"))
        e = np.array(two(fs, ("s1", "s2"))) - actual
        print(f"  {label} ({t1:.5f}): RMSE {100 * h4.rmse(e):.2f}, mean error {100 * e.mean():+.2f} (pricing drift "
              f"{fs.model.f2.kappa * fs.model.f2.theta:.5f}, eta {fs.model.f1.eta:.3f}, rho {fs.model.f1.rho:.3f})")

    # The mystery test, weekly.
    weekly = {}
    for r in ten:
        weekly[r["date"].isocalendar()[:2]] = r                    # last trade of each ISO week (sorted by date)
    wk = [weekly[k] for k in sorted(weekly)]
    for r in wk:
        r["m1"] = h4.model_iv(fb.model, r["v"], r["F"], 10000.0, r["T"])
        r["m2"] = model_iv2(fb2.model, r["v1"], r["v2"], r["F"], 10000.0, r["T"])
    print("\nThe mystery test: weekly changes of the 10,000 strike's traded vol against each reading")
    print("| pairs | reading | RMSE of the change error | slope of traded on predicted (SE) |\n|---|---|---|---|")
    test_pairs = [(p, q_) for p, q_ in zip(wk, wk[1:]) if p["date"] >= TEST_TRADES_START]
    one_week = [(p, q_) for p, q_ in test_pairs if (q_["date"] - p["date"]).days <= 7]
    for label, pairs in (("test window", test_pairs), ("full history", list(zip(wk, wk[1:]))),
                         ("test window, one-week pairs only", one_week)):
        d_tr = np.array([q_["iv"] - p["iv"] for p, q_ in pairs])
        readings = {"slide rule": [0.01 * math.log(q_["S"] / p["S"]) / math.log(0.9) for p, q_ in pairs],
                    "level shift": [q_["vix1y"] - p["vix1y"] for p, q_ in pairs],
                    "sticky strike": [0.0 for _ in pairs],
                    "dynamics, one factor B": [q_["m1"] - p["m1"] for p, q_ in pairs],
                    "dynamics, two factors B2": [q_["m2"] - p["m2"] for p, q_ in pairs]}
        for name, dp in readings.items():
            dp = np.array(dp)
            slope = "-" if name == "sticky strike" else "{:.2f} ({:.2f})".format(*h4.ols(dp, d_tr))
            print(f"| {label} ({len(pairs)}) | {name} | {100 * h4.rmse(d_tr - dp):.2f} | {slope} |")

    # The belief probe, with the model-implied slopes.
    belief = [r for r in ten if r["volume"] >= MIN_CONTRACTS]
    x = np.log([r["vix1y"] for r in belief])
    print("\nThe belief probe: ln(iv / VIX1Y) on ln(VIX1Y), {} days".format(len(belief)))
    print("  traded: slope {:.2f} (SE {:.2f})".format(*h4.ols(x, np.log([r["iv"] / r["vix1y"] for r in belief]))))
    m1 = [h4.model_iv(fb.model, r["v"], r["F"], 10000.0, r["T"]) for r in belief]
    m2 = [model_iv2(fb2.model, r["v1"], r["v2"], r["F"], 10000.0, r["T"]) for r in belief]
    for label, mv in (("one factor B", m1), ("two factors B2", m2)):
        y = np.log(np.array(mv) / np.array([r["vix1y"] for r in belief]))
        print("  model-implied, {}: slope {:.2f} (SE {:.2f})".format(label, *h4.ols(x, y)))
    return 0


if __name__ == "__main__":
    with manifest.run("vol_twofactor_study", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/vol_twofactor_study.py", "scripts/vol_holdout_study.py"), seeds=[SEED]) as run:
        rc = main(sys.argv[1:])
    print(f"\nrun manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
