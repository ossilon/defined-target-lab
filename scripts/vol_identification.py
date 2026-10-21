#!/usr/bin/env python3
"""What can this project's quotes pin down? Heston parameter recovery from noisy synthetic quotes laid out like the
data on 2026-10-21.

    py -I scripts\\vol_identification.py [--reps N] [--workers W]      From target_funds/.

Fixed before the first run (2026-10-21):
- Truths: Gatheral (2006) Table 3.2, a published S&P 500 fit (v0 0.0174, theta 0.0354, kappa 1.3253, eta 0.3877,
  rho -0.7165); and Andersen (2008) Case III, an equity-style stress case (v0 = theta = 0.09, kappa 1, eta 1,
  rho -0.3).
- Layout (34 quotes): the VIX family at 9 days, 30 days, 3, 6 and 12 months; listed S&P 500 calls at 6,500-10,000
  (step 500) for Dec-2027, Dec-2028 and Dec-2029 (1.157, 2.150 and 3.167 years from 2026-10-21); and XX's strikes
  11,000-15,000 for Dec-2029. Index 7,712.57; forwards at an illustrative 2% carry (the carry does not matter here).
- Noise (assumptions, in vol points; roughly half a bid-ask spread, not measured here): vol indices 0.10; options
  0.10 (Dec-2027), 0.25 (Dec-2028), 0.50 (Dec-2029). Normal, independent; the fit weights by the same values.
- Scenarios: A. the VIX family alone (v0, theta, kappa free; eta and rho fixed at the truth, because no
  variance-swap rate depends on them); B. the listed options alone (all five free); C. both (all five free).
- 200 noisy replications per scenario and truth, seed 20261021 (one child seed per replication, so results do not
  depend on the number of workers). Every fit starts from the same generic point (v0 0.03, theta 0.04, kappa 2,
  eta 0.6, rho -0.5), not from the truth.
- Reported: for each parameter the median estimate and its 5-95% range across replications, and the median reported
  standard error against the spread of the estimates (a check of the fit's own error bars); for what XX needs, the
  same for the long-run vol sqrt(theta), the 3.17-year at-the-money-forward vol, the vol at XX's strike (10,000)
  and the 3.17-year variance-swap vol. Fits that fail to converge are counted.
Changed after the first full run (disclosed): that run stopped when one fit ended at a model too extreme to price
(its moment generating function failed near zero); such fits are now counted as failures instead, and the
pricer's moment search starts nearer zero (s = 0.001 instead of 0.05). After the second run: the spread column was
the plain standard deviation, which a few runaway fits (theta and kappa drifting together when only the VIX family
is fitted) blew up to meaningless sizes; it is now the normal-equivalent spread (5-95% range / 3.29), and derived
vols that a fitted model cannot give are skipped and counted.
Synthetic data only; aggregates only; research, not investment advice.
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.volmodel import calibrate, pricers  # noqa: E402
from definedtarget.volmodel.heston import Heston  # noqa: E402

SEED = 20261021
S, CARRY = 7712.57, 0.02
EXPIRIES = (1.157, 2.150, 3.167)
SD = {1.157: 0.0010, 2.150: 0.0025, 3.167: 0.0050}
SD_VIX = 0.0010
VIX_TENORS = (9 / 365, 30 / 365, 0.25, 0.5, 1.0)
TRUTHS = {"Gatheral (2006) Table 3.2": Heston(0.0174, 0.0354, 1.3253, 0.3877, -0.7165),
          "Andersen (2008) Case III": Heston(0.09, 0.09, 1.0, 1.0, -0.3)}
START = Heston(0.03, 0.04, 2.0, 0.6, -0.5)
T_XX, K_XX = 3.167, 10000.0


def options():
    rows = [(T, S * math.exp(CARRY * T), float(K)) for T in EXPIRIES for K in range(6500, 10001, 500)]
    return rows + [(3.167, S * math.exp(CARRY * 3.167), float(K)) for K in (11000, 12000, 13000, 14000, 15000)]


def clean_quotes(truth, scenario):
    vix = VIX_TENORS if scenario in ("A", "C") else ()
    opts = options() if scenario in ("B", "C") else []
    q = calibrate.layout(vix_tenors=vix, options=opts, model=truth, sd_vix=SD_VIX, sd_option=0.001)
    sd = np.where(np.isnan(q.K), SD_VIX, [SD.get(round(t, 3), 0.001) for t in q.T])
    return calibrate.Quotes(T=q.T, K=q.K, F=q.F, vol=q.vol, sd=sd)


def xx_numbers(m):
    F = S * math.exp(CARRY * T_XX)
    iv = pricers.implied_vols(m, F, np.array([F, K_XX]), T_XX)
    return {"sqrt(theta)": math.sqrt(m.theta), "ATMF vol 3.17y": iv[0], "vol at 10,000": iv[1],
            "var-swap vol 3.17y": math.sqrt(m.variance_swap_rate(T_XX))}


def one(args):
    truth_name, scenario, child = args
    truth = TRUTHS[truth_name]
    q = clean_quotes(truth, scenario)
    rng = np.random.default_rng(child)
    noisy = calibrate.Quotes(T=q.T, K=q.K, F=q.F, vol=q.vol + rng.normal(size=q.n) * q.sd, sd=q.sd)
    start = Heston(START.v0, START.theta, START.kappa, truth.eta, truth.rho) if scenario == "A" else START
    fixed = ("eta", "rho") if scenario == "A" else ()
    try:
        f = calibrate.fit_heston(noisy, start, fixed=fixed)
    except (ValueError, ArithmeticError) as err:
        return {"ok": False, "error": str(err)}
    out = {"ok": f.converged, "rmse": f.rmse}
    out.update({p: getattr(f.model, p) for p in calibrate.PARAMS})
    out.update({f"se_{p}": f.se[p] for p in calibrate.PARAMS})
    try:
        with np.errstate(all="ignore"):
            out.update(xx_numbers(f.model))
    except (ValueError, ArithmeticError) as err:      # the fit ended at a model too extreme to price: count it
        return {"ok": False, "error": f"unpriceable fitted model: {err}"}
    return out


def summarize(rows, truth, scenario):
    ok = [r for r in rows if r.get("ok")]
    failed = [r.get("error", "did not converge") for r in rows if not r.get("ok")]
    unpriceable = sum("unpriceable" in e for e in failed)
    lines = [f"converged {len(ok)} of {len(rows)} (failed: {len(failed) - unpriceable} did not converge, "
             f"{unpriceable} ended at a model too extreme to price); median RMSE "
             f"{1e2 * np.median([r['rmse'] for r in ok]):.3f} vol points", "",
             "| quantity | truth | median estimate | 5-95% of estimates | spread (5-95% / 3.29) | median reported SE |",
             "|---|---|---|---|---|---|"]
    true_xx = xx_numbers(truth)
    free = calibrate.PARAMS if scenario != "A" else ("v0", "theta", "kappa")
    def stats(v):
        v = v[np.isfinite(v)]
        lo, mid, hi = np.percentile(v, [5, 50, 95])
        return v.size, lo, mid, hi, (hi - lo) / 3.29          # 3.29 = width of a normal's 5-95% range in sd
    for p in free:
        n, lo, mid, hi, spread = stats(np.array([r[p] for r in ok]))
        se = np.array([r[f"se_{p}"] for r in ok])
        lines.append(f"| {p} | {getattr(truth, p):.4f} | {mid:.4f} | {lo:.4f} to {hi:.4f} | {spread:.4f} | "
                     f"{np.median(se):.4f} |")
    for name, tv in true_xx.items():
        n, lo, mid, hi, spread = stats(100 * np.array([r[name] for r in ok]))
        note = "" if n == len(ok) else f" ({len(ok) - n} fits gave no vol)"
        lines.append(f"| {name} (vol points){note} | {100 * tv:.2f} | {mid:.2f} | {lo:.2f} to {hi:.2f} | "
                     f"{spread:.2f} | - |")
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    children = np.random.SeedSequence(SEED).spawn(len(TRUTHS) * 3 * a.reps)
    jobs, i = [], 0
    for truth_name in TRUTHS:
        for scenario in ("A", "B", "C"):
            for _ in range(a.reps):
                jobs.append((truth_name, scenario, children[i]))
                i += 1
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(one, jobs, chunksize=4))
    labels = {"A": "A. VIX family alone (eta, rho fixed at the truth)", "B": "B. listed options alone",
              "C": "C. VIX family and listed options"}
    k = 0
    for truth_name, truth in TRUTHS.items():
        for scenario in ("A", "B", "C"):
            rows = results[k:k + a.reps]
            k += a.reps
            print(f"\n### {truth_name}: {labels[scenario]}\n")
            print(summarize(rows, truth, scenario))
    return 0


if __name__ == "__main__":
    with manifest.run("vol_identification", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/vol_identification.py",), seeds=[SEED]) as run:
        rc = main(sys.argv[1:])
    print(f"\nrun manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
