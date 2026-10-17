#!/usr/bin/env python3
"""v0.3 step 6: what real history's shape does, judged out of sample first.  From target_funds/:

    py -I scripts\\bootstrap_study.py [--spx PATH] [--french PATH] [--paths N]

Inputs (your own copies; the repository ships no data): S&P 500 daily closes (CSV with date, close; default
data/external/spx_daily_library.csv) and Ken French's daily research factors (F-F_Research_Data_Factors_daily.csv
from his data library at Dartmouth; market total return = Mkt-RF + RF; default data/external/).

Fixed before the first run (2026-10-17; the split was approved in the v0.3 depth pass):

Part A, out-of-sample check. Generators are built ONLY from S&P 500 price log returns of 1950-01-03 to 1999-12-31:
the lognormal with that period's daily mean and spread, and the bootstrap of that period at mean blocks of 21, 126
and 504 sessions (about 1 month, 6 months, 2 years). Test: every 814-session window (the sessions from 2026-10-14
to XX's expiry) starting from 2000 onward, realized in 2000-2026: its log return and its maximum drawdown. For each
window, the share of simulated paths below the realized value (its percentile under the generator). Reported: the
share of test windows below the generator's 10th percentile and above its 90th (ideal: 10% each) and the median
percentile (ideal 50%). The windows overlap: 2000-2026 holds only about 8 separate 814-session spans, so this is a
coarse check, not a test with power.

Part B, the shape effect for XX. Every generator gets the same inputs: XX's 2026-10-13 example (strike 1.27891 x
spot, 814 real sessions to 2030-01-10, vol 16.6%, drift = the forward's growth rate, ln(882.2603 / 778.01) / 3.2465753
a year). The lognormal runs on the real calendar; each bootstrap series is reset to the same daily mean and spread
(equal moments), so differences come from shape only. Histories: S&P 500 1950-2026 and the US market 1926-2026, at
mean blocks 21, 126, 504. Reported with standard errors: P(finish above the strike), P(touch the strike at a close
before expiry), the 10th / 50th / 90th percentile of S_T / S_0, the discounted mean payoff per $1 of spot,
E[S_T] / F, the mean maximum drawdown and P(maximum drawdown above 30%).
Added after the first run (disclosed, 2026-10-17): the value per $1 with each generator's paths rescaled so that
E[S_T] = F exactly ("value, E = F"). Equal log moments leave a skewed history's mean end price below the forward, and
the far out-of-the-money option is sensitive to that; the rescaled column separates shape from that mean shift.
Prints aggregates only.
"""
import argparse
import math
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from definedtarget import calendar as nyse  # noqa: E402
from definedtarget.montecarlo import bootstrap as bs  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.montecarlo import paths as lp  # noqa: E402

S0, F_EX, T_EX, R_EX = 778.01, 882.2603, 3.2465753, 0.0486498
K_RATIO = 995.01 / S0
VOL = 0.166
MU = math.log(F_EX / S0) / T_EX                    # the forward's growth rate (risk-neutral drift)
BLOCKS = (21, 126, 504)
SEED = 20261017


def load_spx(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    c = df["close"].to_numpy(dtype=float)
    if (c <= 0).any():
        raise ValueError("non-positive close in the S&P file")
    return df["date"].dt.date.to_numpy()[1:], np.diff(np.log(c))


def load_french(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [ln.split(",") for ln in lines if ln[:8].strip().isdigit() and len(ln.split(",")) >= 5]
    d = [date(int(r[0][:4]), int(r[0][4:6]), int(r[0][6:8])) for r in rows]
    tr = np.array([float(r[1]) + float(r[4]) for r in rows]) / 100.0
    return np.array(d), np.log1p(tr)


def find(name, explicit):
    p = Path(explicit) if explicit else REPO / "data" / "external" / name
    if not p.exists():
        raise FileNotFoundError(f"{p}: download it (see the docstring) or pass its path")
    return p


def pct(x):
    return f"{100 * x:.1f}%"


def part_a(dates, r, npaths, nsteps):
    build = r[dates <= date(1999, 12, 31)]
    test = r[dates >= date(2000, 1, 1)]
    cum = np.concatenate([[0.0], np.cumsum(test)])
    w_ret = cum[nsteps:] - cum[:-nsteps]
    w_mdd = np.empty(w_ret.size)
    for i in range(w_ret.size):                     # realized maximum drawdown of each window
        seg = cum[i:i + nsteps + 1] - cum[i]
        w_mdd[i] = float((np.maximum.accumulate(seg) - seg).max())
    print(f"Part A: build {dates[0]} to 1999-12-31 ({build.size:,} days); test 2000 onward ({test.size:,} days, "
          f"{w_ret.size:,} overlapping windows of {nsteps} sessions, about {test.size / nsteps:.1f} separate)\n")
    gens = [("lognormal (build mean, spread)", None)] + [(f"bootstrap, mean block {b}", b) for b in BLOCKS]
    m, s = build.mean(), build.std()
    print("| generator | windows below p10 / above p90: return | median percentile | windows below p10 / above "
          "p90: max drawdown | median percentile |")
    print("|---|---|---|---|---|")
    for label, b in gens:
        if b is None:
            t = np.arange(nsteps + 1, dtype=float)
            o = lp.lognormal(SEED, npaths, t, m + 0.5 * s * s, s)       # per-session units: drift m, spread s
        else:
            o = bs.paths(SEED, npaths, nsteps, build, b)
        cols = []
        for real, sim in ((w_ret, o["log_end"]), (w_mdd, o["log_max_drawdown"])):
            sim = np.sort(sim)
            pit = np.searchsorted(sim, real) / sim.size
            cols.append(f"{pct((pit < 0.1).mean())} / {pct((pit > 0.9).mean())} | {np.median(pit):.2f}")
        print(f"| {label} | {cols[0]} | {cols[1]} |")
    print()


def part_b(histories, npaths):
    sessions = nyse.sessions(date(2026, 10, 14), date(2030, 1, 10))
    t = lp.times_from_dates(sessions)
    nsteps = t.size - 1
    T = float(t[-1])
    step_mean, step_sd = (MU - 0.5 * VOL * VOL) * T / nsteps, VOL * math.sqrt(T / nsteps)
    df = math.exp(-R_EX * T)
    gens = [("lognormal (Black-76's model)", lp.lognormal(SEED, npaths, t, MU, VOL, barrier=K_RATIO))]
    for name, r in histories:
        x = bs.equal_moments(r, step_mean, step_sd)
        for b in BLOCKS:
            gens.append((f"{name}, block {b}", bs.paths(SEED, npaths, nsteps, x, b, barrier=K_RATIO)))
    print(f"Part B: XX's example, {nsteps} real sessions (T = {T:.4f} years), vol {VOL:.1%}, drift {MU:.4%} a year, "
          f"strike {K_RATIO:.5f} x spot; {npaths:,} paths each; +/- = 1 standard error\n")
    print("| generator | P(finish above strike) | P(touch strike at a close) | S_T/S_0 p10 / p50 / p90 | "
          "value per $1 | value, E = F | E[S_T]/F | mean max drawdown | P(max drawdown > 30%) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, o in gens:
        s = np.exp(o["log_end"])
        n = s.size
        fin = (s > K_RATIO).mean()
        tch = (o["hit_step"] >= 0).mean()
        pay = df * np.maximum(s - K_RATIO, 0.0)
        mdd = 1.0 - np.exp(-o["log_max_drawdown"])
        big = (mdd > 0.30).mean()
        q = np.quantile(s, [0.1, 0.5, 0.9])
        ef = s.mean() / math.exp(MU * T)
        adj = df * np.maximum(s / ef - K_RATIO, 0.0)     # every path scaled by the same factor: E[S_T] = F
        se = lambda p: math.sqrt(p * (1 - p) / n)  # noqa: E731
        print(f"| {label} | {fin:.4f} +/- {se(fin):.4f} | {tch:.4f} +/- {se(tch):.4f} | "
              f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} | {pay.mean():.5f} +/- {pay.std() / math.sqrt(n):.5f} | {adj.mean():.5f} | "
              f"{ef:.4f} | {mdd.mean():.4f} | {big:.4f} +/- {se(big):.4f} |")
    print()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spx")
    ap.add_argument("--french")
    ap.add_argument("--paths", type=int, default=1_000_000)
    a = ap.parse_args(argv)
    spx_path = Path(a.spx) if a.spx else REPO / "data" / "external" / "spx_daily_library.csv"
    fr_path = find("F-F_Research_Data_Factors_daily.csv", a.french)
    with manifest.run("bootstrap_study", argv, REPO / "results" / "manifests", REPO,
                      ("definedtarget/**/*.py", "scripts/bootstrap_study.py"), seeds=[SEED],
                      inputs={"spx": spx_path, "french": fr_path}, settings={"paths": a.paths, "blocks": BLOCKS}) as m:
        d_spx, r_spx = load_spx(spx_path)
        d_fr, r_fr = load_french(fr_path)
        print(f"S&P 500 price: {d_spx[0]} to {d_spx[-1]}, {r_spx.size:,} daily returns; US market total return "
              f"(French): {d_fr[0]} to {d_fr[-1]}, {r_fr.size:,}\n")
        part_a(d_spx, r_spx, a.paths, 814)
        part_b([("S&P 500 1950-2026", r_spx), ("US market 1926-2026", r_fr)], a.paths)
    print(f"run manifest: results/manifests/{m['path'].name}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
