#!/usr/bin/env python3
"""GPU stress test with the engine's own kernels (BACKLOG B27).  From target_funds/:

    py -I scripts\\gpu_stress_test.py [--minutes 5] [--stop-temp 83]

What it does, in plain words
- Runs the real path kernels (montecarlo.gpu: lognormal and bootstrap, XX's 814 real sessions) back to back at full
  load for a fixed time, each batch on new paths.
- Every second it reads the card's temperature, clocks, power, fan and load (nvidia-smi) and stops early if the
  temperature reaches --stop-temp (83 C by default: where the RTX 3080 starts slowing itself to protect itself).
- After every batch it recomputes 2,000 of that batch's paths on the CPU and compares them: a hot or unstable card can
  make silent errors, and this would catch them (lognormal within 1e-12, bootstrap bit for bit).
- It keeps a running estimate of P(touch +28% before expiry) over all lognormal paths and compares it with the exact
  reflection-principle value: a correctness check at full scale.
- At the end it reports speed in the first and last minute (a drop means thermal throttling), the temperature range,
  clocks, power, any mismatches, and the touch estimate. The per-second log is saved to results/raw/ (git-ignored).
Requires CuPy and an NVIDIA GPU. Synthetic or S&P-shaped inputs only; prints aggregates.
"""
import argparse
import csv
import math
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from definedtarget import calendar as nyse  # noqa: E402
from definedtarget.montecarlo import bootstrap as bs  # noqa: E402
from definedtarget.montecarlo import gpu  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.montecarlo import paths as lp  # noqa: E402

FIELDS = "temperature.gpu,clocks.sm,clocks.mem,power.draw,fan.speed,utilization.gpu"
MU, VOL, BARRIER = 0.0387, 0.166, 1.28


def read_gpu():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
    return [float(v) if v.replace(".", "", 1).isdigit() else float("nan") for v in out]


def history():
    path = REPO / "data" / "external" / "spx_daily_library.csv"
    if path.exists():
        c = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
        r = np.diff(np.log(c))
    else:                                                     # synthetic stand-in when no S&P file is present
        r = np.random.default_rng(1).standard_t(4, 19_000) * 0.007
    return bs.equal_moments(r, (MU - 0.5 * VOL * VOL) / 251.0, VOL / math.sqrt(251.0))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--stop-temp", type=float, default=83.0)
    ap.add_argument("--paths", type=int, default=4_000_000, help="paths per batch")
    a = ap.parse_args(argv)
    if not gpu.available():
        print("no CuPy / NVIDIA GPU: nothing to stress")
        return 1
    import cupy as cp

    t = lp.times_from_dates(nyse.sessions(date(2026, 10, 14), date(2030, 1, 10)))
    T, nsteps = float(t[-1]), t.size - 1
    hist = history()
    nu, b = MU - 0.5 * VOL * VOL, math.log(BARRIER)
    exact_touch = (0.5 * math.erfc(-(nu * T - b) / (VOL * math.sqrt(T)) / math.sqrt(2))
                   + math.exp(2 * nu * b / VOL ** 2) * 0.5 * math.erfc(-(-b - nu * T) / (VOL * math.sqrt(T)) / math.sqrt(2)))

    log, stop = [], threading.Event()
    reason = {"why": "time limit"}
    t_start = time.perf_counter()

    def monitor():
        while not stop.is_set():
            row = read_gpu()
            log.append([round(time.perf_counter() - t_start, 2)] + row)
            if row[0] >= a.stop_temp:
                reason["why"] = f"temperature reached {row[0]:.0f} C"
                stop.set()
            time.sleep(1.0)

    threading.Thread(target=monitor, daemon=True).start()
    print(f"Stress test: {a.minutes:g} minutes, stop at {a.stop_temp:g} C, {a.paths:,} paths per batch x {nsteps} steps")
    print(f"start: {dict(zip(FIELDS.split(','), read_gpu()))}\n")
    batches, done_steps, touch_sum, touch_n, worst_gap, mismatches = [], 0, 0.0, 0, 0.0, 0
    rs = np.random.default_rng(7)
    path0 = 0
    deadline = t_start + 60.0 * a.minutes
    while time.perf_counter() < deadline and not stop.is_set():
        kind = "lognormal" if len(batches) % 2 == 0 else "bootstrap"
        t0 = time.perf_counter()
        if kind == "lognormal":
            o = gpu.lognormal(2026, a.paths, t, MU, VOL, barrier=BARRIER, path0=path0, as_cupy=True)
            touch_sum += float((1.0 - o["p_no_touch"]).sum())
            touch_n += a.paths
        else:
            o = gpu.bootstrap(2026, a.paths, nsteps, hist, 126.0, barrier=BARRIER, path0=path0, as_cupy=True)
        cp.cuda.Stream.null.synchronize()
        sec = time.perf_counter() - t0
        k = int(rs.integers(0, a.paths - 2000))               # spot check: 2,000 of this batch's paths on the CPU
        g_end = cp.asnumpy(o["log_end"][k:k + 2000])
        if kind == "lognormal":
            c_end = lp.lognormal(2026, 2000, t, MU, VOL, barrier=BARRIER, path0=path0 + k)["log_end"]
            gap = float(np.abs(g_end - c_end).max())
            mismatches += int(gap > 1e-12)
            worst_gap = max(worst_gap, gap)
        else:
            c_end = bs.paths(2026, 2000, nsteps, hist, 126.0, barrier=BARRIER, path0=path0 + k)["log_end"]
            mismatches += int(not np.array_equal(g_end, c_end))
        del o
        batches.append((time.perf_counter() - t_start, kind, sec))
        done_steps += a.paths * nsteps
        path0 += a.paths
        if len(batches) % 10 == 0:
            last = log[-1] if log else [0] * 7
            print(f"  {batches[-1][0]:6.0f} s  batch {len(batches):4d}  {last[1]:.0f} C  {last[2]:.0f} MHz  "
                  f"{last[4]:.0f} W  fan {last[5]:.0f}%  mismatches {mismatches}")
    stop.set()
    time.sleep(1.2)
    elapsed = time.perf_counter() - t_start
    out_dir = REPO / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(out_dir / f"gpu_stress_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seconds"] + FIELDS.split(","))
        w.writerows(log)
    arr = np.array([r[1:] for r in log], dtype=float)

    def speed(kind, first):
        rows = [(s, sec) for s, k, sec in batches if k == kind and ((s <= 60) if first else (s >= elapsed - 60))]
        return a.paths * nsteps * len(rows) / max(sum(sec for _, sec in rows), 1e-9) / 1e9

    p = touch_sum / max(touch_n, 1)
    print(f"\nStopped: {reason['why']} after {elapsed:.0f} s; {len(batches)} batches, {path0:,} paths, "
          f"{done_steps / 1e12:.2f} trillion path-steps")
    print(f"speed, billion path-steps/s: lognormal first minute {speed('lognormal', True):.2f}, last minute "
          f"{speed('lognormal', False):.2f}; bootstrap first {speed('bootstrap', True):.2f}, last {speed('bootstrap', False):.2f}")
    print(f"temperature {np.nanmin(arr[:, 0]):.0f}-{np.nanmax(arr[:, 0]):.0f} C; SM clock {np.nanmin(arr[:, 1]):.0f}-"
          f"{np.nanmax(arr[:, 1]):.0f} MHz; power up to {np.nanmax(arr[:, 3]):.0f} W; fan up to {np.nanmax(arr[:, 4]):.0f}%; "
          f"load median {np.nanmedian(arr[:, 5]):.0f}%")
    print(f"CPU spot checks: {len(batches)} batches x 2,000 paths, {mismatches} mismatches (largest lognormal gap "
          f"{worst_gap:.1e})")
    print(f"P(touch +28% before expiry) over {touch_n:,} lognormal paths: {p:.6f} +/- "
          f"{math.sqrt(p * (1 - p) / max(touch_n, 1)):.6f} (exact {exact_touch:.6f})")
    print(f"log: results/raw/gpu_stress_{stamp}.csv")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    with manifest.run("gpu_stress_test", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("definedtarget/**/*.py", "scripts/gpu_stress_test.py"), deterministic=False) as run:
        rc = main(sys.argv[1:])
    print(f"run manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
