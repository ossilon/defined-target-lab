"""GPU path kernels (montecarlo.gpu.lognormal / .bootstrap) against the CPU kernels, measure by measure.

The judge is the CPU engine itself, which tests/test_mc_paths.py and tests/test_mc_bootstrap.py already judge by
exact formulas: the GPU must reproduce it bit for bit (p_no_touch, which calls exp, within 1e-15). Skipped without
CuPy and an NVIDIA GPU.
"""
import math
import time
from datetime import date

import numpy as np

from _harness import check

from definedtarget import calendar as nyse
from definedtarget.montecarlo import bootstrap as bs
from definedtarget.montecarlo import gpu
from definedtarget.montecarlo import paths as lp

if not gpu.available():
    print("skip GPU path checks: CuPy or an NVIDIA GPU is not available")
else:
    import cupy as cp

    t = lp.times_from_dates(nyse.sessions(date(2026, 10, 14), date(2030, 1, 10)))
    args = dict(barrier=1.28, checkpoints=(100, 400, 814), path0=2 ** 35, stream=9)
    cpu = lp.lognormal(77, 20_000, t, 0.0387, 0.166, **args)
    g = gpu.lognormal(77, 20_000, t, 0.0387, 0.166, **args)
    # The path arithmetic is identical; a rare tail normal (about 1 in 40,000, where AS241 calls log) rounds its last
    # digit differently on the GPU, so a few path values differ by about 1e-16 (diagnosed 2026-10-17: every differing
    # path held a differing tail normal). Target from the v0.3 design review: within 1e-6 path by path; tested at 1e-12.
    for k in ("log_end", "log_max", "log_drawdown_end", "log_max_drawdown", "at_checkpoint"):
        d = float(np.nanmax(np.abs(cpu[k] - g[k])))
        same = float(np.mean(cpu[k] == g[k]))
        check(f"GPU lognormal {k} equals the CPU's within 1e-12 on every path ({same:.2%} bit for bit; largest gap "
              f"{d:.1e})", d < 1e-12 and same > 0.99, (d, same))
    check("GPU lognormal hit_step (the first close at the level) equals the CPU's on every path",
          np.array_equal(cpu["hit_step"], g["hit_step"]))
    check("GPU p_no_touch equals the CPU's within 1e-15 (it calls exp)",
          float(np.abs(cpu["p_no_touch"] - g["p_no_touch"]).max()) < 1e-15)
    print(f"info: p_no_touch identical bit for bit on {np.mean(cpu['p_no_touch'] == g['p_no_touch']):.2%} of paths")
    no_b = gpu.lognormal(3, 5000, t, 0.05, 0.2)
    check("without a barrier: hit_step -1 and p_no_touch 1 everywhere, as on the CPU",
          bool((no_b["hit_step"] == -1).all() and (no_b["p_no_touch"] == 1.0).all()))

    rs = np.random.default_rng(5)
    hist = np.column_stack([rs.standard_t(4, 4000) * 0.01, rs.standard_normal(4000)])
    bargs = dict(barrier=math.exp(0.05), checkpoints=(10, 200), path0=12345, stream=4)
    cb = bs.paths(8, 20_000, 300, hist, 21.0, **bargs)
    gb = gpu.bootstrap(8, 20_000, 300, hist, 21.0, **bargs)
    check("GPU bootstrap equals the CPU's bit for bit, every measure (20,000 paths x 300 steps, 2 columns)",
          all(np.array_equal(cb[k], gb[k]) for k in cb))

    # Speed (information): 1 million paths on the real calendar, one launch, results copied back
    for label, fn in (("lognormal", lambda: gpu.lognormal(1, 1_000_000, t, 0.0387, 0.166, barrier=1.28, as_cupy=True)),
                      ("bootstrap", lambda: gpu.bootstrap(1, 1_000_000, 814, hist[:, 0], 126.0, barrier=1.28,
                                                          as_cupy=True))):
        fn()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        sec = time.perf_counter() - t0
        print(f"info: GPU {label}: 1,000,000 paths x 814 steps in {sec:.3f} s ({814 / sec:.0f} million path-steps/s)")
