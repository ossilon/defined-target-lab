"""The Monte Carlo engine's random numbers (definedtarget/montecarlo): judged by outside answers, not by themselves.

Judges: Random123's published known-answer vectors for Philox4x32 (7 and 10 rounds; fetched 2026-10-17 from
github.com/DEShawResearch/random123, tests/kat_vectors); scipy's ndtri for the inverse normal; exact identities
(the same address gives the same number on any thread count, chunk split or device); and loose statistical sanity.
GPU checks run only where CuPy and an NVIDIA GPU are present.
"""
import ast
import math
import time
from pathlib import Path

import numba
import numpy as np
from scipy.special import ndtri

from _harness import check, raises

from definedtarget.montecarlo import gpu, rng

PKG = Path(rng.__file__).resolve().parent
KAT = [  # (rounds, counter, key, expected words)
    (7, (0, 0, 0, 0), (0, 0), (0x5f6fb709, 0x0d893f64, 0x4f121f81, 0x4f730a48)),
    (7, (0xffffffff,) * 4, (0xffffffff,) * 2, (0x5207ddc2, 0x45165e59, 0x4d8ee751, 0x8c52f662)),
    (7, (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
     (0x4dfccaba, 0x190a87f0, 0xc47362ba, 0xb6b5242a)),
    (10, (0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
    (10, (0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
    (10, (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
     (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
]

# 1. Philox against the published answers
for rounds, c, k, want in KAT:
    got = tuple(int(w) for w in rng.philox4x32(*c, *k, rounds))
    check(f"Philox4x32-{rounds} matches Random123's known answer for counter {c[0]:08x}.., key {k[0]:08x}..",
          got == want, [hex(w) for w in got])

# 2. Uniforms: 52 bits, never 0 or 1
check("the smallest uniform is 2^-53 (never 0)", rng.u52(0, 0) == 2.0 ** -53)
check("the largest uniform is 1 - 2^-53 (never 1)", rng.u52(0xffffffff, 0xffffffff) == 1.0 - 2.0 ** -53)
u = rng.uniforms(7, 2000, 51)
check("every uniform lies strictly inside (0, 1)", bool((u > 0).all() and (u < 1).all()))

# 3. The inverse normal against scipy (AS241's stated accuracy is about 1e-16)
ps = np.concatenate([np.logspace(-300, -1, 600), np.linspace(0.0005, 0.9995, 2001), 1.0 - np.logspace(-16, -1, 300),
                     rng.uniforms(11, 1, 20000)[0]])
z = np.array([rng.inv_norm(p) for p in ps])
ref = ndtri(ps)
rel = np.abs(z - ref) / np.maximum(np.abs(ref), 1e-300)
check("AS241 inverse normal agrees with scipy's ndtri to 1e-14 relative, from p = 1e-300 to 1 - 1e-16",
      float(rel[np.abs(ref) > 1e-12].max()) < 1e-14, float(rel[np.abs(ref) > 1e-12].max()))
check("inv_norm(0.5) is exactly 0 and the function is antisymmetric",
      rng.inv_norm(0.5) == 0.0 and abs(rng.inv_norm(0.025) + rng.inv_norm(0.975)) < 1e-15)
check("inv_norm(0.975) = 1.959963984540054 (the familiar 95% two-sided value)",
      abs(rng.inv_norm(0.975) - 1.959963984540054) < 1e-15)

# 4. Addresses: one number per (seed, path, draw, stream), whatever the call
grid = rng.normals(42, 300, 9, path0=1000, stream=3)
check("normals() and the single-draw normal() agree address by address",
      all(grid[i, j] == rng.normal(42, 1000 + i, j, 3) for i in (0, 7, 299) for j in range(9)))
check("uniform() and uniforms() agree address by address",
      rng.uniforms(42, 5, 9, path0=1000, stream=3)[4, 8] == rng.uniform(42, 1004, 8, 3))
whole = rng.normals(5, 1000, 7)
parts = np.vstack([rng.normals(5, 400, 7), rng.normals(5, 600, 7, path0=400)])
check("chunk invariance: paths 0-999 in one call equal two chunks (400 + 600), bit for bit", np.array_equal(whole, parts))
check("a longer path starts with the same draws as a shorter one", np.array_equal(rng.normals(5, 10, 4), whole[:10, :4]))
default_threads = numba.get_num_threads()
numba.set_num_threads(1)
one = rng.normals(5, 1000, 7)
numba.set_num_threads(default_threads)
check(f"thread invariance: 1 thread equals {default_threads} threads, bit for bit", np.array_equal(one, whole))
big = 2 ** 40
check("path numbers above 2^32 use the high word (path 2^40 + 3 differs from path 3)",
      rng.normal(1, big + 3, 0) != rng.normal(1, 3, 0) and rng.normal(1, big + 3, 0) == rng.normals(1, 1, 1, path0=big + 3)[0, 0])
s0, s1 = rng.normals(9, 20000, 10, stream=0).ravel(), rng.normals(9, 20000, 10, stream=1).ravel()
check("streams 0 and 1 are uncorrelated (|corr| < 5 / sqrt(n))", abs(np.corrcoef(s0, s1)[0, 1]) < 5 / math.sqrt(s0.size))
check("different seeds give different numbers", rng.normal(1, 0, 0) != rng.normal(2, 0, 0))
raises("a negative seed is refused", lambda: rng.normals(-1, 1, 1), "64 bits")
raises("a stream beyond 32 bits is refused", lambda: rng.normals(1, 1, 1, stream=2 ** 32), "32 bits")

# 5. Loose statistical sanity on 4 million normals and 1 million uniforms (fixed seed, so the result is fixed)
x = rng.normals(2026, 400_000, 10).ravel()
n = x.size
check("mean of 4M normals within 5 standard errors of 0", abs(x.mean()) < 5 / math.sqrt(n), x.mean())
check("variance within 5 standard errors of 1", abs(x.var() - 1) < 5 * math.sqrt(2 / n), x.var())
check("excess kurtosis within 5 standard errors of 0", abs(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3) < 5 * math.sqrt(24 / n))
cells = np.bincount((rng.uniforms(77, 100_000, 10).ravel() * 100).astype(int), minlength=100)
chi2 = float(((cells - 1e4) ** 2 / 1e4).sum())
check("uniforms fill 100 equal cells evenly (chi-square within 6 sd of its 99 degrees of freedom)",
      abs(chi2 - 99) < 6 * math.sqrt(2 * 99), chi2)

# 6. The package stays project-agnostic: it imports only math, numpy, numba (and cupy in gpu.py)
allowed = {"math", "numpy", "numba", "cupy"}
bad = []
for f in sorted(PKG.glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            bad += [f"{f.name}: {a.name}" for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module.split(".")[0] not in allowed:
            bad.append(f"{f.name}: from {node.module}")
check("montecarlo imports nothing else from definedtarget or elsewhere (other projects can import it as is)", not bad, bad)

# 7. The GPU twin: same words, same uniforms, normals to the last digit or so
if gpu.available():
    import cupy as cp
    words = gpu.philox4x32([c for r, c, k, w in KAT if r == 10], [k for r, c, k, w in KAT if r == 10])
    check("GPU Philox4x32-10 matches Random123's known answers",
          [tuple(int(w) for w in row) for row in words] == [want for r, _, _, want in KAT if r == 10])
    g_u = cp.asnumpy(gpu.uniforms(42, 3001, 9, path0=2 ** 33, stream=5))
    check("GPU uniforms are bit-identical to the CPU's (3,001 paths x 9 draws, paths above 2^33, stream 5)",
          np.array_equal(g_u, rng.uniforms(42, 3001, 9, path0=2 ** 33, stream=5)))
    g_z = cp.asnumpy(gpu.normals(42, 3001, 9, path0=2 ** 33, stream=5))
    c_z = rng.normals(42, 3001, 9, path0=2 ** 33, stream=5)
    diff = float((np.abs(g_z - c_z) / np.maximum(np.abs(c_z), 1.0)).max())
    central = np.abs(rng.uniforms(42, 3001, 9, path0=2 ** 33, stream=5) - 0.5) <= 0.425
    check("GPU normals are bit-identical to the CPU's where AS241 is pure arithmetic (no fused multiply-add)",
          np.array_equal(g_z[central], c_z[central]))
    check("in the tails (which call log) they agree within 1e-15", diff < 1e-15, diff)
    print(f"info: GPU normals identical to the CPU's bit for bit: {np.mean(g_z == c_z):.2%} of {g_z.size:,}")
    t0 = time.perf_counter()
    gpu.normals(1, 1_000_000, 100)
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - t0
    t0 = time.perf_counter()
    rng.normals(1, 1_000_000, 100)
    t_cpu = time.perf_counter() - t0
    print(f"info: 100 million normals: CPU {t_cpu:.2f} s ({100 / t_cpu:.0f} M/s), GPU {t_gpu:.3f} s "
          f"({100 / t_gpu:.0f} M/s, float64, written to GPU memory)")
else:
    print("skip GPU checks: CuPy or an NVIDIA GPU is not available")
