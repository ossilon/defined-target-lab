"""definedtarget.montecarlo: a project-agnostic Monte Carlo engine (v0.3).

It imports nothing else from definedtarget, so other projects can import it unchanged. Modules:
- rng: counter-based random numbers (Philox4x32-10) and the AS241 inverse normal, on the CPU (numba, all threads);
- stats: mergeable moments, Wilson intervals, percentile intervals;
- paths: lognormal paths on any time grid, with one-pass path measures;
- stream: little-memory results for any number of paths (histograms, exact percentiles, run to a precision);
- bootstrap: the circular stationary bootstrap of real history;
- gpu: the same numbers and path kernels on an NVIDIA GPU (optional, needs CuPy);
- causal: the look-ahead checker for rules; manifest: run manifests for reproducible results.
Every draw has an address (seed, path, draw, stream), so results do not depend on threads, chunks or device.
The interface is provisional until v0.3.3 (docs/learn/06-the-monte-carlo-engine.md): names and arguments may change.
"""
from .rng import inv_norm, normal, normals, philox4x32, uniform, uniforms  # noqa: F401
