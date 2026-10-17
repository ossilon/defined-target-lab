"""definedtarget.montecarlo: a project-agnostic Monte Carlo engine (v0.3, being built step by step).

It imports nothing else from definedtarget, so other projects can import it unchanged. Built so far:
- rng: counter-based random numbers (Philox4x32-10) and the AS241 inverse normal, on the CPU (numba, all threads);
- gpu: the same numbers on an NVIDIA GPU (optional, needs CuPy).
Every draw has an address (seed, path, draw, stream), so results do not depend on threads, chunks or device.
"""
from .rng import inv_norm, normal, normals, philox4x32, uniform, uniforms  # noqa: F401
