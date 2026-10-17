"""Counter-based random numbers: Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", 2011).

Plain words
- An ordinary generator hands out numbers one after another, so who draws first changes who gets what: results then
  depend on the number of threads, the chunk size and the device. Philox is a function instead: give it an address
  (a 128-bit counter) and a key, and it returns four random 32-bit words, always the same ones for that address.
- Here the address of every draw is (seed, path, draw, stream): the key holds the 64-bit seed; the counter holds the
  draw pair, the 64-bit path number and a stream number (so separate purposes never share numbers). Any thread, any
  chunk and the GPU (montecarlo.gpu) compute exactly the same numbers for the same address.
- Uniforms use 52 bits of two words: u = (x + 0.5) / 2^52, never 0 and never 1, so the inverse normal is always finite.
- Normals use the inverse of the normal distribution (Wichura's algorithm AS241, relative error about 1e-16), so one
  uniform makes one normal and the mapping keeps order (needed later for antithetic paths and quasi-random methods).

Checked against Random123's published known-answer vectors (tests/test_mc_rng.py). Pure numpy + numba: this package
imports nothing else from definedtarget, so other projects can import it as it is.
"""
import math

import numba
import numpy as np

MASK32 = np.uint64(0xFFFFFFFF)
SH32 = np.uint64(32)
SH26 = np.uint64(26)
SH6 = np.uint64(6)
M0 = np.uint64(0xD2511F53)          # Philox4x32 multipliers
M1 = np.uint64(0xCD9E8D57)
W0 = np.uint64(0x9E3779B9)          # Weyl key increments (golden ratio and sqrt(3) - 1, as 32-bit fractions)
W1 = np.uint64(0xBB67AE85)
TWO_M52 = 2.0 ** -52
ROUNDS = 10


@numba.njit(cache=True)
def philox4x32(c0, c1, c2, c3, k0, k1, rounds=ROUNDS):
    """The four 32-bit output words for counter (c0..c3) and key (k0, k1); all arguments and results uint64 < 2^32."""
    c0, c1, c2, c3 = np.uint64(c0), np.uint64(c1), np.uint64(c2), np.uint64(c3)
    k0, k1 = np.uint64(k0), np.uint64(k1)
    for r in range(rounds):
        if r > 0:                                    # the key moves on before every round but the first
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
        p0 = M0 * c0                                 # 32 x 32 -> 64-bit products: high and low halves
        p1 = M1 * c2
        hi0, lo0 = p0 >> SH32, p0 & MASK32
        hi1, lo1 = p1 >> SH32, p1 & MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


@numba.njit(cache=True)
def _address(seed, path, block, stream):
    """Key and counter of one draw pair: key = the seed's two halves; counter = (pair, path low, path high, stream)."""
    s, p = np.uint64(seed), np.uint64(path)
    return (np.uint64(block) & MASK32, p & MASK32, p >> SH32, np.uint64(stream) & MASK32,
            s & MASK32, s >> SH32)


@numba.njit(cache=True)
def u52(a, b):
    """A uniform in (0, 1) from two 32-bit words: 26 high bits of each make 52 bits; +0.5 keeps it off 0 and 1."""
    x = ((np.uint64(a) >> SH6) << SH26) | (np.uint64(b) >> SH6)
    return (float(x) + 0.5) * TWO_M52


@numba.njit(cache=True)
def uniform(seed, path, draw, stream=0):
    """The draw-th uniform of a path. Draws 2i and 2i+1 share one Philox call (words 0-1 and 2-3)."""
    c0, c1, c2, c3, k0, k1 = _address(seed, path, draw >> 1, stream)
    w0, w1, w2, w3 = philox4x32(c0, c1, c2, c3, k0, k1)
    return u52(w0, w1) if draw % 2 == 0 else u52(w2, w3)


@numba.njit(cache=True)
def inv_norm(p):
    """The standard normal quantile of p in (0, 1): Wichura (1988), Applied Statistics algorithm AS241 (PPND16)."""
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r + 6.7265770927008700853e+4) * r
                   + 4.5921953931549871457e+4) * r + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r
                + 1.3314166789178437745e+2) * r + 3.3871328727963666080e+0)
        den = (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r + 3.9307895800092710610e+4) * r
                   + 2.1213794301586595867e+4) * r + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r
                + 4.2313330701600911252e+1) * r + 1.0)
        return q * num / den
    r = p if q < 0.0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r
                   + 1.27045825245236838258e+0) * r + 3.64784832476320460504e+0) * r + 5.76949722146069140550e+0) * r
                + 4.63033784615654529590e+0) * r + 1.42343711074968357734e+0)
        den = (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r
                   + 1.48103976427480074590e-1) * r + 6.89767334985100004550e-1) * r + 1.67638483018380384940e+0) * r
                + 2.05319162663775882187e+0) * r + 1.0)
    else:
        r -= 5.0
        num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r
                   + 2.65321895265761230930e-2) * r + 2.96560571828504891230e-1) * r + 1.78482653991729133580e+0) * r
                + 5.46378491116411436990e+0) * r + 6.65790464350110377720e+0)
        den = (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r
                   + 7.86869131145613259100e-4) * r + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r
                + 5.99832206555887937690e-1) * r + 1.0)
    z = num / den
    return -z if q < 0.0 else z


@numba.njit(cache=True)
def normal(seed, path, draw, stream=0):
    """The draw-th standard normal of a path (inverse normal of the same-address uniform)."""
    return inv_norm(uniform(seed, path, draw, stream))


@numba.njit(parallel=True, cache=True)
def _fill(seed, path0, npaths, ndraws, stream, as_normal, out):
    for i in numba.prange(npaths):                 # each path is computed from its address: thread layout is irrelevant
        path = path0 + np.uint64(i)                # uint64 + int64 would become a float in numba: cast first
        for b in range((ndraws + 1) // 2):
            c0, c1, c2, c3, k0, k1 = _address(seed, path, b, stream)
            w0, w1, w2, w3 = philox4x32(c0, c1, c2, c3, k0, k1)
            u = u52(w0, w1)
            out[i, 2 * b] = inv_norm(u) if as_normal else u
            if 2 * b + 1 < ndraws:
                u = u52(w2, w3)
                out[i, 2 * b + 1] = inv_norm(u) if as_normal else u


def _check(seed, path0, npaths, ndraws, stream):
    if not (0 <= seed < 2 ** 64 and 0 <= path0 and path0 + npaths <= 2 ** 64 and 0 <= stream < 2 ** 32):
        raise ValueError("seed and path numbers must fit in 64 bits, the stream in 32 bits, all non-negative")
    if npaths < 0 or ndraws < 0 or (ndraws + 1) // 2 > 2 ** 32:
        raise ValueError("npaths and ndraws must be non-negative, with at most 2^33 draws per path")


def normals(seed, npaths, ndraws, path0=0, stream=0):
    """(npaths, ndraws) standard normals for paths path0 .. path0 + npaths - 1 (CPU, all threads)."""
    _check(seed, path0, npaths, ndraws, stream)
    out = np.empty((npaths, ndraws))
    _fill(np.uint64(seed), np.uint64(path0), npaths, ndraws, np.uint64(stream), True, out)
    return out


def uniforms(seed, npaths, ndraws, path0=0, stream=0):
    """(npaths, ndraws) uniforms in (0, 1), same addresses as normals()."""
    _check(seed, path0, npaths, ndraws, stream)
    out = np.empty((npaths, ndraws))
    _fill(np.uint64(seed), np.uint64(path0), npaths, ndraws, np.uint64(stream), False, out)
    return out
