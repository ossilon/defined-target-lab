"""The GPU twin of montecarlo.rng (optional: needs CuPy and an NVIDIA GPU).

Same Philox4x32-10, same addresses (seed, path, draw, stream), same 52-bit uniforms and the same AS241 inverse
normal, written in CUDA C and compiled on first use by NVRTC (through CuPy). The 32-bit words and the uniforms are
bit-identical to the CPU's. Normals are bit-identical where the inverse normal is pure arithmetic (the central 85%);
in the tails it calls log, and a platform's log may round the last digit differently (identical on the RTX 3080 and
Windows numba, 2026-10-17). tests/test_mc_rng.py checks both.

GPU words for a beginner: a *kernel* is a function that thousands of GPU threads run at once, each on its own index;
here one thread makes one draw pair of one path. `__umulhi` gives the high 32 bits of a 32 x 32-bit product in one
instruction, the operation Philox is built on.
"""
import numpy as np

SOURCE = r"""
extern "C" {

__device__ __forceinline__ void philox4x32(unsigned int c[4], unsigned int k0, unsigned int k1, int rounds) {
    for (int r = 0; r < rounds; r++) {
        if (r > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        unsigned int hi0 = __umulhi(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
        unsigned int n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    }
}

__device__ __forceinline__ double u52(unsigned int a, unsigned int b) {
    unsigned long long x = ((unsigned long long)(a >> 6) << 26) | (unsigned long long)(b >> 6);
    return ((double)x + 0.5) * 2.220446049250313080847e-16;    /* 2^-52 */
}

__device__ double inv_norm(double p) {                          /* Wichura AS241 (PPND16), as in rng.inv_norm */
    double q = p - 0.5, r, num, den;
    if (fabs(q) <= 0.425) {
        r = 0.180625 - q * q;
        num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r + 6.7265770927008700853e+4) * r
                 + 4.5921953931549871457e+4) * r + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r
               + 1.3314166789178437745e+2) * r + 3.3871328727963666080e+0);
        den = (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r + 3.9307895800092710610e+4) * r
                 + 2.1213794301586595867e+4) * r + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r
               + 4.2313330701600911252e+1) * r + 1.0);
        return q * num / den;
    }
    r = q < 0.0 ? p : 1.0 - p;
    r = sqrt(-log(r));
    if (r <= 5.0) {
        r -= 1.6;
        num = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r
                 + 1.27045825245236838258e+0) * r + 3.64784832476320460504e+0) * r + 5.76949722146069140550e+0) * r
               + 4.63033784615654529590e+0) * r + 1.42343711074968357734e+0);
        den = (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r
                 + 1.48103976427480074590e-1) * r + 6.89767334985100004550e-1) * r + 1.67638483018380384940e+0) * r
               + 2.05319162663775882187e+0) * r + 1.0);
    } else {
        r -= 5.0;
        num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r
                 + 2.65321895265761230930e-2) * r + 2.96560571828504891230e-1) * r + 1.78482653991729133580e+0) * r
               + 5.46378491116411436990e+0) * r + 6.65790464350110377720e+0);
        den = (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r
                 + 7.86869131145613259100e-4) * r + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r
               + 5.99832206555887937690e-1) * r + 1.0);
    }
    double z = num / den;
    return q < 0.0 ? -z : z;
}

__global__ void philox_words(const unsigned int* ctr, const unsigned int* key, int rounds, unsigned int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int c[4] = {ctr[4 * i], ctr[4 * i + 1], ctr[4 * i + 2], ctr[4 * i + 3]};
    philox4x32(c, key[2 * i], key[2 * i + 1], rounds);
    for (int j = 0; j < 4; j++) out[4 * i + j] = c[j];
}

__global__ void philox_fill(unsigned long long seed, unsigned long long path0, long long npaths, long long ndraws,
                            unsigned int stream, int as_normal, double* out) {
    long long nblocks = (ndraws + 1) / 2;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npaths * nblocks) return;
    long long i = t / nblocks, b = t % nblocks;
    unsigned long long path = path0 + (unsigned long long)i;
    unsigned int c[4] = {(unsigned int)b, (unsigned int)(path & 0xFFFFFFFFull), (unsigned int)(path >> 32), stream};
    philox4x32(c, (unsigned int)(seed & 0xFFFFFFFFull), (unsigned int)(seed >> 32), 10);
    double u = u52(c[0], c[1]);
    out[i * ndraws + 2 * b] = as_normal ? inv_norm(u) : u;
    if (2 * b + 1 < ndraws) {
        u = u52(c[2], c[3]);
        out[i * ndraws + 2 * b + 1] = as_normal ? inv_norm(u) : u;
    }
}

}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        import cupy as cp
        # --fmad=false: no fused multiply-add, so a*b+c rounds twice exactly as on the CPU. With fusion on, half the
        # normals differed from the CPU's in the last digit; with it off all matched (measured 2026-10-17).
        _MODULE = cp.RawModule(code=SOURCE, options=("--std=c++17", "--fmad=false"))
    return _MODULE


def available():
    """True if CuPy imports and sees a GPU."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - any failure means no usable GPU
        return False


def philox4x32(counters, keys, rounds=10):
    """Philox words on the GPU for many (counter, key) pairs: counters (n, 4), keys (n, 2) -> (n, 4) uint32 (numpy)."""
    import cupy as cp
    c = cp.asarray(np.asarray(counters, dtype=np.uint32).reshape(-1, 4))
    k = cp.asarray(np.asarray(keys, dtype=np.uint32).reshape(-1, 2))
    n = c.shape[0]
    out = cp.empty((n, 4), dtype=cp.uint32)
    _module().get_function("philox_words")(((n + 255) // 256,), (256,), (c, k, np.int32(rounds), out, np.int32(n)))
    return cp.asnumpy(out)


def _fill(seed, npaths, ndraws, path0, stream, as_normal):
    import cupy as cp
    from .rng import _check
    _check(seed, path0, npaths, ndraws, stream)
    out = cp.empty((npaths, ndraws), dtype=cp.float64)
    threads = npaths * ((ndraws + 1) // 2)
    if threads:
        _module().get_function("philox_fill")(((threads + 255) // 256,), (256,),
                                              (np.uint64(seed), np.uint64(path0), np.int64(npaths), np.int64(ndraws),
                                               np.uint32(stream), np.int32(as_normal), out))
    return out


def normals(seed, npaths, ndraws, path0=0, stream=0):
    """(npaths, ndraws) standard normals on the GPU (a CuPy array), same addresses as rng.normals."""
    return _fill(seed, npaths, ndraws, path0, stream, 1)


def uniforms(seed, npaths, ndraws, path0=0, stream=0):
    """(npaths, ndraws) uniforms on the GPU (a CuPy array), bit-identical to rng.uniforms."""
    return _fill(seed, npaths, ndraws, path0, stream, 0)
