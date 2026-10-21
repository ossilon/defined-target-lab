# The models, with their mathematics and sources

This file explains each mathematical model in the project: what it assumes, the formulas the code uses, how the
code is checked, and where each formula comes from. It grows with the code. v0.3.1 adds the volatility model,
section by section as each part is built; v0.3.3 adds the earlier models (Black-76 pricing, the fund engine, the
Monte Carlo engine and the bootstrap). Notation follows the code. Time is in years; a variance of 0.04 is a
volatility of 20%.

## 1. The volatility model (v0.3.1, in progress)

Package: `definedtarget/volmodel/`. It imports only Python's standard library, numpy, numba and scipy, never the
rest of the project, so other projects can use it unchanged. Its interface may change until v0.3.3.

Why XX needs it. XX holds one 3-year call on SPY, struck 28% above the index. Its value moves with the index and
with the market's long-dated implied volatility at that strike (about 10% of NAV per vol point, FINDINGS S1). The
simulators so far used one vol, or one vol with a fixed "slide" as the index moves. F17 showed that the slide rule
breaks a basic consistency test (the discounted mark must not drift), so the project needs a model in which the
index and its variance move together and every option price comes from the same model.

### 1.1 The Heston model

Under the pricing measure, in forward terms (F_T is the forward for expiry T, X_T = ln(S_T / F_T)):

    dS / S = sqrt(v) dW1        dv = kappa (theta - v) dt + eta sqrt(v) dW2        dW1 dW2 = rho dt

| Parameter | Meaning | Typical S&P 500 value (Gatheral 2006, Table 3.2) |
|---|---|---|
| v0 | variance now | 0.0174 (13.2% vol) |
| theta | long-run variance | 0.0354 (18.8% vol) |
| kappa | speed of mean reversion, per year | 1.33 (half-life of a variance shock about 6 months) |
| eta | vol of variance | 0.39 |
| rho | correlation of index and variance shocks | -0.72 (variance tends to rise when the index falls) |

The Feller condition 2 kappa theta >= eta^2 keeps the variance away from zero. Fits to index options usually break
it (the values above give 0.62), so the code reports it and does not require it. Source: Heston (1993).

### 1.2 The characteristic function

Everything else is built on phi(u) = E[exp(i u X_T)], known in closed form (Heston 1993). With
beta = kappa - i rho eta u and d = sqrt(beta^2 + eta^2 (u^2 + i u)) (principal square root),

    g = (beta - d) / (beta + d)
    D = (beta - d) / eta^2 * (1 - e^{-dT}) / (1 - g e^{-dT})
    C = kappa theta / eta^2 * [ (beta - d) T - 2 ln( (1 - g e^{-dT}) / (1 - g) ) ]
    phi(u) = exp(C + D v0)

This is the form in Gatheral (2006, eq. 2.12) and Fang and Oosterlee (2008, section 3.3). Heston's original form
writes the same function with e^{+dT}; with the principal logarithm it jumps whenever the logarithm's argument crosses
the negative real axis, which happens at long maturities ("the little Heston trap", Albrecher et al. 2007). Lord and
Kahl (2008) proved that the form above, with the principal logarithm, gives the correct function (as cited by Fang
and Oosterlee 2008). The tests show both behaviors: for Andersen's (2008) Case I at
10 years the original form jumps near u = 0.2, and the form above does not.

Checks in `tests/test_vol_heston.py`: phi(0) = 1; phi(-i) = 1 (the forward is a martingale); |phi| <= 1 and
phi(-u) = conj(phi(u)) on the real line; as eta goes to 0 with rho = 0, phi tends to the normal characteristic
function of a deterministic-variance model, with a gap that shrinks like eta^2.

### 1.3 Moments

Expected total variance from now to T (Gatheral 2006, ch. 11):

    E[W_T] = theta T + (v0 - theta) (1 - e^{-kappa T}) / kappa

It does not depend on eta or rho. The mean of X_T is -E[W_T] / 2. The variance of X_T is Andersen's (2008,
Table A.1) closed form. Fang and Oosterlee (2008, Table 11) print the same variance with the term
theta (6 e^{-kappa T} - 7) where theta (4 e^{-kappa T} - 5) is correct (FINDINGS F21); the tests check Andersen's
form against numerical derivatives of log phi (bound 1e-7; observed below 1e-8).

### 1.4 Two option pricers

Inputs are in forward terms: the forward F, strikes K, maturity T and the discount factor DF. Prices are per unit of
the underlying.

Pricer 1, the COS method (Fang and Oosterlee 2008). The density of X_T on a range [A, B] is replaced by its
Fourier-cosine series. The series coefficients come straight from phi, and the payoff's coefficients are closed
forms, so a European put is

    P = DF K sum'_{k=0}^{N-1} Re[ phi(k pi / (B - A)) e^{-i k pi A / (B - A)} ] U_k(K)

where sum' counts the first term half and U_k is the put payoff's cosine coefficient on the part of [A, B] below the
strike. Calls follow by put-call parity, which the paper recommends because a call's payoff grows without bound. N
doubles from 64 until the added terms fall below 1e-14 of the forward.

The range is where this code departs from the paper. The paper uses the mean plus or minus 12 standard deviations.
Heston's left tail is exponential, not normal, and at long maturities that range cuts off real probability: 0.38 too
low for the December 2029 10,000 call under Andersen's Case I (FINDINGS F21). The code instead bounds each tail with
Chernoff's inequality,

    P(X_T > B) <= E[e^{sX_T}] e^{-sB}  for s > 0,  and the mirror image for the left tail,

minimized over s up to the point where the moment E[e^{sX_T}] becomes infinite (the moment explosion; the code finds
it as the first point where the closed form stops being finite, real and convex). The range then holds all but 1e-15
of the probability on each side, whatever the shape of the distribution.

Pricer 2, the Lewis-Lipton integral (Lewis 2001; Lipton 2002), in Andersen's (2008, Proposition 3) notation:

    C = DF ( F - (K / pi) int_0^inf Re[ exp( (1/2 - i k) ln(F/K) + h1(k) - (k^2 + 1/4) h2(k) v0 ) ] / (k^2 + 1/4) dk )

with h1 and h2 from Andersen's own algebra, so this pricer shares no characteristic-function code with the first. The
integral uses 24-point Gauss-Legendre panels until the integrand's decaying envelope falls below 1e-14. Its puts come
from parity, so deep out-of-the-money puts lose relative accuracy; the COS pricer handles those.

A checked price (`pricers.prices(..., check=True)`) returns the COS price only if the two pricers agree within 1e-9
of the discounted forward, and raises otherwise.

Checks in `tests/test_vol_pricers.py`:
- Fang and Oosterlee's reference values: within 3.5e-10 at 10 years; at one year all three methods here agree to
  1e-14 and sit 1.6e-8 below the printed reference (FINDINGS F21); their Black-Scholes values to 1e-9.
- The two pricers agree within 1e-10 of the forward (in practice about 1e-14) for five parameter sets, six
  maturities from 9 days to 10 years, and 41 strikes spanning 4 standard deviations, including XX's strike.
- scipy's adaptive quadrature of the same integral agrees to 1e-10.
- No arbitrage: call prices lie between DF max(F - K, 0) and DF F, fall as the strike rises, and are convex.
- The lognormal limit, a wider range moving no price, and Lee's (2004) wing bound (implied total variance per unit
  of log-moneyness below 2).

### 1.5 Variance swaps and the VIX family

For any diffusion the fair variance-swap rate equals a strip of out-of-the-money options weighted by 1/K^2
(Demeterfi, Derman, Kamal and Zou 1999; Gatheral 2006, ch. 11):

    E[W_T] = -2 E[ln(S_T / F)] = (2 / DF) [ int_0^F P(K) / K^2 dK + int_F^inf C(K) / K^2 dK ]

Cboe's VIX is 100 times the square root of a 30-day rate computed from a discrete version of this strip on listed
S&P 500 options; VIX9D, VIX3M, VIX6M and VIX1Y do the same at other tenors. In Heston the rate is linear in today's
variance:

    VIX^2(tau) = a + b v0,   b = (1 - e^{-kappa tau}) / (kappa tau),   a = theta (1 - b)

`varswap.strip_variance` computes the strip from any price function. Integrated over the model's own prices, it
must reproduce E[W_T]; `tests/test_vol_varswap.py` checks this to 1e-9 for 22 cases, so the whole smile, wings
included, is judged by one closed form. Cboe's exact discretization is applied when the model is fitted to the real
VIX family (v0.3.1 step 4).

Because the rate does not depend on eta or rho, the VIX family can identify only v0, theta and kappa, and only up
to its longest tenor, one year (section 1.6).

### 1.6 Fitting the model, and what the data can identify

`calibrate.fit_heston` minimizes sum(((model vol - market vol) / sd)^2) over listed implied vols and vol-index
levels, with sd each quote's noise in vol units. The parameters move on an unconstrained scale (logarithms, and the
inverse hyperbolic tangent for rho), so every trial point is a valid model; any of them can be held fixed. The
solver is Levenberg-Marquardt (Levenberg 1944; Marquardt 1963) with central-difference derivatives. Standard errors
come from (J^T J)^{-1} at the answer, carried to the natural scale by the chain rule; a parameter that no quote
depends on gets an infinite standard error.

Checks in `tests/test_vol_calibrate.py`: Rosenbrock's valley and a linear problem; exact recovery of Gatheral's
parameters from noise-free quotes laid out like the project's data, from three starts; agreement with scipy's
`least_squares`; and the VIX-only case, where eta and rho come out unidentified as they should.

What the data can identify (FINDINGS F20, from 200 noisy synthetic copies of that layout): the VIX family alone
leaves the long-run vol wide open (for the Gatheral truth, 17.5 to 25.7 vol points, 5-95%), while adding the listed
Dec-2027 to Dec-2029 strikes pins the vol at XX's strike to within about 0.1 vol points. The remaining question is
therefore whether the model fits the listed strikes and their day-to-day moves at all, which the frozen predictions
H20 and H21 and the v0.3.1 fit test out of sample.

### 1.7 Not built yet

- Simulation of the index and its variance together (Andersen's 2008 quadratic-exponential scheme with its
  martingale correction), judged against the two pricers: v0.3.1 step 3.
- The fit to the real VIX family and listed strikes under the fixed holdout, and the out-of-sample accuracy target:
  step 4.
- A two-factor variance model (Bergomi 2005), for the term structure one speed cannot fit: step 5.

## References

- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. Wilmott Magazine,
  issue 1, 83-92.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. Journal of
  Computational Finance 11(3).
- Bergomi, L. (2005). Smile dynamics II. Risk, October.
- Demeterfi, K., Derman, E., Kamal, M. and Zou, J. (1999). A guide to volatility and variance swaps. Journal of
  Derivatives 6(4).
- Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series
  expansions. SIAM Journal on Scientific Computing 31(2), 826-848.
- Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and
  currency options. Review of Financial Studies 6(2), 327-343.
- Lee, R. W. (2004). The moment formula for implied volatility at extreme strikes. Mathematical Finance 14(3),
  469-480.
- Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. Quarterly of
  Applied Mathematics 2(2).
- Lewis, A. (2001). Option Valuation Under Stochastic Volatility. Finance Press, Newport Beach.
- Lipton, A. (2002). The vol smile problem. Risk 15(2), 61-65.
- Lord, R. and Kahl, C. (2008). Complex logarithms in Heston-like models. Working paper, SSRN 1105998.
- Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. Journal of the
  Society for Industrial and Applied Mathematics 11(2).
