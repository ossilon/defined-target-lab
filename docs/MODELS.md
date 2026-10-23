# The models, with their mathematics and sources

This file explains each mathematical model in the project. For each model it gives the assumptions, the formulas
that the code uses, the tests that check the code, and the source of each formula. The file grows with the code.
Release v0.3.1 adds the volatility model, one part at a time. Release v0.3.3 adds the earlier models: Black-76
pricing, the fund engine, the Monte Carlo engine and the bootstrap.

The notation follows the code. Time is in years. A variance of 0.04 is a volatility of 20%.

## 1. The volatility model (v0.3.1, in progress)

The package is `definedtarget/volmodel/`. It imports only the Python standard library, numpy, numba, scipy and the
sibling Monte Carlo package. It never imports the rest of this project, so other projects can use it as it is. Its
interface can change until v0.3.3.

Why XX needs it. XX holds one 3-year call on SPY with a strike 28% above the index. The value of XX moves with the
index. It also moves with the long-dated implied volatility at that strike: about 10% of the NAV for each vol point
(FINDINGS S1).

The earlier simulators used one vol, or one vol with a fixed slide when the index moves. F17 shows that the slide
rule fails a basic test: the discounted mark of a traded option must not drift. So the project needs a model in
which the index and its variance move together, and in which one model gives every option price.

### 1.1 The Heston model

The model works under the pricing measure and in forward terms. F_T is the forward for expiry T, and
X_T = ln(S_T / F_T).

```math
\frac{dS}{S} = \sqrt{v}\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \eta\sqrt{v}\,dW_2, \qquad dW_1\,dW_2 = \rho\,dt
```

| Parameter | Meaning | A typical S&P 500 value (Gatheral 2006, Table 3.2) |
|---|---|---|
| v0 | the variance now | 0.0174 (a vol of 13.2%) |
| theta | the long-run variance | 0.0354 (a vol of 18.8%) |
| kappa | the speed of mean reversion, each year | 1.33 (a variance shock loses half its size in about 6 months) |
| eta | the vol of the variance | 0.39 |
| rho | the correlation of the index shocks and the variance shocks | -0.72 (the variance tends to rise when the index falls) |

The Feller condition, 2 kappa theta >= eta^2, keeps the variance away from zero. Fits to index options usually fail
this condition: the values above give 0.62. So the code reports the condition but does not require it. The source
of the model is Heston (1993).

### 1.2 The characteristic function

All other parts of the model use the characteristic function phi(u) = E[exp(i u X_T)]. Heston (1993) gives it in
closed form:

```math
\beta = \kappa - i\rho\eta u, \qquad d = \sqrt{\beta^2 + \eta^2 (u^2 + i u)}, \qquad g = \frac{\beta - d}{\beta + d}
```

```math
D = \frac{\beta - d}{\eta^2}\,\frac{1 - e^{-dT}}{1 - g e^{-dT}}, \qquad
C = \frac{\kappa\theta}{\eta^2}\left[(\beta - d)\,T - 2\ln\frac{1 - g e^{-dT}}{1 - g}\right], \qquad
\phi(u) = e^{C + D v_0}
```

The square root is the principal root. This form is the one in Gatheral (2006, equation 2.12) and in Fang and
Oosterlee (2008, section 3.3).

The original form of Heston writes the same function with e^{+dT}. With the principal logarithm, that form jumps
when the argument of the logarithm crosses the negative real axis. This happens at long maturities (Albrecher et al.
2007, "the little Heston trap"). Lord and Kahl (2008) prove that the form above gives the correct function, as Fang
and Oosterlee cite.

The tests in `tests/test_vol_heston.py` check these properties:
- phi(0) = 1, and phi(-i) = 1, so the forward is a martingale;
- the absolute value of phi is at most 1 on the real line, and phi(-u) is the conjugate of phi(u);
- the original form jumps near u = 0.2 for Case I of Andersen (2008) at 10 years, and the form above does not;
- when eta goes to 0 with rho = 0, phi goes to the normal characteristic function, and the gap falls like eta^2.

### 1.3 The moments

The expected total variance from now to T comes from Gatheral (2006, ch. 11):

```math
E[W_T] = \theta T + (v_0 - \theta)\,\frac{1 - e^{-\kappa T}}{\kappa}
```

It does not depend on eta or rho. The mean of X_T is -E[W_T] / 2. The variance of X_T is the closed form of
Andersen (2008, Table A.1).

Fang and Oosterlee (2008, Table 11) print the same variance with an error. Their term theta (6 e^{-kappa T} - 7)
must be theta (4 e^{-kappa T} - 5) (FINDINGS F21). The tests compare the variance of Andersen with the numerical
derivatives of log phi. The limit is 1e-7, and the largest difference is below 1e-8.

### 1.4 Two option pricers

The inputs are in forward terms: the forward F, the strikes K, the maturity T and the discount factor DF. The
prices are for one unit of the underlying.

The first pricer is the COS method of Fang and Oosterlee (2008). It replaces the density of X_T on a range [A, B]
with its Fourier-cosine series. The coefficients of the series come directly from phi. The coefficients of the
payoff have closed forms. A European put is then:

```math
P = DF\,K \sum_{k=0}^{N-1}{}' \,\mathrm{Re}\!\left[\phi\!\left(\frac{k\pi}{B - A}\right) e^{-ik\pi A/(B - A)}\right] U_k(K)
```

The prime on the sum gives the first term half its weight. U_k is the cosine coefficient of the put payoff on the
part of [A, B] below the strike. Calls come from put-call parity, as the paper recommends. The number of terms N
doubles from 64 until the new terms add less than 1e-14 of the forward.

The range [A, B] is different from the range in the paper. The paper uses the mean plus or minus 12 standard
deviations. The left tail of Heston is exponential, not normal. At long maturities that range cuts off real
probability. For example, it gives a price 0.38 too low for the December 2029 10,000 call under Case I of Andersen
(FINDINGS F21). The code bounds each tail with the inequality of Chernoff instead:

```math
P(X_T > B) \le E\!\left[e^{sX_T}\right] e^{-sB} \quad \text{for } s > 0 \text{, and the mirror image for the left tail}
```

The code takes the smallest bound over s, up to the moment explosion. At that point E[e^{sX_T}] becomes infinite.
The code finds it as the first point where the closed form is no longer finite, real and convex. The range then
holds all but 1e-15 of the probability on each side, for any shape of the distribution.

The second pricer is the Lewis-Lipton integral (Lewis 2001; Lipton 2002), in the notation of Andersen (2008,
Proposition 3):

```math
C = DF\left(F - \frac{K}{\pi}\int_0^\infty
\frac{\mathrm{Re}\!\left[\exp\!\left((\tfrac{1}{2} - ik)\ln\tfrac{F}{K} + h_1(k) - (k^2 + \tfrac{1}{4})\,h_2(k)\,v_0\right)\right]}
{k^2 + \tfrac{1}{4}}\,dk\right)
```

The functions h1 and h2 come from the algebra of Andersen. So this pricer shares no code with the first pricer for
the characteristic function. The code integrates with 24-point Gauss-Legendre panels until the decay of the
integrand makes the rest smaller than 1e-14. This pricer gets its puts from parity. So deep out-of-the-money puts
lose relative accuracy here, and the COS pricer gives those puts.

A checked price, `pricers.prices(..., check=True)`, gives the COS price only when the two pricers agree within 1e-9
of the discounted forward. If they do not agree, it stops with an error.

The tests in `tests/test_vol_pricers.py` check these properties:
- The reference values of Fang and Oosterlee agree within 3.5e-10 at 10 years and within 1e-9 for their Black-Scholes
  cases.
- At one year, three methods here agree to 1e-14, and they are 1.6e-8 below the printed reference (FINDINGS F21).
- The two pricers agree within 1e-10 of the forward for five parameter sets, six maturities and 41 strikes. The
  strikes include the strike of XX. In practice the difference is about 1e-14.
- The adaptive quadrature of scipy gives the same integral within 1e-10.
- Each call price lies between DF max(F - K, 0) and DF F, falls when the strike rises, and is convex in the strike.
- The prices go to Black-76 in the lognormal limit, a wider range changes no price, and the wings obey the bound of
  Lee (2004).

### 1.5 Variance swaps and the VIX family

For any diffusion, the fair rate of a variance swap is equal to a strip of out-of-the-money options. Each option
has the weight 1/K^2 (Demeterfi, Derman, Kamal and Zou 1999; Gatheral 2006, ch. 11):

```math
E[W_T] = -2\,E\!\left[\ln\frac{S_T}{F}\right] = \frac{2}{DF}\left[\int_0^F \frac{P(K)}{K^2}\,dK + \int_F^\infty \frac{C(K)}{K^2}\,dK\right]
```

The VIX of Cboe is 100 times the square root of a 30-day rate. Cboe computes it from a discrete version of this
strip on listed S&P 500 options. VIX9D, VIX3M, VIX6M and VIX1Y do the same at other tenors. In Heston the rate is a
linear function of the variance now:

```math
\mathrm{VIX}^2(\tau) = a + b\,v_0, \qquad b = \frac{1 - e^{-\kappa\tau}}{\kappa\tau}, \qquad a = \theta\,(1 - b)
```

The function `varswap.strip_variance` computes the strip from any price function. For the prices of the model, the
strip must give E[W_T]. The file `tests/test_vol_varswap.py` checks this to 1e-9 in 22 cases. So one closed form
checks the whole smile, the wings included. The fit to the real VIX family uses the discrete method of Cboe (v0.3.1,
step 4).

The rate does not depend on eta or rho. So the VIX family can identify only v0, theta and kappa. It can identify
them only up to its longest tenor, one year (section 1.6).

### 1.6 The fit, and what the data can identify

The function `calibrate.fit_heston` minimizes the sum of ((model vol - market vol) / sd)^2. The quotes are listed
implied vols and the levels of vol indices. Each quote has a noise sd in vol units. The parameters move on a scale
without limits: logarithms, and the inverse hyperbolic tangent for rho. So each trial point is a valid model. The
fit can hold any parameter fixed.

The solver is Levenberg-Marquardt (Levenberg 1944; Marquardt 1963) with derivatives from central differences. The
standard errors come from the inverse of J^T J at the answer, and the chain rule takes them to the natural scale. A
parameter that no quote depends on gets an infinite standard error.

The tests in `tests/test_vol_calibrate.py` check these properties:
- the solver finds the answer of the Rosenbrock valley and of a linear problem;
- the fit recovers the parameters of Gatheral exactly from quotes without noise, from three starts;
- the fit agrees with `least_squares` of scipy;
- with only the VIX family, the fit reports eta and rho as unidentified, as it must.

FINDINGS F20 shows what the data can identify. It uses 200 noisy synthetic copies of the quotes of this project. The
VIX family alone leaves the long-run vol wide open. For the parameters of Gatheral, the range is 17.5 to 25.7 vol
points (5% to 95%).

With the listed strikes of December 2027 to December 2029, the fit pins the vol at the strike of XX to
about 0.1 vol points. The open question is therefore whether the model fits the listed strikes and their daily
moves. The frozen predictions H20 and H21 and the out-of-sample fit of v0.3.1 test this.

### 1.7 The simulation of paths

The function `stochvol.simulate` makes Heston paths on any increasing time grid, the real NYSE calendar included.
Each step moves the variance v first and then the log index x. Each path keeps a few numbers, as in
`montecarlo.paths`. These are the end values, the integrated variance, the highest close and the drawdowns. They
also include the first close at a barrier and the values at chosen checkpoints.

The default scheme is the quadratic-exponential (QE) scheme of Andersen (2008) with his martingale correction
(QE-M). These are its three parts:
1. The next variance matches the exact conditional mean m and variance s^2 of the square-root process (his
   equations 17 and 18). If psi = s^2 / m^2 is at most 1.5, the next variance is a (b + Z)^2 with a normal Z
   (equations 23, 27 and 28). Otherwise it is 0 with probability p and exponential in the other cases (equations 24
   to 26, 29 and 30). This handles the mass near zero that breaks Euler schemes when the Feller condition fails.
2. The log index uses the exact relation between the increments of the index and of the variance (equation 11).
   The integral of v over the step uses the trapezoid rule (equations 32 and 33).
3. The martingale correction changes one constant of the step so that E[S(t + dt) | S(t)] = S(t) e^(mu dt) exactly
   (his Propositions 7 and 9).

```math
x_{t+\Delta} = x_t + \mu\Delta + K_0^* + K_1 v_t + K_2 v_{t+\Delta} + \sqrt{K_3 v_t + K_4 v_{t+\Delta}}\;Z
```

The correction exists only if A < 1/(2a) in the quadratic branch, or A < beta in the exponential branch. For
rho <= 0 these conditions are always true. A search over a wide range of positive rho found no case that fails them.
The code still counts each step that fails, and then stops with an error. The scheme "qe" omits the correction. The
scheme "euler" is the full-truncation Euler scheme (his equations 6 and 7), which the code keeps only for
comparison.

With eta = 0, the variance follows its mean path and each step is lognormal. This is the lognormal limit. With
v0 = theta it is `montecarlo.paths.lognormal`, number for number.

The random numbers come from the engine. Draw k of a path on stream s drives the index at step k, as in
`montecarlo.paths`, and the variance uses stream s + 1. Each draw has an address (seed, path, draw, stream). So the
paths do not depend on the number of threads or on the size of the chunks.

The tests in `tests/test_vol_stochvol.py` check these properties:
- In the lognormal limit, each shared output is equal to `montecarlo.paths.lognormal` bit for bit, on the real
  calendar.
- One QE step matches the conditional mean and variance in both branches.
- E[V_T] and Var[V_T] equal the closed forms of Andersen at any step size, and E[S_T] = S_0 e^(mu T).
- The code reproduces 13 biases from the tables of Andersen within 3 combined standard errors, for the Euler, QE and
  QE-M schemes.
- At fine steps, QE-M gives the exact prices within 3 standard errors, and on the daily calendar at the strike of XX.
- The paths are bit-identical in one call or in chunks, and with one thread or with all threads.

### 1.8 Price tables

A path kernel cannot call a Fourier pricer at each step of a million paths. So `pricetable.build` makes a table for
one remaining maturity tau. The table holds the implied vol of the model on a grid of log-moneyness k = ln(K / F)
and of vol s = sqrt(v). The function `pricers.cos_call_surface` gives all the prices of the table in one
calculation. This works because log phi = C + D v, and C and D do not depend on v.

A call is then Black-76 at the interpolated implied vol. The interpolation is a local cubic (Catmull-Rom) in k and
in s. Implied vols are smooth even near expiry, where prices bend sharply at the strike. The table has one extra row
and one extra column at each outer edge, so that each cell has its full stencil. Below s = 0 the table uses the row
at s = ds, because the vol depends on v = s^2 and is symmetric in s.

The default grid covers k from -3 to 3 and vols up to 200%. So an index fall of 95% stays inside the grid. The
COS pricer gives the exact price of each state outside the grid, and the table counts those states. A code review
on 2026-10-23 replaced the first rule, which gave 0 above the strike range and capped the vol at 120%. That rule
priced states with an extreme vol too low.

The tests in `tests/test_vol_pricetable.py` check these properties:
- the surface agrees with the one-at-a-time pricer to 1e-12;
- the table gives the direct COS price within 2e-6 per unit of the forward, at 400 random points;
- this holds at four maturities from one week to 3.2 years;
- outside the grid the table gives the exact COS price;
- the marked option is a martingale along simulated paths (section 1.9).

### 1.9 The error of the simulation, and the martingale identity

The script `scripts/qe_bias_study.py` measures two things for the December 2029 10,000 call under the parameters of
Gatheral (FINDINGS F22). It uses 16,777,216 paths for each case.
1. The bias of the scheme. On daily steps the bias of QE-M is 0.0006 vol points, and the standard error is 0.0037.
   On steps of 63 sessions the bias of QE-M stays below 0.03 vol points. The Euler scheme reaches 0.35 vol points
   on those steps.
2. The martingale identity. The table marks the call along daily QE-M paths. Its expected value at 0.5, 1.5 and 2.5
   years stays within 1.1 standard errors of the price now. Each standard error is at most 0.05% of the price. The
   slide rule gave a drift of -5.55% for the same option at mid-life (FINDINGS F19).

So the drift of F17 is a property of the slide rule. A model in which the index and its variance move together does
not have it.

### 1.10 The fit to real data under the holdout

The module `volmodel.termstructure` fits one speed kappa and one long-run variance theta to a history of vol
indices. It also fits the variance of each day. For given kappa and theta, the variance of a day has a closed form.
The fit then moves only kappa and theta. A moving-block bootstrap of the days gives the band of theta.

The function `calibrate.fit_heston_days` fits the smile to option trades on many days. Each day has its own
variance, from the vol indices of that day. The module `definedtarget/listedvols.py` gives the implied vols of the
listed December 2029 calls from their daily trades.

The script `scripts/vol_holdout_study.py` holds the protocol. Its commit came before any test-window number. The
result is in FINDINGS F23:
- One speed cannot fit the VIX family. The long-run vol of the fit is 54.9%, and the bootstrap gives it no upper end.
- The model misses the accuracy target. Its error is 1.39 vol points, inside the 1.5 target, but a constant 16.6% is
  better. Most of the error is a bias of -1.20 points.
- The fitted dynamics explain the weekly moves of the 3-year strike vol best of four readings.

### 1.11 The two-factor model

One speed cannot fit the VIX family (section 1.10). The two-factor Heston model of Christoffersen, Heston and Jacobs
(2009) has two independent variance factors. Each factor is a Heston factor with its own speed and level:

```math
\frac{dS}{S} = \sqrt{v_1}\,dW_1 + \sqrt{v_2}\,dW_2, \qquad
dv_i = \kappa_i(\theta_i - v_i)\,dt + \eta_i\sqrt{v_i}\,dZ_i, \qquad dW_i\,dZ_i = \rho_i\,dt
```

All other pairs of shocks are independent. So the log price is a sum of two independent Heston log prices, and the
characteristic function is the product of the two Heston functions. The class `heston.DoubleHeston` holds the
model. Both pricers of section 1.4 price it with no other change. With equal kappa, eta and rho the model is the
one-factor model with v0 = v1 + v2 and theta = theta1 + theta2. The tests use this as an exact check.

We chose this model and not the lognormal two-factor model of Bergomi (2005). This model keeps the closed-form
prices, and so it keeps the two pricers as judges. The fits here use one eta and one rho for both factors. This is
a restriction of the model of Christoffersen, Heston and Jacobs. The 3-year trades cannot identify a second eta and
rho.

The term structure. The variance-swap rate is the sum of the rates of the two factors:

```math
\text{rate}(\tau) = \theta_1 (1 - b_1) + b_1 v_1 + c_2 g_2 + b_2 v_2, \qquad g_2 = \frac{1 - b_2}{\kappa_2},
\qquad c_2 = \kappa_2 \theta_2
```

Here b_i is the b of section 1.5 at the speed kappa_i. The function `termstructure.fit_two` fits kappa1, theta1,
kappa2 and c2. For each day, it finds the two variances in closed form, with both variances at 0 or above.

Three findings from the fit window set this form:
1. If the variances can go below 0, any theta1 gives the same rates. The fit then put the variances below 0 on
   1,868 of 2,587 days. With the variances at 0 or above, as the model requires, the fit identifies theta1.
2. The slow speed goes to 0 from every start. So the fit uses the drift c2 of the slow factor, not its long-run
   level. As kappa2 goes to 0, g2 goes to tau / 2, and every formula stays finite. The floor of kappa2 is 1e-6 each
   year (`termstructure.KAPPA2_MIN`).
3. The function `termstructure.vol_of_variance` measures eta from the time series of each factor. A change of
   measure changes drifts, but not the diffusion. So a diffusion model must show the same eta in the indices and in
   the option prices.

The simulation. The function `stochvol.simulate_two` gives each factor its own QE step and its own martingale
correction. The two next variances are independent, so the correction is a product of one term for each factor. The
random numbers of factor 1 are those of the one-factor simulation, so its variance path is the same, bit for bit.

The script `scripts/vol_twofactor_study.py` holds the protocol. Its commit came before any test-window number. The
result is in FINDINGS F24:
- The slow factor does not revert in any of 200 bootstrap refits.
- From VIX and VIX3M alone, the model predicts VIX1Y out of sample with an error of 1.81 vol points. The one-factor
  model gives 3.24.
- The model passes the accuracy target with an error of 1.25 vol points. The constant 16.6% gives 1.35, and the
  difference is within the noise of the 54 test days. The gain over the one-factor model is clear. A bias of -0.96
  points remains.
- The smile needs an eta of 1.31. The time series of the indices shows 0.73. So the model does not fit both.

### 1.12 Price tables for two factors

The function `pricetable.build_two` stores the implied variance of the two-factor model at one remaining maturity.
The grid has three axes: the log-moneyness k and the two vols s1 = sqrt(v1) and s2 = sqrt(v2). The function
`pricers.cos_call_surface_two` prices the whole grid in one calculation. The function `pricetable.calls_two`
interpolates with tricubic Catmull-Rom cubics and prices by Black-76. It prices a state outside the grid with the COS
pricer.

Two choices come from measurements:
1. The table stores the implied variance, not the vol. At short maturities the vol is close to
   sqrt(s1^2 + s2^2). That function has a cone at 0, which a cubic does not fit well. Its square has no cone.
2. The step in k is 0.0125. Near the money at short maturities, the smile bends on a scale of sigma sqrt(tau). A
   step of 0.025 gave an error of 4.8e-5 for each unit of forward at one month. A step of 0.0125 gave 8.7e-6 at one
   month, 2.0e-6 at half a year and 6.4e-7 at 2.2 years. Smaller steps in the vols did not help.

The tests mark the option along two-factor paths with these tables. Its mean at 0.5 and 2.0 years equals the price
now within 3 standard errors (the martingale identity).

The module `definedtarget/forward/fc2.py` uses the tables for the draft forecast model M5 (protocol FC2). It is not
active until the ledger holds FC2.

### 1.13 Parts not built yet

- XX along two-factor paths, repriced each session, and the studies of v0.3.1 step 6.

## References

- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. Wilmott Magazine,
  issue 1, 83-92.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. Journal of
  Computational Finance 11(3).
- Bergomi, L. (2005). Smile dynamics II. Risk, October.
- Christoffersen, P., Heston, S. and Jacobs, K. (2009). The shape and term structure of the index option smirk: why
  multifactor stochastic volatility models work so well. Management Science 55(12), 1914-1932.
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
