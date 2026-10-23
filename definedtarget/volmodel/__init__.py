"""definedtarget.volmodel: volatility models for long-dated index options (v0.3.1, in progress).

Asset-agnostic: it imports only Python's standard library, numpy, numba, scipy and (for simulation) the sibling
package montecarlo, never the rest of definedtarget, so other projects can import it unchanged. Inputs
are forwards, strikes, maturities and discount factors; nothing here knows about XX or the S&P 500.

Modules (step 2: the closed-form judges come first):
- heston: the Heston (1993) model, its characteristic function and its moments;
- pricers: two independent European option pricers built on it, the COS method (Fang and Oosterlee 2008) and the
  Lewis-Lipton integral (Andersen 2008, Proposition 3), and a checked price that requires them to agree;
- varswap: the variance-swap identities behind the VIX family (the log-strip of option prices);
- black: Black-76 and its implied volatility (a copy of the project's pricing code, so this package stands alone;
  tests/test_vol_black.py keeps the two identical);
- calibrate: a Levenberg-Marquardt fit of the Heston parameters to implied vols and variance-swap rates, with
  standard errors and the correlation of the estimates (how well the data pins each parameter down);
- stochvol (step 3): Heston paths by Andersen's (2008) QE scheme with the martingale correction, on montecarlo's
  random streams; eta = 0 reproduces montecarlo.paths.lognormal bit for bit;
- pricetable (step 3): Heston implied vols on a grid of log-moneyness and vol for one remaining maturity, and calls
  priced from it, fast enough to mark options along simulated paths;
- termstructure (step 4): the variance-swap term structure fitted to vol indices, with a bootstrap band;
- two factors (step 5): heston.DoubleHeston, priced by both pricers; termstructure.fit_two (non-negative factor
  variances, the slow factor's drift) and vol_of_variance; calibrate.fit_double_days; stochvol.simulate_two.
The interface is provisional until v0.3.3: names and arguments may change. The mathematics, with sources, is in
docs/MODELS.md.
"""
from .heston import Heston  # noqa: F401
