"""volmodel.black is a copy of definedtarget.pricing (so volmodel stands alone): the two must give identical numbers.

Grid: 9 maturities from 1 day to 15 years x 41 strikes from 0.2F to 5F x 7 vols from 2% to 150%, discount factors
0.6 and 1. Prices and implied vols must be bit-identical (the same operations in the same order); the round trip
price -> implied vol -> price must close to 1e-10 of the time value where the time value is above 1e-12 F.
"""
import numpy as np

from _harness import check

from definedtarget import pricing
from definedtarget.volmodel import black

F = 7712.57
taus = np.array([1 / 365, 9 / 365, 30 / 365, 0.25, 1.0, 2.0, 3.2, 10.0, 15.0])
Ks = F * np.exp(np.linspace(np.log(0.2), np.log(5.0), 41))
vols = np.array([0.02, 0.08, 0.16, 0.25, 0.4, 0.8, 1.5])
grid = np.array(np.meshgrid(taus, Ks, vols, [0.6, 1.0], indexing="ij")).reshape(4, -1)
tau, K, sig, DF = grid
Fa = np.full_like(tau, F)

p_new = black.call_price_array(Fa, K, DF, tau, sig)
p_old = pricing.call_price_array(Fa, K, DF, tau, sig)
check(f"call prices are bit-identical to definedtarget.pricing ({p_new.size} cases)", np.array_equal(p_new, p_old))
iv_new = black.implied_vol_array(p_new, Fa, K, DF, tau)
iv_old = pricing.implied_vol_array(p_old, Fa, K, DF, tau)
check("implied vols are bit-identical too (NaN in the same places)",
      np.array_equal(iv_new, iv_old, equal_nan=True))
tv = p_new - DF * np.maximum(Fa - K, 0)
ok = tv > 1e-12 * F
back = black.call_price_array(Fa[ok], K[ok], DF[ok], tau[ok], iv_new[ok])
gap = np.max(np.abs(back - p_new[ok]) / tv[ok])
check("price -> implied vol -> price closes to 1e-10 of the time value wherever the time value is above 1e-12 F",
      gap < 1e-10 and not np.isnan(iv_new[ok]).any(), gap)
check("puts by parity: P = C - DF (F - K)", black.put_price(F, 9000.0, 0.9, 3.0, 0.2)
      == black.call_price(F, 9000.0, 0.9, 3.0, 0.2) - 0.9 * (F - 9000.0))
check("implied_vols() broadcasts scalars over strike arrays",
      np.array_equal(black.implied_vols(p_new[:41], F, K[:41], tau[:41], DF[:41]), iv_new[:41], equal_nan=True))
