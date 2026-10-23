"""Protocol FC2 (a DRAFT until frozen in the ledger): the v0.3.1 dynamics model as a fifth model, M5, of the monthly
forecast check (FC1 rule 10 allows a later protocol to add models from its own freeze date onward).

Plain words
- M5 is the two-factor Heston model of v0.3.1 (FINDINGS F24; docs/MODELS.md section 1.11), with the parameters fixed
  at the freeze (PARAMS). At each FC1 origin t from FC2's freeze date on, the four vol indices of that close give
  the two factor variances; SPY and both variances move together under the pricing measure from t to the target T
  (volmodel.stochvol, QE with the martingale correction, on the real NYSE calendar); XX is repriced at T with the
  model's own price (a two-factor table at the remaining maturity), and its growth is the model NAV at T over the
  model NAV at t, as for M1-M4.
- M5 has no path vol and no slide: both come out of the model. SPY's month is not lognormal under M5, so the saved
  forecast holds SPY's growth at the same 999 points as XX's, and SPY's PIT reads them the same way.
- Seeds and paths are FC1's (seed = year x 100 + month, one million paths, stream 0): M5's index draws are M1's.
"""
import math

import numpy as np

from ..volmodel import pricers, pricetable, stochvol, termstructure as ts
from ..volmodel.heston import DoubleHeston, Heston
from . import forecast as fc

INDICES = ("VIX", "VIX3M", "VIX6M", "VIX1Y")
TENORS = np.array([30, 93, 184, 366]) / 365.0        # Cboe's tenors (calendar days), as in the v0.3.1 studies
SD_INDEX = 0.005                                     # the step 4 and 5 fits' index noise (vol units)
FREEZE_DATE = None                                   # set at the freeze: the FC2 models run for origins on or after it
PARAMS = None                                        # set at the freeze: {model name: params_from(...)}


def active(origin):
    """{model name: params} of the frozen FC2 models for this origin ({} before the freeze or before its date)."""
    if FREEZE_DATE is None or PARAMS is None or origin < FREEZE_DATE:
        return {}
    return dict(PARAMS)


def params_from(rec):
    """The M5 parameters from a results/fc2_parameters.json entry: the term structure (kappa1, theta1, kappa2, c2)
    that turns indices into states, and the pricing model's theta2, eta and rho."""
    t, s = rec["term_structure"], rec["smile"]
    return {"kappa1": t["kappa1"], "theta1": t["theta1"], "kappa2": t["kappa2"], "c2": t["c2"],
            "theta2": s["theta2"], "eta": s["eta"], "rho": s["rho"]}


def model_at(params, v1, v2):
    """The pricing model at the factor variances v1, v2."""
    p = params
    return DoubleHeston(Heston(v1, p["theta1"], p["kappa1"], p["eta"], p["rho"]),
                        Heston(v2, p["theta2"], p["kappa2"], p["eta"], p["rho"]))


def states(params, index_vols):
    """(v1, v2) from one close's VIX, VIX3M, VIX6M and VIX1Y (decimal vols), with the frozen term structure."""
    p = params
    v = ts.states_two(p["kappa1"], p["theta1"], p["kappa2"], p["c2"], TENORS,
                      np.asarray(index_vols, dtype=float)[None, :], SD_INDEX)[0]
    if not np.all(np.isfinite(v)):
        raise ValueError(f"no M5 state from the indices {index_vols}")
    return float(v[0]), float(v[1])


def index_closes(t, root=None, fetch=True):
    """({index: close as a decimal vol}, {index: provenance}) for close t: each by forecast.index_close (the newest
    saved Cboe file captured by the entry's deadline that has t, as FC1 rule 3 reads the VIX). Raises ValueError if
    one is missing."""
    from . import fetch as fetch_
    root = fetch_.data_dir(root)
    out, recs = {}, {}
    for name in INDICES:
        value, rec = fc.index_close(name, t, root, fetch)
        out[name], recs[name] = value / 100.0, {"name": rec["name"], "fetched_utc_date": rec["fetched_utc"][:10],
                                                "sha256": rec.get("sha256", "")}
    return out, recs


def nav(params, v1, v2, S, tau, rate, q, strike, options, cash, fee_factor, table=None):
    """Model NAV per XX share: fee_factor x options x DF F c(k; v1, v2, tau) + cash, with c from a two-factor table
    (arrays of states) or, without a table, from the COS pricer (one state)."""
    F = S * math.exp((rate - q) * tau) if np.isscalar(S) else S * np.exp((rate - q) * tau)
    DF = math.exp(-rate * tau)
    if table is None:
        c = float(pricers.cos_prices(model_at(params, v1, v2), 1.0, strike / F, tau)[0])
    else:
        c = pricetable.calls_two(table, np.log(strike / F), v1, v2)
    return fee_factor * options * DF * F * c + cash


def simulate_growth(inp, T, index_vols, params, npaths=fc.PATHS):
    """(XX's model growth NAV(T) / NAV(t) per path, SPY's growth S_T / S_t per path, the state (v1, v2) at t)."""
    times, k = fc._grid(inp.origin, T)
    horizon, t_T = float(times[-1]), float(times[k])
    q = inp.rate - math.log(inp.forward / inp.spot) / inp.tau
    v1, v2 = states(params, [index_vols[n] for n in INDICES])
    o = stochvol.simulate_two_model(model_at(params, v1, v2), fc.seed_for(inp.origin), npaths, times[:k + 1],
                                    mu=inp.rate - q)
    S_T = inp.spot * np.exp(o["log_end"])
    nav_t = nav(params, v1, v2, inp.spot, horizon, inp.rate, q, inp.strike, inp.options_per_share,
                inp.cash_per_share, 1.0)
    fee = math.exp(-fc.FEE * t_T)
    if horizon - t_T <= 0.0:                            # the last origin: T is the expiry, XX pays its option
        nav_T = fee * inp.options_per_share * np.maximum(S_T - inp.strike, 0.0) + inp.cash_per_share
    else:
        table = pricetable.build_two(model_at(params, 0.0, 0.0), horizon - t_T)
        nav_T = nav(params, o["v_end"][:, 0], o["v_end"][:, 1], S_T, horizon - t_T, inp.rate, q, inp.strike,
                    inp.options_per_share, inp.cash_per_share, fee, table=table)
    return nav_T / nav_t, S_T / inp.spot, (v1, v2)


def model_record(inp, T, index_vols, params, npaths=fc.PATHS):
    """M5's entry in the saved forecast: XX's and SPY's growth at fc.LEVELS, the state, the indices and parameters."""
    g, s, (v1, v2) = simulate_growth(inp, T, index_vols, params, npaths)
    q12 = lambda x: [float(f"{v:.12g}") for v in np.quantile(x, fc.LEVELS)]  # noqa: E731
    return {"growth_levels": q12(g), "spy_levels": q12(s), "state": [v1, v2],
            "indices": {n: index_vols[n] for n in INDICES}, "params": dict(params)}


def spy_pit(m, spy_t, spy_T):
    """M5's SPY PIT: the share of the saved SPY growth below the one observed (FC1 rule 6's interpolation)."""
    return fc.pit_from_levels(m["spy_levels"], spy_T / spy_t)
