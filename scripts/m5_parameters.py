#!/usr/bin/env python3
"""The two candidate parameter sets for the FC2 forecast model M5 (the v0.3.1 two-factor model, B2 form).

    py -I scripts\\m5_parameters.py <S&P 500 daily closes CSV>      From target_funds/.

(a) validated: step 5's B2 exactly as judged under the holdout (FINDINGS F24): the VIX family 2009-09-18 to 2019-12-31,
    the listed December 2029 trades 2024-06-14 to 2025-06-30.
(b) refit: the same code and settings on every day to date: the VIX family from 2009-09-18 to the last common day, the
    trades from 2024-06-14 to the last trade (5+ contracts, every strike).
Prints both and writes data/fc2_parameters.json in the private data folder: the fit uses licensed trade data, so the
file stays private; the chosen numbers become public only in the FC2 ledger entry. The user picks before the FC2
freeze. Research, not investment advice.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import vol_holdout_study as h4  # noqa: E402
import vol_twofactor_study as h5  # noqa: E402
from definedtarget import curves, listedvols, marks  # noqa: E402
from definedtarget.forward import fetch  # noqa: E402
from definedtarget.montecarlo import manifest  # noqa: E402
from definedtarget.volmodel import termstructure as ts  # noqa: E402


def fit(vols, rows, trade_window):
    sd = np.full(4, h4.SD_INDEX)
    f2 = min((ts.fit_two(h4.TENORS, vols, sd, start=s) for s in h5.STARTS), key=lambda f: f.cost)
    P = f2.params
    for r in rows:
        r["v1"], r["v2"] = (float(x) for x in ts.states_two(*P, h4.TENORS, r["iv_idx"], sd)[0])
    fit_rows = [r for r in rows if trade_window[0] <= r["date"] <= trade_window[1] and r["volume"] >= h4.MIN_CONTRACTS
                and np.isfinite(r["v1"])]
    fb2 = h5.fit_stage2(fit_rows, P[0], P[1], P[2], f2.theta2, ("v1", "v2"))
    m = fb2.model
    return {"term_structure": {"kappa1": P[0], "theta1": P[1], "kappa2": P[2], "c2": P[3], "days": int(vols.shape[0]),
                               "rmse_vol_points": [round(100 * float(x), 3) for x in f2.rmse], "converged": f2.converged},
            "smile": {"theta2": m.f2.theta, "pricing_drift": m.f2.kappa * m.f2.theta, "eta": m.f1.eta, "rho": m.f1.rho,
                      "trades": len(fit_rows), "days": len({r["date"] for r in fit_rows}),
                      "rmse_vol_points": round(100 * fb2.rmse, 3), "converged": fb2.converged,
                      "se": {k: v for k, v in fb2.se.items() if math.isfinite(v)}},
            "windows": {"indices": [None, None], "trades": [trade_window[0].isoformat(), trade_window[1].isoformat()]}}


def main(argv):
    idx = {n: h4.load_index(n) for n in h4.INDICES}
    days = sorted(set.intersection(*(set(idx[n]) for n in h4.INDICES)))
    vols = np.array([[idx[n][d] for n in h4.INDICES] for d in days])
    closes = listedvols.read_closes_csv(argv[0])
    rows = listedvols.trade_vols(listedvols.load_bars(), closes, curves.load_treasury(), marks.load_dividends(),
                                 min_volume=1)
    for r in rows:
        r["iv_idx"] = np.array([[np.nan if idx[n].get(r["date"]) is None else idx[n][r["date"]] for n in h4.INDICES]])
    rows = [r for r in rows if np.isfinite(r["iv"])]
    last_trade = max(r["date"] for r in rows)
    out = {}
    for label, d0, d1, tw in (("validated", h4.FIT_VIX[0], h4.FIT_VIX[1], h4.FIT_TRADES),
                              ("refit", h4.FIT_VIX[0], days[-1], (h4.FIT_TRADES[0], last_trade))):
        mask = np.array([d0 <= d <= d1 for d in days])
        out[label] = fit(vols[mask], rows, tw)
        out[label]["windows"]["indices"] = [d0.isoformat(), d1.isoformat()]
        print(f"{label}: " + json.dumps(out[label], indent=1))
    path = fetch.data_dir() / "fc2_parameters.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: data/{path.name} (private)")
    return 0


if __name__ == "__main__":
    with manifest.run("m5_parameters", sys.argv[1:], REPO / "results" / "manifests", REPO,
                      ("scripts/m5_parameters.py",), seeds=[]) as run:
        rc = main(sys.argv[1:])
    print(f"\nrun manifest: results/manifests/{run['path'].name}")
    sys.exit(rc)
