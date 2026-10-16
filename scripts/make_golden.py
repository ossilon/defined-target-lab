#!/usr/bin/env python3
"""Write web/golden.json: answers from the Python engine that web/selftest.html checks web/engine.js against.

    py -I scripts\\make_golden.py

Every case names a function in engine.js, its arguments and the Python answer. The inputs are synthetic or the
README's dated example (no market data series). tests/test_web_golden.py fails if this file goes stale.
"""
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from definedtarget import pricing, sim  # noqa: E402

OUT = REPO / "web" / "golden.json"
TOL = {"rel": 1e-9, "abs": 1e-12}         # default: |js - python| <= abs + rel x |python|
TIGHT = {"rel": 1e-12, "abs": 1e-300}

# The README's 2026-10-13 example (continuous-yield form) and a few synthetic funds
EX = {"S0": 778.01, "nav": 26.51, "rate": 0.0486498, "T": 3.2465753}
EX["q"] = EX["rate"] - math.log(882.2603 / EX["S0"]) / EX["T"]
XX = {"legs": [{"kind": "call", "sign": 1, "strike": 995.01, "units": 0.48}], "cash": 0.2474, "fee": 0.0099}
SPREAD = {"legs": [{"kind": "call", "sign": 1, "strike": 900.0, "units": 1.0},
                   {"kind": "call", "sign": -1, "strike": 1100.0, "units": 1.0}], "cash": 0.5, "fee": 0.005}
STRANGLE = {"legs": [{"kind": "put", "sign": 1, "strike": 700.0, "units": 1.0},
                     {"kind": "call", "sign": 1, "strike": 900.0, "units": 1.0}], "cash": 0.0, "fee": 0.0}
LISTED = {"legs": [{"kind": "call", "sign": 1, "strike": 10000.0, "units": 100.0}], "cash": 0.0, "fee": 0.0}
MKT = {"rate": EX["rate"], "divYield": EX["q"], "horizon": EX["T"]}
MKT_SPX = {"rate": 0.0480, "divYield": 0.0115, "horizon": 3.19}
MKT_ZERO = {"rate": 0.0, "divYield": 0.0, "horizon": 1.0}


def fund_py(f):
    return sim.Fund(tuple(sim.Leg(L["kind"], L["sign"], L["strike"], L["units"]) for L in f["legs"]),
                    cash=f["cash"], fee=f["fee"])


def mkt_py(m):
    return sim.Market(m["rate"], m["divYield"], m["horizon"])


def enc(x):
    """JSON cannot hold NaN or infinity: write them as strings that selftest.html turns back into numbers."""
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    return x


def build():
    cases = []

    def add(fn, args, expect, tol=TOL, note=""):
        cases.append({"fn": fn, "args": enc(args), "expect": enc(expect), "tol": tol, "note": note})

    # Normal distribution
    for x in (-8.0, -6.0, -3.0, -1.5, -1.4999, -0.5, 0.0, 0.3, 1.0, 1.4999, 1.5, 2.5, 4.0, 6.0, 10.0, 27.0):
        add("erfc", [x], math.erfc(x), TIGHT)
    # pricing.norm_cdf's erfc form; NormalDist().cdf uses 1 + erf(x) and loses ~7 digits deep in the lower tail
    for x in (-5.0, -1.0, 0.0, 0.7, 3.0):
        add("normCdf", [x], 0.5 * math.erfc(-x / math.sqrt(2.0)), TIGHT)
    for p in (1e-10, 1e-4, 0.01, 0.02425, 0.1, 0.5, 0.9, 0.97575, 0.999, 1 - 1e-10):
        add("normInv", [p], NormalDist().inv_cdf(p), {"rel": 1e-12, "abs": 1e-14})

    # Black-76: prices, Greeks, implied vol round trips (and refusals)
    for F in (500.0, 880.0, 995.0, 1200.0, 2000.0):
        for tau in (1e-4, 0.25, 3.25, 10.0):
            for sigma in (0.01, 0.166, 0.6, 2.0):
                DF = math.exp(-0.048 * tau)
                c = pricing.call_price(F, 995.01, DF, tau, sigma)
                add("callPrice", [F, 995.01, DF, tau, sigma], c)
                if sigma == 0.166:
                    add("putPrice", [F, 995.01, DF, tau, sigma], pricing.put_price(F, 995.01, DF, tau, sigma))
                    add("callGreeks", [F, 995.01, DF, tau, sigma], list(pricing.call_greeks(F, 995.01, DF, tau, sigma)))
                iv = pricing.implied_vol(c, F, 995.01, DF, tau)
                tv = c - DF * max(F - 995.01, 0.0)
                if tv > 1e-9 * c and tv > 1e-10:      # vol is only identifiable when time value is not lost to rounding
                    add("impliedVol", [c, F, 995.01, DF, tau], iv, {"rel": 1e-8, "abs": 1e-10}, "round trip")
    add("callPrice", [900.0, 995.01, 0.9, 0.0, 0.2], pricing.call_price(900.0, 995.01, 0.9, 0.0, 0.2), note="expiry")
    add("callPrice", [1100.0, 995.01, 0.9, 1.0, 0.0], pricing.call_price(1100.0, 995.01, 0.9, 1.0, 0.0), note="zero vol")
    add("impliedVol", [1.0, 1100.0, 995.01, 0.9, 1.0], float("nan"), note="price below intrinsic: refused")
    add("impliedVol", [999.0, 1100.0, 995.01, 0.9, 1.0], float("nan"), note="price above DF x F: refused")
    add("impliedVol", [50.0, 900.0, 995.01, 0.9, 0.0], float("nan"), note="no time left: refused")

    # The simulator model
    funds = {"XX": XX, "SPREAD": SPREAD, "STRANGLE": STRANGLE}
    for name, f in funds.items():
        fp, mp = fund_py(f), mkt_py(MKT)
        for S in (400.0, 700.0, 778.01, 900.0, 995.01, 1100.0, 1500.0):
            for t in (0.0, 0.5, 2.0, 3.2465753 - 1 / 365, 3.2465753):
                for vol in (0.12, 0.166, 0.35):
                    add("shareValue", [f, S, t, vol, MKT], sim.share_value(fp, S, t, vol, mp), note=name)
                if t < 3.0:
                    add("shareDelta", [f, S, t, 0.166, MKT], sim.share_delta(fp, S, t, 0.166, mp), note=name)
                    add("leverage", [f, S, t, 0.166, MKT], sim.leverage(fp, S, t, 0.166, mp), note=name)
        add("oneDay", [f, 778.01, 0.166, MKT], list(sim.one_day(fp, 778.01, 0.166, mp)), note=name)
        for mu in (None, 0.04, 0.10, 0.12):
            for vol in (0.166, 0.25):
                add("outcomeRow", [f, 26.51 if name == "XX" else 60.0, 778.01, vol, MKT, mu],
                    sim.outcome_row(fp, 26.51 if name == "XX" else 60.0, 778.01, vol, mp, mu), note=name)
        add("pWorthless", [f, 880.0, 0.166, 3.2465753], sim.p_worthless(fp, 880.0, 0.166, 3.2465753), note=name)
    fp, mp = fund_py(XX), mkt_py(MKT)
    for nav in (5.0, 20.0, 26.51, 40.0, 0.2):
        add("solveVol", [XX, nav, 778.01, MKT], sim.solve_vol(fp, nav, 778.01, mp), {"rel": 1e-9, "abs": 1e-11},
            "NaN when the price is below cash" if nav < 0.25 else "")
    short = {"legs": [{"kind": "call", "sign": -1, "strike": 995.01, "units": 0.48}], "cash": 100.0, "fee": 0.0}
    add("solveVol", [short, 90.0, 778.01, MKT], float("nan"), note="short volatility: type a vol in")
    for target in (26.51, 53.02, 150.0):
        add("levelForValue", [XX, target, MKT], sim.level_for_value(fp, target, mp), note="XX")
    sp, sm = fund_py(SPREAD), mkt_py(MKT)
    for target in (50.0, 150.0, 250.0):
        add("levelForValue", [SPREAD, target, MKT], sim.level_for_value(sp, target, sm), note="spread (250: NaN)")
    lp, lm = fund_py(LISTED), mkt_py(MKT_SPX)
    for S in (6000.0, 7805.45, 10000.0, 12000.0):
        add("shareValue", [LISTED, S, 0.0, 0.17, MKT_SPX], sim.share_value(lp, S, 0.0, 0.17, lm), note="listed call")
    add("outcomeRow", [LISTED, 55000.0, 7805.45, 0.17, MKT_SPX, 0.10],
        sim.outcome_row(lp, 55000.0, 7805.45, 0.17, lm, 0.10), note="listed call")
    zp, zm = fund_py(XX), mkt_py(MKT_ZERO)
    add("shareValue", [XX, 1000.0, 0.3, 0.2, MKT_ZERO], sim.share_value(zp, 1000.0, 0.3, 0.2, zm), note="zero rate")
    for args in ((0.166, 0.0, 0.015, 700.209, 778.01), (0.166, 0.03, 0.015, 900.0, 778.01),
                 (0.05, -0.08, 0.015, 2000.0, 778.01), (0.166, 0.0, 0.0, 500.0, 778.01)):
        add("volAt", list(args), sim.vol_at(*args))
    for mu in (None, 0.0, 0.08, 0.12):
        add("forwardAtTarget", [778.01, MKT, mu], sim.forward_at_target(778.01, mp, mu))
    for lvl in (0.0, 700.0, 995.01, 1300.0):
        add("probAbove", [lvl, 880.0, 0.166, 3.2465753], sim.prob_above(lvl, 880.0, 0.166, 3.2465753))
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        add("quantile", [880.0, 0.166, 3.2465753, p], sim.quantile(880.0, 0.166, 3.2465753, p))
    add("requiredGrowth", [7805.45, 9995.0, 3.2465753], sim.required_growth(7805.45, 9995.0, 3.2465753))
    add("ratioAt", [1.5, 10.03, 10.045, 3.0], sim.ratio_at(1.5, 10.03, 10.045, 3.0))
    return {"about": "Answers from the Python engine (definedtarget/sim.py, pricing.py). Generated by "
                     "scripts/make_golden.py; checked by web/selftest.html. Synthetic inputs and the README's "
                     "dated example only.",
            "cases": cases}


def dumps(doc):
    head = json.dumps(doc["about"])
    lines = ",\n".join(json.dumps(c, sort_keys=True) for c in doc["cases"])
    return f'{{"about": {head},\n"cases": [\n{lines}\n]}}\n'


if __name__ == "__main__":
    doc = build()
    text = dumps(doc)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO)}: {len(doc['cases'])} cases, {len(text.encode()):,} bytes")
