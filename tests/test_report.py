"""Daily report: the attribution adds up, and the page builds from synthetic tables (no real data)."""
from datetime import date

import pandas as pd

from _harness import check

from definedtarget import curves, report, spec, valuation

FLAT = {t: 0.045 for t in (1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)}
HIST = [(date(2025, 12, 19), 1.9), (date(2026, 3, 20), 1.9), (date(2026, 6, 18), 1.9), (date(2026, 9, 18), 1.9)]
K, EXP = 995.01, date(2030, 1, 10)


def model_row(as_of, spot, sigma, ex_div_today=0.0):
    cv = curves.Curve(as_of, FLAT)
    inp = valuation.Inputs(as_of, spot, K, EXP, cv, HIST)
    v = valuation.value(inp, sigma)
    return {"as_of": as_of, "spot": spot, "spx": spot * 10.04, "mark": v["price"], "iv_disc_0bp": sigma,
            "nav": 0.5 * v["price"] + 0.25, "ex_div_today": ex_div_today,
            "xx_theta_decay_pct_nav_per_day": -0.0007, "xx_theta_fwd_pct_nav_per_day": -0.0005,
            "iv_disc_25bp": sigma - 0.004, "iv_disc_50bp": sigma - 0.008, "iv_disc_0bp_g0": sigma - 0.0015,
            "iv_disc_0bp_g10": sigma + 0.0015, "options_per_share": 0.5, "forward": v["forward"], "tau": v["tau"],
            "delta": v["delta"], "gamma": v["gamma"], "vega_pt": v["vega_pt"], "theta_1d": v["theta_1d"],
            "rho_1pct": v["rho_1pct"], "zero_rate_T": cv.zero(v["tau"]), "prob_itm_rn": v["prob_itm"],
            "xx_elasticity": 5.0, "xx_vega_pct_nav_per_pt": 0.1, "xx_rho_pct_nav_per_1pct": 0.15,
            "xx_theta_pct_nav_per_day": -0.0013, "spot_source": "synthetic", "curve_source": "synthetic flat 4.5%"}


m0, m1 = model_row(date(2026, 10, 12), 780.0, 0.165), model_row(date(2026, 10, 13), 785.0, 0.163)
total, parts, days = report.attribution(m0, m1)
check("attribution parts add up exactly to the mark change", abs(sum(parts.values()) - total) < 1e-12)
check("attribution covers one calendar day", days == 1)
check("a rise in SPY shows up as a positive delta term", parts["market (delta)"] > 0)
check("a fall in implied vol shows up as a negative vega term", parts["volatility (vega)"] < 0)
check("with spot, vol and rates explained, the residual is small (< 5% of the move)",
      abs(parts["residual"]) < 0.05 * abs(total), parts)
check("no dividend line on an ordinary day", "SPY dividend (ex-date)" not in parts)

# Ex-dividend day (audit B10a): SPY opens lower by the dividend D but the option never had a claim on it. With
# the market otherwise flat, the price drop is no loss to the option: the dividend line must offset the delta line.
D = 1.95
e0 = model_row(date(2026, 12, 17), 780.0, 0.165)
e1 = model_row(date(2026, 12, 18), 780.0 - D, 0.165, ex_div_today=D)
et, ep, _ = report.attribution(e0, e1)
check("ex-date: the dividend line adds back delta x D", abs(ep["SPY dividend (ex-date)"] - e0["delta"] * D) < 1e-12)
check("ex-date with a flat market: delta + dividend nearly cancel, and the residual stays small",
      abs(ep["market (delta)"] + ep["SPY dividend (ex-date)"]) < 0.01 * abs(ep["market (delta)"])
      and abs(ep["residual"]) < 0.2 * abs(ep["market (delta)"]), ep)
check("ex-date attribution still adds up exactly", abs(sum(ep.values()) - et) < 1e-12)

state = pd.DataFrame([
    {"as_of": d, "nav_published": 0.5 * m["mark"] + 0.25, "market_price": 0.5 * m["mark"] + 0.26,
     "premium": 0.001, "cash_per_share": 0.25, "cash_weight": 0.01, "shares_outstanding": 40000.0,
     "shares_change": 0.0, "flow_day": False}
    for d, m in ((m0["as_of"], m0), (m1["as_of"], m1))])
page = report.build(state, pd.DataFrame([m0, m1]), spec.load("XX"))
check("report names the close date", "close 2026-10-13" in page)
check("report has the odds table with the risk-neutral row and five growth rows, labeled as means",
      "risk-neutral (option price)" in page and page.count("| mean total return") == 5
      and "expected total return" not in page)
check("odds table shows the median path's growth", "Median path: SPY price growth/yr" in page)
check("theta line splits time decay and carry, and shows the along-the-forward figure",
      "time decay -0.07%" in page and "drifts along its forward price | -0.05%" in page)
check("report has the attribution section, in % of NAV with a total row",
      "What moved the price since 2026-10-12" in page and "| Cause | Per XX share | % of NAV |" in page
      and "**total**" in page and "Share of the move" not in page)
check("report carries the not-investment-advice line", "not investment advice" in page)
html = report.to_html(page, "test")
check("HTML version has tables and a dark-mode style", "<table>" in html and "prefers-color-scheme:dark" in html)
check("HTML escapes text (no raw '<' from content)", "<script" not in html)
