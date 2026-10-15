#!/usr/bin/env python3
"""Exploratory (in-sample) statistics we used to design the pre-registered hypotheses.  From the repository folder:

    py -I scripts\\exploratory_stats.py

These numbers come from data that existed BEFORE we froze the hypotheses. They may shape a hypothesis's
wording and thresholds; we may not TEST any hypothesis on them (every test window starts after the freeze).
Prints aggregates only.
"""
import io
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from datetime import date  # noqa: E402

from definedtarget import curves  # noqa: E402
from definedtarget.forward import fetch, parse  # noqa: E402

EXPIRY = date(2030, 1, 10)            # XX's option
LISTED_EXPIRY = date(2029, 12, 21)    # listed SPX Dec-2029 calls (AM-settled)


def ols(x, y, through_origin=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if through_origin:
        b = float((x * y).sum() / (x * x).sum())
        resid = y - b * x
        r2 = 1 - (resid ** 2).sum() / (y ** 2).sum()
        return b, float(r2)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(b), float(r2)


def main():
    root = fetch.data_dir()
    m = pd.read_csv(root / "tables" / "xx_model_daily.csv", parse_dates=["as_of"])
    s = pd.read_csv(root / "tables" / "xx_state.csv", parse_dates=["as_of"])
    d = m.merge(s[["as_of", "flow_day", "day_after_flow", "cash_weight", "premium", "market_price"]], on="as_of",
                suffixes=("", "_s"))
    d["r_spy"] = d["spot"].pct_change()
    d["r_nav"] = d["nav"].pct_change()
    d["d_iv_pts"] = d["iv_disc_0bp"].diff() * 100
    d["d_iv_cont_pts"] = d["iv_cont_0bp"].diff() * 100
    rec = parse.saved_versions(root, "VIX1Y_History.csv")[-1]
    v1y = pd.read_csv(io.StringIO(parse.read_text(root, rec)))
    v1y["as_of"] = pd.to_datetime(v1y["DATE"], format="%m/%d/%Y")
    d = d.merge(v1y[["as_of", "CLOSE"]].rename(columns={"CLOSE": "vix1y"}), on="as_of", how="left")
    d["d_vix1y"] = d["vix1y"].diff()
    d["iv_over_vix1y"] = d["iv_disc_0bp"] * 100 / d["vix1y"]
    clean = d[(~d["flow_day_s"]) & (~d["day_after_flow_s"])].dropna(subset=["r_spy"])
    allc = d.dropna(subset=["r_spy"])

    print(f"closes: {len(d)} ({d['as_of'].min().date()} to {d['as_of'].max().date()}); "
          f"daily changes: {len(allc)}, of which clean (no flow, not the day after a flow): {len(clean)}")
    print("\n[H8] realized beta of NAV % change on SPY % change (through origin):")
    for name, frame in (("all days", allc), ("clean days", clean)):
        b, r2 = ols(frame["r_spy"], frame["r_nav"], through_origin=True)
        print(f"  {name:<10} beta {b:5.2f}  R2 {r2:4.2f}  vs mean model elasticity "
              f"{frame['xx_elasticity'].mean():.2f}  -> ratio {b / frame['xx_elasticity'].mean():.2f}")
    print("\n[spot-vol] implied-vol change (points) per 1% SPY move:")
    for name, frame in (("all days", allc), ("clean days", clean)):
        b, r2 = ols(frame["r_spy"] * 100, frame["d_iv_pts"])
        print(f"  {name:<10} slope {b:+.3f} pts per 1%  R2 {r2:4.2f}")
    print("\n[H4] implied-vol change vs VIX1Y change (clean days):")
    b, r2 = ols(clean["d_vix1y"], clean["d_iv_pts"])
    print(f"  slope {b:+.3f}  R2 {r2:4.2f}  (n={len(clean)})")
    print(f"  daily implied-vol change sd: all {allc['d_iv_pts'].std():.3f} pts, clean {clean['d_iv_pts'].std():.3f} pts")
    print(f"\n[proxy] implied vol / VIX1Y: min {d['iv_over_vix1y'].min():.3f}  max {d['iv_over_vix1y'].max():.3f}  "
          f"mean {d['iv_over_vix1y'].mean():.3f}")
    print(f"\n[H3] cash weight: min {d['cash_weight'].min():.2%}  max {d['cash_weight'].max():.2%}")
    one_contract = (100 * d["mark"] / (d["nav"] * d["options_per_share"] / 0.0048 * 100)).iloc[-1]
    aum = (d["nav"] * 60000).iloc[-1]
    print(f"[H2] one contract = {100 * d['mark'].iloc[-1]:,.0f} USD = {100 * d['mark'].iloc[-1] / aum:.2%} of AUM at "
          f"the last close; a 0.99% yearly fee = {0.0099 * aum / (100 * d['mark'].iloc[-1]):.1f} contracts a year")
    p = d["premium"].abs()
    print(f"\n[H6] |premium|: median {p.median():.2%}; within 1% on {(p < 0.01).mean():.0%} of closes; "
          f"max {p.max():.2%}")
    print(f"[rates] vendor box rate - Treasury same term: "
          f"{((d['vendor_box_rate'] - d['treasury_rate_box_term']) * 1e4).min():.1f} to "
          f"{((d['vendor_box_rate'] - d['treasury_rate_box_term']) * 1e4).max():.1f} bp")

    # [H5 power] deep in-the-money vendor marks: day-to-day change minus SPY's change
    vm = parse.load_holdings(root, accounts=None, all_versions=False)
    deep = vm[vm["is_option"] & (vm["opt_underlying"] == "SPY") & (vm["opt_cp"] == "C") & (vm["opt_strike"] < 100)]
    spot = d.set_index(d["as_of"].dt.date)["spot"]
    print("\n[H5 power] deep in-the-money SPY call marks (vendor), daily change minus SPY change:")
    for (acct, k, exp), g in deep.groupby(["account", "opt_strike", "opt_expiry"]):
        g = g.sort_values("as_of").drop_duplicates("as_of")
        g = g[g["as_of"].isin(spot.index)]
        dc = g["price"].diff().to_numpy()[1:]
        ds = spot.loc[g["as_of"]].diff().to_numpy()[1:]
        res = dc - ds
        print(f"  {acct} {k:g}C {exp}: n={len(res)}  residual mean {res.mean():+.3f}  sd {res.std(ddof=1):.3f} USD "
              f"(SPY's next quarterly dividend is about 1.9 USD)")

    lottery_checks(root, d, allc, clean)
    listed_checks(root, d)
    return 0


def lottery_checks(root, d, allc, clean):
    """Exploratory numbers behind H14, H16 and H18 (added 2026-10-14, before the freeze)."""
    from definedtarget import marks, valuation
    d = d.copy()
    d["E_prev"] = d["xx_elasticity"].shift()                  # leverage known at the previous close
    d["vega_prev"] = d["xx_vega_pct_nav_per_pt"].shift()
    d["r_mkt"] = d["market_price"].pct_change()
    d["d_rate_bp"] = d["zero_rate_T"].diff() * 1e4
    allc = d.dropna(subset=["r_spy"])
    clean = allc[(~allc["flow_day_s"]) & (~allc["day_after_flow_s"])]

    print("\n[H8/H14] does the volatility slide explain the extra daily leverage, and does it last?")
    for name, f in (("all days", allc), ("clean days", clean)):
        beta, _ = ols(f["r_spy"], f["r_nav"], through_origin=True)
        slide, _ = ols(f["r_spy"] * 100, f["d_iv_pts"])
        E, vega = f["E_prev"].mean(), f["vega_prev"].mean()
        print(f"  {name:<10} measured {beta / E:.2f} x elasticity; slide {slide:+.3f} pts per 1% "
              f"-> 1 + vega x slide / elasticity = {1 + vega * 100 * slide / E:.2f}")
    a, b = d.iloc[0], d.iloc[-1]
    move = b["spot"] / a["spot"] - 1
    slide_all, _ = ols(allc["r_spy"] * 100, allc["d_iv_pts"])
    print(f"  {a['as_of'].date()} -> {b['as_of'].date()}: SPY {move:+.2%}, implied vol "
          f"{100 * (b['iv_disc_0bp'] - a['iv_disc_0bp']):+.2f} pts (a lasting slide would add "
          f"{slide_all * 100 * move:+.2f}); noise over {len(allc)} independent daily changes would be about "
          f"{allc['d_iv_pts'].std() * math.sqrt(len(allc)):.2f} pts, so one span cannot separate the two")

    # The frozen H4/H10 test is ONE regression with both regressors; the bands above came from one-regressor slopes.
    f = allc.dropna(subset=["d_vix1y"])
    X = np.column_stack([np.ones(len(f)), f["r_spy"] * 100, f["d_vix1y"]])
    coef, *_ = np.linalg.lstsq(X, f["d_iv_pts"].to_numpy(), rcond=None)
    print(f"\n[H4/H10] the frozen two-regressor test on all {len(f)} changes (indicative only): SPY coefficient "
          f"{coef[1]:+.3f} pts per 1%, VIX1Y coefficient {coef[2]:+.2f}")

    print("\n[H16] leverage of the exchange price (official closing price) against NAV:")
    for name, f in (("all days", allc), ("clean days", clean)):
        b_nav, _ = ols(f["r_spy"], f["r_nav"], through_origin=True)
        b_mkt, _ = ols(f["r_spy"], f["r_mkt"], through_origin=True)
        print(f"  {name:<10} NAV beta {b_nav:.2f}, price beta {b_mkt:.2f}, ratio {b_mkt / b_nav:.2f}")

    print("\n[H18] which rate does the vendor use? Implied vol re-solved under three rate conventions:")
    st = d.set_index(d["as_of"].dt.date)
    cbd, divs = curves.load_treasury(root), marks.load_dividends(root)
    days = list(st.index)
    rho_bp, vega_last = b["xx_rho_pct_nav_per_1pct"] / 100, b["xx_vega_pct_nav_per_pt"]
    print(f"  a vendor rate that never moves shows {-rho_bp / vega_last:+.4f} pts per bp; daily rate change sd "
          f"{allc['d_rate_bp'].std():.1f} bp")
    for name, cdays in (("same-day curve", days), ("previous-day curve", [days[0]] + days[:-1]),
                        ("fixed first-day curve", [days[0]] * len(days))):
        ivs = []
        for day, cday in zip(days, cdays):
            r = st.loc[day]
            cv = curves.curve_on(cbd, cday, spread=0.0)
            inp = valuation.Inputs(day, r["spot"], 995.01, EXPIRY, cv, divs)
            ivs.append(valuation.implied_vol(inp, r["mark"]) * 100)
        div = np.diff(ivs)
        X = np.column_stack([np.ones(len(div)), allc["r_spy"] * 100, allc["d_rate_bp"]])
        coef, *_ = np.linalg.lstsq(X, div, rcond=None)
        print(f"  {name:<22} implied-vol change sd {div.std(ddof=1):.3f} pts; rate coefficient "
              f"{coef[2]:+.4f} pts per bp (all 9 changes; indicative only)")


def listed_checks(root, d):
    """H13, H17, H19: the listed S&P 500 Dec-2029 calls from a local quote snapshot (TradingView, display-only)."""
    from definedtarget import analytics as an, marks, pricing, valuation
    files = sorted((root / "external").glob("listed_spx_dec2029_*.csv"))
    if not files:
        print("\n[listed] no quote snapshot in data/external; skipped")
        return
    q = pd.read_csv(files[-1])
    close = pd.Timestamp(q["close_date"].iloc[0])
    row = d[d["as_of"] == close].iloc[0]
    xx_iv = row["iv_disc_0bp"]
    spx, spy = row["spx"], row["spot"]
    ratio = spx / spy
    cv = curves.curve_on(curves.load_treasury(root), close.date(), spread=0.0)
    inp = valuation.Inputs(close.date(), spy, 995.01, LISTED_EXPIRY, cv, marks.load_dividends(root))
    F, DF, tau, _ = valuation.forward(inp)
    bid, ask = dict(zip(q["strike"], q["bid"])), dict(zip(q["strike"], q["ask"]))
    mid = {k: (bid[k] + ask[k]) / 2 for k in bid}
    iv = {k: [pricing.implied_vol(x / ratio, F, k / ratio, DF, tau) for x in (bid[k], mid[k], ask[k])] for k in bid}
    print(f"\n[listed] {files[-1].name}: S&P {spx:.2f}, S&P/SPY {ratio:.4f}, discount factor to "
          f"{LISTED_EXPIRY} {DF:.4f}")
    b, m_, a = iv[10000]
    print(f"[H13] 10,000 call implied vol bid / mid / ask {b:.2%} / {m_:.2%} / {a:.2%}; XX {xx_iv:.2%}")
    sl = (iv[10200][1] - iv[9800][1]) / math.log(10200 / 9800)
    print(f"[H17] listed skew 9,800-10,200: {sl:+.3f} vol per unit log-strike")
    flat = an.lognormal_prob_above(F * ratio, 10000, xx_iv, tau)
    flat_mid = an.lognormal_prob_above(F * ratio, 10000, m_, tau)
    for k1, k2 in ((9600, 10400), (9000, 11000)):
        p = an.call_spread_prob(mid[k1], mid[k2], k1, k2, DF)
        lo, hi = an.call_spread_prob(bid[k1], ask[k2], k1, k2, DF), an.call_spread_prob(ask[k1], bid[k2], k1, k2, DF)
        print(f"[H19] P(S&P above 10,000) from the {k1}-{k2} call spread {p:.1%} (bid/ask bounds {lo:.0%}-{hi:.0%})"
              f"; simple model {flat:.1%}; gap {100 * (p - flat):+.1f} points")
    print(f"[F10] the simple model at the listed call's own mid vol ({m_:.2%}) gives {flat_mid:.1%}, so the skew, not "
          f"the vol level, accounts for {100 * (an.call_spread_prob(mid[9600], mid[10400], 9600, 10400, DF) - flat_mid):.1f} "
          "points of the gap")

    # The lottery ladder: S&P level each XX multiple needs, and the odds two ways (analytics.prob_above_from_spreads)
    pairs = ((9600, 10400), (10600, 10800), (11000, 12000), (12000, 13000), (13000, 14000), (14000, 15000))
    centers = [(k1 + k2) / 2 for k1, k2 in pairs]
    p_mid = [an.call_spread_prob(mid[a_], mid[b_], a_, b_, DF) for a_, b_ in pairs]
    p_lo = [max(an.call_spread_prob(bid[a_], ask[b_], a_, b_, DF), 1e-4) for a_, b_ in pairs]
    p_hi = [an.call_spread_prob(ask[a_], bid[b_], a_, b_, DF) for a_, b_ in pairs]
    cash_ps = row["nav"] - row["options_per_share"] * row["mark"]
    print("[ladder] buyer at NAV; S&P levels use the central finish-line ratio 9,995 / 995.01")
    for label, mult in (("options pay", None), ("1x", 1.0), ("2x", 2.0), ("3x", 3.0), ("5x", 5.0), ("10x", 10.0)):
        lvl = 995.01 if mult is None else an.breakeven_spot(row["nav"], 995.01, row["options_per_share"], cash_ps,
                                                             0.0099, row["tau"], multiple=mult)
        lvl *= 9995 / 995.01
        lm, lo, hi = (an.prob_above_from_spreads(centers, arr, lvl)[0] for arr in (p_mid, p_lo, p_hi))
        beyond = lvl > centers[-1]                  # 9,995 sits 5 points below the first center: not flagged
        print(f"  {label:<12} S&P {lvl:>7,.0f}: simple model {an.lognormal_prob_above(F * ratio, lvl, xx_iv, tau):5.1%}"
              f", listed {lm:5.1%} (quotes {lo:.0%}-{hi:.0%}){' (extrapolated)' if beyond else ''}")


if __name__ == "__main__":
    sys.exit(main())
