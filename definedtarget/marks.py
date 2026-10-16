"""What the fund's daily option marks imply: implied volatility, Greeks per XX share, and the vendor's rate.

For each close date:
- spot: Roundhill's own reference SPY close (daily_instrument_pricing.csv) when captured, else a close you add in
  data/external/closes.csv (or tv_closes.csv); see load_spots. Every row records the source (ASSUMPTIONS.md A5).
- curve: that day's US Treasury par curve (or the previous one on bond-market holidays), plus a financing spread.
- dividends: SPY history from data/external/spy_dividends.csv, projected forward (discrete) or as a trailing
  yield (continuous).
- implied vol: the code solves it for every spread x dividend case, so the sensitivity is always visible.
- vendor box rate: the discount rate implied by XBOX's SPY box spread in the same holdings file (a check on
  which rate the pricing vendor uses; short-dated, so it is evidence, not the rate for 2030).
"""
import csv
import io
from datetime import date, timedelta

import pandas as pd

from . import curves, valuation
from . import dividends as dv
from .forward import fetch, parse

SPREADS = (0.0, 0.0025, 0.0050)
GROWTH_SENSITIVITY = (0.0, 0.10)
BASE_SPREAD, BASE_MODE = 0.0, "discrete"


def load_dividends(root=None):
    path = fetch.data_dir(root) / "external" / "spy_dividends.csv"
    if not path.exists():
        raise FileNotFoundError("data/external/spy_dividends.csv missing - run scripts/import_spy_dividends.py")
    with open(path, encoding="utf-8") as fh:
        return sorted((date.fromisoformat(r["ex_date"]), float(r["amount"])) for r in csv.DictReader(fh))


USER_CLOSE_FILES = (("tv_closes.csv", "TradingView close"), ("closes.csv", "user-supplied close"))


def load_spots(root=None):
    """{date: (spy_close, spx_close, source)}.

    Order of preference (a later source overwrites an earlier one for the same date):
    1. data/external/tv_closes.csv, then data/external/closes.csv: closes you add yourself, one row per date and
       symbol (columns date, symbol, close; symbol SPY or SPX, with or without an exchange prefix such as AMEX:).
       Use official closing prices: other feeds can differ (one feed differed by 0.9% on a volatile day).
    2. Roundhill's own reference closes (daily_instrument_pricing.csv), captured daily; the file shows only the
       latest day, so it covers only days on which the capture ran.
    """
    out = {}
    for name, label in USER_CLOSE_FILES:
        path = fetch.data_dir(root) / "external" / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["sym"] = df["symbol"].astype(str).str.split(":").str[-1].str.upper()
        for d, g in df.groupby("date"):
            vals = dict(zip(g["sym"], g["close"]))
            if "SPY" in vals:
                out[date.fromisoformat(str(d))] = (float(vals["SPY"]), float(vals.get("SPX", "nan")),
                                                   f"{label} ({name})")
    for rec in parse.saved_versions(root, "daily_instrument_pricing.csv"):
        df = pd.read_csv(io.StringIO(parse.read_text(root, rec)), dtype=str)
        for d, g in df.groupby("Rate Date"):
            vals = dict(zip(g["Fund Ticker"].str.strip(), g["Daily Value"].astype(float)))
            if "SPY" in vals:
                day = pd.to_datetime(d, format="%m/%d/%Y").date()
                out[day] = (vals["SPY"], vals.get("SPX", float("nan")), "Roundhill reference close")
    return out


def box_rates(vendor_marks):
    """{as_of: (rate, expiry, fund, leg_decimals, flow_day)} from a 4-leg SPY box (two strikes, calls and puts) in
    any one fund.

    The box value accrues to the file's `Date` column (the NEXT session), not to the close date in the file name:
    from Thursday to Friday it grows by about three days' interest (audit A1, 2026-10-14). So the code solves the
    rate from the effective date. Using the close date instead gives false dips of about 11 bp on Fridays.

    leg_decimals: the most decimals among the four leg prices. 2 decimals on every leg looks like traded prices
    (a creation day), not the pricing vendor's 4-decimal model marks. flow_day: the fund's shares outstanding
    changed since its previous file. Both flag days whose box rate is weaker evidence of the vendor's rate
    (audit B10a: XBOX's 10/15 rate came from a creation day with 2-decimal legs).
    """
    shares = {}
    if "shares_outstanding" in vendor_marks.columns:
        for (as_of, acct), so in vendor_marks.groupby(["as_of", "account"])["shares_outstanding"].first().items():
            shares.setdefault(acct, []).append((as_of, so))
    prev_shares = {}
    for acct, seq in shares.items():
        seq.sort()
        for (d0, s0), (d1, s1) in zip(seq, seq[1:]):
            prev_shares[(d1, acct)] = (s0, s1)
    out = {}
    for (as_of, acct), g in vendor_marks.groupby(["as_of", "account"]):
        if len(g) != 4 or set(g["opt_cp"]) != {"C", "P"} or g["opt_strike"].nunique() != 2:
            continue
        signs = g["shares"].apply(lambda s: 1.0 if s > 0 else -1.0)
        value = float((signs * g["price"]).sum())
        k_lo, k_hi = sorted(g["opt_strike"].unique())
        expiry = g["opt_expiry"].iloc[0]
        valued_on = g["effective_date"].iloc[0]
        decimals = int(g["price_decimals"].max()) if "price_decimals" in g.columns else -1
        pair = prev_shares.get((as_of, acct))
        flow = bool(pair is not None and pair[0] != pair[1])
        out[as_of] = (curves.box_implied_rate(k_lo, k_hi, value, valued_on, expiry), expiry, acct, decimals, flow)
    return out


def ex_dividend_on(dividends, day, growth=None):
    """SPY dividend going ex on `day`: the known amount if the history has it, else the projection; 0.0 if none."""
    for d, a in dividends:
        if d == day:
            return a
    kw = {} if growth is None else {"growth": growth}
    return sum(x.amount for x in dv.project(dividends, day - timedelta(days=1), day, **kw))


def model_daily(state, curves_by_date, dividends, spots, vendor_marks=None, skip_missing_spot=False):
    """One row per close: inputs, implied vols for every case, base-case Greeks per option and per XX share.

    A close with no SPY price raises an error, or, with skip_missing_spot=True, gets no row; the skipped dates are
    listed in the result's .attrs["skipped_no_spot"], so a caller can report them (never a silent gap; audit A1).
    """
    boxes = box_rates(vendor_marks) if vendor_marks is not None and len(vendor_marks) else {}
    rows, skipped = [], []
    for r in state.itertuples():
        if r.as_of not in spots:
            if skip_missing_spot:
                skipped.append(r.as_of)
                continue
            raise ValueError(f"no SPY spot price for {r.as_of}: add the close (Roundhill reference file or "
                             "data/external/closes.csv) before pricing")
        spy, spx, spot_src = spots[r.as_of]
        row = {"as_of": r.as_of, "spot": spy, "spx": spx, "spot_source": spot_src, "mark": r.mark,
               "flow_day": r.flow_day, "day_after_flow": r.day_after_flow,
               "options_per_share": r.contracts_per_share * 100.0, "nav": r.nav_published}
        for mode in ("discrete", "continuous"):
            for s in SPREADS:
                cv = curves.curve_on(curves_by_date, r.as_of, spread=s)
                inp = valuation.Inputs(r.as_of, spy, r.strike, r.expiry, cv, dividends, mode=mode)
                row[f"iv_{mode[:4]}_{int(round(s * 1e4))}bp"] = valuation.implied_vol(inp, r.mark)
        for g in GROWTH_SENSITIVITY:                               # ASSUMPTIONS A3: dividend growth 0% / 10%
            cv = curves.curve_on(curves_by_date, r.as_of, spread=BASE_SPREAD)
            inp = valuation.Inputs(r.as_of, spy, r.strike, r.expiry, cv, dividends, mode="discrete", growth=g)
            row[f"iv_disc_0bp_g{int(round(g * 100))}"] = valuation.implied_vol(inp, r.mark)
        cv = curves.curve_on(curves_by_date, r.as_of, spread=BASE_SPREAD)
        row["curve_source"] = cv.source
        row["zero_rate_T"] = cv.zero((r.expiry - r.as_of).days / 365.0)
        inp = valuation.Inputs(r.as_of, spy, r.strike, r.expiry, cv, dividends, mode=BASE_MODE)
        sigma = row[f"iv_{BASE_MODE[:4]}_0bp"]
        if sigma != sigma:                                           # NaN: mark outside no-arbitrage bounds
            raise ValueError(f"{r.as_of}: mark {r.mark} has no implied vol (outside no-arbitrage bounds)")
        v = valuation.value(inp, sigma)
        ops = row["options_per_share"]
        row.update({
            "forward": v["forward"], "df": v["df"], "tau": v["tau"],
            "delta": v["delta"], "gamma": v["gamma"], "vega_pt": v["vega_pt"], "theta_1d": v["theta_1d"],
            "theta_decay_1d": v["theta_decay_1d"], "theta_fwd_1d": v["theta_fwd_1d"],
            "ex_div_next_day": v["ex_div_next_day"], "ex_div_today": ex_dividend_on(dividends, r.as_of),
            "rho_1pct": v["rho_1pct"], "elasticity_option": v["elasticity"], "prob_itm_rn": v["prob_itm"],
            "breakeven_cash_spy": v["breakeven_cash"], "breakeven_bills_spy": v["breakeven_vs_bills"],
            # per XX share: the fund holds `ops` one-share options per XX share, plus cash
            "xx_delta_spy_shares": v["delta"] * ops,
            "xx_elasticity": v["delta"] * ops * spy / r.nav_published,
            "xx_vega_pct_nav_per_pt": v["vega_pt"] * ops / r.nav_published,
            "xx_theta_pct_nav_per_day": v["theta_1d"] * ops / r.nav_published,
            "xx_theta_decay_pct_nav_per_day": v["theta_decay_1d"] * ops / r.nav_published,
            "xx_theta_fwd_pct_nav_per_day": v["theta_fwd_1d"] * ops / r.nav_published,
            "xx_rho_pct_nav_per_1pct": v["rho_1pct"] * ops / r.nav_published,
            "spx_spy_ratio": spx / spy if spx == spx else float("nan"),
        })
        if r.as_of in boxes:
            (row["vendor_box_rate"], row["vendor_box_expiry"], row["vendor_box_fund"],
             row["box_leg_decimals"], row["box_flow_day"]) = boxes[r.as_of]
            # Treasury zero rate (continuous) for the same term, for a like-for-like comparison
            row["treasury_rate_box_term"] = cv.zero((row["vendor_box_expiry"] - r.effective_date).days / 365.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["skipped_no_spot"] = skipped
    return out
