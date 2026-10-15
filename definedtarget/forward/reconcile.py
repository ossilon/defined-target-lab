"""Daily fund state rebuilt from the issuer's holdings, with the accounting identities checked.

Identities (each one is a check; a failure is a STOP for the daily log):
1. Date rule: a holdings file named for close date D carries Date = the next NYSE session after D.
2. NetAssets / SharesOutstanding rounds to the published NAV (within half a cent).
3. Every option line: MarketValue = contracts x 100 x mark (within 1 cent).
4. The listed lines (options + money market + Cash & Other) add up to NetAssets within half a cent per share.
   They are not exact. The gap changes sign from day to day (a few dollars to a few hundred, in both
   directions), so it is NOT simply the accrued management fee (that would always lower NetAssets); its cause is
   unexplained (OPEN). The table keeps the gap (`lines_gap_usd`) visible, so anyone can study it.

Definitions that later steps use:
- contracts_per_share = option contracts / shares outstanding (XX: 0.0048, i.e. calls on 0.48 SPY shares)
- cash = every non-option line (money-market fund FGXXX + "Cash & Other"); while a creation settles, the two
  lines can swing by large, nearly equal and opposite amounts, so always add them together.
- flow day = shares outstanding changed since the previous close (a creation or a redemption).
- a mark with <= 2 meaningful decimals looks like a traded price (the fund's own FLEX trade); 4 decimals looks
  like a pricing vendor's model value.
"""
from datetime import date

import pandas as pd

from .. import calendar

HALF_CENT = 0.005

# Identity failures we investigated and explained. Never widen a tolerance instead of adding a reason here.
KNOWN_EXCEPTIONS = {
    (date(2026, 9, 30), "check_lines"): (
        "seed day (first close): NetAssets equals shares x the published NAV exactly, but the listed lines exceed "
        "it by just over half a cent per share, enough to round the NAV up one cent. Cause unexplained (seed-day "
        "accounting); every later day is inside half a cent. Investigated 2026-10-14, reason corrected after "
        "audit A1."),
}


def fund_state(holdings, fund_daily):
    """One row per close date (latest saved version of each holdings file) plus check columns."""
    h = holdings[holdings["latest_version"]] if "latest_version" in holdings else holdings
    navs = dict(zip(fund_daily["as_of"], fund_daily["nav"]))
    mkts = dict(zip(fund_daily["as_of"], fund_daily["market_price"]))
    rows = []
    for as_of, g in sorted(h.groupby("as_of"), key=lambda kv: kv[0]):
        opt = g[g["is_option"]]
        shares = float(g["shares_outstanding"].iloc[0])
        net_assets = float(g["net_assets"].iloc[0])
        contracts = float(opt["shares"].sum())
        opt_value = float(opt["market_value"].sum())
        cash = float(g.loc[~g["is_option"], "market_value"].sum())
        lines_total = opt_value + cash
        mark = float(opt["price"].iloc[0]) if len(opt) == 1 else float("nan")
        published = navs.get(as_of, float("nan"))
        rows.append({
            "as_of": as_of,
            "effective_date": g["effective_date"].iloc[0],
            "n_option_lines": len(opt),
            "strike": float(opt["opt_strike"].iloc[0]) if len(opt) == 1 else float("nan"),
            "expiry": opt["opt_expiry"].iloc[0] if len(opt) == 1 else None,
            "contracts": contracts,
            "mark": mark,
            "mark_text": opt["price_text"].iloc[0] if len(opt) == 1 else "",
            "mark_decimals": int(opt["price_decimals"].max()) if len(opt) else 0,
            "shares_outstanding": shares,
            "net_assets": net_assets,
            "option_value": opt_value,
            "cash": cash,
            "lines_total": lines_total,
            "lines_gap_usd": lines_total - net_assets,
            "nav_net_assets": net_assets / shares,
            "nav_lines": lines_total / shares,
            "nav_published": published,
            "market_price": mkts.get(as_of, float("nan")),
            "contracts_per_share": contracts / shares,
            "option_per_share": opt_value / shares,
            "cash_per_share": cash / shares,
            "cash_weight": cash / net_assets,
            "max_mv_error": float((opt["market_value"] - opt["shares"] * 100.0 * opt["price"]).abs().max())
            if len(opt) else 0.0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["premium"] = df["market_price"] / df["nav_published"] - 1.0
    df["shares_change"] = df["shares_outstanding"].diff().fillna(0.0)
    df["flow_day"] = df["shares_change"] != 0                  # creation (+) or redemption (-) since last close
    df["day_after_flow"] = df["flow_day"].shift(1, fill_value=False)
    df["traded_looking_mark"] = df["mark_decimals"] <= 2
    df["check_date_rule"] = [e == calendar.next_session(a) for a, e in zip(df["as_of"], df["effective_date"])]
    df["check_nav"] = (df["nav_net_assets"] - df["nav_published"]).abs() <= HALF_CENT + 1e-9
    df["check_mv"] = df["max_mv_error"] <= 0.01 + 1e-9
    df["check_lines"] = (df["nav_lines"] - df["nav_net_assets"]).abs() <= HALF_CENT + 1e-9
    df["check_one_option"] = df["n_option_lines"] == 1
    return df


CHECK_COLUMNS = ["check_date_rule", "check_nav", "check_mv", "check_lines", "check_one_option"]


def failures(state, known=KNOWN_EXCEPTIONS):
    """(unexplained, explained): lists of (as_of, check name) for failed identities."""
    unexplained, explained = [], []
    for col in CHECK_COLUMNS:
        for as_of in state.loc[~state[col], "as_of"]:
            (explained if (as_of, col) in known else unexplained).append((as_of, col))
    return unexplained, explained


def revisions(holdings, fund="XX"):
    """Days whose holdings file the capture saved more than once, and whether the fund's mark or shares changed."""
    h = holdings[(holdings["account"] == fund) & holdings["is_option"]]
    out = []
    for as_of, g in h.groupby("as_of"):
        if g["version"].nunique() > 1:
            out.append({"as_of": as_of, "versions": g["version"].nunique(),
                        "marks": list(g.sort_values("version")["price_text"]),
                        "contracts": list(g.sort_values("version")["shares"])})
    return pd.DataFrame(out)
