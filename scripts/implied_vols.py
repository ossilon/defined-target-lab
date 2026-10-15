#!/usr/bin/env python3
"""Implied volatility and Greeks of XX's daily option marks.  From target_funds/:

    py -I scripts\\implied_vols.py

Writes data/tables/xx_model_daily.csv (git-ignored) and prints a summary table.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from definedtarget import curves, marks  # noqa: E402
from definedtarget.forward import fetch, parse, reconcile  # noqa: E402


def main():
    root = fetch.data_dir()
    state = reconcile.fund_state(parse.load_holdings(root, accounts=("XX",)), parse.load_fund_daily(root, "XX"))
    vm = parse.load_holdings(root, accounts=None, all_versions=False)
    vm = vm[vm["is_option"] & (vm["opt_underlying"] == "SPY")]
    df = marks.model_daily(state, curves.load_treasury(root), marks.load_dividends(root), marks.load_spots(root), vm)
    out = root / "tables" / "xx_model_daily.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    show = df[["as_of", "spot", "mark", "iv_disc_0bp", "iv_disc_25bp", "iv_disc_50bp", "iv_cont_0bp",
               "zero_rate_T", "vendor_box_rate", "delta", "elasticity_option", "xx_elasticity",
               "xx_vega_pct_nav_per_pt", "xx_theta_pct_nav_per_day", "xx_rho_pct_nav_per_1pct", "prob_itm_rn",
               "breakeven_cash_spy", "flow_day"]].copy()
    for c in ("iv_disc_0bp", "iv_disc_25bp", "iv_disc_50bp", "iv_cont_0bp", "zero_rate_T", "vendor_box_rate",
              "xx_vega_pct_nav_per_pt", "xx_theta_pct_nav_per_day", "xx_rho_pct_nav_per_1pct"):
        show[c] = (show[c] * 100).round(3)
    print(show.round(4).to_string(index=False))
    print("\n(iv_*, rates and xx_* % columns are in percent; spot source per row is in the CSV)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
