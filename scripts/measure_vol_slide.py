#!/usr/bin/env python3
"""How far does long-dated implied volatility rise when the S&P falls? Sets the simulator's default "vol slide".

    py -I scripts\\measure_vol_slide.py <S&P daily closes CSV> [<vol index CSV>]

Inputs (yours; never published): a CSV of S&P 500 daily closes (columns date, close; any date format pandas reads)
and Cboe's VIX1Y history (the newest data/raw/cboe/<date>/VIX1Y_History.csv by default; Cboe publishes it free).

Method: regress the change in the vol index (points) on the S&P's log return, over non-overlapping windows of 1, 5,
21 and 63 trading days, for the whole history and two halves. The slide is the vol change for a 10% fall:
    slide = beta x ln(0.9)        (beta = vol points per unit of log return; negative when vol rises in falls)
The simulator uses the same log scale, so a slide measured here plugs straight in. Prints the slope per window with
its standard error and the R-squared. Only these summary numbers are worth publishing; the data stays local.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WINDOWS = (1, 5, 21, 63)


def read_closes(path, date_col=None, close_col=None):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    d = date_col or cols.get("date") or cols.get("time")
    c = close_col or cols.get("close")
    s = pd.Series(df[c].astype(float).values, index=pd.to_datetime(df[d], format="mixed").dt.normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


def slope(x, y):
    """OLS slope of y on x with its standard error and R-squared."""
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x - x.mean(), y - y.mean()
    b = (xm * ym).sum() / (xm * xm).sum()
    resid = ym - b * xm
    se = math.sqrt((resid * resid).sum() / (len(x) - 2) / (xm * xm).sum())
    r2 = 1 - (resid * resid).sum() / (ym * ym).sum()
    return b, se, r2


def measure(spx, vol, label=""):
    both = pd.concat({"spx": spx, "vol": vol}, axis=1, join="inner").dropna()
    out = []
    for w in WINDOWS:
        sub = both.iloc[::w]                                   # non-overlapping windows
        r = np.log(sub["spx"]).diff().dropna()
        dv = sub["vol"].diff().loc[r.index]
        b, se, r2 = slope(r, dv)
        out.append({"period": label, "window_days": w, "n": len(r), "slide_pts_per_-10%": b * math.log(0.9),
                    "se": abs(se * math.log(0.9)), "r2": r2,
                    "first": both.index[0].date(), "last": both.index[-1].date()})
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    spx = read_closes(argv[0])
    if len(argv) > 1:
        vol_path = Path(argv[1])
    else:
        vol_path = sorted((REPO / "data" / "raw" / "cboe").glob("*/VIX1Y_History.csv"))[-1]
    vol = read_closes(vol_path)
    rows = measure(spx, vol, "all")
    mid = vol.index[0] + (vol.index[-1] - vol.index[0]) / 2
    rows += measure(spx[spx.index < mid], vol[vol.index < mid], "first half")
    rows += measure(spx[spx.index >= mid], vol[vol.index >= mid], "second half")
    print(f"vol index: {vol_path.name}; S&P: {Path(argv[0]).name}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
