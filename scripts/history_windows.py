#!/usr/bin/env python3
"""How fast has the S&P 500 grown over stretches as long as a defined-target fund's life? (FINDINGS F13)

    py -I scripts\\history_windows.py <path to Shiller's ie_data.xls> [years, default 3.25]

Input: Robert Shiller's free monthly S&P file (https://shillerdata.com/, "U.S. Stock Markets 1871-Present"), read
with xlrd (values only). Its price is a monthly average of daily closes, which smooths the windows a little.
Prints, for every window of the given length (overlapping, one per month): the annualized total return (dividends
reinvested monthly) and price-only growth, their percentiles since 1871, 1950 and 1988, and how often price growth
cleared a defined-target fund's bars (XX on 2026-10-13: 7.9%/yr to reach the strike, 9.8%/yr to break even at NAV).
Only these summary numbers are worth publishing; the file stays with you.
"""
import sys

import numpy as np
import pandas as pd

BARS = (("reach XX's strike", 0.079), ("XX break-even at NAV", 0.098), ("the 16% row's median path", 0.133))


def read_shiller(path):
    import xlrd                                        # local import: only this script needs it
    sh = xlrd.open_workbook(path).sheet_by_name("Data")
    hdr = next(r for r in range(sh.nrows) if str(sh.cell_value(r, 0)).strip() == "Date")
    rows = []
    for r in range(hdr + 1, sh.nrows):
        d, p, dv = sh.cell_value(r, 0), sh.cell_value(r, 1), sh.cell_value(r, 2)
        if isinstance(d, float) and isinstance(p, float) and p > 0:
            rows.append((d, p, dv if isinstance(dv, float) else np.nan))
    df = pd.DataFrame(rows, columns=["d", "price", "div"])
    df["div"] = df["div"].ffill()                       # the latest months carry no dividend yet
    df = df.dropna().reset_index(drop=True)
    df["year"] = df.d.astype(int) + ((df.d * 100).round().astype(int) % 100 - 1) / 12.0
    growth = (df.price + df["div"] / 12.0) / df.price.shift(1)
    df["tr"] = growth.fillna(1.0).cumprod()             # total return index, dividends reinvested monthly
    return df


def windows(df, col, months, start):
    s = df.loc[df.year >= start, col].to_numpy()
    return (s[months:] / s[:-months]) ** (12.0 / months) - 1.0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    df = read_shiller(argv[0])
    months = int(round(float(argv[1]) * 12)) if len(argv) > 1 else 39
    print(f"S&P 500, every {months}-month window (Shiller monthly {df.year.iloc[0]:.0f}-{df.year.iloc[-1]:.2f}), annualized")
    for col, label in (("tr", "total return"), ("price", "price only")):
        print(f"\n{label}:")
        for start in (1871, 1950, 1988):
            w = windows(df, col, months, start)
            q = np.percentile(w, [50, 75, 90, 95])
            line = (f"  from {start}: {len(w)} windows (~{len(w) // months} separate)  median {q[0]:.1%}  top 25% {q[1]:.1%}"
                    f"  top 10% {q[2]:.1%}  top 5% {q[3]:.1%}  best {w.max():.1%}")
            if col == "price":
                line += "  | cleared: " + ", ".join(f"{name} {np.mean(w >= bar):.0%}" for name, bar in BARS)
            print(line)
    w = windows(df, "tr", months, 1871)
    best = df.year.iloc[int(np.argmax(w))]
    print(f"\nbest total-return window starts {best:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
