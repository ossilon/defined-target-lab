"""Implied vols of the listed S&P 500 December 2029 calls' daily last trades (local TradingView bars), for the
v0.3.1 volatility fit. TradingView's data is display-only: the bars stay in the git-ignored data/ folder.

Plain words
- Bars: data/external/listed_bars/SPX291221C<strike>_daily.json (lists t, c, v: bar time, the day's last trade,
  contracts) and, for the 10,000 strike, data/external/listed_spx_10000c_dec2029_daily.py (the same three lists,
  read as data, never run).
- Trade date: a bar's time is 20:15 New York on the evening before its trade date (the start of Cboe's overnight
  session), so the trade date is the first NYSE session after the bar time's New York date (data/README.md).
- Forward and discounting: F = S e^{-qT} / DF(T), with S the S&P 500 close on the trade date, DF from that day's
  Treasury zero curve (curves.curve_on, no spread), T = (2029-12-21 - trade date) / 365 (ACT/365), and q SPY's
  trailing 12-month dividends over SPY's price, with SPY's price taken as S / RATIO (RATIO = 10.04, ASSUMED: the
  S&P/SPY ratio, about 10.03-10.04 in 2024-26; FINDINGS M1). These are F12's conventions.
- Implied vol: Black-76 at the forward (definedtarget.pricing). A trade's last price is not at the index close, so
  each vol carries timing noise (FINDINGS F12).
"""
import ast
import csv
import json
import math
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from . import calendar, curves, dividends, pricing
from .forward import fetch

EXPIRY = date(2029, 12, 21)
RATIO = 10.04
NEW_YORK = ZoneInfo("America/New_York")
TEN_K_FILE = "listed_spx_10000c_dec2029_daily.py"


def trade_date(t):
    """The NYSE trade date of a TradingView daily option bar stamped t (unix seconds)."""
    ny = datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone(NEW_YORK)
    return calendar.next_session(ny.date())


def load_bars(root=None):
    """{strike: list of (t, last trade, contracts)} from the local bar files (both formats)."""
    ext = fetch.data_dir(root) / "external"
    out = {}
    for p in sorted((ext / "listed_bars").glob("SPX291221C*_daily.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        strike = float(p.name[len("SPX291221C"):].split("_")[0])
        out[strike] = list(zip(d["t"], d["c"], d["v"]))
    tenk = ext / TEN_K_FILE
    if tenk.exists() and 10000.0 not in out:
        lists = {}
        for node in ast.parse(tenk.read_text(encoding="utf-8")).body:      # literal data only, never executed
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                lists[node.targets[0].id] = ast.literal_eval(node.value)
        out[10000.0] = list(zip(lists["t"], lists["c"], lists["v"]))
    return out


def forward(S, day, curve, history):
    """(F, DF, T) for the December 2029 expiry on trade date `day` (see the module docstring)."""
    T = (EXPIRY - day).days / 365.0
    df = curve.df(T)
    q = dividends.trailing_yield(history, day, S / RATIO)
    return S * math.exp(-q * T) / df, df, T


def trade_vols(bars, closes, curves_by_date, history, min_volume=1):
    """Rows (dict) for every bar with at least min_volume contracts and an S&P close on its trade date: date, strike,
    price, volume, S, F, DF, T, iv (NaN when no vol reproduces the price). closes: {date: S&P 500 close}."""
    rows = []
    for strike, items in sorted(bars.items()):
        for t, c, vol in items:
            if vol < min_volume:
                continue
            day = trade_date(t)
            S = closes.get(day)
            if S is None:
                continue
            F, df, T = forward(S, day, curves.curve_on(curves_by_date, day), history)
            iv = pricing.implied_vol(float(c), F, strike, df, T)
            rows.append({"date": day, "strike": strike, "price": float(c), "volume": int(vol), "S": S, "F": F,
                         "DF": df, "T": T, "iv": iv})
    return sorted(rows, key=lambda r: (r["date"], r["strike"]))


def read_closes_csv(path):
    """{date: close} from a CSV with columns date (YYYY-MM-DD) and close (the price library's format)."""
    with open(path, encoding="utf-8") as fh:
        return {date.fromisoformat(r["date"][:10]): float(r["close"]) for r in csv.DictReader(fh)}
