"""Command line:  py -m definedtarget <command>   (run from the repository folder)

  calc         "What must the S&P do?" for one share (no data needed; see --help for the inputs)
  daily        capture today's files, rebuild tables, price the marks, write the daily report
  report       print the latest daily report (after `daily`)
  prospectus   recompute and print the prospectus's payoff illustration
"""
import argparse
import math
import sys
from datetime import date

from . import calc


def _latest_model():
    """Defaults from the latest captured close, or None if no capture has saved data yet."""
    try:
        import pandas as pd
        from .forward import fetch
        root = fetch.data_dir()
        m = pd.read_csv(root / "tables" / "xx_model_daily.csv").iloc[-1]
        s = pd.read_csv(root / "tables" / "xx_state.csv").iloc[-1]
        return {"price": float(m["nav"]), "spot": float(m["spot"]), "sigma": float(m["iv_disc_0bp"]),
                "rate": float(m["zero_rate_T"]), "tau": float(m["tau"]), "forward": float(m["forward"]),
                "options_per_share": float(m["options_per_share"]), "cash_per_share": float(s["cash_per_share"]),
                "index_now": float(m["spx"]) if m["spx"] == m["spx"] else None, "as_of": str(m["as_of"])}
    except (FileNotFoundError, KeyError, IndexError):
        return None


def cmd_calc(a):
    d = _latest_model() or {}
    if a.price is None and not d:
        print("The tool found no captured data: give at least --price --spot --iv --rate (and --as-of or --years).")
        return 2
    tau = a.years if a.years else ((date(2030, 1, 10) - date.fromisoformat(a.as_of)).days / 365.0 if a.as_of
                                   else d.get("tau"))
    explicit = any(v is not None for v in (a.spot, a.iv, a.rate, a.years, a.as_of))
    given = {"--price": a.price if a.price is not None else d.get("price"),
             "--spot": a.spot if a.spot is not None else d.get("spot"),
             "--iv": a.iv if a.iv is not None else d.get("sigma"),
             "--rate": a.rate if a.rate is not None else d.get("rate"),
             "--as-of or --years": tau}
    missing = [flag for flag, v in given.items() if v is None]
    if missing:
        print("Missing input(s): " + ", ".join(missing) + ". Without captured data, give --price --spot --iv --rate "
              "and --as-of (or --years).")
        return 2
    c = calc.CalcInputs(
        price=given["--price"],
        spot=given["--spot"],
        sigma=given["--iv"],
        rate=given["--rate"],
        tau=tau,
        options_per_share=a.options_per_share or d.get("options_per_share", 0.48),
        cash_per_share=a.cash if a.cash is not None else d.get("cash_per_share", 0.25),
        fee=a.fee, div_yield=a.div_yield,
        forward=None if explicit else d.get("forward"),
        index_now=None if explicit else d.get("index_now"))
    if d and not explicit:
        print(f"(using defaults from the latest captured close, {d['as_of']})\n")
    print(calc.render(calc.calculate(c)))
    return 0


def cmd_daily(_a):
    from .forward import routine
    note, stops = routine.run()
    print("\n".join(note))
    for s in stops:
        print("STOP:", s)
    return 1 if stops else 0


def cmd_report(_a):
    from .forward.routine import REPO
    files = sorted((REPO / "forward" / "daily").glob("*_report.md"))
    if not files:
        print("No report yet: run  py -m definedtarget daily")
        return 2
    print(files[-1].read_text(encoding="utf-8"))
    return 0


def cmd_prospectus(_a):
    print("The prospectus illustration, recomputed by this tool: one option, ETF strike $1,000 (index 10,000), "
          "premium $50\n")
    print("  Index    ETF     Payout   Profit")
    for idx, etf, pay, prof in calc.prospectus_table():
        print(f"  {idx:>6,}  {etf:>6,}  {pay:>6,.0f}   {prof:>6,.0f}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="definedtarget", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calc", help="what must the S&P do for one share")
    c.add_argument("--price", type=float, help="the price you pay per fund share (default: latest NAV)")
    c.add_argument("--spot", type=float, help="SPY price now")
    c.add_argument("--iv", type=float, help="option implied volatility as a decimal, e.g. 0.166")
    c.add_argument("--rate", type=float, help="interest rate to the target date, decimal, e.g. 0.0486")
    c.add_argument("--div-yield", type=float, default=0.0097, help="SPY dividend yield, decimal (default 0.0097)")
    c.add_argument("--as-of", help="valuation date YYYY-MM-DD (sets the years left to 2030-01-10)")
    c.add_argument("--years", type=float, help="years left to the target date (instead of --as-of)")
    c.add_argument("--options-per-share", type=float, help="one-share options per fund share (XX: 0.48)")
    c.add_argument("--cash", type=float, help="cash per fund share")
    c.add_argument("--fee", type=float, default=0.0099, help="yearly fee (default 0.0099)")
    c.set_defaults(fn=cmd_calc)
    sub.add_parser("daily", help="capture + tables + pricing + daily report").set_defaults(fn=cmd_daily)
    sub.add_parser("report", help="print the latest daily report").set_defaults(fn=cmd_report)
    sub.add_parser("prospectus", help="recompute the prospectus payoff illustration").set_defaults(fn=cmd_prospectus)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
