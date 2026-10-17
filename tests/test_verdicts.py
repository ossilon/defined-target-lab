"""Verdict rules (definedtarget/verdicts.py) on synthetic data built so each answer is known in advance (no real data).

The synthetic fund: daily closes from 2026-10-14; SPY moves randomly; NAV moves k x elasticity x SPY's move; implied
vol, VIX1Y and the 3-year rate move by chosen coefficients plus small noise. Each check sets one coefficient inside or
outside a frozen band and expects pass or fail; others test pending, extension, inconclusive and void.
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from _harness import check

from definedtarget import calendar as nyse
from definedtarget import pricing, verdicts as V

DAYS = nyse.sessions(date(2026, 10, 13), date(2028, 2, 28))      # 2026-10-13 is the pre-test base close


def synth(seed=1, k_lev=1.3, iv_spy=0.15, iv_vix=0.5, iv_rate=0.0, cash=0.01, prem=0.002, box_bp=50.0,
          creations=(), creation_jump=0.0, cps_sales=(), last=None, gap_listed=None, listed_from=V.LISTED_START,
          skew_gap=0.10, listed_slide=-0.05, flows_after=False):
    rng = np.random.default_rng(seed)
    days = [d for d in DAYS if last is None or d <= last]
    n = len(days)
    r_spy = rng.normal(0, 0.009, n)
    r_spy[0] = 0.0
    spy = 778.01 * np.cumprod(1 + r_spy)
    E = np.full(n, 5.6)
    vix = 21 + np.cumsum(rng.normal(0, 0.6, n))
    rate = 480 + np.cumsum(rng.normal(0, 4.0, n))
    iv = np.empty(n)
    iv[0] = 16.6
    shares = np.full(n, 60000.0)
    cps = np.full(n, 0.0048)
    for i in range(1, n):
        d_iv = iv_spy * r_spy[i] * 100 + iv_vix * (vix[i] - vix[i - 1]) + iv_rate * (rate[i] - rate[i - 1])
        iv[i] = iv[i - 1] + d_iv + rng.normal(0, 0.01)
        shares[i] = shares[i - 1]
        cps[i] = cps[i - 1]
        if days[i] in creations:
            shares[i] += 10000
        if days[i] in cps_sales:
            cps[i] = cps[i - 1] - 1 / shares[i]
    nav = np.empty(n)
    nav[0] = 26.51
    for i in range(1, n):
        nav[i] = nav[i - 1] * (1 + k_lev * E[i - 1] * r_spy[i])
    flow = [days[i] in creations for i in range(n)]
    after = [i > 0 and flow[i - 1] for i in range(n)]
    for i in range(1, n):                              # creation days lift the implied vol; it falls back next day
        if flow[i] and creation_jump:
            iv[i] += creation_jump
            if i + 1 < n:
                iv[i + 1] -= creation_jump
    state = pd.DataFrame({"as_of": days, "shares_outstanding": shares, "contracts": cps * shares,
                          "contracts_per_share": cps, "cash_weight": cash, "nav_published": nav,
                          "market_price": nav * (1 + prem * np.sign(rng.normal(size=n))), "flow_day": flow,
                          "day_after_flow": after, "strike": 995.01, "expiry": date(2030, 1, 10)})
    model = pd.DataFrame({"as_of": days, "iv_disc_0bp": iv / 100, "xx_elasticity": E, "zero_rate_T": rate / 1e4,
                          "vendor_box_rate": (rate + box_bp) / 1e4, "treasury_rate_box_term": rate / 1e4,
                          "prob_itm_rn": 0.29})
    prices = {d: (float(spy[i]), float(spy[i] * 10.033)) for i, d in enumerate(days)}
    vix1y = {d: float(vix[i]) for i, d in enumerate(days)}
    # listed quotes: the 10,000 call priced at XX's vol + gap; spreads give a probability skew_gap above the model
    F_DF = lambda d: (spy[days.index(d)] * 1.12, 0.86, (V.LISTED_EXPIRY - d).days / 365)  # noqa: E731
    rows, lvol = [], 16.9
    for i, d in enumerate(days):
        if d < listed_from or (gap_listed is None and listed_slide is None):
            continue
        F, DF, tau = F_DF(d)
        ratio = 10.033
        lvol = (lvol + listed_slide * r_spy[i] * 100) if listed_slide is not None else lvol
        vol = (iv[i] / 100 + (gap_listed or 0) / 100) if gap_listed is not None else lvol / 100
        c = pricing.call_price(F, 10000 / ratio, DF, tau, vol) * ratio
        rows.append({"close_date": d, "strike": 10000, "bid": c * 0.97, "ask": c * 1.03})
        p_target = 0.29 + skew_gap                                  # build K1/K2 mids with that spread probability
        c2 = 50.0
        c1 = c2 + p_target * 800 * DF
        rows += [{"close_date": d, "strike": 9600, "bid": c1, "ask": c1}, {"close_date": d, "strike": 10400,
                                                                          "bid": c2, "ask": c2}]
    listed = pd.DataFrame(rows, columns=["close_date", "strike", "bid", "ask"])
    return V.Inputs(state=state, model=model, prices=prices, vix1y=vix1y, dividends=[], listed=listed,
                    listed_forward=F_DF)


def provider_of(inp):
    """Simulated capture: a close's files arrive the next day, so a cutoff on day D sees closes before D."""
    def provider(cutoff):
        d = date.fromisoformat(cutoff[:10])
        keep = lambda df, col="as_of": df[df[col] < d] if df is not None else df  # noqa: E731
        return V.Inputs(state=keep(inp.state), model=keep(inp.model),
                        prices={k: v for k, v in inp.prices.items() if k < d},
                        vix1y={k: v for k, v in inp.vix1y.items() if k < d}, dividends=inp.dividends,
                        calls=keep(inp.calls) if inp.calls is not None else None,
                        listed=keep(inp.listed, "close_date"), listed_forward=inp.listed_forward, voids=inp.voids)
    return provider


def status(hid, today, inp):
    return [r for r in V.all_results(today, provider_of(inp)) if r.id == hid][0]


AFTER = date(2028, 2, 1)                               # after every review except H7
base = synth()

# 1. Before the review date: pending, with a provisional number.
r = status("H8", date(2026, 12, 1), base)
check("before its review a hypothesis is pending, with a provisional number", r.status == "pending" and "ratio" in r.stats)

# 2. Band tests: one coefficient inside the band passes, outside fails.
check("H8 passes when NAV moves 1.3x the elasticity (band 1.05-1.60)", status("H8", AFTER, base).status == "pass",
      status("H8", AFTER, base).line())
check("H8 fails when NAV moves exactly with the elasticity (1.0)", status("H8", AFTER, synth(k_lev=1.0)).status == "fail")
check("H10 passes with +0.15 vol points per 1% SPY (band +0.05 to +0.35)", status("H10", AFTER, base).status == "pass",
      status("H10", AFTER, base).line())
check("H10 fails with a negative slide (-0.10)", status("H10", AFTER, synth(iv_spy=-0.10)).status == "fail")
check("H4 passes with 0.5 of VIX1Y passed through (band 0.1-0.9)", status("H4", AFTER, base).status == "pass")
check("H4 fails with no pass-through", status("H4", AFTER, synth(iv_vix=0.0)).status == "fail")
check("H18 passes when the vendor follows Treasury moves (coefficient near 0)", status("H18", AFTER, base).status == "pass")
check("H18 fails when the vendor's rate never moves (-0.0156)", status("H18", AFTER, synth(iv_rate=-0.0156)).status == "fail",
      status("H18", AFTER, synth(iv_rate=-0.0156)).line())
check("H3 passes with 1% cash", status("H3", AFTER, base).status == "pass")
check("H3 fails with 2% cash", status("H3", AFTER, synth(cash=0.02)).status == "fail")
check("H6 passes with a 0.2% premium", status("H6", AFTER, base).status == "pass")
check("H6 fails with a 0.8% premium (median above 0.5%)", status("H6", AFTER, synth(prem=0.008)).status == "fail")
check("H12 passes with a steady +50 bp box spread", status("H12", AFTER, base).status == "pass")
check("H12 fails with a +90 bp spread", status("H12", AFTER, synth(box_bp=90)).status == "fail")
check("H16 passes when the exchange price tracks NAV (ratio near 1)", status("H16", AFTER, base).status == "pass",
      status("H16", AFTER, base).line())
r = status("H14", AFTER, synth(k_lev=1.0))
check("H14 passes when 10-session moves match the elasticity (slope near 1)", r.status == "pass" and r.n >= 8, r.line())
check("H14 fails when the extra leverage persists (1.3)", status("H14", AFTER, base).status == "fail")
check("H14 reports the slope's standard error (P2)", "slope_se" in status("H14", AFTER, base).stats)
r = status("H15", AFTER, base)
check("H15 passes when NAV falls 1.3x the elasticity on bad days (band 1.00-1.60)", r.status == "pass", r.line())
check("H11 fails when implied vol / VIX1Y drifts out of 0.72-0.82", status("H11", AFTER, base).status == "fail")

# 3. H2: the fee is paid by selling whole contracts.
sales = DAYS[40:250:70][:3]
r = status("H2", AFTER, synth(cps_sales=sales))
check("H2 passes: three one-contract sales cut contracts per share about 1% (band 0.5-1.5%)", r.status == "pass", r.line())
check("H2 fails with no sales (fall 0%)", status("H2", AFTER, base).status == "fail")

# 4. Listed tests.
r = status("H13", AFTER, synth(gap_listed=0.3))
check("H13 passes when XX's vol sits between the listed bid - 0.5 and the mid", r.status == "pass", r.line())
check("H13 fails when XX's vol sits above the listed mid", status("H13", AFTER, synth(gap_listed=-1.0)).status == "fail")
r = status("H19", AFTER, base)
check("H19 passes with a +10 point gap between listed and model odds (band +5 to +15)", r.status == "pass", r.line())
check("H19 fails with a +20 point gap", status("H19", AFTER, synth(skew_gap=0.20)).status == "fail")
r = status("H17", AFTER, base)
check("H17 passes when listed vol falls 0.05 per 1% rally (band -0.20 to +0.05)", r.status == "pass", r.line())
check("H17 reports XX's H10 coefficient on the same closes", "xx_h10_coef_same_closes" in r.stats)
check("H17 fails when listed vol slides up like the vendor (+0.165)",
      status("H17", AFTER, synth(listed_slide=0.165)).status == "fail")

# 5. Minimum samples: the review extends up to three months, then reads inconclusive.
short = synth(last=date(2026, 11, 25))                              # data stop long before 40 clean days
r = status("H8", date(2027, 6, 1), short)
check("too few clean days, even after three months: inconclusive (extended)",
      r.status == "inconclusive" and "extended" in r.review, r.line())
r = status("H8", date(2027, 2, 1), short)
check("inside the extension window it stays pending", r.status == "pending" and "extending" in r.review, r.line())
late = synth(listed_from=date(2027, 1, 20))                          # listed data start late: H13 extends, then decides
r = status("H13", date(2027, 7, 20), late)
check("a test that reaches its minimum during the extension gets a real verdict at the extended review",
      r.status in ("pass", "fail") and "extended" in r.review and r.n >= 60, r.line())

# 6. Void events: a roll before the review voids the hypothesis.
v = synth()
v.voids = [(date(2026, 12, 1), "option line changed")]
check("a void event before the review gives 'void', naming it", status("H8", AFTER, v).status == "void"
      and "option line" in status("H8", AFTER, v).note)

# 7. Event tests (H1): 15 creations; the mark jumps on each and falls back next day. Quiet implied vol (no SPY or
#    VIX1Y moves), so the planted effect is the only signal: with noisy vol, a one-sided 5% test will flag about 1 in 20
#    seeds by chance (seed 3 does), which is P1's 'a failure marks a lead, not proof'.
cre = set(DAYS[5:5 + 15 * 8:8])
r = status("H1", AFTER, synth(creations=cre, creation_jump=0.3, seed=3, iv_spy=0.0, iv_vix=0.0))
check("H1 passes when the day after each of 15 creations falls 0.3 points (more than 0.10, p < 0.05)",
      r.status == "pass" and r.n == 15 and r.stats["p_value"] < 0.05, r.line())
r = status("H1", AFTER, synth(creations=cre, creation_jump=0.0, seed=3, iv_spy=0.0, iv_vix=0.0))
check("H1 fails when creations leave no trace", r.status == "fail", r.line())
r = status("H1", date(2027, 1, 5), synth(creations=cre, creation_jump=0.3, seed=3, iv_spy=0.0, iv_vix=0.0))
check("H1 is pending before its 15th event", r.status == "pending" and r.n < 15, r.line())
r = status("H1", date(2027, 10, 20), synth(creations=set(list(cre)[:5]), seed=3))
check("H1 with fewer than 15 events by 2027-10-14 is inconclusive", r.status == "inconclusive", r.line())

# 8. H5 on deep in-the-money calls around an ex-date.
h5 = synth()
ex = date(2026, 12, 18)
d0 = nyse.previous_session(nyse.previous_session(ex))
h5.dividends = [(date(2025, 12, 19), 1.99), (date(2026, 3, 20), 1.80), (date(2026, 6, 18), 1.90),
                (date(2026, 9, 18), 1.89), (ex, 2.0)]
s0, s1 = h5.prices[d0][0], h5.prices[ex][0]
rows = []
for k in (20, 40, 60):
    rows += [{"as_of": d0, "account": "TPAY", "strike": k, "expiry": date(2027, 12, 17), "price": s0 - k},
             {"as_of": ex, "account": "TPAY", "strike": k, "expiry": date(2027, 12, 17), "price": s1 - k + 2.0}]
rows += [{"as_of": d0, "account": "OTHER", "strike": 20, "expiry": date(2027, 12, 17), "price": s0 - 20},
         {"as_of": ex, "account": "OTHER", "strike": 20, "expiry": date(2027, 12, 17), "price": s1 - 20 + 2.0}]
h5.calls = pd.DataFrame(rows)
r = status("H5", date(2026, 12, 22), h5)
check("H5 passes when deep calls ignore the ex-date drop (ratio 1.0); a contract in two funds counts once",
      r.status == "pass" and r.n == 3 and abs(r.stats["median_ratio"] - 1.0) < 1e-9, r.line())
check("H5 is pending before 2026-12-21", status("H5", date(2026, 12, 19), h5).status == "pending")
h5.calls = h5.calls.assign(price=np.where(h5.calls["as_of"] == ex, h5.calls["price"] - 2.0, h5.calls["price"]))
check("H5 fails when the marks drop by the dividend (continuous yield, ratio 0)",
      status("H5", date(2026, 12, 22), h5).status == "fail")

# 9. H7a checkpoints: drift inside the band each January passes; the overall verdict waits for the last.
h7 = synth()
for y in (2027, 2028):
    for d in nyse.sessions(date(y, 1, 1), date(y, 1, 31)):
        spy = h7.prices.get(d, (700.0, 0))[0]
        h7.prices[d] = (spy, spy * V.H7A_BASELINE * (1 + 0.0003 * (y - 2026)))
r = status("H7a", date(2027, 2, 1), h7)
check("H7a: the 2027 checkpoint passes with +0.03% drift; the overall verdict stays pending",
      r.status == "pending" and r.stats.get("2027") == "pass", r.line())
check("the 7-session January median uses sessions with both closes", V.h7a_ratio(h7, 2027)[1] == 7)

# 10. Windows and conventions.
check("a review window ends with the last close before the review date",
      V.changes(V.frame(base), ["nav", "spy"], V.TEST_START, date(2027, 1, 14))["as_of"].max() < date(2027, 1, 14))
check("add_months keeps month ends valid", V.add_months(date(2026, 11, 30), 3) == date(2027, 2, 28))
check("a cutoff counts the whole review day (UTC)", V.cutoff_for(date(2027, 1, 14)) == "2027-01-14T23:59:59Z")

# 11. Which listed readings count: P2's read window now; P3's market-time window once P3 is in the ledger.
import json  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from definedtarget import ledger  # noqa: E402
from definedtarget.forward import listed as L  # noqa: E402


def reply(mode, ask=590.0):
    data = {L.option_symbol(10000): {"bid": 530.0, "ask": ask, "update_mode": mode},
            "SP:SPX": {"close": 7800.0}, "AMEX:SPY": {"close": 777.0}}
    return json.dumps({"success": True, "data": data})


with tempfile.TemporaryDirectory() as tmp:
    raw = Path(tmp) / "raw" / "tradingview_chain" / "2026-11-02"
    raw.mkdir(parents=True)
    # Mon 2026-11-02 (New York on standard time): 21:20 UTC = 16:20 NY (market 16:05), 22:40 UTC = 17:40 NY
    (raw / "20261102T212000Z.json").write_text(reply("delayed_streaming_900", ask=591.0), encoding="utf-8")
    (raw / "20261102T224000Z.json").write_text(reply("delayed_streaming_900", ask=592.0), encoding="utf-8")
    led = Path(tmp) / "ledger.jsonl"
    ledger.append({"kind": "protocol", "id": "P1", "rules": ["synthetic"]}, path=led)
    p2 = V.counted_listed(Path(tmp), None, ledger_path=led)
    check("under P2 the reading inside 16:15-20:15 New York counts (17:40), the 16:20 one too; the first is used",
          list(p2["ask"]) == [591.0], p2)
    ledger.append({"kind": "protocol", "id": "P3", "rules": ["synthetic"]}, path=led,
                  now=datetime(2026, 10, 30, tzinfo=timezone.utc))
    p3 = V.counted_listed(Path(tmp), None, ledger_path=led)
    check("under P3 only the reading whose market time is 16:00-16:15 counts (16:20 read, 15-min delay)",
          list(p3["ask"]) == [591.0], p3)
    (raw / "20261102T213400Z.json").write_text(reply("delayed_streaming_900", ask=593.0), encoding="utf-8")
    p3b = V.counted_listed(Path(tmp), None, ledger_path=led)
    check("under P3 a 16:34 read (market 16:19) does not count; P3 keeps the latest qualifying reading",
          list(p3b["ask"]) == [591.0], p3b)
    check("a cutoff hides readings taken after it",
          V.counted_listed(Path(tmp), "2026-11-02T21:00:00Z", ledger_path=led).empty)

# 12. H9: creations from 2026-11-01 widen XX's vol against the listed mid; H7 waits for 2030.
cre9 = set(d for d in DAYS if d >= date(2026, 11, 2))
cre9 = set(sorted(cre9)[3:3 + 15 * 8:8])
h9 = synth(creations=cre9, seed=5, iv_spy=0.0, iv_vix=0.0, gap_listed=0.0)
lv = h9.listed.copy()
for d in cre9:                                     # on creation days the listed market stays put while XX's mark rises
    i = h9.model.index[h9.model["as_of"] == d][0]
    h9.model.loc[i, "iv_disc_0bp"] += 0.003
r = status("H9", AFTER, h9)
check("H9 passes when XX's vol sits 0.3 points above the listed mid on each of 15 creation days", r.status == "pass" and
      r.n == 15, r.line())
check("H7 is pending until 2030-01-10", status("H7", AFTER, base).status == "pending")

# 13. Audit B10c fixes (2026-10-17): each check would have failed before the fix.
import importlib.util  # noqa: E402

from definedtarget import marks  # noqa: E402

# B1: --record never writes from data that does not exist yet.
spec = importlib.util.spec_from_file_location("verdicts_cli", Path(__file__).resolve().parents[1] / "scripts" / "verdicts.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
UTC = timezone.utc
final = [V.Result("H8", "pass", "2027-01-14", 40, {"ratio": 1.3}, decided=date(2027, 1, 14))]
ready, waiting = V.to_record(final, [], "2027-01-14T23:00:00Z")
check("B1: a verdict waits while its deciding day (UTC) has not ended", not ready and len(waiting) == 1)
ready, waiting = V.to_record(final, [], "2027-01-15T00:00:01Z")
check("B1: the next day it is ready to record", len(ready) == 1 and not waiting)
check("B1: a final word without a deciding date is never recordable",
      not V.to_record([V.Result("H8", "pass", "x")], [], "2099-01-01T00:00:00Z")[0])
with tempfile.TemporaryDirectory() as tmp:
    led = Path(tmp) / "ledger.jsonl"
    ledger.append({"kind": "hypothesis", "id": "H8", "claim": "c", "test": "t", "pass_if": "p",
                   "review_on": "2027-01-14"}, path=led)

    def boom(today):
        raise AssertionError("results must not be computed for a future --as-of")
    rc = cli.main(["--as-of", "2027-02-01", "--record", "due"], ledger_path=led,
                  now=datetime(2027, 1, 15, 1, 0, tzinfo=UTC), results_fn=boom)
    check("B1: --record with an --as-of after today is refused before anything is computed",
          rc == 1 and len(ledger.read(led)) == 1)
    rc = cli.main(["--record", "due"], ledger_path=led, now=datetime(2027, 1, 14, 23, 0, tzinfo=UTC),
                  results_fn=lambda d: final)
    check("B1: --record on the deciding day itself records nothing", rc == 1 and len(ledger.read(led)) == 1)
    rc = cli.main(["--record", "due"], ledger_path=led, now=datetime(2027, 1, 15, 1, 0, tzinfo=UTC),
                  results_fn=lambda d: final)
    entries = ledger.read(led)
    check("B1: the day after, it records once, with the data cutoff in the evidence",
          rc == 0 and len(entries) == 2 and entries[-1]["evidence"]["data_cutoff"] == "2027-01-14T23:59:59Z")
    rc = cli.main(["--record", "due"], ledger_path=led, now=datetime(2027, 1, 16, 1, 0, tzinfo=UTC),
                  results_fn=lambda d: final)
    check("B1: a recorded verdict is never recorded twice", rc == 1 and len(ledger.read(led)) == 2)
    h7r = [V.Result("H7a", "pending", "x", stats={"2027": "pass", "2028": "pending"})]
    ready, waiting = V.to_record(h7r, [], "2027-01-21T00:00:00Z")
    check("B1: an H7a checkpoint is recordable after its own checkpoint day", [c[1] for c in ready] == ["2027"])

# B2: H5 never skips its first ex-date because an input is missing.
h5b = synth()
h5b.dividends = h5.dividends + [(date(2027, 3, 19), 1.95)]
h5b.calls = pd.DataFrame(rows)
del h5b.prices[d0]
r = status("H5", date(2026, 12, 22), h5b)
check("B2: a missing reference close gives 'no data' for the December ex-date", r.status == "no data"
      and "2026-12-18" in r.note and "reference close" in r.note, r.line())
r = status("H5", date(2027, 3, 25), h5b)
check("B2: ... and it does not move on to the March ex-date", r.status == "no data" and "2026-12-18" in r.note, r.line())
h5c = synth()
h5c.dividends = h5.dividends[:-1] + [(ex, float("nan"))]
h5c.calls = pd.DataFrame(rows)
r = status("H5", date(2026, 12, 22), h5c)
check("B2: a December row without an amount gives 'no data', never a verdict", r.status == "no data"
      and "dividend amount" in r.note, r.line())
h5d = synth()
h5d.dividends = h5.dividends + [(date(2027, 3, 19), 1.95)]
h5d.calls = pd.DataFrame(rows[:2])                       # one contract only: fewer than 3 -> next ex-date
r = status("H5", date(2026, 12, 22), h5d)
check("B2: fewer than 3 contracts (inputs present) still moves to the next ex-date", r.status == "pending"
      and "2027-03-19" in r.note, r.line())

# B3: P2's price rule admits issuer and user-supplied closes, never TradingView closes.
with tempfile.TemporaryDirectory() as tmp:
    ext = Path(tmp) / "external"
    ext.mkdir()
    (ext / "tv_closes.csv").write_text("date,symbol,close\n2026-11-02,AMEX:SPY,700\n2026-11-02,SP:SPX,7020\n",
                                       encoding="utf-8")
    (ext / "closes.csv").write_text("date,symbol,close\n2026-11-03,SPY,701\n2026-11-03,SPX,7031\n", encoding="utf-8")
    spots = marks.load_spots(Path(tmp))
    kept = V.price_rule(spots)
    check("B3: a TradingView close (tv_closes.csv) counts as missing under P2", date(2026, 11, 2) not in kept, spots)
    check("B3: a user-supplied close (closes.csv) counts", kept.get(date(2026, 11, 3)) == (701.0, 7031.0), kept)

# S1: H19 skips a close whose 9,600 quote is missing for a day; it falls back only when the strike lapses.
s1 = synth()
extra = []
for d, g in s1.listed.groupby("close_date"):
    m = 50 + (0.29 + 0.20) * 600 * 0.86                  # a 9,800 call that would put the gap at +20 points
    extra.append({"close_date": d, "strike": 9800, "bid": m, "ask": m})
s1.listed = pd.concat([s1.listed, pd.DataFrame(extra)], ignore_index=True)
closes = sorted(s1.listed["close_date"].unique())
gap_days = set(closes[::3])
s1.listed = s1.listed[~((s1.listed["strike"] == 9600) & s1.listed["close_date"].isin(gap_days))]
r = status("H19", AFTER, s1)
check("S1: a one-day gap at 9,600 is a missing close, not a switch to 9,800", r.status == "pass"
      and r.stats["missing_closes"] > 0, r.line())
s1b = synth()
s1b.listed = pd.concat([s1b.listed, pd.DataFrame(extra)], ignore_index=True)
s1b.listed = s1b.listed[~((s1b.listed["strike"] == 9600) & (s1b.listed["close_date"] >= date(2027, 3, 1)))]
n_before = status("H19", AFTER, s1b).n
r = status("H19", date(2027, 4, 14), s1b)
check("S1: a lapsed 9,600 strike moves to the nearest quoted strike below (missing only the newest close)",
      r.stats.get("missing_closes") == 1, r.line())

# S2: a zero bid or an unsolvable quote is a missing close, not a miss.
s2 = synth(gap_listed=0.3)
ten = s2.listed[s2.listed["strike"] == 10000].index[:10]
s2.listed.loc[ten, "bid"] = 0.0
r = status("H13", AFTER, s2)
check("S2: ten zero bids count as missing closes (share unchanged, n down by 10)",
      r.status == "pass" and r.stats["unusable_quotes"] == 10 and r.stats["share_in_band"] == 1.0, r.line())
check("S2: the capture's parser drops a zero bid as one-sided",
      L.parse(json.dumps({"success": True, "data": {L.option_symbol(10000): {"bid": 0, "ask": 5.0}}}))[0] == {})

# S3: the extension's last allowed close completes the sample; a void during the extension is caught.
s3 = synth()
keep = set(DAYS[:40]) | {date(2027, 4, 14)}               # 39 changes in the window, the 40th ends 2027-04-14
s3.state = s3.state[s3.state["as_of"].isin(keep)]
s3.model = s3.model[s3.model["as_of"].isin(keep)]
s3.prices = {d: v for d, v in s3.prices.items() if d in keep}
r = status("H8", date(2027, 4, 20), s3)
check("S3: a sample completed by the last allowed close gets a verdict the next session",
      r.status in ("pass", "fail") and r.review.startswith("2027-04-15") and r.n == 40, r.line())
sv = synth(last=date(2026, 11, 25))
sv.voids = [(date(2027, 3, 1), "liquidation notice")]
r = status("H8", date(2027, 6, 1), sv)
check("S3: a void during the extension gives 'void', not 'inconclusive'", r.status == "void", r.line())

# S4: H1 waits until the 15th event is in the data captured by the review; no crash without clean days.
late_inp = synth(creations=cre, creation_jump=0.3, seed=3, iv_spy=0.0, iv_vix=0.0)
ev15 = V.h1_stat(late_inp)[0][14][0]


def provider_late(inp, late_day, lag=7):
    base_p = provider_of(inp)

    def provider(cutoff):
        got = base_p(cutoff)
        d = date.fromisoformat(cutoff[:10])
        if d <= late_day + timedelta(days=lag):          # a capture outage: closes from the 15th event's arrive
            got.state = got.state[got.state["as_of"] < late_day]   # a week late
            got.model = got.model[got.model["as_of"] < late_day]
        return got
    return provider


r = [x for x in V.all_results(AFTER, provider_late(late_inp, ev15)) if x.id == "H1"][0]
check("S4: a late capture moves H1's review until the data hold 15 events (never n = 14)",
      r.n == 15 and date.fromisoformat(r.review) > ev15 + timedelta(days=7), r.line())
busy = synth(creations=set(DAYS[1:31:2]), seed=3)
r = status("H1", AFTER, busy)
check("S4: no clean comparison days gives 'no data', not a crash or a verdict", r.status == "no data", r.line())

# S5: H2 lets contracts per share recover on the day after a creation (bought T+1).
c_day = DAYS[30]
s5 = synth(creations={c_day}, cps_sales=sales)
i = s5.state.index[s5.state["as_of"] == c_day][0]
s5.state.loc[i, "contracts"] = s5.state.loc[i - 1, "contracts"]
s5.state.loc[i, "contracts_per_share"] = s5.state.loc[i, "contracts"] / s5.state.loc[i, "shares_outstanding"]
n, ok, st = V.t_h2(s5, AFTER)
check("S5: a rise on the day after a flow day is not an unexplained rise", st["unexplained_rises"] == 0, st)

# S6: void detection.
st6 = synth().state.copy()
j = st6.index[st6["as_of"] == date(2027, 2, 1)][0]
st6.loc[j:, "shares_outstanding"] = st6.loc[j:, "shares_outstanding"] / 2
st6.loc[j:, "nav_published"] = st6.loc[j:, "nav_published"] * 2
check("S6: an exact 1-for-2 reverse split is found", (date(2027, 2, 1), "reverse split") in V.find_voids(st6))
st7 = synth().state.copy()
st7["strike"] = st7["strike"].astype(float)
st7.loc[j, "strike"] = float("nan")                      # one close with two option lines
check("S6: a one-day two-line glitch is not a roll", V.find_voids(st7) == [])
st7.loc[j:, "strike"] = 1100.0
check("S6: a new option line on two consecutive closes is a roll",
      V.find_voids(st7) == [(date(2027, 2, 1), "option line changed to 1100.0 2030-01-10")], V.find_voids(st7))
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "void_events.csv"
    p.write_text("date,event\n2027-03-01,liquidation\n", encoding="utf-8")
    try:
        V.find_voids(None, events_csv=p)
        refused = False
    except ValueError:
        refused = True
    check("S6: a hand-listed void event without a source is refused", refused)
    p.write_text("date,event,source\n2027-03-01,liquidation,497 filing 2027-02-27\n", encoding="utf-8")
    check("S6: with a source it counts, naming the source",
          V.find_voids(None, events_csv=p) == [(date(2027, 3, 1), "liquidation (497 filing 2027-02-27)")])

# S7: fund events do not void the index-only ratio hypotheses.
h7.voids = [(date(2026, 12, 1), "option line changed")]
check("S7: H7a ignores fund void events (index closes only)", status("H7a", date(2027, 2, 1), h7).stats.get("2027")
      == "pass")

# C7: P3 applies only to readings taken after its ledger entry was recorded.
with tempfile.TemporaryDirectory() as tmp:
    raw = Path(tmp) / "raw" / "tradingview_chain" / "2026-11-02"
    raw.mkdir(parents=True)
    (raw / "20261102T212000Z.json").write_text(reply("delayed_streaming_900", ask=591.0), encoding="utf-8")
    (raw / "20261102T212900Z.json").write_text(reply("delayed_streaming_900", ask=594.0), encoding="utf-8")
    led = Path(tmp) / "ledger.jsonl"
    ledger.append({"kind": "protocol", "id": "P3", "rules": ["synthetic"]}, path=led,
                  now=datetime(2026, 11, 2, 21, 40, tzinfo=UTC))
    got = V.counted_listed(Path(tmp), None, ledger_path=led)
    check("C7: readings before P3 was recorded follow P2 (first in window)", list(got["ask"]) == [591.0], got)

# C10: one crashing hypothesis shows 'error' and leaves the others.
bad = synth()
bad.voids = None                                          # breaks every void check, but nothing else
res = V.all_results(AFTER, provider_of(bad))
check("C10: a crash in one review shows as 'error' for that hypothesis only",
      any(x.status == "error" for x in res) and any(x.status != "error" for x in res))

# 14. Independent external audit of v0.2.2 (2026-10-17): each check covers a defect the audit reproduced on v0.2.2,
#     or documents how a frozen rule is read (A04).
import copy  # noqa: E402

from definedtarget.forward import parse as P  # noqa: E402

# A01: H1 and H9 never extend: a 15th event after 2027-10-14 leaves them inconclusive.
cre_late = set(sorted(d for d in DAYS if d >= date(2026, 11, 2))[::12][:14]) | {date(2027, 10, 20)}
late = synth(creations=cre_late, creation_jump=0.3, seed=3, iv_spy=0.0, iv_vix=0.0, gap_listed=0.0)
for d in cre_late:
    late.model.loc[late.model["as_of"] == d, "iv_disc_0bp"] += 0.003
for hid in ("H1", "H9"):
    r = status(hid, date(2027, 11, 1), late)
    check(f"A01: {hid} with its 15th event after 2027-10-14 stays inconclusive (P2: never extends)",
          r.status == "inconclusive" and r.review == "2027-10-14" and r.n == 14, r.line())

# A02: a void event before the deadline gives 'void', not 'inconclusive', below 15 events.
gone = synth()
gone.voids = [(date(2027, 2, 1), "fund liquidated (synthetic notice)")]
for hid in ("H1", "H9"):
    r = status(hid, date(2027, 10, 15), gone)
    check(f"A02: {hid} below 15 events with a liquidation is void, naming it", r.status == "void"
          and "liquidated" in r.note, r.line())

# A03: impossible H19 quotes (above the no-arbitrage bound) count as missing, never as a pass.
imp = synth()
imp.listed.loc[imp.listed["strike"].isin([9600, 10400]), ["bid", "ask"]] += 100000
r = status("H19", AFTER, imp)
check("A03: H19 call prices above their upper bound are missing closes (no pass)", r.status != "pass" and r.n == 0,
      r.line())
nb = synth()
ext_rows = []
for d, g in nb.listed.groupby("close_date"):
    c9600 = float(g.loc[g["strike"] == 9600, "bid"].iloc[0])
    ext_rows += [{"close_date": d, "strike": 9500, "bid": c9600 + 60.0, "ask": c9600 + 60.0},
                 {"close_date": d, "strike": 10500, "bid": 40.0, "ask": 40.0}]
nb.listed = pd.concat([nb.listed, pd.DataFrame(ext_rows)], ignore_index=True)
for bad_quote in ("impossible", "one-sided"):
    x = copy.deepcopy(nb)
    named = x.listed["strike"].isin([9600, 10400])
    if bad_quote == "impossible":
        x.listed.loc[named, ["bid", "ask"]] += 100000
    else:
        x.listed.loc[named, "bid"] = 0.0
    r = status("H19", AFTER, x)
    check(f"B10e: a 9,600 / 10,400 quote that is {bad_quote} is a missing close, not a switch to 9,500 / 10,500",
          r.status != "pass" and r.n == 0, r.line())

# A04 (reading, not a defect): a flow-day fall beyond rounding is a fall; the pass rule fails only rises.
x4 = synth()
x4.state.loc[20:, "shares_outstanding"] = 50000.0
x4.state.loc[20:, "contracts_per_share"] = 237 / 50000
x4.state["contracts"] = x4.state["shares_outstanding"] * x4.state["contracts_per_share"]
x4.state.loc[20, "flow_day"] = True
x4.state.loc[21, "day_after_flow"] = True
r = status("H2", AFTER, x4)
check("A04: a 3-contract flow-day shortfall is reported for information; H2 follows its frozen pass rule",
      r.status == "pass" and r.stats["flow_day_shortfalls"] == 1, r.line())

# A05: H7 reads one fixed cutoff, the end of 2030-01-10.
h7base = synth(last=date(2026, 10, 20))


def h7_provider(cutoff):
    x = copy.copy(h7base)
    trig = 9990.0 if cutoff <= V.cutoff_for(V.XX_EXPIRY) else 10100.0
    x.prices = {V.XX_EXPIRY: (1000.0, 1000.0 * trig / V.XX_STRIKE)}
    return x


a5, b5 = V.review_h7(date(2030, 1, 12), h7_provider), V.review_h7(date(2030, 1, 19), h7_provider)
check("A05: H7 gives the same word whenever it runs, decided by the data captured by 2030-01-10",
      a5.status == b5.status == "pass" and a5.decided == b5.decided == V.XX_EXPIRY, (a5.line(), b5.line()))
check("A05: without the expiry closes by that date H7 reads 'no data'",
      V.review_h7(date(2030, 1, 12), lambda c: copy.copy(h7base)).status == "no data")


# A07 and A08 through the real loader: synthetic raw files, captured like the real ones.
def save_raw(root, name, text, at=datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc)):
    return fetch.capture("https://example.invalid/" + name, name, "synthetic", None, root=root, now=at,
                         getter=lambda u: (200, {}, text.encode()))


def raw_fixture(root, bad=False):
    ext = root / "external"
    ext.mkdir(parents=True)
    (ext / "spy_dividends.csv").write_text("ex_date,amount\n2025-12-19,2\n2026-03-20,2\n2026-06-18,2\n2026-09-18,2\n",
                                           encoding="utf-8")
    (ext / "closes.csv").write_text("date,symbol,close\n2026-10-14,SPY,778\n2026-10-14,SPX,7780\n", encoding="utf-8")
    mv = 2000000 if bad else 1440000
    body = ",".join(P.HOLDINGS_COLUMNS) + "\n"
    body += f"10/15/2026,XX,4SPY  300110C00995010,x,Call,288,50,{mv},99%,1454545.45,60000,6,\n"
    body += "10/15/2026,XX,FGXXX,x,Cash,14545.45,1,14545.45,1%,1454545.45,60000,6,Y\n"
    save_raw(root, "holdings_10142026.csv", body)
    save_raw(root, "xx_daily.csv", "Fund Ticker,Rate Date,NAV,Market Price\nXX,10/14/2026,24.24,24.24\n")
    save_raw(root, "par_yield_curve_2026.csv",
             "Date,1 Mo,6 Mo,1 Yr,2 Yr,3 Yr,5 Yr,10 Yr,30 Yr\n10/14/2026,4,4,4,4,4,4,4,4\n")
    save_raw(root, "VIX1Y_History.csv", "DATE,CLOSE\n10/14/2026,22\n")
    P.snapshot_external(root, now=datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc))


from definedtarget.forward import fetch  # noqa: E402

cut = V.cutoff_for(date(2027, 4, 14))
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    raw_fixture(root)
    a7 = V.load(root, cut)
    (root / "external" / "closes.csv").write_text("date,symbol,close\n2026-10-14,SPY,850\n2026-10-14,SPX,8500\n",
                                                 encoding="utf-8")
    (root / "external" / "spy_dividends.csv").write_text(
        "ex_date,amount\n2025-12-19,3\n2026-03-20,3\n2026-06-18,3\n2026-09-18,3\n", encoding="utf-8")
    (root / "external" / "void_events.csv").write_text("date,event,source\n2027-01-01,liquidation,notice added later\n",
                                                      encoding="utf-8")
    P.snapshot_external(root, now=datetime(2027, 5, 1, tzinfo=timezone.utc))      # copies saved after the cutoff
    b7 = V.load(root, cut)
    check("A07: later edits to closes, dividends and void notices do not reach an earlier review",
          a7.prices == b7.prices and a7.dividends == b7.dividends and b7.voids == []
          and a7.model["iv_disc_0bp"].tolist() == b7.model["iv_disc_0bp"].tolist(), (a7.prices, b7.prices, b7.voids))
    c7 = V.load(root, V.cutoff_for(date(2027, 5, 2)))
    check("A07: a review after the copies were saved sees the edits",
          c7.prices[date(2026, 10, 14)][0] == 850.0 and len(c7.voids) == 1, (c7.prices, c7.voids))
    check("A07: a good synthetic file passes every accounting identity", a7.problems == [], a7.problems)
    # B10e: the review itself reads the cutoff copy (no void), but recording waits on the late notice (P1 rule 6)
    r_late = V.review_window_test("H3", date(2027, 4, 15), lambda _: a7)
    blockers = V.record_blockers(a7, c7, date(2027, 4, 14))
    ready, waiting = V.to_record([r_late], [], "2027-04-22T00:00:00Z", lambda d: V.record_blockers(a7, c7, d))
    check("B10e: a void notice dated before the review but saved after it blocks recording a pass or fail",
          r_late.status == "pass" and len(blockers) == 1 and "void" in blockers[0] and not ready
          and "void" in waiting[0][4], (r_late.line(), blockers, waiting))

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    raw_fixture(root, bad=True)
    x8 = V.load(root, cut)
    r8 = V.review_window_test("H3", date(2027, 4, 15), lambda _: x8)
    ready, waiting = V.to_record([r8], [], "2027-04-16T00:00:00Z", lambda d: V.record_blockers(x8, x8, d))
    check("A08: an unexplained accounting failure keeps a fund-data verdict from being recorded",
          bool(x8.problems) and not ready and "accounting" in waiting[0][4], (x8.problems, waiting))
    r7 = V.Result("H7", "pass", "2030-01-10", 1, {"trigger": 9990.0}, decided=V.XX_EXPIRY)
    check("A08: index-only H7 does not wait on fund accounting",
          len(V.to_record([r7], [], "2030-01-11T00:00:01Z", lambda d: V.record_blockers(x8, x8, d))[0]) == 1)

# A15: without a December 2026 row, H5 reads 'no data' and never moves to March.
x15 = synth()
x15.dividends = [(date(2026, 9, 18), 1.89), (date(2027, 3, 19), 1.95)]
r = status("H5", date(2027, 4, 15), x15)
check("A15: no December 2026 ex-date in the file: H5 'no data', not a March verdict", r.status == "no data"
      and "December" in r.note, r.line())

# Rank: a regressor that never moves has no coefficient: 'error', never a verdict.
rk = synth()
rk.vix1y = {d: 22.0 for d in rk.vix1y}
r = status("H4", AFTER, rk)
check("a rank-deficient regression (VIX1Y never moves) gives 'error', not a verdict",
      r.status == "error" and "rank" in r.note, r.line())
r = status("H17", AFTER, rk)
check("B10e: H17's information stat cannot block H17's own verdict when VIX1Y never moves",
      r.status in ("pass", "fail") and "not computable" in str(r.stats.get("xx_h10_coef_same_closes")), r.line())

# A12: the verdicts' runtime dependency is declared; openpyxl is an optional extra.
pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
check("A12: pyproject declares scipy, and openpyxl as the 'flex' extra",
      '"scipy"' in pyproject.split("dependencies = ", 1)[1].split("\n", 1)[0] and 'flex = ["openpyxl"]' in pyproject)
