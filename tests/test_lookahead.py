"""No look-ahead (v0.3 design review): nothing reported for session k may depend on anything after k.

Judges, all exact (bit for bit):
1. montecarlo.causal.check, the reusable checker for later rules (v0.3.2's dip and ladder studies): it passes
   causal rules and catches each classic look-ahead (a centered average, tomorrow's value, a full-sample statistic,
   a one-row peek at the end).
2. Every path engine (lognormal and bootstrap on the CPU, both on the GPU where present, and the XX layer): a run
   stopped at session k reports exactly what a longer run with the same seed shows at session k, whatever comes
   after; measures that only grow (highest level, largest drawdown) never shrink when the future is added.
3. The bootstrap's choice of history row at step k depends only on steps before it.
4. The forecast check (FC1) builds its inputs from the origin close only: changing every later row in the tables
   and the VIX file changes nothing.
"""
import math
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from _harness import check, raises

from definedtarget import calendar as nyse, sim
from definedtarget import xxpaths as xp
from definedtarget.montecarlo import bootstrap as bs
from definedtarget.montecarlo import causal, gpu
from definedtarget.montecarlo import paths as lp

# ------------------------------------------------------------------ 1. the checker itself
rng = np.random.default_rng(5)
r = rng.standard_normal(600) * 0.01
level = np.cumsum(r)


def trailing_mean(x, w=20):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def drawdown_signal(x):
    """Buy after a 10% fall from the running high (the shape of a v0.3.2 dip rule)."""
    lv = np.exp(np.cumsum(x))
    return (lv / np.maximum.accumulate(lv) <= 0.9).astype(np.int64)


def centered_mean(x, w=21):
    return np.convolve(x, np.ones(w) / w, mode="same")


def tomorrow(x):
    return np.append(x[1:], np.nan)


def zscore_full_sample(x):
    return (x - x.mean()) / x.std()


def peek_last_row(x):
    return x + (x[-1] > 0)


for name, rule in (("trailing 20-session mean", trailing_mean), ("running drawdown signal", drawdown_signal),
                   ("cumulative level", np.cumsum)):
    check(f"causal.check passes a causal rule: {name}", causal.check(rule, r) == [])
two_cols = np.column_stack([r, np.abs(r)])
check("causal.check handles a 2-column series (rows = sessions)",
      causal.check(lambda x: np.cumsum(x, axis=0), two_cols) == [])
for name, rule in (("centered 21-session mean", centered_mean), ("tomorrow's value", tomorrow),
                   ("a full-sample z-score", zscore_full_sample), ("a one-row peek at the last row", peek_last_row)):
    bad = causal.check(rule, r)
    check(f"causal.check catches look-ahead: {name} ({len(bad)} cuts)", len(bad) > 0)
bad = causal.check(tomorrow, r, cuts=[100])
check("it reports the first output that read the future (tomorrow's value at cut 100: output 100, first future "
      "tried)", bad == [(100, 100, "reversed")], bad)


def month_end_peek(x):
    """Peeks one row ahead only on every 21st session (a rare, date-shaped peek)."""
    out = x.copy()
    for i in range(20, x.size - 1, 21):
        out[i] += x[i + 1]
    return out


def crash_peek(x):
    """Cuts exposure only when a fall of more than 5% is coming within 5 sessions (peeks only before extremes)."""
    out = np.ones(x.size)
    for i in range(x.size):
        if x[i + 1:i + 6].size and x[i + 1:i + 6].min() < -0.05:
            out[i] = 0.0
    return out


bad = causal.check(month_end_peek, r)
month_ends = [k for k in range(20, r.size - 1, 21)]
check(f"a peek on month-ends only is caught at exactly those {len(month_ends)} cuts and no others (only every cut "
      "point, the default up to 1,000 rows, is sure to include them)", [k for k, _, _ in bad] == month_ends,
      [k for k, _, _ in bad][:5])
check("r has no fall of more than 5%, so crash_peek never fires on the real series", crash_peek(r).min() == 1.0)
bad = causal.check(crash_peek, r)
check(f"a rule that peeks only when a crash is coming is caught ({len(bad)} cuts)", len(bad) > 0 and
      all(f in causal.FUTURES for _, _, f in bad))
for name, rule in (("dividing by the series length", lambda x: x / len(x)),
                   ("switching on once the series is long", lambda x: x * float(x.size > 299))):
    bad = causal.check(rule, r)
    check(f"causal.check catches a rule that reads the series length: {name} ({len(bad)} cuts)",
          len(bad) > 0 and {f for _, _, f in bad} <= {"none", "longer"}, bad[:3])
t0 = __import__("time").perf_counter()
causal.check(trailing_mean, r)
check("checking 600 sessions at every cut point under 6 futures takes under 10 s",
      __import__("time").perf_counter() - t0 < 10)
c5 = causal.default_cuts(5000)
check("longer series get 100 spread cut points plus the first and last", len(c5) == 101 and c5[0] == 0
      and c5[-1] == 4998, len(c5))
raises("a rule must return one output per session", lambda: causal.check(lambda x: x[:-1], r), "one per row")

# ------------------------------------------------------------------ 2. path engines: a prefix never sees the future
times = lp.times_from_dates(nyse.sessions(date(2026, 10, 14), date(2030, 1, 10)))
N = times.size - 1
CUTS = (1, 2, 21, 252, 401, N - 1)
NP = 20_000


def prefix_checks(label, full_run, cut_run):
    ok_level = ok_hit = ok_grow = True
    for k in CUTS:
        full, cut = full_run(k), cut_run(k)
        ok_level &= np.array_equal(cut["log_end"], full["at_checkpoint"][:, 0])
        fh, ch = full["hit_step"], cut["hit_step"]
        ok_hit &= np.array_equal(np.where(fh < k, fh, -1), ch)
        ok_grow &= bool((full["log_max"] >= cut["log_max"]).all() and
                        (full["log_max_drawdown"] >= cut["log_max_drawdown"]).all())
    check(f"{label}: the level at session k equals the run stopped at k, bit for bit ({len(CUTS)} cuts)", ok_level)
    check(f"{label}: the first barrier close is the same whenever it came by session k", ok_hit)
    check(f"{label}: the highest level and largest drawdown never shrink when the future is added", ok_grow)


prefix_checks("lognormal (CPU)",
              lambda k: lp.lognormal(9, NP, times, 0.04, 0.17, barrier=1.1, checkpoints=(k,)),
              lambda k: lp.lognormal(9, NP, times[:k + 1], 0.04, 0.17, barrier=1.1))
hist = rng.standard_normal((3000, 2)) * 0.01
prefix_checks("bootstrap (CPU)",
              lambda k: bs.paths(9, NP, N, hist, 21, barrier=1.1, checkpoints=(k,)),
              lambda k: bs.paths(9, NP, k, hist, 21, barrier=1.1))
idx = bs.indices(9, 500, N, 3000, 21)
check("bootstrap: the history rows chosen up to step k do not depend on how long the path runs",
      all(np.array_equal(bs.indices(9, 500, k, 3000, 21), idx[:, :k]) for k in CUTS))
if gpu.available():
    prefix_checks("lognormal (GPU)",
                  lambda k: gpu.lognormal(9, NP, times, 0.04, 0.17, barrier=1.1, checkpoints=(k,)),
                  lambda k: gpu.lognormal(9, NP, times[:k + 1], 0.04, 0.17, barrier=1.1))
    prefix_checks("bootstrap (GPU)",
                  lambda k: gpu.bootstrap(9, NP, N, hist, 21, barrier=1.1, checkpoints=(k,)),
                  lambda k: gpu.bootstrap(9, NP, k, hist, 21, barrier=1.1))
else:
    print("skip GPU prefix checks (no CuPy / NVIDIA GPU)")

# The XX layer: NAV at session k (priced with the target date, which is known) is the same whether the path stops
# at k (one last step to expiry, as the forecast check runs it) or runs on to expiry.
S0, R = 778.01, 0.0486498
T = float(times[-1])
M = sim.Market(R, R - math.log(882.2603 / S0) / T, T)
XX = sim.Fund((sim.Leg("call", 1, 995.01, 0.48),), cash=0.2474, fee=0.0099)
ok_nav = ok_mult = True
for k in CUTS[:-1]:
    full = xp.simulate(4, NP, XX, M, S0, times, 0.17, 0.166, slide=0.01, checkpoints=(k,))
    short = np.append(times[:k + 1], T)
    cut = xp.simulate(4, NP, XX, M, S0, short, 0.17, 0.166, slide=0.01, checkpoints=(k,))
    ok_nav &= np.array_equal(full["nav_at_checkpoint"], cut["nav_at_checkpoint"])
    f, c = full["first_multiple"], cut["first_multiple"]
    ok_mult &= np.array_equal(np.where((f >= 0) & (f < k), f, -1), np.where((c >= 0) & (c < k), c, -1))
check(f"XX layer: NAV at session k is the same whatever the path does after k ({len(CUTS) - 1} cuts)", ok_nav)
check("XX layer: the first session at 2x / 5x / 10x is the same whenever it came before k", ok_mult)

# ------------------------------------------------------------------ 4. the forecast check reads its origin only
from definedtarget.forward import fetch, forecast as fc  # noqa: E402


def make_root(tmp, later):
    """A data folder with tables for 2026-10-30 and the next session, and a VIX file holding both."""
    root = Path(tmp)
    (root / "tables").mkdir(parents=True)
    t, t1 = "2026-10-30", "2026-11-02"
    (root / "tables" / "xx_model_daily.csv").write_text(
        "as_of,spot,iv_disc_0bp,zero_rate_T,forward,tau,options_per_share,nav\n"
        f"{t},778.01,0.166,0.0486,882.26,3.2,0.48,26.5\n"
        f"{t1},{later['spot']},{later['iv']},0.05,900.0,3.19,0.47,{later['nav']}\n", encoding="utf-8")
    (root / "tables" / "xx_state.csv").write_text(
        "as_of,strike,cash_per_share,nav_published\n"
        f"{t},995.01,0.2474,26.5\n{t1},995.01,{later['cash']},{later['nav']}\n", encoding="utf-8")
    body = f"DATE,OPEN,HIGH,LOW,CLOSE\n10/30/2026,1,1,1,17.5\n11/02/2026,1,1,1,{later['vix']}\n".encode()
    fetch.capture("https://example.invalid/VIX_History.csv", "VIX_History.csv", "cboe", b"DATE,", root=root,
                  getter=lambda url: (200, {}, body))
    return root


with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
    calm = make_root(a, {"spot": 780.0, "iv": 0.165, "nav": 26.6, "cash": 0.2474, "vix": 17.0})
    crash = make_root(b, {"spot": 600.0, "iv": 0.30, "nav": 9.0, "cash": 0.1, "vix": 80.0})
    ia = fc.load_inputs(date(2026, 10, 30), calm, fetch_vix=False)
    ib = fc.load_inputs(date(2026, 10, 30), crash, fetch_vix=False)
    check("FC1's inputs for an origin ignore every later row (a calm and a crash future give the same inputs)",
          ia == ib and ia.vix == 17.5 and ia.spot == 778.01, (ia, ib))
