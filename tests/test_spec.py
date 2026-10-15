"""Fund spec files: XX loads, and the loader refuses malformed specs."""
import copy
import json
from datetime import date

from _harness import check, raises

from definedtarget import spec

xx = spec.load("XX")
leg = xx.legs[0]
check("XX spec: one long SPY call, strike 995.01, expiry 2030-01-10",
      len(xx.legs) == 1 and leg.underlying.value == "SPY" and leg.strike.value == 995.01
      and leg.expiry.value == "2030-01-10" and leg.sign.value == 1)
check("XX spec: 0.48 one-share options per fund share", abs(leg.options_per_share() - 0.48) < 1e-12)
check("XX spec: target date parses to 2030-01-10", xx.expiry == date(2030, 1, 10))
check("XX spec: the roll rule is OPEN with no value", xx.roll_rule.status == "OPEN" and xx.roll_rule.value is None)
check("XX spec: every field carries a source", all(f.source for f in
      [xx.name, xx.index, xx.index_target, xx.target_date, xx.fee] + [getattr(leg, k) for k in spec.LEG_KEYS]))

raw = json.loads((spec.FUNDS_DIR / "xx.json").read_text(encoding="utf-8"))
bad = copy.deepcopy(raw)
bad["fee"]["status"] = "GUESSED"
raises("an unknown status word is refused", lambda: spec.from_dict(bad), "unknown status")
bad = copy.deepcopy(raw)
bad["fee"]["source"] = ""
raises("a field without a source is refused", lambda: spec.from_dict(bad), "needs a source")
bad = copy.deepcopy(raw)
bad["roll_rule"] = {"value": "same moneyness", "status": "OPEN", "source": "x"}
raises("an OPEN field holding a value is refused", lambda: spec.from_dict(bad), "OPEN field must have value null")
bad = copy.deepcopy(raw)
bad["legs"] = []
raises("a fund with no option legs is refused", lambda: spec.from_dict(bad), "at least one option leg")
bad = copy.deepcopy(raw)
del bad["legs"][0]["strike"]
raises("a leg missing its strike is refused", lambda: spec.from_dict(bad), "missing field 'strike'")
