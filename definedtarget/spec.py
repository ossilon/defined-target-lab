"""Fund specification: everything the engine needs to know about one target-date call fund, in a JSON file.

Every field is {"value": ..., "status": ..., "source": ...} so the engine uses nothing without saying where it
came from. Status words (as in FUND_FACTS.md): VERIFIED, MEASURED, PROBABLE, ASSUMED, OPEN. A field that is
OPEN has value null and the engine must not rely on it (e.g. the roll rule).

A fund holds one or more option "legs". XX has one: a long European cash-settled SPY call. Sister funds may use
other ETFs, several legs (one per ETF), or spreads, so each leg carries its own underlying, type, sign, style and
settlement.
"""
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

STATUSES = {"VERIFIED", "MEASURED", "PROBABLE", "ASSUMED", "OPEN"}
FUNDS_DIR = Path(__file__).resolve().parent / "funds"


@dataclass(frozen=True)
class Field:
    value: object
    status: str
    source: str

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r} (use one of {sorted(STATUSES)})")
        if self.status == "OPEN" and self.value is not None:
            raise ValueError("an OPEN field must have value null")
        if self.status != "OPEN" and self.value is None:
            raise ValueError("a field with a value of null must be OPEN")
        if not self.source:
            raise ValueError("every field needs a source")


@dataclass(frozen=True)
class Leg:
    underlying: Field
    option_type: Field          # "call" or "put"
    sign: Field                 # +1 long, -1 short
    exercise: Field             # "european"
    settlement: Field           # "cash"
    strike: Field
    expiry: Field               # ISO date
    contracts_per_share: Field  # option contracts per fund share (x multiplier = one-share options per fund share)
    multiplier: Field

    def options_per_share(self):
        return float(self.contracts_per_share.value) * float(self.multiplier.value)


@dataclass(frozen=True)
class FundSpec:
    ticker: str
    name: Field
    index: Field
    index_target: Field
    target_date: Field
    fee: Field
    cash_if_worthless: Field
    roll_rule: Field
    legs: tuple

    @property
    def expiry(self):
        return date.fromisoformat(self.target_date.value)


def _field(d, key):
    if key not in d:
        raise ValueError(f"missing field {key!r}")
    f = d[key]
    return Field(f.get("value"), f.get("status", ""), f.get("source", ""))


LEG_KEYS = ["underlying", "option_type", "sign", "exercise", "settlement", "strike", "expiry",
            "contracts_per_share", "multiplier"]
FUND_KEYS = ["name", "index", "index_target", "target_date", "fee", "cash_if_worthless", "roll_rule"]


def from_dict(d):
    legs = tuple(Leg(**{k: _field(leg, k) for k in LEG_KEYS}) for leg in d.get("legs", []))
    if not legs:
        raise ValueError("a fund needs at least one option leg")
    spec = FundSpec(d["ticker"], *(_field(d, k) for k in FUND_KEYS), legs)
    for leg in legs:
        if leg.option_type.value not in ("call", "put") or leg.sign.value not in (1, -1):
            raise ValueError("leg option_type must be call/put and sign +1/-1")
        date.fromisoformat(leg.expiry.value)
    date.fromisoformat(spec.target_date.value)
    return spec


def load(ticker, folder=FUNDS_DIR):
    path = Path(folder) / f"{ticker.lower()}.json"
    return from_dict(json.loads(path.read_text(encoding="utf-8")))
