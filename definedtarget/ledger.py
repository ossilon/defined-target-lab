"""Pre-registered predictions: an append-only, hash-chained ledger (forward/predictions.jsonl).

Plain words
- A forward test only means something if the predictions go on record BEFORE the data that tests them.
- Each entry is one JSON line. It stores the SHA-256 fingerprint of the previous line ("prev_hash") and of itself
  ("row_hash"). Changing an earlier entry changes its fingerprint and breaks every later link, so `verify()`
  catches a careless edit. It cannot catch a deliberate rewrite: anyone can recompute every later hash, and an empty
  or shortened file also verifies. Proof of WHEN the entries existed comes from outside the file: an
  OpenTimestamps proof of the frozen bytes, the public git history, and a test that pins the frozen entries'
  fingerprint (tests/test_ledger.py).
- Never edit or delete an entry. To resolve a hypothesis, append a "verdict" entry that names it.
- Appends keep the file's existing line endings, so the bytes of earlier lines never change and a timestamp proof of
  them stays valid.

Kinds: "protocol" (the shared rules every test follows: definitions, data windows, void events), "hypothesis"
(frozen, with a test rule and a review date), "parked" (written down, not yet testable), "verdict" (the result,
appended later; a hypothesis with yearly checkpoints may receive several).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "forward" / "predictions.jsonl"
GENESIS = "0" * 64
KINDS = {"protocol", "hypothesis", "parked", "verdict"}
REQUIRED = {"protocol": ("id", "rules"),
            "hypothesis": ("id", "claim", "test", "pass_if", "review_on"),
            "parked": ("id", "claim", "why_parked", "better_test"),
            "verdict": ("id", "result", "evidence")}


def _hash(prev_hash, body):
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def read(path=DEFAULT_PATH):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(path=DEFAULT_PATH):
    """(ok, message): every line links to the one before and its own fingerprint matches its content."""
    prev = GENESIS
    for i, e in enumerate(read(path), 1):
        body = {k: v for k, v in e.items() if k not in ("prev_hash", "row_hash")}
        if e.get("prev_hash") != prev:
            return False, f"line {i}: chain broken (prev_hash does not match line {i - 1})"
        if _hash(prev, body) != e.get("row_hash"):
            return False, f"line {i}: content changed after it was written"
        prev = e["row_hash"]
    return True, "chain intact"


def append(entry, path=DEFAULT_PATH, now=None):
    """Append one entry (validated), returning it with its hashes. Refuses a broken chain or a duplicate."""
    kind = entry.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    missing = [k for k in REQUIRED[kind] if not entry.get(k)]
    if missing:
        raise ValueError(f"missing fields for a {kind}: {missing}")
    ok, msg = verify(path)
    if not ok:
        raise ValueError(f"refusing to append: {msg}")
    existing = read(path)
    if kind == "protocol" and not isinstance(entry.get("rules"), list):
        raise ValueError("a protocol's rules must be a list of sentences")
    if kind != "verdict" and any(e["id"] == entry["id"] and e["kind"] != "verdict" for e in existing):
        raise ValueError(f"{entry['id']} is already in the ledger (resolve it with a verdict, never re-freeze)")
    if kind == "verdict" and not any(e["id"] == entry["id"] and e["kind"] == "hypothesis" for e in existing):
        raise ValueError(f"no frozen hypothesis {entry['id']} to give a verdict on")
    body = dict(entry)
    # The UTC date only, never the clock time (project rule: public files carry no times of the author's activity;
    # entries frozen before 2026-10-24 keep the times they were frozen with). Code that needs an instant reads a bare
    # date as the end of that UTC day, so nothing taken before a freeze ever counts under it (verdicts.counted_listed).
    body["recorded_utc"] = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    prev = existing[-1]["row_hash"] if existing else GENESIS
    full = {**body, "prev_hash": prev, "row_hash": _hash(prev, body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Match the line ending already in the file (a ledger first written on Windows uses CRLF); a new file uses LF.
    eol = "\r\n" if path.exists() and b"\r\n" in path.read_bytes() else "\n"
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(full, ensure_ascii=False, sort_keys=True) + eol)
    return full


def render(path=DEFAULT_PATH):
    """Markdown view of the ledger (for forward/HYPOTHESES.md)."""
    entries = read(path)
    ok, msg = verify(path)
    verdicts = {}
    for e in entries:
        if e["kind"] == "verdict":
            verdicts.setdefault(e["id"], []).append(e)
    L = ["# Pre-registered predictions (forward test)", "",
         "Generated from `forward/predictions.jsonl` by `definedtarget.ledger.render()`; do not edit by hand.",
         f"Chain check: **{msg}** ({len(entries)} entries). README.md, \"Check the freeze\", shows how to verify "
         "that the entries existed before the data that tests them.", ""]
    for e in entries:
        if e["kind"] == "protocol":
            L += [f"## Protocol {e['id']} (frozen {e['recorded_utc'][:10]})", ""] + [f"- {r}" for r in e["rules"]] + [""]
    hyps = [e for e in entries if e["kind"] == "hypothesis"]
    L += ["## Hypotheses", "", "| ID | Frozen (UTC) | Claim | Review on | Status |", "|---|---|---|---|---|"]
    for e in hyps:
        status = "; ".join(v["result"] for v in verdicts.get(e["id"], [])) or "open"
        L.append(f"| {e['id']} | {e['recorded_utc'][:10]} | {e['claim']} | {e['review_on']} | {status} |")
    for e in hyps:
        L += ["", f"### {e['id']}: {e['claim']}", "", f"- **Test:** {e['test']}", f"- **Pass if:** {e['pass_if']}"]
        if e.get("if_fail_means"):
            L.append(f"- **If it fails:** {e['if_fail_means']}")
        if e.get("exploratory"):
            L.append(f"- **Before the freeze (shaped the band; tests nothing):** {e['exploratory']}")
        L.append(f"- **Review on:** {e['review_on']}")
    parked = [e for e in entries if e["kind"] == "parked"]
    if parked:
        L += ["", "## Parked (written down, not yet testable)", "", "| ID | Recorded | Claim | Why parked | Better test |",
              "|---|---|---|---|---|"]
        for e in parked:
            L.append(f"| {e['id']} | {e['recorded_utc'][:10]} | {e['claim']} | {e['why_parked']} | {e['better_test']} |")
    if verdicts:
        L += ["", "## Verdicts", ""]
        for e in entries:
            if e["kind"] == "verdict":
                L.append(f"- **{e['id']}** ({e['recorded_utc'][:10]}): {e['result']}. {e['evidence']}")
    return "\n".join(L) + "\n"
