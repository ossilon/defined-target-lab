"""Hash-chained prediction ledger: appends, verifies, and catches any later edit."""
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from _harness import check, raises

from definedtarget import ledger

T = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)
with tempfile.TemporaryDirectory() as tmp:
    q = Path(tmp) / "protocol.jsonl"
    pr = ledger.append({"kind": "protocol", "id": "P1", "rules": ["rule one", "rule two"]}, q, now=T)
    check("a protocol entry is accepted and rendered at the top", "## Protocol P1" in ledger.render(q)
          and "- rule two" in ledger.render(q))
    raises("a protocol with rules as plain text (not a list) is refused",
           lambda: ledger.append({"kind": "protocol", "id": "P2", "rules": "one long string"}, q), "list of sentences")
    check("a new entry records the UTC date only, never the clock time (rule from 2026-10-24)",
          pr["recorded_utc"] == "2026-10-14" and "09:00" not in q.read_text(encoding="utf-8"), pr["recorded_utc"])

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "predictions.jsonl"
    h = {"kind": "hypothesis", "id": "H99", "claim": "c", "test": "t", "pass_if": "x", "review_on": "2027-01-01"}
    e1 = ledger.append(h, p, now=T)
    check("first entry links to the all-zero genesis hash", e1["prev_hash"] == ledger.GENESIS)
    e2 = ledger.append({"kind": "parked", "id": "H98", "claim": "c", "why_parked": "w", "better_test": "b"}, p, now=T)
    check("second entry links to the first entry's hash", e2["prev_hash"] == e1["row_hash"])
    check("an untouched ledger verifies", ledger.verify(p) == (True, "chain intact"))
    raises("re-freezing the same hypothesis is refused", lambda: ledger.append(h, p, now=T), "already in the ledger")
    raises("a verdict on an unknown hypothesis is refused",
           lambda: ledger.append({"kind": "verdict", "id": "H1", "result": "r", "evidence": "e"}, p), "no frozen")
    raises("a hypothesis without a pass rule is refused",
           lambda: ledger.append({"kind": "hypothesis", "id": "H97", "claim": "c", "test": "t", "review_on": "d"}, p),
           "missing fields")
    v = ledger.append({"kind": "verdict", "id": "H99", "result": "holds", "evidence": "e"}, p, now=T)
    check("a verdict is appended, never edited in", v["prev_hash"] == e2["row_hash"])
    md = ledger.render(p)
    check("rendered table shows the verdict as the status", "| holds |" in md and "chain intact" in md)
    # tamper with the first line
    lines = p.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["claim"] = "edited later"
    lines[0] = json.dumps(first, sort_keys=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, msg = ledger.verify(p)
    check("editing an earlier entry is detected", not ok and "line 1" in msg, msg)
    raises("nothing can be appended to a tampered ledger",
           lambda: ledger.append({"kind": "parked", "id": "H96", "claim": "c", "why_parked": "w", "better_test": "b"}, p),
           "refusing to append")

    md = ledger.render(p.parent / "none.jsonl")
    check("an empty ledger renders without error", "0 entries" in md)

# Appends keep the file's line endings, so a timestamp proof of the earlier bytes stays valid.
with tempfile.TemporaryDirectory() as tmp:
    crlf, lf = Path(tmp) / "crlf.jsonl", Path(tmp) / "lf.jsonl"
    ledger.append({"kind": "protocol", "id": "P1", "rules": ["r"]}, lf, now=T)
    ledger.append({"kind": "protocol", "id": "P2", "rules": ["r"]}, lf, now=T)
    check("a new ledger uses LF line endings", b"\r\n" not in lf.read_bytes() and lf.read_bytes().count(b"\n") == 2)
    crlf.write_bytes(lf.read_bytes().split(b"\n")[0] + b"\r\n")           # one CRLF line, as written on Windows
    before = crlf.read_bytes()
    ledger.append({"kind": "protocol", "id": "P2", "rules": ["r"]}, crlf, now=T)
    after = crlf.read_bytes()
    check("an append keeps a CRLF ledger's earlier bytes and its CRLF ending",
          after.startswith(before) and after.endswith(b"\r\n") and after.count(b"\r\n") == 2)

# The frozen entries (2026-10-14). OpenTimestamps stamped the SHA-256 of exactly these 21 lines, as bytes
# (forward/predictions_2026-10-14.jsonl.ots). Later entries append below them; these bytes must never change.
FROZEN_LINES = 21
FROZEN_SHA256 = "bc2b73a352bb46ebebf184ee4fdbbb13a7576c22fbba259e897d9c96d0e8888b"
FROZEN_HEAD = "bdc033f7fa542609daeba64b7c2e3adb616f1ccdf589af8bf3ef0360f90dfeb2"
real = ledger.DEFAULT_PATH
if real.exists():
    check("the project's prediction ledger verifies", ledger.verify(real)[0], ledger.verify(real)[1])
    raw = real.read_bytes().splitlines(keepends=True)
    frozen = b"".join(raw[:FROZEN_LINES])
    check(f"the ledger still holds the {FROZEN_LINES} frozen entries", len(raw) >= FROZEN_LINES, len(raw))
    check("the frozen entries' bytes match the timestamped SHA-256",
          hashlib.sha256(frozen).hexdigest() == FROZEN_SHA256, hashlib.sha256(frozen).hexdigest())
    check("the frozen entries end at the pinned head hash", ledger.read(real)[FROZEN_LINES - 1]["row_hash"] == FROZEN_HEAD)
    # Protocol P2 (clarifications), appended 2026-10-15 before any test-period close was captured; a second
    # OpenTimestamps proof covers the first 22 lines (forward/predictions_2026-10-15.jsonl.ots).
    p2 = hashlib.sha256(b"".join(raw[:22])).hexdigest()
    check("the first 22 lines (P1, 20 hypotheses, P2) match their pinned SHA-256",
          p2 == "a213dfabc265a43a3a88e8b5e96529dc402df60675fb8316d22b6127d45f339e", p2)
