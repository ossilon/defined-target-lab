"""Importing cloud captures (a clone of the private capture repository) into the local data folder.

Synthetic files only. Checks: files copied once with their manifest records; a different local copy is kept, not
overwritten; an altered cloud file is refused; "latest" follows download time even when imports arrive out of order;
TradingView replies are filed through listed.store with the read time from the file name.
"""
import hashlib
import json
import tempfile
from pathlib import Path

from _harness import check

from definedtarget.forward import cloud_import, fetch, listed, parse


def put(root, rel, body):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return {"sha256": hashlib.sha256(body).hexdigest(), "path": rel, "bytes": len(body)}


def manifest(root, recs):
    p = Path(root) / "raw" / "manifest.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


with tempfile.TemporaryDirectory() as tmp:
    clone, local = Path(tmp) / "research-captures", Path(tmp) / "data"
    (clone / ".git").mkdir(parents=True)
    cloud = clone / "roundhill"
    a = put(cloud, "raw/roundhill/2026-10-17/dailynav.csv", b"Fund Name,Fund Ticker,CUSIP,Net Assets\ncloud,XX\n")
    b = put(cloud, "raw/roundhill/2026-10-17/xx_daily.csv", b"cloud version\n")
    c = put(cloud, "raw/roundhill/2026-10-17/bidask_xx.json", b"{}")
    recs = [
        {"url": "u/dailynav", "name": "dailynav.csv", "source": "roundhill", "fetched_utc": "2026-10-17T04:00:00Z",
         "result": "saved", **a},
        {"url": "u/xx_daily", "name": "xx_daily.csv", "source": "roundhill", "fetched_utc": "2026-10-17T03:00:00Z",
         "result": "saved", **b},
        {"url": "u/bidask", "name": "bidask_xx.json", "source": "roundhill", "fetched_utc": "2026-10-17T04:00:00Z",
         "result": "saved", **c},
        {"url": "u/x", "name": "x.csv", "source": "roundhill", "fetched_utc": "2026-10-17T04:00:00Z",
         "result": "network_error"},
    ]
    (cloud / c["path"]).write_bytes(b"{tampered}")              # the file no longer matches its fingerprint
    manifest(cloud, recs)
    # Local: a newer, different xx_daily.csv already saved under the same path.
    d = put(local, "raw/roundhill/2026-10-17/xx_daily.csv", b"local newer version\n")
    manifest(local, [{"url": "u/xx_daily", "name": "xx_daily.csv", "source": "roundhill",
                      "fetched_utc": "2026-10-17T05:00:00Z", "result": "saved", **d}])

    n = cloud_import.import_files(clone, root=local)
    check("saved cloud files are imported; an altered one is refused; errors are skipped",
          n == {"imported": 2, "already": 0, "refused": 1}, n)
    check("a cloud file lands at the same path", (local / a["path"]).read_bytes().startswith(b"Fund Name"))
    check("a different local copy is kept, and the cloud copy sits beside it",
          (local / d["path"]).read_bytes() == b"local newer version\n"
          and (local / "raw/roundhill/2026-10-17/xx_daily__cloud.csv").read_bytes() == b"cloud version\n")
    check("imported records are marked as coming from the cloud",
          sum(1 for r in fetch.read_manifest(local) if r.get("via") == "cloud") == 2)
    latest = parse.saved_versions(local, "xx_daily")[-1]
    check("'latest' is the most recent download, even though the cloud record was appended later",
          latest["fetched_utc"] == "2026-10-17T05:00:00Z", latest)
    again = cloud_import.import_files(clone, root=local)
    check("a second import copies nothing", again == {"imported": 0, "already": 2, "refused": 1}, again)

    # TradingView replies: filed through listed.store with the file name's read time.
    reply = json.dumps({"success": True, "data": {listed.option_symbol(10000): {"bid": 530.0, "ask": 590.0},
                                                  "SP:SPX": {"close": 7800.0}, "AMEX:SPY": {"close": 777.0}}})
    qdir = clone / "tradingview" / "spx_dec2029_calls"
    qdir.mkdir(parents=True)
    (qdir / "20261016T213356Z.json").write_text(reply, encoding="utf-8")   # Fri 17:33 New York: inside
    q = cloud_import.import_quotes(clone, root=local)
    check("a cloud reply is filed with its own read time and P2's window check",
          len(q) == 1 and str(q[0]["close"]) == "2026-10-16" and q[0]["in_window"], q)
    check("importing the same reply again files nothing new", cloud_import.import_quotes(clone, root=local) == [])
