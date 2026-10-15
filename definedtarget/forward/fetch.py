"""Capture an issuer's public fund files exactly as served. Standard library only.

What it does, in plain words
- Downloads each file once, with a polite pause between requests and a clear user agent.
- Saves the raw bytes UNCHANGED under  <data>/raw/<source>/<UTC fetch date>/<file name>.
- Records every fetch, including failures, as one JSON line in  <data>/raw/manifest.jsonl  (append-only;
  the code never rewrites it). Each line holds the URL, the UTC fetch time, the HTTP status, the server's
  Last-Modified time, the SHA-256 fingerprint of the bytes, and where the code saved the bytes.
- If the bytes are identical to the last saved copy of the same URL, the manifest records the fetch as
  "unchanged" and the code saves nothing new. So running the capture twice is harmless (it is "idempotent").
- A file only counts as valid if it starts with the header we expect. The code saves a web page or error text
  served in place of a CSV for inspection but marks it invalid, so nobody can mistake it for data.

Why keep raw bytes: we re-derive every table we build later from these files, so we can always fix a parsing
bug and re-run, and nobody has to trust our tidy tables.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "defined-target-lab/0.1 (research; one request per file per day)"
PAUSE_SECONDS = 1.0                      # wait between requests to the same site

REPO = Path(__file__).resolve().parents[2]


def data_dir(path=None):
    """Data folder: the argument, else $DEFINEDTARGET_DATA, else <repo>/data (git-ignored)."""
    if path is not None:
        return Path(path)
    env = os.environ.get("DEFINEDTARGET_DATA")
    return Path(env) if env else REPO / "data"


def http_get(url, timeout=60):
    """Return (status, headers, body bytes). HTTP errors come back as their status code, not an exception."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers or {}), err.read() or b""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(root):
    """All manifest records, oldest first (empty list if none yet). root=None means the default data folder."""
    path = data_dir(root) / "raw" / "manifest.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append_manifest(root, record):
    path = Path(root) / "raw" / "manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def last_saved(records, url):
    """The most recent record for this URL whose bytes the code saved (status 'saved'), or None."""
    for rec in reversed(records):
        if rec["url"] == url and rec["result"] == "saved":
            return rec
    return None


def capture(url, name, source, expect_prefix, root=None, now=None, getter=http_get, records=None):
    """Fetch one URL and record it. Returns the manifest record (a dict).

    expect_prefix: bytes the body must start with to count as valid (e.g. b"Date,Account"), or None for any.
    result is one of: "saved" (new bytes, valid), "unchanged" (same bytes as last saved copy),
    "invalid" (saved for inspection, but not the expected content), "invalid_repeat" (the same invalid content
    again: recorded, not saved twice), "http_error", "network_error".
    """
    root = data_dir(root)
    now = now or datetime.now(timezone.utc)
    records = read_manifest(root) if records is None else records
    rec = {"url": url, "name": name, "source": source,
           "fetched_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        status, headers, body = getter(url)
    except Exception as err:  # noqa: BLE001 - the manifest records network failures; capture never raises them
        rec.update(result="network_error", error=f"{type(err).__name__}: {err}")
        _append_manifest(root, rec)
        return rec

    rec.update(status=status, bytes=len(body), sha256=sha256(body),
               last_modified=headers.get("Last-Modified"), content_type=headers.get("Content-Type"))
    if status != 200:
        rec["result"] = "http_error"
        _append_manifest(root, rec)
        return rec

    valid = expect_prefix is None or body.lstrip(b"\xef\xbb\xbf").startswith(expect_prefix)
    if not valid and any(r["url"] == url and r.get("sha256") == rec["sha256"] and r["result"] == "invalid"
                         for r in records):
        rec["result"] = "invalid_repeat"              # the same error page again: record it, don't re-save it
        _append_manifest(root, rec)
        return rec
    prev = last_saved(records, url)
    if valid and prev is not None and prev["sha256"] == rec["sha256"]:
        rec["result"] = "unchanged"
        rec["same_as"] = prev["path"]
        _append_manifest(root, rec)
        return rec

    day_dir = root / "raw" / source / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    target = day_dir / name
    if target.exists():                               # a second, different copy on the same day
        stem, dot, ext = name.rpartition(".")
        target = day_dir / (f"{stem}__{now.strftime('%H%M%S')}.{ext}" if dot else f"{name}__{now.strftime('%H%M%S')}")
    target.write_bytes(body)
    rec["path"] = target.relative_to(root).as_posix()
    rec["result"] = "saved" if valid else "invalid"
    _append_manifest(root, rec)
    return rec


def pause():
    time.sleep(PAUSE_SECONDS)
