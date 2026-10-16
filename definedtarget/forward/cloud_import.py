"""Bring the cloud routines' captures (a local clone of the private capture repository) into the local data folder.

The cloud routines write into the capture repository:
- `roundhill/`: a data root of its own (`raw/manifest.jsonl` plus `raw/<source>/<UTC date>/<file>`), written by
  `roundhill.capture_all` and `cboe_flex.capture_all` exactly as on this computer.
- `tradingview/spx_dec2029_calls/<read time UTC>.json`: each TradingView reply, unchanged.

`import_files` copies every file the cloud saved into the local data root and appends its manifest record (marked
"via": "cloud"); `import_quotes` files each reply through `listed.store`, with the read time from its file name, so
protocol P2's window check is the same one used locally. Both are safe to run again: nothing is copied twice and
nothing local is overwritten.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import fetch, listed

CLOUD_DATA = "roundhill"                                   # the cloud routines' data root inside the repository
QUOTES_DIR = Path("tradingview") / "spx_dec2029_calls"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_files(capture_repo, root=None):
    """Copy files the cloud saved into the local data root. Returns counts: imported, already, refused."""
    cloud_root = Path(capture_repo) / CLOUD_DATA
    root = fetch.data_dir(root)
    local = fetch.read_manifest(root)
    seen = {(r["url"], r["fetched_utc"]) for r in local}
    counts = {"imported": 0, "already": 0, "refused": 0}
    for rec in fetch.read_manifest(cloud_root):
        if rec.get("result") != "saved" or "path" not in rec:
            continue                                       # errors and "unchanged" add no data
        if (rec["url"], rec["fetched_utc"]) in seen:
            counts["already"] += 1
            continue
        src = cloud_root / rec["path"]
        if not src.exists() or _sha256(src) != rec.get("sha256"):
            counts["refused"] += 1                         # missing or altered: never import it
            continue
        dst_rel = Path(rec["path"])
        dst = root / dst_rel
        if dst.exists() and _sha256(dst) != rec["sha256"]:  # a different local copy: keep both
            dst_rel = dst_rel.with_name(f"{dst_rel.stem}__cloud{dst_rel.suffix}")
            dst = root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
        fetch._append_manifest(root, {**rec, "path": dst_rel.as_posix(), "via": "cloud"})
        seen.add((rec["url"], rec["fetched_utc"]))
        counts["imported"] += 1
    return counts


def import_quotes(capture_repo, root=None):
    """File each cloud TradingView reply through listed.store. Returns a list of store() summaries for new ones."""
    out = []
    for path in sorted((Path(capture_repo) / QUOTES_DIR).glob("*.json")):
        read_utc = datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        try:
            s = listed.store(path.read_text(encoding="utf-8"), read_utc, root=root)
        except (ValueError, json.JSONDecodeError) as err:
            out.append({"file": path.name, "refused": str(err)})
            continue
        if s["new"]:
            out.append(s)
    return out
