"""web/golden.js must match a fresh run of scripts/make_golden.py within each case's tolerance (so the browser
self-test checks today's engine, on any platform),
and the browser files stay small and self-contained (no outside scripts, fonts or images)."""
import importlib.util
import json
import re
from pathlib import Path

from _harness import check

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_golden", REPO / "scripts" / "make_golden.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

doc = mg.build()
on_disk = (REPO / "web" / "golden.js").read_text(encoding="utf-8")


def parse_golden(text):
    body = text.split("window.GOLDEN = ", 1)[1].rstrip().rstrip(";")
    return json.loads(body)


def close(a, b, tol):
    """Same structure; numbers within abs + rel x |b| (the case's own tolerance). Another platform's libm can move
    the last digits of a fresh run (external audit A14), so exact text equality is too strict."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol["abs"] + tol["rel"] * abs(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    return False


def matches(disk, fresh):
    """(ok, first difference): the stored golden file against a fresh run of make_golden.py."""
    for key in ("about", "tol", "objects"):
        if disk[key] != fresh[key]:
            return False, key
    if len(disk["cases"]) != len(fresh["cases"]):
        return False, f"{len(disk['cases'])} cases on disk, {len(fresh['cases'])} fresh"
    for i, (a, b) in enumerate(zip(disk["cases"], fresh["cases"])):
        tol = b.get("tol", fresh["tol"])
        same = a.get("fn") == b.get("fn") and a.get("tol") == b.get("tol") and a.get("note") == b.get("note")
        if not (same and close(a["args"], b["args"], fresh["tol"]) and close(a["expect"], b["expect"], tol)):
            return False, f"case {i}: {b.get('fn')} {b.get('note', '')}"
    return True, ""


fresh = json.loads(json.dumps(doc))                     # the same JSON types the file holds
ok, where = matches(parse_golden(on_disk), fresh)
check("web/golden.js matches a fresh run of make_golden.py within each case's tolerance (re-run it after any "
      "engine change)", ok, where)
nudged = json.loads(json.dumps(fresh))
for c in nudged["cases"]:
    if isinstance(c["expect"], float) and c["expect"] != 0.0:
        c["expect"] *= 1 + 1e-13                         # a last-digits platform difference
check("a last-digits difference inside the tolerance still matches (another platform's libm)",
      matches(nudged, fresh)[0])
moved = json.loads(json.dumps(fresh))
first = next(c for c in moved["cases"] if isinstance(c["expect"], float) and abs(c["expect"]) > 1.0)
first["expect"] *= 1.001
check("a real change beyond the tolerance is still caught", not matches(moved, fresh)[0])
check("the golden file covers every engine function the page uses",
      {c["fn"] for c in doc["cases"]} >= {"callPrice", "impliedVol", "shareValue", "shareDelta", "leverage", "solveVol",
                                           "oneDay", "outcomeRow", "levelForValue", "volAt", "probAbove", "quantile",
                                           "normInv", "erfc"})
check("several hundred cases", len(doc["cases"]) > 500, len(doc["cases"]))

engine = (REPO / "web" / "engine.js").read_text(encoding="utf-8")
exported = set(re.findall(r"(\w+): \w+[,\s]", engine.split("var api = {", 1)[1].split("};", 1)[0]))
check("every function the golden file calls is exported by engine.js",
      {c["fn"] for c in doc["cases"]} <= exported, {c["fn"] for c in doc["cases"]} - exported)

for page in sorted((REPO / "web").glob("*.html")) + sorted((REPO / "web").glob("*.js")):
    text = page.read_text(encoding="utf-8")
    outside = re.findall(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", text) + re.findall(r"@import|url\(\s*['\"]?http", text)
    check(f"{page.name} loads nothing from other sites", not outside, outside)
check("engine.js stays small (< 40 KB)", len(engine.encode()) < 40_000, len(engine.encode()))
tags = {name: set(re.findall(r'src="(?:engine|app|golden)\.js\?v=([^"]+)"', (REPO / "web" / name).read_text(encoding="utf-8")))
        for name in ("index.html", "selftest.html")}
check("both pages load the scripts with one shared version tag (so browsers never mix an old engine with new pages)",
      len(tags["index.html"] | tags["selftest.html"]) == 1, tags)
# Load speed is decided by the bytes sent, and web servers (GitHub Pages included) send these files compressed. So the
# guard is the compressed size; the looser cap on the stored size stops bloat while keeping the code commented and
# readable (changed 2026-10-17 from "stored < 60 KB", deliberately: the page reached 60.8 KB stored, 20.6 KB sent).
import gzip  # noqa: E402
page_files = [(REPO / "web" / f).read_bytes() for f in ("index.html", "app.js", "engine.js")]
sent = sum(len(gzip.compress(b, 9)) for b in page_files)
check("the simulator page sends under 25 KB compressed (index.html + app.js + engine.js)", sent < 25_000, sent)
check("... and stays under 75 KB stored", sum(len(b) for b in page_files) < 75_000, sum(len(b) for b in page_files))
app = (REPO / "web" / "app.js").read_text(encoding="utf-8")
check("the page sends nothing anywhere: no fetch, XMLHttpRequest, beacons, storage, image or script loads, "
      "navigation, forms or dynamic imports in app.js",
      not re.search(r"\bfetch\(|XMLHttpRequest|sendBeacon|localStorage|sessionStorage|indexedDB|document\.cookie|"
                    r"WebSocket|EventSource|new Image|\.src\s*=|location\.(href|assign|replace)\s*[=(]|window\.open|"
                    r"<form|\bimport\(|createElement\(\s*['\"](script|img|iframe|link)", app))
used = set(re.findall(r"\bE\.(\w+)\(", app))
check("every engine function app.js calls has its own cases in the golden file",
      used <= {c["fn"] for c in doc["cases"]}, used - {c["fn"] for c in doc["cases"]})
for name in ("app.js", "engine.js", "index.html", "selftest.html"):
    raw = (REPO / "web" / name).read_bytes()
    check(f"{name} is pure ASCII (box and block characters written as escapes, so no encoding can garble them)",
          all(b < 128 for b in raw), sorted({b for b in raw if b >= 128})[:5])
