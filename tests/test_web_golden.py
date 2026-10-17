"""web/golden.js must equal a fresh run of scripts/make_golden.py (so the browser self-test checks today's engine),
and the browser files stay small and self-contained (no outside scripts, fonts or images)."""
import importlib.util
import re
from pathlib import Path

from _harness import check

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_golden", REPO / "scripts" / "make_golden.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

doc = mg.build()
on_disk = (REPO / "web" / "golden.js").read_text(encoding="utf-8")
check("web/golden.js matches a fresh run of make_golden.py (re-run it after any engine change)",
      on_disk == mg.dumps(doc))
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
check("the page sends nothing anywhere: no fetch, XMLHttpRequest, beacons or storage in app.js",
      not re.search(r"\bfetch\(|XMLHttpRequest|sendBeacon|localStorage|sessionStorage|document\.cookie|WebSocket", app))
