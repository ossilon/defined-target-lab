"""web/golden.json must equal a fresh run of scripts/make_golden.py (so the browser self-test checks today's engine),
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
on_disk = (REPO / "web" / "golden.json").read_text(encoding="utf-8")
check("web/golden.json matches a fresh run of make_golden.py (re-run it after any engine change)",
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

for page in sorted((REPO / "web").glob("*.html")) + [REPO / "web" / "engine.js"]:
    text = page.read_text(encoding="utf-8")
    outside = re.findall(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", text) + re.findall(r"@import|url\(\s*['\"]?http", text)
    check(f"{page.name} loads nothing from other sites", not outside, outside)
check("engine.js stays small (< 40 KB)", len(engine.encode()) < 40_000, len(engine.encode()))
