"""docs/MODELS.md must not drift from the code (BACKLOG B34): every file path and every module, function or class
it names in backticks must exist.

Names are resolved inside definedtarget.volmodel and definedtarget.montecarlo (`pricers.prices`, `montecarlo.paths`,
`montecarlo.paths.lognormal`); a call written with arguments, such as `pricers.prices(..., check=True)`, is checked by
its name. Paths are relative to the repository. Names from other libraries are listed in EXTERNAL.
"""
import importlib
import re
from pathlib import Path

from _harness import check

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "MODELS.md"
EXTERNAL = {"least_squares"}                      # scipy.optimize.least_squares, named in the text as scipy's


def resolve(name):
    """True if a dotted name exists as a module or attribute under definedtarget.volmodel or definedtarget."""
    parts = name.split(".")
    for base in ("definedtarget.volmodel", "definedtarget"):
        for cut in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join([base] + parts[:cut]))
            except ImportError:
                continue
            try:
                for attr in parts[cut:]:
                    obj = getattr(obj, attr)
                return True
            except AttributeError:
                break
    return False


text = DOC.read_text(encoding="utf-8")
text = re.sub(r"```.*?```", "", text, flags=re.S)                 # math and code blocks are not names
spans = sorted(set(re.findall(r"`([^`]+)`", text)))
paths, names, unknown = [], [], []
for s in spans:
    s = s.strip()
    if "/" in s:
        paths.append(s)
    elif re.fullmatch(r"[A-Za-z_][\w.]*(\(.*\))?", s):
        names.append(re.sub(r"\(.*\)$", "", s))
for p in paths:
    if not (ROOT / p.rstrip("/")).exists():
        unknown.append(p)
for n in names:
    if n not in EXTERNAL and not resolve(n):
        unknown.append(n)
check(f"docs/MODELS.md names only files and code that exist ({len(paths)} paths, {len(names)} names)",
      not unknown and len(paths) + len(names) > 10, unknown)
