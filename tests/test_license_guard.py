"""License guard: nothing that git would publish may hold issuer files, market data or personal identifiers.

The data licenses (Roundhill's site terms, TradingView's display-only license) forbid publishing the raw files or
daily series built from them. This test runs scripts/scan_identifiers.py's checks inside the test suite, so a data
file or a pasted daily series fails the tests before it can reach a push.
"""
import importlib.util
import tempfile
from pathlib import Path

from _harness import ROOT, check

spec = importlib.util.spec_from_file_location("scan_identifiers", ROOT / "scripts" / "scan_identifiers.py")
scan_ids = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan_ids)

# 1. The .gitignore keeps every data location and private document out of the repository.
pats = scan_ids.ignore_patterns()
for rel in ("data/raw/roundhill/2026-10-14/xx_daily.csv", "data/tables/xx_state.csv",
            "data/external/listed_spx_dec2029_2026-10-13.csv", "forward/daily/2026-10-13_report.md",
            "results/raw/run.csv", "tests/realdata_local.py", "PLAN.md", "PROMPT.md", "BACKLOG.md",
            "identifiers_local.txt", "docs/README_DRAFT.md", "forward/hypotheses_draft.json",
            "definedtarget.egg-info/PKG-INFO", "build/lib/definedtarget/pricing.py", "dist/x.whl", ".venv/x.py",
            ".env", "definedtarget/__pycache__/x.pyc"):
    check(f".gitignore keeps {rel} private", scan_ids.ignored(rel, pats))
for rel in ("data/README.md", "README.md", "forward/HYPOTHESES.md", "forward/predictions.jsonl",
            "forward/predictions_2026-10-14.jsonl.ots", "definedtarget/funds/xx.json", "LICENSE", ".gitattributes"):
    check(f".gitignore publishes {rel}", not scan_ids.ignored(rel, pats))

# 1b. The matcher follows git's rules (audit A2): a pattern with a slash is anchored to the repository root, '*' never
#     crosses a folder, and a file inside an ignored folder cannot be re-included.
rules = ["data/*", "!data/README.md", "forward/daily/", "tests/*_local.py", "*.egg-info/", "keep/", "!keep/x.md"]
for rel, want in (("data/README.md", False), ("data/raw/x.csv", True), ("a/forward/daily/x.csv", False),
                  ("forward/daily/x.md", True), ("tests/sub/x_local.py", False), ("tests/x_local.py", True),
                  ("pkg.egg-info/PKG-INFO", True), ("sub/pkg.egg-info/PKG-INFO", True), ("keep/x.md", True)):
    check(f"git rule: {rel} is {'ignored' if want else 'published'}", scan_ids.ignored(rel, rules) == want)

# 2. The daily-series detector: 5 or more dated rows with numbers is a series; prose dates are not.
series = "\n".join(f"| 2026-10-{d:02d} | 26.{d:02d} | 778.{d:02d} |" for d in range(1, 13))
check("12 dated numeric rows count as a daily series", scan_ids.daily_series_rows(series) == 12)
prose = "Review on 2027-04-14.\n2026-10-13: one note with 1.5 and 2.5 in it\n| 2026-10-13 | only 1.0 |"
check("dates in prose or rows with one number do not count", scan_ids.daily_series_rows(prose) == 0,
      scan_ids.daily_series_rows(prose))

# 2b. The British-spelling check (public style rule): the plural noun "analyses" is American too and passes.
brit = scan_ids.GENERIC["British spelling"]
for stem, end in (("analy", "se"), ("analy", "sed"), ("analy", "sing"), ("Analy", "ser"), ("behavio", "ur"),
                  ("mode", "lled"), ("labe", "lled"), ("colo", "urs")):  # split, so this file passes the scan itself
    check(f"British spelling '{stem}{end}' is caught", brit.search(f"the {stem}{end} here") is not None)
for word in ("analyses of the saved forecasts", "analyzes", "analyzed", "analysis", "behavior", "modeled", "center"):
    check(f"'{word}' passes", brit.search(f"the {word} here") is None)

# 2c. The forecast-check log (FC1) and run manifests are published and pass, even after 39 months of entries; a JSON
#     file elsewhere still fails, and their contents are still scanned.
from datetime import datetime, timezone  # noqa: E402

from definedtarget.forward import forecast as fc  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "forecast_log.jsonl"
    when = datetime(2030, 2, 1, tzinfo=timezone.utc)
    for t, T in fc.origins():
        rec = {"origin": t.isoformat(), "target": T.isoformat(), "seed": fc.seed_for(t), "paths": fc.PATHS,
               "models": {m: {} for m in fc.MODELS}}
        fc.append_log(fc.forecast_entry(rec, code={"definedtarget/forward/forecast.py": "0" * 64}), log, now=when)
        pits = {m: {"xx": 0.1234, "spy": 0.5678} for m in fc.MODELS}
        fc.append_log(fc.outcome_entry(rec, pits), log, now=when)
    man = Path(tmp) / "m.json"
    man.write_text('{"kind": "run manifest", "seeds": [20261017], "date_utc": "2026-10-18"}\n', encoding="utf-8")
    leak = Path(tmp) / "leak.json"
    leak.write_text('{"input": "C:\\\\Users\\\\someone\\\\x.csv"}\n', encoding="utf-8")
    probs = scan_ids.scan([("forward/forecast_log.jsonl", log), ("results/manifests/m.json", man),
                           ("results/manifests/leak.json", leak), ("results/other.json", man)], handle="", url="")
    check("a full 39-month forecast log passes the scan", not any(p.startswith("forward/forecast_log") for p in probs),
          probs[:3])
    check("a run manifest passes the scan", not any(p.startswith("results/manifests/m.json") for p in probs), probs)
    check("a manifest's contents are still scanned (a home path fails)",
          any(p.startswith("results/manifests/leak.json") for p in probs), probs)
    check("a JSON file outside results/manifests/ still fails", any(p.startswith("results/other.json") for p in probs),
          probs)

# 2d. The allowed JSON places accept only the shapes the tools write (audit of v0.3.0, D12): a private forecast or a
#     price series saved there fails.
import json as _json  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    priv = Path(tmp) / "private.json"
    priv.write_text(_json.dumps({"protocol": "FC1", "origin": "2026-10-30", "inputs": {"spot": 778.01, "nav": 26.51},
                                 "models": {"M1": {"growth_levels": [0.9, 1.0, 1.1]}}}), encoding="utf-8")
    prices = Path(tmp) / "prices.json"
    prices.write_text(_json.dumps([{"date": f"2026-10-{d:02d}", "close": 100 + d / 10} for d in range(1, 21)]),
                      encoding="utf-8")
    probs = scan_ids.scan([("results/manifests/private.json", priv), ("results/manifests/prices.json", prices),
                           ("forward/forecast_log.jsonl", priv)], handle="", url="")
    for rel in ("results/manifests/private.json", "results/manifests/prices.json", "forward/forecast_log.jsonl"):
        check(f"D12: a private forecast or price series as {rel} fails the scan", any(p.startswith(rel) for p in probs), probs)

# 2e. The package metadata parses as the README's `pip install .` needs (audit of v0.3.0, D20)
import tomllib  # noqa: E402

meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
check("D20: keywords and classifiers sit under [project]; the only optional extra is 'flex'",
      "monte-carlo" in meta.get("keywords", []) and any("Python :: 3" in c for c in meta.get("classifiers", []))
      and set(meta.get("optional-dependencies", {})) == {"flex"}, meta.get("optional-dependencies"))

# 3. The scan refuses a data file and a pasted series (synthetic files in a temporary folder).
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "prices.csv"
    csv.write_text("date,close\n2026-10-13,778.01\n", encoding="utf-8")
    md = Path(tmp) / "notes.md"
    md.write_text(series, encoding="utf-8")
    five = "\n".join(series.splitlines()[:5])
    (Path(tmp) / "five.md").write_text(five, encoding="utf-8")
    (Path(tmp) / "shot.png").write_bytes(b"\x89PNG")
    (Path(tmp) / "data.json").write_text("{}", encoding="utf-8")
    probs = scan_ids.scan([("notes/prices.csv", csv), ("notes/notes.md", md), ("notes/five.md", Path(tmp) / "five.md"),
                           ("notes/shot.png", Path(tmp) / "shot.png"), ("notes/data.json", Path(tmp) / "data.json")],
                          handle="", url="")
    check("the scan refuses a CSV outside the allowed folders", any(p.startswith("notes/prices.csv") for p in probs),
          probs)
    check("the scan refuses a pasted daily series", any(p.startswith("notes/notes.md") for p in probs), probs)
    check("the scan refuses as few as 5 dated numeric rows", any(p.startswith("notes/five.md") for p in probs), probs)
    check("the scan refuses an image outside docs/img/", any(p.startswith("notes/shot.png") for p in probs), probs)
    check("the scan refuses a JSON file outside the fund specs", any(p.startswith("notes/data.json") for p in probs),
          probs)

# 3b. Attribution: the handle may stand only right after "Copyright (c) YYYY " (LICENSE), after "- name:" (CITATION.cff)
#     and after "> " (the README citation line), and the exact repository URL may appear in CITATION.cff and README.
#     "zzhandle" and "zzcity" stand in for private terms (audit A2's bypass cases included).
fake = [scan_ids.re.compile("zzhandle"), scan_ids.re.compile("zzcity")]
URL = "https://github.com/zzhandle/tool"
with tempfile.TemporaryDirectory() as tmp:
    lic, readme, cff = Path(tmp) / "LICENSE", Path(tmp) / "README.md", Path(tmp) / "CITATION.cff"
    lic.write_text("Copyright (c) 2026 zzhandle and contributors\n", encoding="utf-8")
    cff.write_text('authors:\n  - name: "zzhandle"\nrepository-code: "https://github.com/zzhandle/tool"\n',
                   encoding="utf-8")
    readme.write_text("> zzhandle (2026). *tool*. https://github.com/zzhandle/tool\n"
                      "see https://github.com/zzhandle/tool from zzcity\n"
                      "written by zzhandle\n"
                      "> zzcity wrote this quote\n"
                      "mirror at https://github.com/zzcity/tool\n", encoding="utf-8")
    probs = scan_ids.scan([("LICENSE", lic), ("README.md", readme), ("CITATION.cff", cff)], personal=fake,
                          handle="zzhandle", url=URL)
    check("the handle passes in the copyright line, the citation line, CITATION.cff and the repository URL",
          not any(p.startswith(("LICENSE", "CITATION.cff", "README.md:1:")) for p in probs), probs)
    for line, why in ((2, "another private term on a URL line"), (3, "the handle anywhere else"),
                      (4, "another private term after '> '"), (5, "another private term in a different URL")):
        check(f"{why} still fails", any(p.startswith(f"README.md:{line}:") for p in probs), probs)
# 3c. The simulator's GitHub Pages address (https://<owner>.github.io/<repo>/...) may appear in README.md and
#     web/index.html only; anywhere else, or another owner's Pages address, still fails.
check("the Pages address is worked out from the repository URL (owner lower-cased)",
      scan_ids.pages_prefix("https://github.com/ZZhandle/tool") == "https://zzhandle.github.io/tool/")
with tempfile.TemporaryDirectory() as tmp:
    readme, page, other = Path(tmp) / "README.md", Path(tmp) / "index.html", Path(tmp) / "notes.md"
    readme.write_text("Try it: https://zzhandle.github.io/tool/web/\n"
                      "Fork: https://zzhandle.github.io/othertool/web/\n", encoding="utf-8")
    page.write_text('<meta property="og:image" content="https://zzhandle.github.io/tool/docs/img/card.png">\n'
                    '<meta name="author" content="zzhandle">\n', encoding="utf-8")
    other.write_text("see https://zzhandle.github.io/tool/web/\n", encoding="utf-8")
    probs = scan_ids.scan([("README.md", readme), ("web/index.html", page), ("notes/notes.md", other)],
                          personal=fake, handle="zzhandle", url=URL)
    check("the simulator's own Pages address passes in README.md and web/index.html",
          not any(p.startswith(("README.md:1:", "web/index.html:1:")) for p in probs), probs)
    check("a Pages address for a different repository still fails", any(p.startswith("README.md:2:") for p in probs),
          probs)
    check("the handle elsewhere in web/index.html still fails", any(p.startswith("web/index.html:2:") for p in probs),
          probs)
    check("the Pages address in any other file still fails", any(p.startswith("notes/notes.md") for p in probs), probs)
check("the scan reads the handle and the URL from the repository's CITATION.cff",
      all(scan_ids.attribution()), scan_ids.attribution())

# 4. The real repository: zero problems (personal terms too, when the private list exists on this machine).
personal = scan_ids.personal_terms() or []
problems = scan_ids.scan(personal=personal)
check(f"every publishable file passes the scan ({'with' if personal else 'without'} the private term list)",
      not problems, "; ".join(problems[:5]))
