"""Run manifests (definedtarget/montecarlo/manifest.py): what they record, what they never record, and the
reproducibility check they enable.

Judges: the output fingerprint equals SHA-256 of exactly what was printed; input and code fingerprints equal
hashlib's; two identical runs give the same output, a changed print does not; a run that fails writes nothing and
gives the screen back; privacy (file names only, the date only, no machine or user name).
"""
import hashlib
import json
import platform
import re
import sys
import tempfile
from pathlib import Path

from _harness import check, raises

from definedtarget.montecarlo import manifest

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "code").mkdir()
    (tmp / "code" / "a.py").write_bytes(b"x = 1\n")
    (tmp / "code" / "sub").mkdir()
    (tmp / "code" / "sub" / "b.py").write_bytes(b"y = 2\n")
    data = tmp / "inputs.csv"
    data.write_bytes(b"date,close\n2026-10-13,1.0\n")
    out = tmp / "manifests"

    def study(text):
        with manifest.run("demo", ["--input", str(data), "--paths", "1000"], out, tmp, ("code/**/*.py",),
                          seeds=[7], inputs={"prices": data}, settings={"paths": 1000}) as m:
            print(text, end="")
        return m

    real_stdout = sys.stdout
    m1 = study("line one\nline two\n")
    check("the screen is given back after the run", sys.stdout is real_stdout)
    rec = manifest.read(m1["path"])
    check("output fingerprint = SHA-256 of exactly what was printed",
          rec["output"] == {"sha256": hashlib.sha256(b"line one\nline two\n").hexdigest(), "lines": 2}, rec["output"])
    check("input fingerprint, size and file name only", rec["inputs"]["prices"] == {
        "file": "inputs.csv", "bytes": data.stat().st_size, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
    check("every code file matching the patterns is fingerprinted, with '/' paths",
          rec["code"] == {"code/a.py": hashlib.sha256(b"x = 1\n").hexdigest(),
                          "code/sub/b.py": hashlib.sha256(b"y = 2\n").hexdigest()}, rec["code"])
    check("an argument that is a path is recorded as its file name only",
          rec["argv"] == ["--input", "inputs.csv", "--paths", "1000"], rec["argv"])
    check("seeds, settings and the deterministic flag are kept",
          rec["seeds"] == [7] and rec["settings"] == {"paths": 1000} and rec["deterministic"] is True)
    raw = m1["path"].read_text(encoding="utf-8")
    check("the date is recorded, never a clock time (no HH:MM anywhere in the file)",
          len(rec["date_utc"]) == 10 and re.search(r"\d{1,2}:\d{2}", raw) is None)
    check("no folders, machine or user names in the file",
          str(tmp) not in raw and tmp.name not in raw and platform.node() not in raw)
    check("library versions recorded for the libraries in use", rec["versions"]["python"] == platform.python_version())
    check("the file is LF only, no byte-order mark", b"\r\n" not in m1["path"].read_bytes()
          and not m1["path"].read_bytes().startswith(b"\xef\xbb\xbf"))

    m2 = study("line one\nline two\n")
    check("the same run again reproduces the output (same_output)", manifest.same_output(rec, manifest.read(m2["path"])) == (True, []))
    m3 = study("line one\nline 2\n")
    same, why = manifest.same_output(rec, manifest.read(m3["path"]))
    check("a changed result is caught", not same and why == ["output differs"], why)
    (tmp / "code" / "a.py").write_bytes(b"x = 2\n")
    m4 = study("line one\nline two\n")
    check("a changed code file is caught even when the output is the same",
          manifest.same_output(rec, manifest.read(m4["path"])) == (False, ["code differ"]))

    # The code a run loads is fingerprinted automatically; a file it never loads is not (audit of v0.3: a new,
    # unrelated module made every published manifest look stale)
    lib = tmp / "lib"
    lib.mkdir()
    (lib / "used_mod_v03.py").write_bytes(b"VALUE = 42\n")
    (lib / "unused_mod_v03.py").write_bytes(b"VALUE = 0\n")
    sys.path.insert(0, str(lib))
    try:
        with manifest.run("loaded", [], out, tmp, "code/a.py") as m5:     # one pattern as a plain string
            import used_mod_v03                                           # noqa: F401
    finally:
        sys.path.remove(str(lib))
        sys.modules.pop("used_mod_v03", None)
    code = manifest.read(m5["path"])["code"]
    check("a module the run imported is fingerprinted, a file it never loaded is not, and a single-string pattern works",
          "lib/used_mod_v03.py" in code and "lib/unused_mod_v03.py" not in code and "code/a.py" in code, sorted(code))

    before = sorted(out.iterdir())

    def failing():
        with manifest.run("broken", [], out, tmp, ("code/*.py",)):
            print("half a table")
            raise RuntimeError("study crashed")
    raises("a failing study raises its own error", failing, "study crashed")
    check("a failing run writes no manifest and gives the screen back", sorted(out.iterdir()) == before
          and sys.stdout is real_stdout)
    notman = tmp / "other.json"
    notman.write_text(json.dumps({"kind": "something else"}), encoding="utf-8")
    raises("read refuses a file that is not a run manifest", lambda: manifest.read(notman), "not a run manifest")
