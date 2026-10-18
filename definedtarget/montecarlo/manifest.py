"""Run manifests: a small record of exactly what produced a printed result, so anyone can reproduce it.

Plain words
- Wrap a study's main body in `with manifest.run(...)`. Everything it prints still goes to the screen, and a copy is
  fingerprinted (SHA-256). On a clean finish, one JSON file is written to `out_dir` with: the run's name and
  arguments, its seeds, each input file's name, size and SHA-256, the SHA-256 of every code file the run loaded
  from `code_root` (plus any named by pattern, such as the script itself), the library versions, the run time in
  seconds, and the output's fingerprint and line count.
- Rerun the same command on the same inputs and code: a seeded study must print the same output, so the new
  manifest's output fingerprint must match the old one (`same_output`). That is the reproducibility check; a timing
  study (deterministic=False) prints different times each run, so only its settings are comparable.
- Privacy: file names only, never folders (a path can reveal a user name); arguments that are existing paths are
  reduced to their file names; the date is recorded, never the clock time; no machine or user names.
- Standard library only, and nothing from outside montecarlo, like the rest of the engine.
"""
import hashlib
import io
import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LIBRARIES = ("numpy", "numba", "scipy", "pandas", "cupy")
KIND = "run manifest"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _public_arg(a):
    """An argument as recorded: an existing file or folder becomes its name only."""
    try:
        p = Path(a)
        if len(a) > 2 and (p.exists() or "\\" in a or "/" in a):
            return p.name
    except (OSError, ValueError):
        pass
    return a


def _versions():
    out = {"python": platform.python_version(), "system": platform.system()}
    for name in LIBRARIES:
        mod = sys.modules.get(name)                # only libraries the run actually imported
        if mod is not None:
            out[name] = getattr(mod, "__version__", "unknown")
    return out


def loaded_files(root):
    """The source files under root of every module this Python process has imported (the code a run used)."""
    root = Path(root).resolve()
    out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py"):
            p = Path(f).resolve()
            if root in p.parents:
                out.add(p)
    return out


def code_fingerprints(root, patterns=(), loaded=True):
    """{relative path: SHA-256} for the files under root matching the glob patterns, plus (loaded=True) every module
    under root the process has imported: the code the run used, and nothing it did not (so an unrelated new file
    elsewhere in the package does not make an old manifest look stale; audit of v0.3). Sorted, '/' separators."""
    root = Path(root).resolve()
    patterns = (patterns,) if isinstance(patterns, str) else patterns     # one pattern given as a plain string
    files = {p.resolve() for pat in patterns for p in root.glob(pat) if p.is_file() and "__pycache__" not in p.parts}
    if loaded:
        files |= loaded_files(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}


class _Tee(io.TextIOBase):
    def __init__(self, screen):
        self.screen, self.copy = screen, io.StringIO()

    def write(self, s):
        self.copy.write(s)
        return self.screen.write(s)

    def flush(self):
        self.screen.flush()


@contextmanager
def run(name, argv, out_dir, code_root, code_patterns=(), seeds=(), inputs=None, settings=None, deterministic=True):
    """Record a manifest for the code run inside the `with` block (written only if it finishes without an error).

    code_patterns: globs under code_root always fingerprinted (e.g. the script itself); every module under
    code_root that the process has imported by the end of the run is fingerprinted too. inputs: {label: path} of
    every data file read; settings: any extra values worth keeping (path counts, grids).
    Yields a dict; after the block it holds the manifest and "path", the file written."""
    t0 = time.perf_counter()
    inputs = {k: Path(v) for k, v in (inputs or {}).items()}
    record = {"kind": KIND, "name": name, "argv": [_public_arg(str(a)) for a in argv], "seeds": list(seeds),
              "settings": settings or {}, "deterministic": bool(deterministic),
              "inputs": {k: {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                         for k, p in sorted(inputs.items())}}
    tee = _Tee(sys.stdout)
    real, sys.stdout = sys.stdout, tee
    result = {}
    try:
        yield result
    finally:
        sys.stdout = real
    text = tee.copy.getvalue().replace("\r\n", "\n")
    record.update({"code": code_fingerprints(code_root, code_patterns),     # after the run: everything it imported
                   "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "seconds": round(time.perf_counter() - t0, 1),
                   "versions": _versions(),
                   "output": {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "lines": text.count("\n")}})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{record['date_utc']}_{record['output']['sha256'][:10]}.json"
    path.write_bytes((json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    result.update(record)
    result["path"] = path


def read(path):
    rec = json.loads(Path(path).read_bytes())
    if rec.get("kind") != KIND:
        raise ValueError(f"{path} is not a run manifest")
    return rec


def same_output(a, b):
    """(same, reasons): whether two manifests show the same result from the same settings, inputs and code."""
    reasons = [f"{k} differ" for k in ("name", "argv", "seeds", "settings", "inputs", "code") if a.get(k) != b.get(k)]
    if a["output"]["sha256"] != b["output"]["sha256"]:
        reasons.append("output differs")
    return not reasons, reasons
