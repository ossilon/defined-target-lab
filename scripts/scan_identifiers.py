#!/usr/bin/env python3
"""Refuse to publish personal identifiers, secrets or data files.  Run before every push:

    py -I scripts\\scan_identifiers.py          (must print "0 problem(s)")

It checks every file that git would publish (everything .gitignore does not exclude, matched the way git matches):
1. personal terms from the PRIVATE list identifiers_local.txt (git-ignored, so the list itself is never published),
   in file contents and file names. The author's public handle may appear only where attribution needs it: the
   LICENSE copyright line, the CITATION.cff author line, the README citation line and the repository URL. The scan
   reads the handle and the URL from CITATION.cff, so this code never names them;
2. generic leaks: Windows/Unix home-directory paths, email addresses, things that look like API keys or tokens,
   plus British spellings (the public style rule is plain American English);
3. data files that must never be published (CSV/JSON/JSONL/XLS/PDF/Parquet outside the fund specs and docs), image
   files outside docs/img/ (a screenshot can carry licensed data), and text files with 5 or more dated numeric rows
   (a daily series pasted into a document).

tests/test_license_guard.py runs the same scan, so the test suite fails before any data can reach a push.
"""
import functools
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".html", ".js", ".css",
                 ".csv", ".gitignore", ".cff", ".svg", ""}
GENERIC = {
    "home directory path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+|/home/[a-z0-9_-]+|/Users/[A-Za-z0-9_-]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "possible secret": re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    # public style rule: plain American English (catches common British spellings)
    "British spelling": re.compile(r"(?i)\b\w*(behaviour|licence|organis|recognis|summaris|centre|neighbour|favour"
                                   r"|colour|programme|labell|modell|cancell|catalogue|whilst|towards|realis|vectoris"
                                   r"|itemis|minimis|maximis|optimis|prioritis|parallelis|analyse)\w*"),
}
ALLOWED_EMAILS = {"info@roundhillinvestments.com"}         # public issuer contact, cited in docs
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".xls", ".xlsx", ".pdf", ".parquet"}
ALLOWED_DATA_DIRS = ("definedtarget/funds/", "docs/")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
ALLOWED_IMAGE_DIRS = ("docs/img/",)
# The frozen prediction ledger must be public (it proves pre-registration). It holds text only; the scan still
# checks its contents like any text file.
ALLOWED_DATA_FILES = {"forward/predictions.jsonl",
                      "web/golden.json"}                # engine answers on synthetic inputs (scripts/make_golden.py)
# Attribution: where the handle may stand, per file (the handle must follow this prefix directly).
ATTRIBUTION_PREFIX = {"LICENSE": r"^Copyright \(c\) \d{4} ",
                      "CITATION.cff": r"^\s*-\s*(?:name|alias):\s*\"?",
                      "README.md": r"^> "}
URL_FILES = {"CITATION.cff", "README.md"}                 # files that may carry the repository URL


def attribution(repo=REPO):
    """(handle, repository URL) from CITATION.cff; (None, None) if the file is missing."""
    path = Path(repo) / "CITATION.cff"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    h = re.search(r"^\s*-\s*(?:name|alias):\s*\"?([^\"\n]+?)\"?\s*$", text, re.MULTILINE)
    u = re.search(r"^repository-code:\s*\"?([^\"\n]+?)\"?\s*$", text, re.MULTILINE)
    return (h.group(1) if h else None), (u.group(1) if u else None)


def without_attribution(rel, line, handle, url):
    """The line with only the permitted attribution removed, so a personal term anywhere else still counts."""
    if url and rel in URL_FILES:
        line = line.replace(url, "")
    if handle and rel in ATTRIBUTION_PREFIX:
        line = re.sub(f"({ATTRIBUTION_PREFIX[rel]}){re.escape(handle)}", r"\1", line)
    return line


def ignore_patterns():
    pats = []
    for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pats.append(line)
    return pats


@functools.lru_cache(maxsize=None)
def _rule(pattern):
    """(regex, negated, directory_only) for one .gitignore line, following gitignore(5).

    A slash at the start or in the middle anchors the pattern to the repository root; otherwise it matches at any
    depth. A trailing slash matches directories only. '*' and '?' never cross a '/'; '**' does.
    """
    negated = pattern.startswith("!")
    pat = pattern[1:] if negated else pattern
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")
    anchored = "/" in pat
    pat = pat.lstrip("/")
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pat.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pat[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pat[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pat[i]), i + 1
    return re.compile(("" if anchored else "(?:.*/)?") + out + r"\Z"), negated, dir_only


def ignored(rel, pats):
    """True if git would ignore the file at `rel` (a path relative to the repository, with '/').

    Like git: each parent folder is checked first, and a file inside an ignored folder stays ignored even if a later
    '!' pattern names it. For the file itself, the last matching pattern wins.
    """
    rules = [_rule(p) for p in pats]
    parts = rel.split("/")
    for depth in range(1, len(parts) + 1):
        sub, is_dir = "/".join(parts[:depth]), depth < len(parts)
        state = None
        for rx, negated, dir_only in rules:
            if dir_only and not is_dir:
                continue
            if rx.match(sub):
                state = not negated
        if is_dir and state:
            return True
        if not is_dir:
            return bool(state)
    return False


def published_files():
    pats = ignore_patterns()
    out = []
    for path in sorted(REPO.rglob("*")):
        if path.is_dir() or ".git" in path.relative_to(REPO).parts:
            continue
        rel = path.relative_to(REPO).as_posix()
        if not ignored(rel, pats):
            out.append((rel, path))
    return out


# A dated row: a line whose first field is an ISO date, followed by at least two decimal numbers. Several such rows in
# one file look like a daily price or NAV series, which the data licenses forbid publishing.
DATED_ROW = re.compile(r"^\s*\|?\s*(19|20)\d\d-\d\d-\d\d\s*[|,;\t]")
DECIMAL = re.compile(r"-?\d+\.\d+")
MAX_DATED_ROWS = 4


def daily_series_rows(text):
    """Number of lines in `text` that look like rows of a dated numeric series."""
    return sum(1 for line in text.splitlines() if DATED_ROW.match(line) and len(DECIMAL.findall(line)) >= 2)


def personal_terms():
    local = REPO / "identifiers_local.txt"
    if not local.exists():
        return None
    out = []
    for line in local.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(re.compile(line, re.IGNORECASE))
    return out


def scan(files=None, personal=(), handle=None, url=None):
    """Problems (list of strings) in the files git would publish: data files, images, daily series, personal terms
    and leaks. `handle` and `url` default to the ones in CITATION.cff."""
    files = published_files() if files is None else files
    if handle is None and url is None:
        handle, url = attribution()
    problems = []
    for rel, path in files:
        for rx in personal:
            if rx.search(rel):
                problems.append(f"{rel}: personal term /{rx.pattern}/ in the file name")
        suffix = path.suffix.lower()
        allowed_file = rel in ALLOWED_DATA_FILES
        if suffix in DATA_SUFFIXES and not rel.startswith(ALLOWED_DATA_DIRS) and not allowed_file:
            problems.append(f"{rel}: data file; git would publish it")
            continue
        if suffix in IMAGE_SUFFIXES and not rel.startswith(ALLOWED_IMAGE_DIRS):
            problems.append(f"{rel}: image file; publish only images this code generates, under docs/img/")
            continue
        if suffix not in TEXT_SUFFIXES and not allowed_file:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = daily_series_rows(text)
        if rows > MAX_DATED_ROWS:
            problems.append(f"{rel}: {rows} dated numeric rows; looks like a daily series")
        for n, line in enumerate(text.splitlines(), 1):
            checked = without_attribution(rel, line, handle, url)
            for rx in personal:
                if rx.search(checked):
                    problems.append(f"{rel}:{n}: personal term /{rx.pattern}/")
            for name, rx in GENERIC.items():
                if name == "British spelling" and rel == "scripts/scan_identifiers.py":
                    continue                                   # this file lists the spellings it looks for
                for m in rx.finditer(line):
                    if name == "email address" and m.group(0).lower() in ALLOWED_EMAILS:
                        continue
                    problems.append(f"{rel}:{n}: {name}: {m.group(0)[:40]}")
    return problems


def main():
    personal = personal_terms()
    if personal is None:
        print("WARNING: identifiers_local.txt not found; only generic checks run")
        personal = []
    files = published_files()
    problems = scan(files, personal)
    for p in problems:
        print(p)
    print(f"\nscanned {len(files)} publishable files; {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
