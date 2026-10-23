#!/usr/bin/env python3
"""ASD-STE100 lint for the public Markdown files of this repository (the rules: ASD-STE100, Simplified Technical
English, with American spelling; copied from the 2x-ledger project on 2026-10-23).

    py -I scripts\\ste_check.py                 every public .md file; prints "N problem(s)", exit 1 if N > 0
    py -I scripts\\ste_check.py README.md ...   named files only

Rules the script can test: sentence length (25 words; 20 for an instruction), passive voice, -ing forms used as
nouns or as a progressive tense, banned words (slang, vague words, Latin abbreviations), British spelling,
paragraph length (6 sentences). It skips fenced code, inline code, link targets, headings' length and HTML.
What it cannot test (the approved-word list, one topic per sentence) a reader checks.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_WORDS, MAX_INSTRUCTION, MAX_SENTENCES = 25, 20, 6

# The public files written to ASD-STE100 so far. The rest of the repository predates the rule (2026-10-22) and is
# converted in v0.3.3 (BACKLOG B36); each converted file joins this list.
PUBLIC = ["docs/MODELS.md"]
IGNORE_FILE = re.compile(r"(_draft|_DRAFT)\.md$|^(PLAN|PROMPT|BACKLOG)\.md$")

IMPERATIVE = {"run", "read", "use", "open", "copy", "install", "start", "check", "do", "set", "pass", "call", "write",
              "add", "remove", "commit", "push", "put", "keep", "see", "download", "save", "load", "type", "enter", "click"}

BANNED = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "name the items, or write 'and so on'", "etc": "and so on",
    "vs.": "against", "vs": "against", "via": "through", "cf.": "compare", "approx.": "about", "ca.": "about",
    "basically": "delete", "very": "delete", "really": "delete", "quite": "delete", "pretty": "delete", "a lot": "many",
    "lots of": "many", "stuff": "name the items", "kind of": "delete", "sort of": "delete", "whilst": "while",
    "amongst": "among", "thus": "so", "hence": "so", "moreover": "also", "furthermore": "also", "nevertheless": "but",
    "notwithstanding": "but", "albeit": "but", "hereby": "delete", "thereof": "of it", "aforementioned": "name it",
    "in order to": "to", "due to the fact that": "because", "as well as": "and", "utilize": "use", "utilise": "use",
    "in terms of": "rephrase", "with regard to": "about", "regarding": "about", "prior to": "before",
    "subsequent to": "after", "in the event that": "if", "at this point in time": "now", "a number of": "several",
    "cherry-pick": "choose", "rule of thumb": "rule", "bottom line": "result", "ballpark": "about", "gotcha": "trap",
    "hard-wired": "fixed", "hand-wavy": "vague", "under the hood": "inside", "off the shelf": "standard",
    "state of the art": "best available", "garbage": "bad data", "sanity check": "check", "eat": "cost",
}

BRITISH = re.compile(
    r"\b(behaviour|colour|favour|flavour|honour|labour|neighbour|rumour|vapour|humour|licence|defence|offence|pretence"
    r"|centre|metre|litre|fibre|calibre|theatre|programme|catalogue|analogue|dialogue|grey|cheque|aluminium|tyre|mould"
    r"|enrol|enrolment|fulfil|fulfilment|instalment|skilful|wilful|jewellery|manoeuvre|learnt|spelt|dreamt|practise"
    r"|judgement|realis(?:e|ed|es|ing|ation)|recognis(?:e|ed|es|ing)|organis(?:e|ed|es|ing|ation)|optimis(?:e|ed|es|ing|ation)"
    r"|minimis(?:e|ed|es|ing)|maximis(?:e|ed|es|ing)|normalis(?:e|ed|es|ing|ation)|summaris(?:e|ed|es|ing)|analys(?:e|ed|es|ing)"
    r"|standardis(?:e|ed|es|ing)|capitalis(?:e|ed|es|ing|ation)|categoris(?:e|ed|es|ing)|emphasis(?:e|ed|es|ing)"
    r"|finalis(?:e|ed|es|ing)|initialis(?:e|ed|es|ing)|serialis(?:e|ed|es|ing)|prioritis(?:e|ed|es|ing)|stabilis(?:e|ed|es|ing)"
    r"|visualis(?:e|ed|es|ing)|annualis(?:e|ed|es|ing)|winsoris(?:e|ed|es|ing)|parameteris(?:e|ed|es|ing)|generalis(?:e|ed|es|ing|ation)"
    r"|modell(?:ing|ed|er)|labell(?:ing|ed)|cancell(?:ing|ed)|travell(?:ing|ed)|levell(?:ing|ed)|signall(?:ing|ed)"
    r"|totall(?:ing|ed)|channell(?:ing|ed)|fuell(?:ing|ed)|marvellous)\b", re.I)

# -ing words that are not gerunds, or technical names with one meaning in docs/GLOSSARY.md
ALLOWED_ING = {
    "nothing", "something", "anything", "everything", "during", "morning", "evening", "thing", "things", "string",
    "strings", "ceiling", "sibling", "siblings", "spring", "bring", "king", "ring", "sing", "wing", "swing", "sterling",
    "according", "including", "following", "regarding", "meaning", "building", "setting", "settings", "heading", "headings",
    "underlying", "rebalancing", "financing", "compounding", "trading", "listing", "listings", "delisting", "holding", "holdings",
    "timing", "borrowing", "lending", "pricing", "rounding", "funding", "tracking", "trailing", "rolling", "closing", "opening",
    "outstanding", "remaining", "existing", "ranking", "rankings", "reporting", "accounting", "missing", "leading", "lagging",
    "working", "warning", "warnings", "offering", "offerings", "scaling", "sampling", "testing", "scoring", "hedging",
    "stop-loss", "ongoing", "upcoming", "forthcoming", "incoming", "outgoing", "wording", "spelling", "reading", "readings",
    "earnings", "savings", "proceedings", "bidding", "clearing", "writing", "filing", "filings", "capping", "shorting",
    "swing", "springer", "beijing", "boeing", "wyoming", "reading", "flooring", "ending", "endings", "being", "clothing",
}
ADJECTIVE_PARTICIPLES = {"based", "known", "named", "dated", "closed", "listed", "related", "limited", "located", "interested",
                         "split-adjusted", "levered", "borrowed", "unverified", "verified", "frozen", "fixed", "called",
                         "open", "set", "given", "left", "right", "due", "equal", "worth", "ready", "sure", "likely"}
IRREGULAR = ("built|done|made|seen|given|taken|shown|held|kept|put|read|run|split|cut|hit|let|paid|said|sold|told|found|bought|"
             "brought|caught|taught|thought|written|driven|chosen|broken|spoken|hidden|drawn|thrown|grown|flown|blown|withdrawn|"
             "begun|won|lost|met|fed|led|spent|sent|lent|bent|meant|dealt|felt|understood|added|rebuilt|reset|rerun|set")
PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being|am|get|gets|got)\s+(?:(?:not|also|then|still|never|always|often|now|first|only)\s+)?"
                     r"((?:[a-z]+-)?[a-z]+ed|" + IRREGULAR + r")\b", re.I)
PROGRESSIVE = re.compile(r"\b(is|are|was|were|be|been|am)\s+(?:(?:not|also|still|now)\s+)?([a-z]+ing)\b", re.I)
GERUND = re.compile(r"(?:^|\b(?:by|of|for|when|while|before|after|without|instead of|avoid|keep|keeps|stop|stops|start|starts|"
                    r"continue|continues|means|mean|needs|need|require|requires|allow|allows|about|from|to|through|on|in|with)\s+)"
                    r"([a-z]+ing)\b", re.I)

SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`*])")
CODE_SPAN = re.compile(r"`[^`]*`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"https?://\S+")
HTML = re.compile(r"<[^>]+>")


def public_files(root=ROOT):
    files = [root / p for p in PUBLIC if (root / p).exists()]
    return [f for f in files if not IGNORE_FILE.search(f.name)]


QUOTED = re.compile(r'"[^"\n]{3,}"')          # a quoted title or phrase is someone else's words


def _quote(m):
    """Skip a quotation of at most 15 words (a title, a term); a longer one is our own prose and stays checked."""
    return " QUOTE " if len(m.group(0).split()) <= 15 else m.group(0)


def strip_markup(text):
    text = CODE_SPAN.sub(" CODE ", text)
    text = QUOTED.sub(_quote, text)
    text = LINK.sub(r"\1", text)
    text = URL.sub(" URL ", text)
    text = HTML.sub(" ", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    return text


def blocks(lines):
    """Yield (line_no, kind, text) for paragraphs, list items, table cells and headings; skip fenced code."""
    in_code, para, start = False, [], 0
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            in_code = not in_code
            if para:
                yield start, "para", " ".join(para)
                para = []
            continue
        if in_code:
            continue
        s = line.strip()
        if not s:
            if para:
                yield start, "para", " ".join(para)
                para = []
            continue
        if s.startswith("#"):
            if para:
                yield start, "para", " ".join(para)
                para = []
            yield i, "heading", s.lstrip("#").strip()
            continue
        if s.startswith("|"):
            if para:
                yield start, "para", " ".join(para)
                para = []
            if re.fullmatch(r"\|[\s:|-]+\|?", s):
                continue
            for cell in s.strip("|").split("|"):
                cell = cell.strip()
                if cell:
                    yield i, "cell", cell
            continue
        if re.match(r"^(\d+\.|[-*+])\s+", s):
            if para:
                yield start, "para", " ".join(para)
            para, start = [re.sub(r"^(\d+\.|[-*+])\s+", "", s)], i
            continue
        if not para:
            start = i
        para.append(s)
    if para:
        yield start, "para", " ".join(para)


def sentences(text):
    return [s.strip() for s in SENT_END.split(text.strip()) if s.strip()]


def words(sentence):
    return [w for w in re.split(r"\s+", sentence) if re.search(r"[A-Za-z0-9]", w)]


def check_sentence(sent):
    out = []
    low = sent.lower()
    ws = words(sent)
    first = re.sub(r"[^a-z]", "", ws[0].lower()) if ws else ""
    limit = MAX_INSTRUCTION if first in IMPERATIVE else MAX_WORDS
    if len(ws) > limit:
        out.append(f"{len(ws)} words (limit {limit})")
    for bad, fix in BANNED.items():
        if re.search(r"(?<![a-z])" + re.escape(bad) + r"(?![a-z])", low):
            out.append(f"banned word '{bad}' -> {fix}")
    m = BRITISH.search(sent)
    if m:
        out.append(f"British spelling '{m.group(0)}'")
    for m in PASSIVE.finditer(sent):
        if m.group(2).lower() not in ADJECTIVE_PARTICIPLES:
            out.append(f"passive voice '{m.group(0)}'")
            break
    for m in PROGRESSIVE.finditer(sent):
        if m.group(2).lower() not in ALLOWED_ING:
            out.append(f"progressive tense '{m.group(0)}'")
            break
    for m in GERUND.finditer(sent):
        w = m.group(1).lower()
        if w not in ALLOWED_ING and len(w) > 5:
            out.append(f"-ing form '{m.group(1)}' used as a noun")
            break
    return out


def check_file(path):
    problems = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, kind, text in blocks(lines):
        text = strip_markup(text)
        if kind == "heading":
            for bad, fix in BANNED.items():
                if re.search(r"(?<![a-z])" + re.escape(bad) + r"(?![a-z])", text.lower()):
                    problems.append((line_no, f"banned word '{bad}' -> {fix}", text))
            m = BRITISH.search(text)
            if m:
                problems.append((line_no, f"British spelling '{m.group(0)}'", text))
            continue
        sents = sentences(text)
        if kind == "para" and len(sents) > MAX_SENTENCES:
            problems.append((line_no, f"paragraph of {len(sents)} sentences (limit {MAX_SENTENCES})", sents[0][:60] + "..."))
        for s in sents:
            for why in check_sentence(s):
                problems.append((line_no, why, s))
    return problems


def main(argv):
    files = [Path(a) if Path(a).is_absolute() else ROOT / a for a in argv] or public_files()
    total = 0
    for f in files:
        probs = check_file(f)
        total += len(probs)
        for line_no, why, s in probs:
            shown = f.relative_to(ROOT).as_posix() if f.resolve().is_relative_to(ROOT) else str(f)
            print(f"{shown}:{line_no}: {why}\n    {s[:160]}")
    print(f"\n{total} problem(s) in {len(files)} file(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
