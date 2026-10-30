"""PDF -> clean text blocks (with page numbers) -> section-aware passages."""
import re
import unicodedata
from collections import Counter

import pymupdf


class PDFError(Exception):
    """Raised for PDFs we refuse to / cannot process."""


# ------------------------------------------------------------------ sections
HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXL]+)[.)]?\s+)?"
    r"(abstract|introduction|related work|background|preliminaries|"
    r"methods?|methodology|materials and methods|proposed (?:method|approach)|approach|"
    r"experiments?|experimental (?:setup|results)|results?|results and discussion|"
    r"evaluation|discussion|limitations?|conclusions?|conclusions? and future work|"
    r"future work|references|bibliography|acknowledg(?:e)?ments?|appendix)\s*:?$",
    re.I,
)
ABSTRACT_INLINE_RE = re.compile(r"^abstract\s*[—–:\-.]\s*(.+)$", re.I | re.S)


def canonical_section(heading: str) -> str:
    h = re.sub(r"^(?:\d+(?:\.\d+)*|[IVXL]+)[.)]?\s+", "", heading.strip().lower()).rstrip(":")
    if "abstract" in h:
        return "abstract"
    if "reference" in h or "bibliograph" in h:
        return "references"
    if "acknowledg" in h:
        return "acknowledgements"
    if "appendix" in h:
        return "appendix"
    if "future" in h:
        return "future_work"
    if "limitation" in h:
        return "limitations"
    if "conclusion" in h:
        return "conclusion"
    if "introduction" in h:
        return "introduction"
    if "related" in h or "background" in h or "preliminar" in h:
        return "background"
    if "discussion" in h:
        return "discussion"
    if "result" in h or "evaluation" in h:
        return "results"
    if "method" in h or "approach" in h or "experiment" in h or "material" in h:
        return "methods"
    return "other"


# ------------------------------------------------------------------ text utils
_ABBREV = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "cf.", "approx.", "Sec.", "No."]
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(\[])")


def split_sentences(text: str):
    protected = text
    for a in _ABBREV:
        protected = protected.replace(a, a.replace(".", "․"))
    parts = _SPLIT_RE.split(protected)
    return [p.replace("․", ".").strip() for p in parts if p.strip()]


def clean_block(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    text = text.replace("­", "")
    return re.sub(r"\s+", " ", text).strip()


def _alpha_ratio(t: str) -> float:
    return sum(c.isalpha() for c in t) / max(len(t), 1)


def _norm_for_repeat(t: str) -> str:
    return re.sub(r"\d+", "#", t.strip().lower())


# ------------------------------------------------------------------ loading
def load_document(path, max_pages: int):
    """Return dict(title, n_pages, pages=[[block_text, ...], ...])."""
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # corrupted / not a pdf
        raise PDFError("The file could not be opened as a PDF.") from exc
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise PDFError("Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise PDFError("The PDF has no pages.")
        if doc.page_count > max_pages:
            raise PDFError(f"The PDF has {doc.page_count} pages; the limit is {max_pages}.")
        pages = []
        for page in doc:
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            blocks = _reading_order(blocks, page.rect.width)
            pages.append([clean_block(b[4]) for b in blocks])
        title = _guess_title(doc) or (doc.metadata or {}).get("title") or ""
        n_pages = doc.page_count

    pages = _drop_headers_footers(pages)
    return {"title": title.strip()[:250], "n_pages": n_pages, "pages": pages}


def _reading_order(blocks, page_width):
    """Order text blocks the way a human reads them, including 2-column papers.

    Full-width blocks (title, abstract, wide tables) split the page into horizontal
    bands; inside each band the left column is read first, then the right column.
    """
    mid, tol = page_width / 2, 0.05 * page_width
    ordered, left, right = [], [], []

    def flush():
        ordered.extend(sorted(left, key=lambda b: b[1]))
        ordered.extend(sorted(right, key=lambda b: b[1]))
        left.clear()
        right.clear()

    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        x0, x1 = b[0], b[2]
        if x0 < mid - tol and x1 > mid + tol:   # spans both columns
            flush()
            ordered.append(b)
        elif x1 <= mid + tol:
            left.append(b)
        else:
            right.append(b)
    flush()
    return ordered


def _guess_title(doc):
    try:
        data = doc[0].get_text("dict")
        lines = []
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                txt = "".join(s["text"] for s in ln["spans"]).strip()
                if txt and ln["spans"] and not HEADING_RE.match(txt) and ln["bbox"][1] < 0.4 * doc[0].rect.height:
                    lines.append((max(s["size"] for s in ln["spans"]), ln["bbox"][1], txt))
        if not lines:
            return ""
        biggest = max(sz for sz, _, _ in lines)
        title_lines = [t for sz, y, t in sorted(lines, key=lambda x: x[1]) if sz >= biggest - 0.5]
        title = " ".join(title_lines[:3])
        return title if 8 <= len(title) <= 250 else ""
    except Exception:
        return ""


def _drop_headers_footers(pages):
    n = len(pages)
    counts = Counter()
    for blocks in pages:
        for norm in {_norm_for_repeat(b) for b in blocks if len(b) < 120}:
            counts[norm] += 1
    out = []
    for blocks in pages:
        keep = []
        for b in blocks:
            if re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", b.strip(), re.I):
                continue
            if n >= 4 and len(b) < 120 and counts[_norm_for_repeat(b)] / n >= 0.4:
                continue
            keep.append(b)
        out.append(keep)
    return out


# ------------------------------------------------------------------ passages
def build_passages(pages, target_words: int = 140, min_words: int = 12):
    """Split pages into passages of ~target_words, remembering page + section.

    Passages never cross a page boundary (so page references are exact) or a section
    boundary. References / acknowledgements are excluded from the index.
    One sentence of overlap is carried between consecutive passages on a page.
    """
    passages = []
    section = "front"
    skipping = False

    for page_no, blocks in enumerate(pages, start=1):
        state = {"cur": [], "words": 0, "new": 0, "section": section}

        def flush(keep_last=False, state=state, page_no=page_no):
            cur = state["cur"]
            if cur and state["new"] > 0:
                text = " ".join(cur).strip()
                if len(text.split()) >= min_words:
                    passages.append({"page": page_no, "section": state["section"], "text": text})
            if keep_last and cur:
                state["cur"] = [cur[-1]]
                state["words"] = len(cur[-1].split())
            else:
                state["cur"], state["words"] = [], 0
            state["new"] = 0

        for block in blocks:
            body = block
            m_inline = ABSTRACT_INLINE_RE.match(block)
            if m_inline:
                flush()
                section = state["section"] = "abstract"
                skipping = False
                body = m_inline.group(1)
            elif len(block) < 80 and HEADING_RE.match(block.strip()):
                flush()
                section = state["section"] = canonical_section(block)
                skipping = section in {"references", "acknowledgements"}
                continue

            if skipping or len(body.split()) < 6 or _alpha_ratio(body) < 0.55:
                continue

            for sent in split_sentences(body):
                state["cur"].append(sent)
                state["words"] += len(sent.split())
                state["new"] += 1
                if state["words"] >= target_words:
                    flush(keep_last=True)
        flush()

    for i, p in enumerate(passages):
        p["idx"] = i
    return passages