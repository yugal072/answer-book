"""PDF ingestion: question-paper PDF -> canonical ``Paper``.

The heavy lifting lives in :mod:`app.ingestion.text_parser` (structure-aware,
deterministic, no model calls) and :mod:`app.ingestion.validation`
(what to do when the result does not add up). This module owns the I/O,
the fingerprint/id convention, header metadata, and the
validate-or-raise contract.

Public surface (unchanged for existing callers)::

    extract_pdf_pages(path)            -> list[list[str]]   # pages of lines
    extract_paper_metadata(pages)      -> dict               # subject/class/board
    parse_questions(pages)             -> list[Question]
    ingest_paper(path)                 -> Paper

and two additions for callers that want to see the diagnostics::

    ingest_paper_with_report(path)     -> (Paper, ValidationReport)
    ingest_paper(..., diagnostics=[])  # appends findings in place

Failure behaviour
----------------
* A missing/empty/corrupt PDF raises :class:`~app.ingestion.errors.DocumentError`.
* A PDF with no usable text layer (a pure scan) raises
  ``DocumentError`` naming the image path as the alternative - it never
  returns an empty ``Paper`` dressed up as success.
* A result that fails structural validation raises
  :class:`~app.ingestion.errors.ValidationError` carrying the report.

Secrets: none. This path is fully local and deterministic.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader

from app.ingestion.errors import DocumentError
from app.ingestion.text_parser import ParseResult, parse_pages
from app.ingestion.validation import (
    ValidationReport,
    raise_if_invalid,
    validate_paper,
)
from app.models.loaders_models import Paper, Question

log = logging.getLogger(__name__)

#: Text below this many characters across the whole document means the PDF
#: has no usable text layer (a pure scan) and needs the image path.
_MIN_TEXT_CHARS = 40


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _read_pages(file_path: Path) -> Tuple[List[List[str]], int]:
    """Read the PDF text layer. Returns (pages of lines, page count)."""
    path = Path(file_path)
    if not path.exists():
        raise DocumentError(f"PDF not found: {path}")
    if not path.is_file():
        raise DocumentError(f"Not a file: {path}")
    if path.stat().st_size == 0:
        raise DocumentError(f"PDF is empty: {path}")
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentError(f"PDF cannot be read ({path.name}): {exc}") from exc

    try:
        raw_pages = reader.pages
    except Exception as exc:  # pragma: no cover - encrypted/corrupt
        raise DocumentError(
            f"PDF page tree is unreadable ({path.name}): {exc}"
        ) from exc

    pages: List[List[str]] = []
    for number, page in enumerate(raw_pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise DocumentError(
                f"Page {number} of {path.name} could not be decoded: {exc}"
            ) from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pages.append(lines)

    if sum(len(p) for p in pages) == 0 or sum(
        len(line) for page in pages for line in page
    ) < _MIN_TEXT_CHARS:
        raise DocumentError(
            f"{path.name} has no usable text layer "
            f"({sum(len(p) for p in pages)} text lines). It looks like a "
            f"scan; render the pages to PNG and use the image ingestion "
            f"path (app.ingestion.image_loader.extract_paper_from_image_file)."
        )
    return pages, len(pages)


def extract_pdf_pages(file_path: Path) -> list[list[str]]:
    """Extract the text layer of every page as a list of lines.

    Repeated running heads/footers and table headers are removed by
    :func:`app.ingestion.text_parser.strip_running_heads`; question-level
    parsing happens in :func:`parse_questions`.
    """
    pages, _ = _read_pages(Path(file_path))
    from app.ingestion.text_parser import strip_running_heads

    return strip_running_heads(pages)


# --------------------------------------------------------------------------
# Header metadata
# --------------------------------------------------------------------------

#: "Subject: Mathematics Part 1  |  Class: Class 9"
_SUBJECT = re.compile(r"Subject\s*:\s*(?P<value>[^|]+)", re.IGNORECASE)
_CLASS = re.compile(r"\bClass\s*:\s*(?P<value>[^|]+)", re.IGNORECASE)
#: "Total Marks: 80  |  Time Allowed: 3h  |  CBSE" - the board is the last
#: field of that specific metadata line, never a stray capitalised word.
_TOTAL_MARKS = re.compile(r"Total\s+Marks\s*:\s*(?P<value>\d+)", re.IGNORECASE)
_BOARD_TAIL = re.compile(r"\|\s*(?P<value>[A-Z][A-Za-z. ]{1,20})\s*$")
_TIME_ALLOWED = re.compile(r"Time\s+Allowed\s*:\s*(?P<value>[^|]+)", re.IGNORECASE)


def extract_header_facts(pages: list[list[str]]) -> dict:
    """Everything the paper header actually states.

    Keys: ``subject``, ``class``, ``board``, ``total_marks``,
    ``time_allowed``. Only fields the paper really prints are returned;
    nothing is inferred and nothing is invented.
    """
    meta: dict = {}
    for lines in pages:
        for line in lines:
            if "subject" not in meta:
                match = _SUBJECT.search(line)
                if match:
                    meta["subject"] = match.group("value").strip() or None

            if "class" not in meta:
                match = _CLASS.search(line)
                if match:
                    meta["class"] = match.group("value").strip() or None

            if "time_allowed" not in meta:
                match = _TIME_ALLOWED.search(line)
                if match:
                    meta["time_allowed"] = match.group("value").strip() or None

            # The board only counts on the header line that also carries
            # the other header fields, so a capitalised word in ordinary
            # question text can never become a board name.
            if "board" not in meta and _TOTAL_MARKS.search(line):
                match = _BOARD_TAIL.search(line)
                if match:
                    meta["board"] = match.group("value").strip()

            if "total_marks" not in meta:
                match = _TOTAL_MARKS.search(line)
                if match:
                    meta["total_marks"] = int(match.group("value"))
    return {k: v for k, v in meta.items() if v}


def extract_paper_metadata(pages: list[list[str]]) -> dict:
    """Subject / class / board from the paper header text layer.

    This is the canonical three-key view of the header, so PDF ingestion
    fills exactly the same ``Paper`` fields as image ingestion. Anything
    else the header states (declared total marks, time allowed) is
    available through :func:`extract_header_facts`.
    """
    facts = extract_header_facts(pages)
    return {k: facts[k] for k in ("subject", "class", "board") if k in facts}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_questions(pages: list[list[str]]) -> list[Question]:
    """Parse pages of lines into canonical ``Question`` objects."""
    return parse_pages(pages).questions


def _parse_with_result(pages: list[list[str]]) -> ParseResult:
    return parse_pages(pages)


# --------------------------------------------------------------------------
# Paper
# --------------------------------------------------------------------------

def _build_paper(
    file_path: Path,
    raw_pages: list[list[str]],
    parse_result: ParseResult,
) -> Tuple[Paper, int, dict]:
    """Assemble the canonical Paper. Returns (paper, total_pages, header facts)."""
    total_pages = len(raw_pages)
    file_bytes = file_path.read_bytes()
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    facts = extract_header_facts(raw_pages)

    questions = parse_result.questions
    sections = list(
        dict.fromkeys(q.section for q in questions if q.section)
    )

    marks_known = [q.marks for q in questions if q.marks is not None]
    if not questions or len(marks_known) != len(questions):
        # Never present a partial sum as the paper's total.
        total_marks: Optional[int] = None
    else:
        total_marks = sum(marks_known)

    return Paper(
        paper_id=f"pap_{fingerprint[:8]}",
        fingerprint=fingerprint,
        status="ready",
        subject=facts.get("subject"),
        class_name=facts.get("class"),
        board=facts.get("board"),
        questions=questions,
        total_questions=len(questions),
        total_marks=total_marks,
        sections=sections,
    ), total_pages, facts


def ingest_paper_with_report(
    file_path: Path,
) -> Tuple[Paper, ValidationReport]:
    """Ingest a PDF and return the paper together with its diagnostics.

    Raises ``DocumentError`` for unusable input and ``ValidationError``
    when the extraction is structurally unsound.
    """
    path = Path(file_path)
    pages, total_pages = _read_pages(path)
    parse_result = _parse_with_result(pages)
    paper, _total, facts = _build_paper(path, pages, parse_result)

    report = validate_paper(
        paper,
        source="pdf",
        total_pages=total_pages,
        declared_total_marks=facts.get("total_marks"),
        declared_question_count=parse_result.declared_count,
        declared_sections=set(parse_result.sections_seen) or None,
    )
    for note in parse_result.notes:
        report.warn("parser_note", note)
    for line in parse_result.rejected:
        report.warn(
            "rejected_question_candidate",
            f"line looked like a question start but was kept as text: "
            f"{line!r}",
        )
    raise_if_invalid(report, context=f"ingestion of {path.name}")
    return paper, report


def ingest_paper(
    file_path: Path,
    *,
    diagnostics: Optional[list] = None,
) -> Paper:
    """Ingest a question-paper PDF into the canonical ``Paper``.

    ``diagnostics``, when given, is appended to with the
    :class:`~app.ingestion.validation.Diagnostic` findings (warnings and
    errors). The call raises on errors, so what lands in the list is
    information the caller may want to log, not a hidden failure.
    """
    paper, report = ingest_paper_with_report(file_path)
    if diagnostics is not None:
        diagnostics.extend(report.diagnostics)
    for warning in report.warnings:
        log.warning("%s: %s", getattr(file_path, "name", file_path), warning)
    return paper


# --------------------------------------------------------------------------
# Private aliases
#
# PR #13 renamed these three helpers to underscore-prefixed names, on the
# assumption they were internal. They are not: the repository's own tests
# import them by their public names, so a straight rename breaks the suite.
# Both spellings therefore resolve to the same function - the internal-only
# intent of the rename is respected, and existing importers keep working.
# --------------------------------------------------------------------------

_extract_pdf_pages = extract_pdf_pages
_extract_paper_metadata = extract_paper_metadata
_parse_questions = parse_questions


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    project_root = Path(__file__).resolve().parents[2]
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "data" / "papers"

    pdf_files = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.glob("*.pdf"))
    )
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
    for pdf_file in pdf_files:
        try:
            paper, report = ingest_paper_with_report(pdf_file)
        except Exception as exc:
            print(f"{pdf_file.name}: FAILED {type(exc).__name__}: {exc}")
            continue
        print(
            f"{pdf_file.name}: {paper.total_questions} questions, "
            f"{paper.total_marks} marks, sections {paper.sections}, "
            f"warnings {len(report.warnings)}"
        )
        if report.warnings:
            for warning in report.warnings:
                print(f"    {warning}")
