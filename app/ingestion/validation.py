"""Structural post-extraction validation and diagnostics.

Extraction mistakes are silent by nature: a lost question, a shifted page
number or a marks value copied onto the wrong question all produce a
perfectly valid ``Paper``. This module is the layer that notices.

It is deliberately *structural* - it checks the extracted result against
the canonical schema and against internal consistency and against what the
source document itself declared. It never rewrites a value to make a
check pass.

Severities
----------
``error``
    The result must not be trusted or handed downstream
    (duplicate question numbers on the PDF path, an empty question list,
    an out-of-range page number, a schema violation).
``warning``
    Suspicious but usable: the caller should know. Non-contiguous
    numbering, a marks total that disagrees with the paper's declared
    total, a question count that disagrees with the section instructions,
    low vision confidence.

``ValidationError`` (see :mod:`app.ingestion.errors`) carries the report, so
"why did this fail" is never reduced to a bare message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from app.ingestion.errors import ValidationError
from app.models.loaders_models import Paper, Question

CANONICAL_TYPES = {"numerical", "mcq", "short", "long"}

ERROR = "error"
WARNING = "warning"


@dataclass
class Diagnostic:
    """One structural finding about an extraction result."""

    code: str
    severity: str
    message: str
    page: Optional[int] = None
    question: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - formatting only
        where = []
        if self.question:
            where.append(f"question {self.question}")
        if self.page is not None:
            where.append(f"page {self.page}")
        suffix = f" ({', '.join(where)})" if where else ""
        return f"[{self.severity}] {self.code}: {self.message}{suffix}"


@dataclass
class ValidationReport:
    """Result of :func:`validate_paper`."""

    source: str = "pdf"
    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        page: Optional[int] = None,
        question: Optional[str] = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(code, severity, message, page, question)
        )

    def error(self, code: str, message: str, **kw) -> None:
        self.add(code, ERROR, message, **kw)

    def warn(self, code: str, message: str, **kw) -> None:
        self.add(code, WARNING, message, **kw)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if not self.diagnostics:
            return "no structural problems found"
        head = "no errors" if self.ok else f"{len(self.errors)} error(s)"
        return f"{len(self.diagnostics)} finding(s), {head}"


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def _check_canonical_schema(paper: Paper, report: ValidationReport) -> None:
    """Re-validate through pydantic: a hand-built object that drifted from
    the canonical contract must not slip through."""
    try:
        Paper.model_validate(paper.model_dump())
    except Exception as exc:  # pragma: no cover - defensive
        report.error(
            "schema_invalid", f"result violates the canonical schema: {exc}"
        )
        return
    for q in paper.questions:
        try:
            Question.model_validate(q.model_dump())
        except Exception as exc:
            report.error(
                "schema_invalid",
                f"question violates the canonical schema: {exc}",
                question=q.number,
                page=q.page,
            )


def _check_numbers(
    questions: Sequence[Question], source: str, report: ValidationReport
) -> None:
    seen: dict = {}
    for q in questions:
        if not q.number or not q.number.strip():
            report.error(
                "missing_number",
                "question has no number",
                question=q.number,
                page=q.page,
            )
            continue
        seen.setdefault(q.number.strip(), []).append(q)

    duplicates = {n: qs for n, qs in seen.items() if len(qs) > 1}
    for number, group in duplicates.items():
        pages = sorted({q.page for q in group if q.page is not None})
        # On the vision path the same question can legitimately appear on
        # two overlapping input images; on the deterministic PDF path a
        # repeated number means the parser merged or repeated content.
        severity = WARNING if source == "image" else ERROR
        report.add(
            "duplicate_number",
            severity,
            f"question number {number} appears {len(group)} times "
            f"on pages {pages}",
            question=number,
        )

    # The same wording twice is a different failure: overlapping vision
    # tiles routinely report one question under two numbers ("1(i)" and
    # "i"), which no numbering check can see.
    by_text: dict = {}
    for q in questions:
        key = re.sub(
            r"[^a-z0-9]+", "", re.sub(r"\s+", " ", q.text).strip().lower()
        )[:160]
        if not key:
            continue
        by_text.setdefault(key, []).append(q)
    for group in by_text.values():
        if len(group) < 2:
            continue
        numbers = [q.number for q in group]
        pages = sorted({q.page for q in group if q.page is not None})
        report.add(
            "duplicate_text",
            WARNING if source == "image" else ERROR,
            f"identical question wording appears {len(group)} times as "
            f"{numbers} on pages {pages}",
            question=numbers[0],
        )

    numeric = [
        (n, qs[0]) for n, qs in seen.items() if n.isdigit()
    ]
    if numeric:
        ordered = [int(n) for n, _ in numeric]
        if ordered != sorted(ordered):
            report.error(
                "impossible_ordering",
                f"question numbers are not in ascending order: {ordered[:12]}",
            )
        unique = sorted(set(ordered))
        if unique and unique[0] != 1 and len(unique) == len(ordered):
            report.warn(
                "numbering_starts_above_one",
                f"numbering starts at {unique[0]}, not 1",
            )
        missing = [
            n
            for n in range(min(unique), max(unique) + 1)
            if n not in set(unique)
        ]
        if missing:
            report.warn(
                "numbering_gap",
                f"question numbers missing from the sequence: {missing[:12]}"
                + (" ..." if len(missing) > 12 else ""),
            )


def _check_content(
    questions: Sequence[Question], report: ValidationReport
) -> None:
    for q in questions:
        if not q.text or not q.text.strip():
            report.error(
                "empty_text",
                "question has no text",
                question=q.number,
                page=q.page,
            )
        elif len(q.text.strip()) < 3:
            report.warn(
                "suspiciously_short_text",
                f"question text is very short: {q.text.strip()!r}",
                question=q.number,
                page=q.page,
            )
        if q.page is not None and q.page < 1:
            report.error(
                "invalid_page",
                f"page number {q.page} is not a valid page",
                question=q.number,
            )
        if q.type not in CANONICAL_TYPES:
            report.error(
                "invalid_type",
                f"question type {q.type!r} is not canonical",
                question=q.number,
            )
        if q.marks is not None and (q.marks < 0 or q.marks > _MAX_MARKS):
            report.error(
                "invalid_marks",
                f"marks value {q.marks} is out of range",
                question=q.number,
            )
        if q.options is not None:
            if not q.options:
                report.warn(
                    "empty_options",
                    "options list is present but empty",
                    question=q.number,
                )
            for option in q.options:
                if not option or not option.strip():
                    report.error(
                        "invalid_option",
                        "empty option string",
                        question=q.number,
                    )
        if q.type == "mcq" and not q.options:
            report.warn(
                "mcq_without_options",
                "question is typed mcq but carries no options",
                question=q.number,
            )
        if q.has_figure and not _FIGURE_REF.search(q.text):
            report.warn(
                "figure_without_reference",
                "has_figure is set but the text references no figure",
                question=q.number,
            )


def _check_section_consistency(
    questions: Sequence[Question],
    declared_sections: Optional[set],
    report: ValidationReport,
) -> None:
    """Sections must be real, and must not reappear after moving on.

    A paper that runs A, B, C, A again means the parser lost a heading, so
    it is reported. A question whose section is not among the sections the
    caller declared is an error. Having *no* sections at all is normal for
    a paper that prints none, so only partial coverage is flagged.
    """
    present = {q.section for q in questions if q.section}
    with_section = sum(1 for q in questions if q.section)
    if present and with_section < len(questions):
        report.warn(
            "partial_sections",
            f"{len(questions) - with_section} of {len(questions)} questions "
            f"have no section although the paper declares {sorted(present)}",
        )
    if declared_sections:
        for q in questions:
            if q.section and q.section not in declared_sections:
                report.error(
                    "inconsistent_section",
                    f"section {q.section!r} is not among "
                    f"{sorted(declared_sections)}",
                    question=q.number,
                )

    runs: List[str] = []
    for q in questions:
        if not q.section:
            continue
        if not runs or runs[-1] != q.section:
            if q.section in runs:
                report.warn(
                    "section_reappears",
                    f"section {q.section!r} reappears after "
                    f"{runs[-1]!r}; a section heading may have been missed",
                    question=q.number,
                )
            runs.append(q.section)


def _check_pages(
    questions: Sequence[Question], total_pages: Optional[int], report: ValidationReport
) -> None:
    if not total_pages:
        return
    for q in questions:
        if q.page is not None and q.page > total_pages:
            report.error(
                "page_out_of_range",
                f"page {q.page} exceeds the document's {total_pages} pages",
                question=q.number,
            )
    pages = sorted({q.page for q in questions if q.page})
    if pages and pages[0] != 1 and questions:
        report.warn(
            "no_question_on_first_page",
            f"first extracted question is on page {pages[0]}",
        )


def _check_marks_total(
    paper: Paper,
    declared_total: Optional[int],
    report: ValidationReport,
) -> None:
    known = [q.marks for q in paper.questions if q.marks is not None]
    unknown = sum(1 for q in paper.questions if q.marks is None)
    if unknown:
        report.warn(
            "unknown_marks",
            f"{unknown} of {len(paper.questions)} questions have no marks",
        )
    if declared_total is None:
        return
    if not known:
        report.warn(
            "marks_total_unknown",
            f"paper declares {declared_total} marks but none were extracted",
        )
    elif sum(known) != declared_total:
        report.warn(
            "marks_total_mismatch",
            f"extracted marks total {sum(known)} but the paper header "
            f"declares {declared_total}",
        )


def _check_declared_count(
    paper: Paper,
    declared: Optional[int],
    report: ValidationReport,
) -> None:
    if declared is None:
        return
    if declared != len(paper.questions):
        report.warn(
            "question_count_mismatch",
            f"extracted {len(paper.questions)} questions but the section "
            f"instructions declare {declared}",
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

_MAX_MARKS = 100

_FIGURE_REF = re.compile(
    r"\b(?:fig(?:ure)?\.?|diagram|graph|plot|sketch)\s*"
    r"(?:\d+|(?:below|above|here|shown|given|accompanying|adjacent|opposite)\b)"
    r"|\b(?:see|shown in|as shown in|refer to)\s+(?:the\s+|given\s+)?"
    r"(?:fig(?:ure)?\.?|diagram|graph|plot)\b",
    re.IGNORECASE,
)


def validate_paper(
    paper: Paper,
    *,
    source: str = "pdf",
    total_pages: Optional[int] = None,
    declared_total_marks: Optional[int] = None,
    declared_question_count: Optional[int] = None,
    declared_sections: Optional[set] = None,
    confidences: Optional[dict] = None,
) -> ValidationReport:
    """Validate an extracted :class:`Paper` structurally.

    ``confidences`` maps question number -> vision confidence, for the
    image path; low confidence is reported as a warning so a caller can see
    that a value was hard to read rather than silently trusting it.
    """
    report = ValidationReport(source=source)

    if not paper.questions:
        report.error(
            "no_questions", "extraction produced no questions at all"
        )
        return report

    _check_canonical_schema(paper, report)
    _check_numbers(paper.questions, source, report)
    _check_content(paper.questions, report)
    _check_section_consistency(paper.questions, declared_sections, report)
    _check_pages(paper.questions, total_pages, report)
    _check_marks_total(paper, declared_total_marks, report)
    _check_declared_count(paper, declared_question_count, report)

    if paper.total_questions != len(paper.questions):
        report.error(
            "total_questions_mismatch",
            f"total_questions={paper.total_questions} but "
            f"{len(paper.questions)} questions are present",
        )
    if confidences:
        for q in paper.questions:
            value = confidences.get(q.number)
            if value is not None and value < 0.5:
                report.warn(
                    "low_confidence",
                    f"vision confidence {value:.2f} - the reading may be "
                    f"unreliable",
                    question=q.number,
                    page=q.page,
                )
    return report


def raise_if_invalid(report: ValidationReport, context: str = "extraction") -> None:
    """Raise :class:`ValidationError` when the report holds an error."""
    if report.errors:
        summary = "; ".join(d.code for d in report.errors[:5])
        raise ValidationError(
            f"{context} failed structural validation: {summary}",
            report.diagnostics,
        )
