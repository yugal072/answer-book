"""Graph Adapter: canonical ``Paper`` -> validated, normalized ``WorkItem`` list.

This is the seam between ingestion and the graph. It does not guess: missing
marks are flagged (``marks_assumed``), an MCQ without options is re-routed to a
text strategy with a warning, and unknown paper metadata stays unknown.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.models.loaders_models import Paper
from app.tools.hash_utils import generate_question_signature
from app.workflow.schemas import QUESTION_TYPES, Diagnostic, PaperContext, WorkItem


class AdapterError(ValueError):
    """The paper cannot be turned into work items (maps to HTTP 422)."""


@dataclass
class AdaptedPaper:
    context: PaperContext
    items: List[WorkItem]
    diagnostics: List[Diagnostic] = field(default_factory=list)


def _clean(s: Optional[str]) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").replace("\r", "")).strip()


def adapt_paper(
    paper: Paper,
    *,
    limit: Optional[int] = None,
    source_file: Optional[str] = None,
) -> AdaptedPaper:
    if not paper.questions:
        raise AdapterError(f"Paper '{paper.paper_id}' contains no questions.")
    if limit is not None and limit < 1:
        raise AdapterError("limit must be >= 1")

    ctx = PaperContext(
        paper_id=paper.paper_id,
        fingerprint=paper.fingerprint,
        subject=_clean(paper.subject) or None,
        class_name=_clean(paper.class_name) or None,
        board=_clean(paper.board) or None,
        total_marks=paper.total_marks,
        sections=list(paper.sections or []),
        source_file=source_file,
    )

    questions = paper.questions[:limit] if limit else paper.questions
    diagnostics: List[Diagnostic] = []
    items: List[WorkItem] = []
    for idx, q in enumerate(questions, start=1):
        warnings: List[str] = []
        text = _clean(q.text)
        options = [_clean(o) for o in (q.options or []) if _clean(o)]

        marks_assumed = q.marks is None or q.marks < 1
        marks = 1 if marks_assumed else int(q.marks)
        if marks_assumed:
            warnings.append("marks not stated in the source; assumed 1 (mark-split sum is not enforced)")

        declared = q.type
        qtype = declared if declared in QUESTION_TYPES else "short"
        if qtype == "mcq" and len(options) < 2:
            qtype = "short"
            warnings.append(
                f"declared MCQ but {len(options)} option(s) were extracted; routed to the 'short' strategy"
            )
        if qtype != "mcq" and options:
            # keep options as context (e.g. match-the-following) but they do not make it an MCQ
            warnings.append("non-MCQ question carries options; shown to the solver as context")

        item = WorkItem(
            item_id=f"q{idx:03d}",
            index=idx,
            number=str(q.number),
            text=text,
            marks=marks,
            marks_assumed=marks_assumed,
            type=qtype,
            declared_type=declared,
            options=options,
            has_figure=bool(q.has_figure),
            section=q.section,
            section_title=q.section_title,
            page=q.page,
            choice_group=q.choice_group,
            chapter=q.chapter,
            signature=generate_question_signature(
                text, marks, ctx.board or "unspecified", ctx.class_name or ""
            ),
            warnings=warnings,
        )
        items.append(item)
        for w in warnings:
            diagnostics.append(
                Diagnostic(stage="adapter", severity="warning", code="normalized", message=f"Q{item.number}: {w}")
            )
    return AdaptedPaper(context=ctx, items=items, diagnostics=diagnostics)
