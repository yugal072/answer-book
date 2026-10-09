"""Knowledge-base and paper-repository interfaces plus their implementations.

* :class:`KnowledgeBase` - the reusable Question-Solution store. Reads return
  *candidates* (the policy decides); writes are idempotent upserts keyed by
  ``context_key`` and never downgrade an existing stronger row.
* :class:`PaperRepository` - per-paper bookkeeping (status + per-question
  checkpoint rows) in the existing ``papers`` / ``solved_questions`` tables.
* ``Sql*`` classes run on SQLAlchemy (PostgreSQL in production). ``InMemory*``
  are deterministic fakes for graph tests; they are NOT evidence that
  PostgreSQL works - see the dedicated database tests for that.

Graph nodes receive these through closures, never through graph state, so no
session or connection is ever checkpointed.
"""
from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.store.knowledge import VerifiedSolutionTable
from app.store.models import PaperTable, SolvedQuestionTable
from app.tools.hash_utils import normalize_text
from app.workflow.cache_policy import StoredSolution, context_key
from app.workflow.schemas import (
    ConfidenceAssessment,
    PaperContext,
    QuestionResult,
    SolverDraft,
    VerificationReport,
    WorkItem,
)
from app.workflow.settings import EVIDENCE_RANK, PROMPT_VERSION, SCHEMA_VERSION


class StoreOutcome(BaseModel):
    kb_id: Optional[int]
    action: str  # inserted | updated | kept_existing


class KnowledgeBase(Protocol):
    def find_candidates(self, signature: str) -> List[StoredSolution]: ...

    def store_verified(
        self,
        item: WorkItem,
        ctx: PaperContext,
        draft: SolverDraft,
        solution: dict,
        assessment: ConfidenceAssessment,
        report: VerificationReport,
    ) -> StoreOutcome: ...


class PaperRepository(Protocol):
    def begin(self, ctx: PaperContext, total_questions: int) -> None: ...
    def save_question(self, ctx: PaperContext, result: QuestionResult) -> None: ...
    def finish(self, ctx: PaperContext, status: str, total_questions: int) -> None: ...


def _record_fields(
    item: WorkItem, ctx: PaperContext, draft: SolverDraft, solution: dict,
    assessment: ConfidenceAssessment, report: VerificationReport,
) -> dict:
    return dict(
        context_key=context_key(item, ctx),
        question_signature=item.signature,
        question_text_norm=normalize_text(item.text),
        marks=item.marks,
        board=(ctx.board or "").strip(),
        class_name=(ctx.class_name or "").strip(),
        subject=(ctx.subject or "").strip(),
        question_type=item.type,
        options=list(item.options),
        has_figure=item.has_figure,
        solution=solution,
        draft=draft.model_dump(),
        verification_status="verified",
        evidence=assessment.evidence,
        verification_score=assessment.verification_score,
        confidence=assessment.final_confidence,
        self_confidence=assessment.self_confidence,
        schema_version=SCHEMA_VERSION,
        prompt_version=draft.prompt_version or PROMPT_VERSION,
        solver_model=draft.model,
        verifier_model=report.verifier_model,
        source_paper_id=ctx.paper_id,
        source_question_number=item.number,
    )


def _is_upgrade(new_evidence: str, new_conf: float, old_evidence: str, old_conf: float) -> bool:
    return (EVIDENCE_RANK.get(new_evidence, 0), new_conf) >= (EVIDENCE_RANK.get(old_evidence, 0), old_conf)


# =========================================================================== SQL
class SqlKnowledgeBase:
    def __init__(self, session_factory: Callable[[], AbstractContextManager]) -> None:
        self._session = session_factory

    def find_candidates(self, signature: str) -> List[StoredSolution]:
        with self._session() as session:
            rows = (
                session.query(VerifiedSolutionTable)
                .filter(VerifiedSolutionTable.question_signature == signature)
                .order_by(VerifiedSolutionTable.id.asc())
                .all()
            )
            return [self._to_stored(r) for r in rows]

    @staticmethod
    def _to_stored(r: VerifiedSolutionTable) -> StoredSolution:
        return StoredSolution(
            id=r.id,
            question_signature=r.question_signature,
            question_text_norm=r.question_text_norm,
            marks=r.marks,
            board=r.board,
            class_name=r.class_name,
            subject=r.subject,
            question_type=r.question_type,
            options=list(r.options or []),
            has_figure=r.has_figure,
            solution=dict(r.solution),
            draft=dict(r.draft),
            verification_status=r.verification_status,
            evidence=r.evidence,
            verification_score=r.verification_score,
            confidence=r.confidence,
            self_confidence=r.self_confidence,
            schema_version=r.schema_version,
            prompt_version=r.prompt_version,
            solver_model=r.solver_model,
            verifier_model=r.verifier_model,
            source_paper_id=r.source_paper_id,
            source_question_number=r.source_question_number,
            updated_at=r.updated_at.isoformat() if r.updated_at else None,
        )

    def store_verified(self, item, ctx, draft, solution, assessment, report) -> StoreOutcome:
        fields = _record_fields(item, ctx, draft, solution, assessment, report)
        with self._session() as session:  # commits on success, rolls back on error
            row = session.query(VerifiedSolutionTable).filter_by(context_key=fields["context_key"]).first()
            if row is None:
                try:
                    with session.begin_nested():  # savepoint: a lost insert race must not poison the txn
                        row = VerifiedSolutionTable(**fields)
                        session.add(row)
                        session.flush()
                    return StoreOutcome(kb_id=row.id, action="inserted")
                except IntegrityError:
                    row = session.query(VerifiedSolutionTable).filter_by(context_key=fields["context_key"]).first()
                    if row is None:
                        raise
            if _is_upgrade(fields["evidence"], fields["confidence"], row.evidence, row.confidence):
                for k, v in fields.items():
                    setattr(row, k, v)
                session.flush()
                return StoreOutcome(kb_id=row.id, action="updated")
            return StoreOutcome(kb_id=row.id, action="kept_existing")


class SqlPaperRepository:
    """Per-paper rows in the pre-existing ``papers`` / ``solved_questions`` tables."""

    def __init__(self, session_factory: Callable[[], AbstractContextManager]) -> None:
        self._session = session_factory

    def begin(self, ctx: PaperContext, total_questions: int) -> None:
        with self._session() as session:
            paper = session.query(PaperTable).filter_by(paper_id=ctx.paper_id).first()
            if paper is None:
                paper = PaperTable(paper_id=ctx.paper_id, fingerprint=ctx.fingerprint)
                session.add(paper)
            paper.status = "solving"
            paper.subject = ctx.subject
            paper.class_name = ctx.class_name
            paper.board = ctx.board
            paper.total_questions = total_questions
            paper.total_marks = ctx.total_marks or 0
            paper.sections = ctx.sections
            paper.source_file = ctx.source_file

    def save_question(self, ctx: PaperContext, result: QuestionResult) -> None:
        if result.status == "failed":
            return  # nothing worth checkpointing; a re-run will retry it
        q = result.question
        sol = result.solution
        number = str(q.get("number"))
        sig = str(q.get("signature", ""))
        with self._session() as session:
            row = (
                session.query(SolvedQuestionTable)
                .filter_by(paper_id=ctx.paper_id, question_number=number, question_signature=sig)
                .first()
            )
            if row is None:
                row = SolvedQuestionTable(paper_id=ctx.paper_id, question_number=number, question_signature=sig)
                session.add(row)
            row.section = q.get("section")
            row.question_text = q.get("text", "")
            row.marks = int(q.get("marks", 1))
            row.question_type = q.get("type")
            row.options = q.get("options")
            row.has_figure = bool(q.get("has_figure", False))
            row.answer = sol.answer
            row.steps = sol.steps
            row.mark_split = sol.mark_split
            row.common_mistakes = sol.common_mistakes
            row.confidence = sol.confidence
            row.verified_by = sol.verified_by[:100]
            row.needs_teacher_check = sol.needs_teacher_check

    def finish(self, ctx: PaperContext, status: str, total_questions: int) -> None:
        with self._session() as session:
            paper = session.query(PaperTable).filter_by(paper_id=ctx.paper_id).first()
            if paper is not None:
                paper.status = status
                paper.updated_at = func.now()


# =========================================================================== in-memory fakes
class InMemoryKnowledgeBase:
    def __init__(self) -> None:
        self._rows: Dict[str, StoredSolution] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self.fail_on_find: Optional[Exception] = None
        self.fail_on_store: Optional[Exception] = None
        self.find_calls = 0
        self.store_calls = 0

    def find_candidates(self, signature: str) -> List[StoredSolution]:
        self.find_calls += 1
        if self.fail_on_find:
            raise self.fail_on_find
        with self._lock:
            return [r.model_copy(deep=True) for r in self._rows.values() if r.question_signature == signature]

    def store_verified(self, item, ctx, draft, solution, assessment, report) -> StoreOutcome:
        self.store_calls += 1
        if self.fail_on_store:
            raise self.fail_on_store
        f = _record_fields(item, ctx, draft, solution, assessment, report)
        key = f.pop("context_key")
        with self._lock:
            old = self._rows.get(key)
            if old is not None and not _is_upgrade(f["evidence"], f["confidence"], old.evidence, old.confidence):
                return StoreOutcome(kb_id=old.id, action="kept_existing")
            row = StoredSolution(id=old.id if old else self._next_id, updated_at=str(self._next_id), **f)
            if old is None:
                self._next_id += 1
            self._rows[key] = row
            return StoreOutcome(kb_id=row.id, action="updated" if old else "inserted")

    # test helpers
    def all(self) -> List[StoredSolution]:
        return list(self._rows.values())

    def put(self, row: StoredSolution, key: Optional[str] = None) -> StoredSolution:
        with self._lock:
            row = row.model_copy(update={"id": row.id or self._next_id})
            self._next_id = max(self._next_id, (row.id or 0) + 1)
            self._rows[key or f"k{row.id}"] = row
            return row


class InMemoryPaperRepository:
    def __init__(self) -> None:
        self.papers: Dict[str, dict] = {}
        self.questions: Dict[Tuple[str, str, str], QuestionResult] = {}
        self.events: List[str] = []

    def begin(self, ctx: PaperContext, total_questions: int) -> None:
        self.papers[ctx.paper_id] = {"status": "solving", "total": total_questions}
        self.events.append("begin")

    def save_question(self, ctx: PaperContext, result: QuestionResult) -> None:
        if result.status != "failed":
            self.questions[(ctx.paper_id, result.item_id, result.question.get("number", ""))] = result

    def finish(self, ctx: PaperContext, status: str, total_questions: int) -> None:
        self.papers[ctx.paper_id]["status"] = status
        self.events.append(f"finish:{status}")
