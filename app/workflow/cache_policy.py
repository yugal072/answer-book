"""Cache-acceptance policy: when may a stored solution be reused?

A database row is a *candidate*, never proof. A candidate is accepted only if
EVERY check below passes, and every rejection records its reason, so reuse is
deterministic, auditable and testable. The policy is pure: it takes values and
returns a decision; it performs no I/O.

Checks
------
1.  status            stored.verification_status == "verified"
2.  schema            stored.schema_version == current SCHEMA_VERSION
3.  prompt            stored.prompt_version is in settings.accepted_prompt_versions
4.  figure            neither question depends on a figure (text alone is not the whole question)
5.  marks_known       the current question states its marks
6.  text              normalized question text is identical (guards signature collisions)
7.  marks             same marks
8.  context           same board, class and subject (unknown only matches unknown)
9.  type              same question type
10. options           same options in the same order (letters map to the same text)
11. evidence          stored evidence >= minimum for the type, confidence >= store gate
12. contract          the stored solution re-passes the DETERMINISTIC verification
                      checks against the CURRENT question (mark-split sum, option
                      validity, answer/choice consistency, arithmetic recompute)

Among accepted candidates the strongest evidence wins, then confidence, then
the most recent, then the lowest id (so ties are stable).
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from app.tools.hash_utils import normalize_text
from app.workflow.schemas import CacheDecision, PaperContext, SolverDraft, WorkItem
from app.workflow.settings import EVIDENCE_RANK, SCHEMA_VERSION, WorkflowSettings
from app.workflow.verification import deterministic_checks


class StoredSolution(BaseModel):
    """A knowledge-base row as the policy sees it (storage-agnostic)."""

    id: Optional[int] = None
    question_signature: str
    question_text_norm: str
    marks: int
    board: str = ""
    class_name: str = ""
    subject: str = ""
    question_type: str
    options: List[str] = Field(default_factory=list)
    has_figure: bool = False
    solution: dict
    draft: dict
    verification_status: str
    evidence: str
    verification_score: float
    confidence: float
    self_confidence: float = 0.0
    schema_version: str
    prompt_version: str
    solver_model: str = ""
    verifier_model: str = ""
    source_paper_id: Optional[str] = None
    source_question_number: Optional[str] = None
    updated_at: Optional[str] = None


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def options_fingerprint(options: Sequence[str]) -> str:
    return json.dumps([normalize_text(o) for o in options], ensure_ascii=False)


def context_key(item: WorkItem, ctx: PaperContext) -> str:
    """Idempotency key of a knowledge-base row."""
    parts = [
        item.signature,
        _norm(ctx.subject),
        item.type,
        options_fingerprint(item.options),
        "fig" if item.has_figure else "nofig",
        SCHEMA_VERSION,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def evaluate_candidate(
    item: WorkItem, ctx: PaperContext, stored: StoredSolution, s: WorkflowSettings
) -> List[str]:
    """Return the list of rejection reasons ([] means acceptable)."""
    why: List[str] = []
    if stored.verification_status != "verified":
        why.append(f"stored result is '{stored.verification_status}', not verified")
    if stored.schema_version != SCHEMA_VERSION:
        why.append(f"schema_version {stored.schema_version} != {SCHEMA_VERSION}")
    if stored.prompt_version not in s.accepted_prompt_versions:
        why.append(f"prompt_version {stored.prompt_version} is not accepted")
    if item.has_figure or stored.has_figure:
        why.append("question depends on a figure; text alone does not identify it")
    if item.marks_assumed:
        why.append("current question does not state its marks")
    if normalize_text(item.text) != stored.question_text_norm:
        why.append("normalized question text differs")
    if item.marks != stored.marks:
        why.append(f"marks differ ({item.marks} vs stored {stored.marks})")
    if _norm(ctx.board) != _norm(stored.board):
        why.append("board differs")
    if _norm(ctx.class_name) != _norm(stored.class_name):
        why.append("class differs")
    if _norm(ctx.subject) != _norm(stored.subject):
        why.append("subject differs")
    if item.type != stored.question_type:
        why.append(f"question type differs ({item.type} vs stored {stored.question_type})")
    if options_fingerprint(item.options) != options_fingerprint(stored.options):
        why.append("options differ")
    need = s.min_evidence.get(item.type, "cross_check")
    if EVIDENCE_RANK.get(stored.evidence, 0) < EVIDENCE_RANK.get(need, 99):
        why.append(f"stored evidence '{stored.evidence}' is below '{need}'")
    if stored.confidence < s.store_min_confidence:
        why.append(f"stored confidence {stored.confidence:.2f} below {s.store_min_confidence:.2f}")
    if why:
        return why  # contract re-validation is pointless on an already-rejected row

    # 12. contract: re-run the deterministic checks against the current question
    try:
        draft = SolverDraft.model_validate(stored.draft)
    except Exception as exc:  # noqa: BLE001
        return [f"stored draft no longer matches the output contract: {type(exc).__name__}"]
    findings, _ = deterministic_checks(item, draft, s.numeric_rel_tol)
    bad = [f"{f.check}: {f.message}" for f in findings if not f.passed and f.severity == "error"]
    if bad:
        why.append("stored solution fails current deterministic checks: " + "; ".join(bad))
    return why


def decide(
    item: WorkItem, ctx: PaperContext, candidates: Sequence[StoredSolution], s: WorkflowSettings
) -> Tuple[CacheDecision, Optional[StoredSolution]]:
    """Pick the best acceptable candidate, or explain why none was."""
    if not candidates:
        return CacheDecision(looked_up=True, candidates=0, accepted=False, reasons=["no stored candidate"]), None

    accepted: List[StoredSolution] = []
    reasons: List[str] = []
    for cand in candidates:
        why = evaluate_candidate(item, ctx, cand, s)
        if why:
            reasons.append(f"candidate {cand.id}: " + "; ".join(why))
        else:
            accepted.append(cand)

    if not accepted:
        return CacheDecision(looked_up=True, candidates=len(candidates), accepted=False, reasons=reasons), None

    accepted.sort(
        key=lambda c: (
            -EVIDENCE_RANK.get(c.evidence, 0),
            -c.confidence,
            _neg_str(c.updated_at),
            c.id if c.id is not None else 0,
        )
    )
    best = accepted[0]
    return (
        CacheDecision(
            looked_up=True,
            candidates=len(candidates),
            accepted=True,
            reasons=[f"candidate {best.id} passed all {12} acceptance checks"] + reasons,
            kb_id=best.id,
            source_paper_id=best.source_paper_id,
            stored_at=best.updated_at,
        ),
        best,
    )


def _neg_str(v: Optional[str]) -> Tuple[int, ...]:
    """Sort key making later timestamps come first (string comparison, descending)."""
    return tuple(-ord(c) for c in (v or ""))
