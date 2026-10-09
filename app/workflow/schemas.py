"""Typed contracts for the Answer Book workflow.

Everything that crosses a node boundary is one of these models (as a plain
``dict`` inside LangGraph state, so checkpoints stay serializable). Nothing
here holds credentials, sessions, or clients.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict

from app.workflow.settings import SCHEMA_VERSION

QuestionType = Literal["mcq", "numerical", "short", "long"]
QUESTION_TYPES = ("mcq", "numerical", "short", "long")

EvidenceLevel = Literal["none", "structural", "llm_review", "cross_check", "symbolic"]
Severity = Literal["error", "warning", "info"]
#: verified   - verification passed AND evidence/confidence gates met (stored in KB)
#: cached     - an earlier *verified* result was accepted by the cache policy
#: unverified - an answer exists but did not meet the verified criteria (review required)
#: failed     - no usable answer could be produced (error, bad input, provider failure)
QuestionStatus = Literal["verified", "cached", "unverified", "failed"]
#: complete     - every question verified or cached
#: needs_review - every question has an answer, at least one is unverified
#: partial      - at least one question failed, at least one has an answer
#: failed       - no question produced an answer
PaperOutcome = Literal["complete", "needs_review", "partial", "failed"]


# --------------------------------------------------------------------------- input
class PaperContext(BaseModel):
    paper_id: str
    fingerprint: str
    subject: Optional[str] = None
    class_name: Optional[str] = None
    board: Optional[str] = None
    total_marks: Optional[int] = None
    sections: List[str] = Field(default_factory=list)
    source_file: Optional[str] = None
    #: True for ad-hoc single-question runs: no paper rows, no JSON export
    #: (the knowledge base is still consulted and fed).
    ephemeral: bool = False

    @property
    def subject_key(self) -> str:
        return (self.subject or "").strip().lower()


class WorkItem(BaseModel):
    """One question, normalized and ready to be solved independently."""

    item_id: str  # positional + unique even when the paper repeats a number
    index: int
    number: str
    text: str
    marks: int = Field(ge=1)
    marks_assumed: bool = False  # True when the source did not state marks
    type: QuestionType
    declared_type: str  # the type ingestion reported (differs if re-routed)
    options: List[str] = Field(default_factory=list)
    has_figure: bool = False
    section: Optional[str] = None
    section_title: Optional[str] = None
    page: Optional[int] = None
    choice_group: Optional[str] = None
    chapter: Optional[str] = None
    signature: str  # question-level cache signature (text+marks+board+class)
    warnings: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- drafts
class MarkSplitItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    marks: float = Field(gt=0)
    for_: str = Field(alias="for", min_length=1)


class CommonMistake(BaseModel):
    wrong: str = Field(min_length=1)
    why: str = Field(min_length=1)


class _DraftBase(BaseModel):
    """Fields every solver output must have (the frozen contract)."""

    answer: str = Field(min_length=1)
    steps: List[str] = Field(min_length=1)
    mark_split: List[MarkSplitItem] = Field(min_length=1)
    common_mistakes: List[CommonMistake] = Field(default_factory=list)
    self_confidence: float = Field(ge=0.0, le=1.0, description="The model's own confidence; NOT evidence.")

    @field_validator("steps")
    @classmethod
    def _no_blank_steps(cls, v: List[str]) -> List[str]:
        v = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not v:
            raise ValueError("steps must contain at least one non-empty step")
        return v


class McqDraft(_DraftBase):
    chosen_option: str = Field(description="Single letter of the chosen option, e.g. 'B'.")


class NumericalDraft(_DraftBase):
    final_value: float = Field(description="Final numeric answer as a number (no units).")
    unit: Optional[str] = None
    arithmetic_expression: Optional[str] = Field(
        default=None,
        description="Pure arithmetic (numbers, + - * / ** parentheses, sqrt, pi) that evaluates to final_value.",
    )


class TextDraft(_DraftBase):
    pass


class SolverDraft(BaseModel):
    """Type-independent view of a draft, stored in state and in the result."""

    strategy: QuestionType
    answer: str
    steps: List[str]
    mark_split: List[Dict[str, Any]]  # [{"marks": float, "for": str}]
    common_mistakes: List[Dict[str, str]]
    self_confidence: float
    chosen_option: Optional[str] = None
    final_value: Optional[float] = None
    unit: Optional[str] = None
    arithmetic_expression: Optional[str] = None
    prompt_version: str = ""
    model: str = ""


# --------------------------------------------------------------------------- independent solves / reviews
class BlindMcq(BaseModel):
    chosen_option: str
    reasoning: str = ""


class BlindNumeric(BaseModel):
    final_value: float
    reasoning: str = ""


class ReviewCheck(BaseModel):
    name: str
    passed: bool
    note: str = ""


class ReviewResult(BaseModel):
    checks: List[ReviewCheck] = Field(min_length=1)
    issues: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- verification
class Finding(BaseModel):
    check: str
    passed: bool
    severity: Severity = "error"
    message: str = ""
    method: Literal["deterministic", "independent_solve", "llm_review"] = "deterministic"


class VerificationReport(BaseModel):
    #: passed       - no error-level finding failed
    #: failed       - at least one error-level finding failed (repairable)
    #: inconclusive - the verifier itself could not run (provider failure);
    #:                nothing is known about the draft, repair would not help
    state: Literal["passed", "failed", "inconclusive"]
    evidence: EvidenceLevel = "none"
    findings: List[Finding] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)  # share of checks passed
    verifier_model: str = ""
    error: Optional[str] = None

    def failed_messages(self) -> List[str]:
        return [f"{f.check}: {f.message}" for f in self.findings if not f.passed and f.severity == "error"]


class AttemptRecord(BaseModel):
    """One draft plus what verification said about it. Append-only history."""

    attempt: int
    draft: Dict[str, Any]
    verification: Dict[str, Any]


# --------------------------------------------------------------------------- confidence
class ConfidenceAssessment(BaseModel):
    self_confidence: float  # what the model claimed (input only)
    evidence: EvidenceLevel
    evidence_cap: float
    verification_state: str
    verification_score: float  # "accuracy": share of checks that passed
    final_confidence: float
    verified: bool  # safe to present AND store as verified
    reasons: List[str] = Field(default_factory=list)  # why verified is False


# --------------------------------------------------------------------------- cache
class CacheDecision(BaseModel):
    looked_up: bool = False
    candidates: int = 0
    accepted: bool = False
    reasons: List[str] = Field(default_factory=list)  # rejection reasons / acceptance notes
    kb_id: Optional[int] = None
    source_paper_id: Optional[str] = None
    stored_at: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- results
class Diagnostic(BaseModel):
    stage: str
    severity: Severity
    code: str
    message: str


class SolutionOut(BaseModel):
    """The frozen SolutionContract, plus additive fields."""

    model_config = ConfigDict(populate_by_name=True)
    question_number: str
    answer: str = ""
    steps: List[str] = Field(default_factory=list)
    mark_split: List[Dict[str, Any]] = Field(default_factory=list)
    common_mistakes: List[Dict[str, str]] = Field(default_factory=list)
    confidence: float = 0.0
    verified_by: str = "none"
    needs_teacher_check: bool = True


class QuestionResult(BaseModel):
    item_id: str
    index: int
    question: Dict[str, Any]  # original question as given to the graph
    status: QuestionStatus
    origin: Literal["cache", "generated", "none"]
    solution: SolutionOut
    verification: Optional[VerificationReport] = None
    confidence_assessment: Optional[ConfidenceAssessment] = None
    cache: CacheDecision = Field(default_factory=CacheDecision)
    attempts: List[AttemptRecord] = Field(default_factory=list)
    stored_in_kb: bool = False
    diagnostics: List[Diagnostic] = Field(default_factory=list)
    prompt_version: str = ""


class PaperSummary(BaseModel):
    total: int
    verified: int
    cached: int
    unverified: int
    failed: int


class SolvedPaper(BaseModel):
    schema_version: str = SCHEMA_VERSION
    paper_id: str
    fingerprint: str
    status: str  # PaperOutcome (kept `str` so legacy readers never break)
    outcome: PaperOutcome
    metadata: Dict[str, Any]
    total_questions: int
    total_marks: Optional[int] = None
    sections: List[str] = Field(default_factory=list)
    summary: PaperSummary
    #: Contract-compatible list (one entry per question, in paper order).
    solutions: List[Dict[str, Any]]
    #: Full per-question detail (same order).
    results: List[QuestionResult]
    diagnostics: List[Diagnostic] = Field(default_factory=list)


# --------------------------------------------------------------------------- graph states
def _merge_results(left: Optional[List[dict]], right: Optional[List[dict]]) -> List[dict]:
    """Reducer for parallel question branches: idempotent, keyed by item_id.

    Branches may finish in any order and a resumed run may re-deliver a
    result; keying by ``item_id`` makes both harmless. Ordering is restored
    from ``index`` at aggregation time, never from arrival order.
    """
    merged: Dict[str, dict] = {r["item_id"]: r for r in (left or [])}
    for r in right or []:
        merged[r["item_id"]] = r
    return list(merged.values())


class QuestionState(TypedDict, total=False):
    """State of ONE question's sub-graph."""

    item: dict  # WorkItem
    ctx: dict  # PaperContext
    cache: dict  # CacheDecision
    stored: Optional[dict]  # accepted KB payload (cache hit only)
    draft: Optional[dict]  # SolverDraft (current)
    verification: Optional[dict]  # VerificationReport (latest)
    attempts: Annotated[List[dict], operator.add]  # AttemptRecord history
    repair_count: int
    repair_feedback: List[str]
    confidence: Optional[dict]  # ConfidenceAssessment
    stored_in_kb: bool
    independent: Optional[dict]  # blind solve by the verifier model (computed once per question)
    tainted: bool  # an MCQ repair saw cross-check feedback, so later agreement is not independent
    repair_failed: bool
    fatal: bool  # an unrecoverable problem for THIS question only
    diagnostics: Annotated[List[dict], operator.add]
    # The ONLY key shared with the paper graph (merged by item_id):
    results: Annotated[List[dict], _merge_results]


class PaperState(TypedDict, total=False):
    """State of the whole-paper graph."""

    paper_context: dict
    work_items: List[dict]
    expected_ids: List[str]
    results: Annotated[List[dict], _merge_results]
    paper_diagnostics: Annotated[List[dict], operator.add]
    solved_paper: dict
    output: dict
