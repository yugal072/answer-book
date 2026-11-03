"""State schemas for the AnswerBook question-solving workflow.

PaperState is the parent orchestration state.
QuestionState is the state for solving one individual question.
"""

import operator
from typing import Annotated, Any, Literal
from typing_extensions import NotRequired, TypedDict


# --- Helper Types matching the Frozen API Contract ---

class CommonMistakeItem(TypedDict):
    wrong: str
    why: str


# Using functional syntax because 'for' is a reserved keyword in Python
MarkSplitItem = TypedDict(
    "MarkSplitItem",
    {
        "marks": int,
        "for": str,
    },
)


class SolutionContract(TypedDict):
    """Normalized final output contract for one solved question. 
    
    Mentioned in section 3 of brief.
    """
    
    question_number: str
    answer: str
    steps: list[str]
    mark_split: list[MarkSplitItem]
    common_mistakes: list[CommonMistakeItem]
    confidence: float
    verified_by: str
    needs_teacher_check: bool


# --- Main LangGraph States ---

QuestionType = Literal["numerical", "mcq", "short", "long"]

class QuestionInput(TypedDict):
    """Question data produced by ingestion and normalized by the adapter."""
    # 1. Inputs (from Ingestion / static JSON)
    number: str
    section: str | None
    text: str
    marks: int | None
    type: QuestionType
    options: list[str] | None
    has_figure: bool
    page: int | None
    choice_group: str | None
    section_title: str | None
    chapter: str | None
   
 
class VerificationState(TypedDict, total=False):
    """Results from solution verification."""
    
    status: Literal["not_checked", "passed", "failed", "needs_review"]
    verified_by: str | None
    reasons: list[str]
    checks: dict[str, Any]


class QuestionWorkItem(QuestionInput, total=False):
    """Question input plus identity and initial execution state.

    This is the item stored in PaperState and dispatched to a worker.
    """
    
    work_item_id: str
    question_index: int

    selected_route: dict[str, Any] | None
    attempts: int
    max_attempts: int

    raw_solution: dict[str, Any] | None
    verification: VerificationState | None
    confidence: float | None

    cache_status: Literal["not_checked", "hit", "miss", "error"]
    solution: SolutionContract | None
    error: str | None


class QuestionState(QuestionWorkItem):
    """State received and updated by a single question-solving subgraph."""
    
    paper_id: str
    paper_metadata: dict[str, Any]
    selected_route: dict[str, Any] | None
    plan_error: str | None
    cache_error: str | None
    cached_solution: dict[str, Any] | None
    question_signature: str | None
    generation_error: str | None
    repair_feedback: list[str]
    figure_image_bytes: bytes | None
    question_result: dict[str, Any] | None
    generated_diagram_base64: str | None
    figure_interpretation: dict[str, Any] | None
    model_used: str | None
    tool_results: list[dict[str, Any]] | None
    
    
# Paper Level Orchestration State

class PaperMetadata(TypedDict):
    subject: str | None
    class_name: str | None
    board: str | None
    total_marks: int | None
    sections: list[str]
    declared_total_questions: int | None
    
class QuestionResult(TypedDict):
    """A completed question result associated with its original position."""

    work_item_id: str
    question_index: int
    question_number: str
    solution: SolutionContract | None
    status: Literal["completed", "failed", "needs_review"]
    error: str | None
    artifacts: NotRequired[list[dict[str, Any]]]
    
class PaperState(TypedDict, total=False):
    """Parent state for orchestrating all questions in a paper."""

    paper_id: str
    fingerprint: str
    status: str
    paper_metadata: dict[str, Any]

    # All question work items, preserving ingestion order.
    questions: list[QuestionWorkItem]
    total_questions: int

    # Parent-level progress and collected worker outputs.
    current_index: int
    results: list[QuestionResult]

    # Paper-level failure information.
    errors: list[str]

    # Parallel workers append their individual results through this reducer
    worker_results: Annotated[list[QuestionResult], operator.add]