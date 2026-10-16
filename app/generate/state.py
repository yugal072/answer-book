"""State schemas for the AnswerBook question-solving workflow.

PaperState is the parent orchestration state.
QuestionState is the state for solving one individual question.
"""

import operator
from typing import List, Optional, Dict, Any, Literal, Annotated
from typing_extensions import TypedDict


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
    Mentioned in section 3 of brief."""
    
    question_number: str
    answer: str
    steps: List[str]
    mark_split: List[MarkSplitItem]
    common_mistakes: List[CommonMistakeItem]
    confidence: float
    verified_by: str
    needs_teacher_check: bool


# --- Main LangGraph States ---

QuestionType = Literal["numerical", "mcq", "short", "long"]

class QuestionInput(TypedDict):
    """Question data produced by ingestion and normalized by the adapter."""
    # 1. Inputs (from Ingestion / static JSON)
    number: str
    section: Optional[str]
    text: str
    marks: Optional[int]
    type: QuestionType
    options: Optional[List[str]]
    has_figure: bool
    page: Optional[int]
    choice_group: Optional[str]
    section_title: Optional[str]
    chapter: Optional[str]
   
 
class VerificationState(TypedDict, total=False):
    """Results from solution verification."""
    status: Literal["not_checked", "passed", "failed", "needs_review"]
    verified_by: Optional[str]
    reasons: list[str]
    checks: dict[str, Any]


class QuestionWorkItem(QuestionInput, total=False):
    """Question input plus identity and initial execution state.

    This is the item stored in PaperState and dispatched to a worker.
    """
    
    work_item_id: str
    question_index: int

    selected_route: Optional[str]
    attempts: int
    max_attempts: int

    raw_solution: Optional[dict[str, Any]]
    verification: Optional[VerificationState]
    confidence: Optional[float]

    cache_status: Literal["not_checked", "hit", "miss", "error"]
    solution: Optional[SolutionContract]
    error: Optional[str]


class QuestionState(QuestionWorkItem):
    """State received and updated by a single question-solving execution.

    A worker receives one question, never the complete paper's question list.
    """
    
# Paper Level Orchestration State

class PaperMetadata(TypedDict):
    subject: Optional[str]
    class_name: Optional[str]
    board: Optional[str]
    total_marks: Optional[int]
    sections: list[str]
    declared_total_questions: Optional[int]
    
class QuestionResult(TypedDict):
    """A completed question result associated with its original position."""

    work_item_id: str
    question_index: int
    question_number: str
    solution: Optional[SolutionContract]
    status: Literal["completed", "failed", "needs_review"]
    error: Optional[str]
    
class PaperState(TypedDict, total=False):
    """Parent state for orchestrating all questions in a paper."""

    paper_id: str
    fingerprint: str
    status: str
    paper_metadata: PaperMetadata

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