"""Paper-level caching and status management tools.

Implements Recommendation A:
- Fast pre-flight cache check using raw file SHA-256 fingerprint.
- Checkpoint loader for smart resume after crashes.
- Paper lifecycle status updates ('parsing' -> 'solving' -> 'ready' / 'failed').
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import func
from app.store.connection import get_db_session
from app.store.models import PaperTable, SolvedQuestionTable


def check_paper_cache(fingerprint: str) -> Optional[Dict[str, Any]]:
    """Checks if a paper has already been completely solved.

    Args:
        fingerprint: SHA-256 hash of the input paper file.

    Returns:
        dict: Complete Solution Book payload if status == 'ready', else None.
    """
    if not fingerprint:
        return None

    with get_db_session() as session:
        paper = (
            session.query(PaperTable)
            .filter(PaperTable.fingerprint == fingerprint)
            .first()
        )

        if not paper or paper.status != "ready":
            return None

        # Fetch all solved questions for this paper, ordered by database ID
        q_rows = (
            session.query(SolvedQuestionTable)
            .filter(SolvedQuestionTable.paper_id == paper.paper_id)
            .order_by(SolvedQuestionTable.id.asc())
            .all()
        )

        solutions: List[Dict[str, Any]] = []
        for q in q_rows:
            solutions.append({
                "question_number": q.question_number,
                "answer": q.answer,
                "steps": q.steps or [],
                "mark_split": q.mark_split or [],
                "common_mistakes": q.common_mistakes or [],
                "confidence": q.confidence or 1.0,
                "verified_by": f"cache:paper_fingerprint:{q.verified_by or 'db'}",
                "needs_teacher_check": q.needs_teacher_check,
            })

        return {
            "paper_id": paper.paper_id,
            "fingerprint": paper.fingerprint,
            "status": "ready",
            "metadata": {
                "subject": paper.subject,
                "class_name": paper.class_name,
                "board": paper.board,
            },
            "total_questions": len(solutions),
            "total_marks": paper.total_marks,
            "sections": paper.sections or [],
            "solutions": solutions,
        }


def get_paper_resume_info(paper_id: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Fetches already-solved questions for crash recovery.

    Args:
        paper_id: The ID of the paper being processed.

    Returns:
        tuple: (list of already solved SolutionContract dicts, set of solved question numbers)
    """
    if not paper_id:
        return [], set()

    with get_db_session() as session:
        q_rows = (
            session.query(SolvedQuestionTable)
            .filter(SolvedQuestionTable.paper_id == paper_id)
            .order_by(SolvedQuestionTable.id.asc())
            .all()
        )

        existing_solutions: List[Dict[str, Any]] = []
        solved_numbers: Set[str] = set()

        for q in q_rows:
            solved_numbers.add(q.question_number)
            existing_solutions.append({
                "question_number": q.question_number,
                "answer": q.answer,
                "steps": q.steps or [],
                "mark_split": q.mark_split or [],
                "common_mistakes": q.common_mistakes or [],
                "confidence": q.confidence or 1.0,
                "verified_by": f"checkpoint:{q.verified_by or 'db'}",
                "needs_teacher_check": q.needs_teacher_check,
            })

        return existing_solutions, solved_numbers


def upsert_paper_record(
    paper_id: str,
    fingerprint: str,
    status: str = "solving",
    subject: Optional[str] = None,
    class_name: Optional[str] = None,
    board: str = "CBSE",
    total_questions: int = 0,
    total_marks: int = 0,
    sections: Optional[List[str]] = None,
    source_file: Optional[str] = None,
) -> None:
    """Inserts or updates the paper record when ingestion starts."""
    with get_db_session() as session:
        paper = session.query(PaperTable).filter(PaperTable.paper_id == paper_id).first()

        if paper:
            paper.status = status
            if subject:
                paper.subject = subject
            if class_name:
                paper.class_name = class_name
            if board:
                paper.board = board
            if total_questions:
                paper.total_questions = total_questions
            if total_marks:
                paper.total_marks = total_marks
            if sections:
                paper.sections = sections
            if source_file:
                paper.source_file = source_file
            paper.updated_at = func.now()
        else:
            paper = PaperTable(
                paper_id=paper_id,
                fingerprint=fingerprint,
                status=status,
                subject=subject,
                class_name=class_name,
                board=board,
                total_questions=total_questions,
                total_marks=total_marks,
                sections=sections,
                source_file=source_file,
            )
            session.add(paper)


def mark_paper_status(paper_id: str, status: str) -> None:
    """Updates the status of a paper (e.g. 'ready' or 'failed')."""
    with get_db_session() as session:
        paper = session.query(PaperTable).filter(PaperTable.paper_id == paper_id).first()
        if paper:
            paper.status = status
            paper.updated_at = func.now()
