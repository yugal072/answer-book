"""Atomic question-level checkpointing tool for crash resilience.

Commits each question solution immediately to PostgreSQL upon completion.
Guarantees zero data loss if execution is interrupted mid-paper.
"""

from typing import Any, Dict
from sqlalchemy import func
from app.store.connection import get_db_session
from app.store.models import PaperTable, SolvedQuestionTable


def save_question_checkpoint(
    paper_id: str,
    signature: str,
    question_item: Dict[str, Any],
    solution_contract: Dict[str, Any],
) -> bool:
    """Saves a solved question immediately to PostgreSQL within its own transaction.

    Args:
        paper_id: The ID of the paper being processed.
        signature: Composite SHA-256 question signature.
        question_item: The input Question dictionary from ingestion.
        solution_contract: The verified solution dictionary adhering to Section 3.

    Returns:
        bool: True on successful commit.
    """
    q_num = str(question_item.get("number", solution_contract.get("question_number", "")))

    with get_db_session() as session:
        # Check if this question was already saved for this paper
        existing_row = (
            session.query(SolvedQuestionTable)
            .filter(
                SolvedQuestionTable.paper_id == paper_id,
                SolvedQuestionTable.question_number == q_num,
            )
            .first()
        )

        if existing_row:
            # Update existing row
            existing_row.question_signature = signature
            existing_row.section = question_item.get("section")
            existing_row.question_text = question_item.get("text", "")
            existing_row.marks = int(question_item.get("marks", 1))
            existing_row.question_type = question_item.get("type")
            existing_row.options = question_item.get("options")
            existing_row.has_figure = bool(question_item.get("has_figure", False))

            existing_row.answer = solution_contract.get("answer", "")
            existing_row.steps = solution_contract.get("steps", [])
            existing_row.mark_split = solution_contract.get("mark_split", [])
            existing_row.common_mistakes = solution_contract.get("common_mistakes", [])
            existing_row.confidence = float(solution_contract.get("confidence", 1.0))
            existing_row.verified_by = solution_contract.get("verified_by", "groq")
            existing_row.needs_teacher_check = bool(solution_contract.get("needs_teacher_check", False))
        else:
            # Create new row
            new_row = SolvedQuestionTable(
                paper_id=paper_id,
                question_signature=signature,
                question_number=q_num,
                section=question_item.get("section"),
                question_text=question_item.get("text", ""),
                marks=int(question_item.get("marks", 1)),
                question_type=question_item.get("type"),
                options=question_item.get("options"),
                has_figure=bool(question_item.get("has_figure", False)),
                answer=solution_contract.get("answer", ""),
                steps=solution_contract.get("steps", []),
                mark_split=solution_contract.get("mark_split", []),
                common_mistakes=solution_contract.get("common_mistakes", []),
                confidence=float(solution_contract.get("confidence", 1.0)),
                verified_by=solution_contract.get("verified_by", "groq"),
                needs_teacher_check=bool(solution_contract.get("needs_teacher_check", False)),
            )
            session.add(new_row)

        # Update parent paper's updated_at timestamp as heartbeat
        paper = session.query(PaperTable).filter(PaperTable.paper_id == paper_id).first()
        if paper:
            paper.updated_at = func.now()

    return True
