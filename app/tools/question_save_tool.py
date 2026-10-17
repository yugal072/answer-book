"""Atomic question-level checkpointing tool for crash resilience.

Commits each question solution immediately to PostgreSQL upon completion.
Guarantees zero data loss if execution is interrupted mid-paper.
"""

from __future__ import annotations

from typing import Any
from sqlalchemy import func

from app.store.connection import get_db_session
from app.store.models import PaperTable, SolvedQuestionTable


def save_question_checkpoint(
    paper_id: str,
    signature: str,
    question_item: dict[str, Any],
    solution_contract: dict[str, Any],
) -> bool:
    """Save a solved question immediately to PostgreSQL within its own transaction.

    Args:
        paper_id: The ID of the paper being processed.
        signature: Composite SHA-256 question signature.
        question_item: The input Question dictionary from ingestion.
        solution_contract: The verified solution dictionary adhering to Section 3.

    Returns:
        True on successful commit.
    """
    q_num = str(
        question_item.get("number", solution_contract.get("question_number", ""))
    )

    with get_db_session() as session:
        # Check if this question was already saved for this paper
        work_item_id = question_item.get("work_item_id")
        query = session.query(SolvedQuestionTable).filter(
            SolvedQuestionTable.paper_id == paper_id
        )
        if work_item_id:
            query = query.filter(
                SolvedQuestionTable.work_item_id == str(work_item_id)
            )
        else:
            # Compatibility path for callers that have not yet adopted work_item_id.
            query = query.filter(SolvedQuestionTable.question_number == q_num)

        existing_row = query.first()

        selected_route = question_item.get("selected_route") or {}
        verification = question_item.get("verification") or {}

        model_used = question_item.get("model_used")
        prompt_version = selected_route.get("prompt_version")
        verification_status = verification.get("status")
        verification_reasons = verification.get("reasons", [])

        section = question_item.get("section")
        question_text = question_item.get("text", "")
        marks = int(question_item.get("marks") or 0)
        question_type = question_item.get("type")
        options = question_item.get("options")
        has_figure = bool(question_item.get("has_figure", False))

        answer = solution_contract.get("answer", "")
        steps = solution_contract.get("steps", [])
        mark_split = solution_contract.get("mark_split", [])
        common_mistakes = solution_contract.get("common_mistakes", [])
        confidence = float(solution_contract.get("confidence", 0.0))
        verified_by = solution_contract.get("verified_by", "groq")
        needs_teacher_check = bool(
            solution_contract.get("needs_teacher_check", True)
        )

        if existing_row:
            existing_row.question_signature = signature
            existing_row.work_item_id = (
                str(work_item_id) if work_item_id else existing_row.work_item_id
            )
            existing_row.question_index = question_item.get("question_index")
            existing_row.model_used = model_used
            existing_row.prompt_version = prompt_version
            existing_row.verification_status = verification_status
            existing_row.verification_reasons = verification_reasons
            existing_row.section = section
            existing_row.question_text = question_text
            existing_row.marks = marks
            existing_row.question_type = question_type
            existing_row.options = options
            existing_row.has_figure = has_figure

            existing_row.answer = answer
            existing_row.steps = steps
            existing_row.mark_split = mark_split
            existing_row.common_mistakes = common_mistakes
            existing_row.confidence = confidence
            existing_row.verified_by = verified_by
            existing_row.needs_teacher_check = needs_teacher_check
        else:
            new_row = SolvedQuestionTable(
                paper_id=paper_id,
                question_signature=signature,
                question_number=q_num,
                work_item_id=str(work_item_id) if work_item_id else None,
                question_index=question_item.get("question_index"),
                model_used=model_used,
                prompt_version=prompt_version,
                verification_status=verification_status,
                verification_reasons=verification_reasons,
                section=section,
                question_text=question_text,
                marks=marks,
                question_type=question_type,
                options=options,
                has_figure=has_figure,
                answer=answer,
                steps=steps,
                mark_split=mark_split,
                common_mistakes=common_mistakes,
                confidence=confidence,
                verified_by=verified_by,
                needs_teacher_check=needs_teacher_check,
            )
            session.add(new_row)

        # Update parent paper's updated_at timestamp as heartbeat
        paper = (
            session.query(PaperTable)
            .filter(PaperTable.paper_id == paper_id)
            .first()
        )
        if paper:
            paper.updated_at = func.now()

    return True
