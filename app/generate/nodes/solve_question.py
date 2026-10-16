
"""Initial question worker for the AnswerBook LangGraph workflow.

This version is deterministic and does not call an LLM.
Its placeholder output is intentionally marked for teacher review.
"""

from __future__ import annotations

from typing import Any

from app.generate.state import (
    QuestionResult,
    QuestionState,
    SolutionContract,
)


def solve_question(state: QuestionState) -> dict[str, Any]:
    """Process one question and return one reducer-compatible result."""
    work_item_id = state.get("work_item_id")
    question_index = state.get("question_index")
    question_number = state.get("number")
    question_text = state.get("text")

    # A malformed Send payload should fail explicitly.
    missing = [
        name
        for name, value in (
            ("work_item_id", work_item_id),
            ("question_index", question_index),
            ("number", question_number),
            ("text", question_text),
        )
        if value is None
    ]

    if missing:
        error = f"Worker input is missing required fields: {', '.join(missing)}."
        result: QuestionResult = {
            "work_item_id": str(work_item_id or "unknown"),
            "question_index": (
                question_index if isinstance(question_index, int) else -1
            ),
            "question_number": str(question_number or "unknown"),
            "solution": None,
            "status": "failed",
            "error": error,
        }
        return {
            "worker_results": [result],
        }

    assert isinstance(work_item_id, str)
    assert isinstance(question_index, int)
    assert isinstance(question_number, str)
    assert isinstance(question_text, str)

    if not question_text.strip():
        error = "Question text is empty."
        result = {
            "work_item_id": work_item_id,
            "question_index": question_index,
            "question_number": question_number,
            "solution": None,
            "status": "failed",
            "error": error,
        }
        return {
            "worker_results": [result],
        }

    # ---------------------------------------------------------
    # Initial placeholder implementation.
    # Replace this section with the actual solving pipeline.
    # ---------------------------------------------------------

    solution: SolutionContract = {
        "question_number": question_number,
        "answer": (
            "[PLACEHOLDER — no answer has been generated. "
            "Connect the appropriate solver before using this output.]"
        ),
        "steps": [
            "This is a test result confirming that the question "
            "was dispatched to a worker.",
            "No mathematical or factual solution has been generated.",
        ],
        "mark_split": [],
        "common_mistakes": [],
        "confidence": 0.0,
        "verified_by": "not_verified",
        "needs_teacher_check": True,
    }

    result = {
        "work_item_id": work_item_id,
        "question_index": question_index,
        "question_number": question_number,
        "solution": solution,
        "status": "needs_review",
        "error": None,
    }

    return {
        "worker_results": [result],
    }
