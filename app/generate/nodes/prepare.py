"""Validate PaperState, PaperMetadata, and QuestionWorkItems before fan-out.

Prepare a PaperState for question-level LangGraph execution.
"""

from __future__ import annotations

from typing import Any

from app.generate.state import PaperState, QuestionWorkItem

REQUIRED_QUESTION_FIELDS = ("work_item_id", "question_index", "number", "text")


def prepare_paper(state: PaperState) -> dict[str, Any]:
    """Validate and prepare question work items before fan-out.

    Returns a partial state update. Invalid parent input fails the paper
    gracefully instead of dispatching malformed worker tasks.
    """
    paper_id = state.get("paper_id")
    questions = state.get("questions", [])

    if not paper_id:
        return {
            "status": "failed",
            "errors": ["PaperState is missing paper_id."],
            "total_questions": 0,
            "worker_results": [],
        }

    if not isinstance(questions, list):
        return {
            "status": "failed",
            "errors": ["PaperState.questions must be a list."],
            "total_questions": 0,
            "worker_results": [],
        }

    prepared: list[QuestionWorkItem] = []
    errors = list(state.get("errors", []))
    seen_ids: set[str] = set()

    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            errors.append(f"Question at index {index} is not a dictionary.")
            continue

        missing = [
            field
            for field in REQUIRED_QUESTION_FIELDS
            if field not in item or item[field] is None
        ]
        if missing:
            errors.append(
                f"Question at index {index} is missing required fields: "
                f"{', '.join(missing)}."
            )
            continue

        work_item_id = item["work_item_id"]
        if not isinstance(work_item_id, str) or not work_item_id.strip():
            errors.append(
                f"Question at index {index} has an invalid work_item_id."
            )
            continue

        if work_item_id in seen_ids:
            errors.append(f"Duplicate work_item_id: {work_item_id}.")
            continue
        seen_ids.add(work_item_id)

        if not isinstance(item["number"], str) or not item["number"].strip():
            errors.append(f"Question at index {index} has an invalid number.")
            continue

        if not isinstance(item["text"], str) or not item["text"].strip():
            errors.append(f"Question at index {index} has empty question text.")
            continue

        # Copy the item; do not mutate the incoming state.
        normalized = dict(item)
        normalized["question_index"] = index
        normalized["attempts"] = 0
        normalized.setdefault("max_attempts", 2)
        normalized["selected_route"] = None
        normalized["raw_solution"] = None
        normalized["verification"] = None
        normalized["confidence"] = None
        normalized["cache_status"] = "not_checked"
        normalized["solution"] = None
        normalized["error"] = None

        prepared.append(normalized)  # type: ignore[arg-type]

    # A malformed item should not silently disappear from the paper.
    if errors:
        return {
            "status": "failed",
            "errors": errors,
            "questions": prepared,
            "total_questions": len(questions),
            "worker_results": [],
        }

    return {
        "status": "solving",
        "questions": prepared,
        "total_questions": len(prepared),
        "current_index": 0,
        "results": [],
        "errors": [],
        "worker_results": [],
    }
