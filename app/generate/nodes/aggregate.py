"""Aggregate parallel question-worker results into the parent paper state."""

from __future__ import annotations

from typing import Any

from app.generate.state import PaperState, QuestionResult


def aggregate_results(state: PaperState) -> dict[str, Any]:
    """Collect worker results and return the ordered parent-level output."""
    questions = state.get("questions", [])
    worker_results = state.get("worker_results", [])
    errors = list(state.get("errors", []))

    expected_by_id: dict[str, int] = {
        question["work_item_id"]: question["question_index"]
        for question in questions
        if "work_item_id" in question and "question_index" in question
    }

    results_by_id: dict[str, QuestionResult] = {}

    for result in worker_results:
        work_item_id = result.get("work_item_id")

        if not work_item_id:
            errors.append("A worker returned a result without work_item_id.")
            continue

        if work_item_id not in expected_by_id:
            errors.append(
                f"Worker returned an unexpected work_item_id: {work_item_id}."
            )
            continue

        if work_item_id in results_by_id:
            errors.append(
                f"Duplicate worker result for work_item_id: {work_item_id}."
            )
            continue

        # Check that the worker preserved the expected original index.
        expected_index = expected_by_id[work_item_id]
        if result.get("question_index") != expected_index:
            errors.append(
                f"Incorrect question_index for {work_item_id}: "
                f"expected {expected_index}, "
                f"received {result.get('question_index')}."
            )
            continue

        results_by_id[work_item_id] = result

    ordered_results: list[QuestionResult] = []

    for question in sorted(
        questions,
        key=lambda item: item.get("question_index", 0),
    ):
        work_item_id = question.get("work_item_id")

        if not work_item_id:
            continue

        result = results_by_id.get(work_item_id)

        if result is None:
            error = f"No worker result received for {work_item_id}."
            errors.append(error)

            ordered_results.append(
                {
                    "work_item_id": work_item_id,
                    "question_index": question.get("question_index", -1),
                    "question_number": question.get("number", "unknown"),
                    "solution": None,
                    "status": "failed",
                    "error": error,
                }
            )
            continue

        ordered_results.append(result)

    failed_count = sum(
        result["status"] == "failed" for result in ordered_results
    )

    # "ready" means processing finished, not that every answer is verified.
    if state.get("status") == "failed" or (
        questions and failed_count == len(questions)
    ):
        paper_status = "failed"
    else:
        paper_status = "ready"

    return {
        "results": ordered_results,
        "errors": errors,
        "total_questions": len(questions),
        "current_index": len(questions),
        "status": paper_status,
    }
