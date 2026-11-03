"""Pure contracts and deterministic checks used by the question solver."""

from __future__ import annotations

from typing import Any

REQUIRED_SOLUTION_KEYS = {"answer", "steps", "mark_split", "common_mistakes"}


def validate_solution_payload(
    payload: Any, question_number: str, marks: int | None
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate structure and invariants without pretending to prove correctness."""
    errors: list[str] = []

    if not isinstance(payload, dict):
        return None, ["Solution must be a JSON object."]

    missing = sorted(REQUIRED_SOLUTION_KEYS - set(payload))
    if missing:
        errors.append("Missing required fields: " + ", ".join(missing))

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer must be a non-empty string.")

    steps = payload.get("steps")
    if not isinstance(steps, list) or any(
        not isinstance(x, str) or not x.strip() for x in steps
    ):
        errors.append("steps must be a list of non-empty strings.")

    mark_split = payload.get("mark_split")
    if not isinstance(mark_split, list):
        errors.append("mark_split must be a list.")
    else:
        total = 0
        for i, item in enumerate(mark_split):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("marks"), int)
                or item["marks"] < 0
                or not isinstance(item.get("for"), str)
            ):
                errors.append(
                    f"mark_split[{i}] must contain a non-negative integer marks and string 'for'."
                )
            else:
                total += item["marks"]

        if marks is not None and mark_split and total > marks:
            errors.append(
                f"mark_split totals {total}, exceeding question marks ({marks})."
            )

    mistakes = payload.get("common_mistakes")
    if not isinstance(mistakes, list):
        errors.append("common_mistakes must be a list.")
    else:
        for i, item in enumerate(mistakes):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("wrong"), str)
                or not isinstance(item.get("why"), str)
            ):
                errors.append(
                    f"common_mistakes[{i}] must contain string fields 'wrong' and 'why'."
                )

    if errors:
        return None, errors

    normalized: dict[str, Any] = {
        "question_number": str(question_number),
        "answer": answer.strip(),
        "steps": [s.strip() for s in steps],
        "mark_split": [
            {"marks": x["marks"], "for": x["for"].strip()} for x in mark_split
        ],
        "common_mistakes": [
            {"wrong": x["wrong"].strip(), "why": x["why"].strip()}
            for x in mistakes
        ],
    }
    return normalized, []


def deterministic_confidence(
    solution: dict[str, Any],
    verification: dict[str, Any],
    marks: int | None,
) -> float:
    """Conservative operational score, not a calibrated probability of truth."""
    score = 0.25

    if solution.get("answer", "").strip():
        score += 0.15

    if solution.get("steps"):
        score += 0.10

    if marks is not None and solution.get("mark_split"):
        total = sum(
            x.get("marks", 0)
            for x in solution["mark_split"]
            if isinstance(x, dict)
        )
        if total <= marks:
            score += 0.10

    checks = verification.get("checks", {})
    if checks.get("structure_valid"):
        score += 0.10
    if checks.get("answer_review_passed"):
        score += 0.20
    if checks.get("independent_check_passed"):
        score += 0.10

    if verification.get("status") != "passed":
        score = min(score, 0.59)

    return round(max(0.0, min(score, 0.99)), 2)
