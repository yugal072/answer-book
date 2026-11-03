from __future__ import annotations

from typing import Any

from app.generate.solver_contracts import (
    deterministic_confidence,
    validate_solution_payload,
)


def valid_payload() -> dict[str, Any]:
    return {
        "answer": "x = 2",
        "steps": ["Subtract 1 from both sides."],
        "mark_split": [{"marks": 2, "for": "Correct method and result"}],
        "common_mistakes": [
            {"wrong": "x = 3", "why": "Incorrect arithmetic"}
        ],
    }


def test_valid_solution_payload_is_normalized():
    normalized, errors = validate_solution_payload(valid_payload(), "Q1", 2)
    assert errors == []
    assert normalized is not None
    assert normalized["question_number"] == "Q1"
    assert normalized["answer"] == "x = 2"


def test_rejects_missing_answer_and_bad_steps():
    normalized, errors = validate_solution_payload(
        {"answer": " ", "steps": [3]}, "1", 2
    )
    assert normalized is None
    assert any("answer" in error for error in errors)
    assert any("steps" in error for error in errors)


def test_rejects_mark_split_exceeding_question_marks():
    payload = valid_payload()
    payload["mark_split"] = [{"marks": 3, "for": "method"}]
    normalized, errors = validate_solution_payload(payload, "1", 2)
    assert normalized is None
    assert any("exceeding" in error for error in errors)


def test_confidence_is_capped_when_verification_did_not_pass():
    payload, errors = validate_solution_payload(valid_payload(), "1", 2)
    assert not errors
    assert payload is not None
    score = deterministic_confidence(
        payload,
        {
            "status": "failed",
            "checks": {
                "structure_valid": True,
                "answer_review_passed": True,
                "independent_check_passed": True,
            },
        },
        2,
    )
    assert score <= 0.59
