"""End-to-end pipeline integration test for LangGraph AnswerBook solver.

Exercises:
1. Ingested Paper / Question Pydantic validation.
2. State adaptation (paper_to_graph_state).
3. Graph execution (prepare -> dispatch -> solve_question -> aggregate).
4. Question-level planning, tool execution, candidate generation, verification, and aggregation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from app.generate.adapter import paper_to_graph_state
from app.generate.prompts import PLANNER_SYSTEM, SOLUTION_SYSTEM, VERIFY_SYSTEM
from app.models.loaders_models import Paper, Question


def mock_call_groq_json(
    system_prompt: str,
    user_prompt: str,
    max_retries: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Deterministic mock for LLM calls during pipeline testing."""
    if system_prompt == PLANNER_SYSTEM:
        return {
            "task_summary": "Solve numerical arithmetic question",
            "complexity": "easy",
            "model_tier": "cheap",
            "needs_vision": False,
            "needs_image_generation": False,
            "needs_external_tools": ["calculator"],
            "tool_requests": [
                {
                    "name": "calculator",
                    "expression": "25 * 4",
                    "purpose": "evaluate product",
                }
            ],
            "verification_focus": ["arithmetic", "units"],
            "rationale": "Simple direct calculation.",
        }

    if system_prompt == SOLUTION_SYSTEM:
        return {
            "answer": "100",
            "steps": ["Multiply 25 by 4 to obtain 100."],
            "mark_split": [{"marks": 2, "for": "Correct product and steps"}],
            "common_mistakes": [
                {"wrong": "90", "why": "Arithmetic error during multiplication"}
            ],
        }

    if system_prompt == VERIFY_SYSTEM:
        return {
            "passed": True,
            "correctness": 1.0,
            "issues": [],
            "checks": {
                "structure_valid": True,
                "answer_review_passed": True,
                "independent_check_passed": True,
            },
            "suggested_repairs": [],
        }

    raise ValueError(f"Unexpected system prompt: {system_prompt[:50]}...")


def test_full_pipeline_execution():
    """Verify end-to-end execution of the LangGraph workflow on a sample paper."""
    try:
        from app.generate.graph import build_graph
    except ImportError as exc:
        # LangGraph or related dependencies not yet installed in this Python environment
        import pytest

        pytest.skip(f"LangGraph dependencies not installed: {exc}")

    # 1. Define ingested sample Paper
    sample_paper = Paper(
        paper_id="paper_test_001",
        fingerprint="dummy_fingerprint_abc123",
        status="parsing",
        subject="Mathematics",
        class_name="10",
        board="CBSE",
        total_questions=2,
        total_marks=4,
        sections=["Section A"],
        questions=[
            Question(
                number="1",
                section="Section A",
                text="Evaluate 25 multiplied by 4.",
                marks=2,
                type="numerical",
                has_figure=False,
            ),
            Question(
                number="2",
                section="Section A",
                text="Calculate the product of 25 and 4.",
                marks=2,
                type="numerical",
                has_figure=False,
            ),
        ],
    )

    # 2. Adapt Paper into initial LangGraph PaperState
    initial_state = paper_to_graph_state(sample_paper)
    assert initial_state["paper_id"] == "paper_test_001"
    assert len(initial_state["questions"]) == 2
    assert initial_state["total_questions"] == 2

    # 3. Compile graph and run with mocked LLM calls
    graph = build_graph()

    with patch(
        "app.generate.nodes.solve_question.call_groq_json",
        side_effect=mock_call_groq_json,
    ):
        final_state = graph.invoke(initial_state)

    # 4. Assert end-to-end pipeline invariants
    assert final_state["status"] == "ready"
    assert len(final_state["results"]) == 2
    assert len(final_state["errors"]) == 0

    first_result = final_state["results"][0]
    assert first_result["work_item_id"] == "paper_test_001:q:0000"
    assert first_result["question_number"] == "1"
    assert first_result["status"] == "completed"

    solution = first_result["solution"]
    assert solution is not None
    assert solution["answer"] == "100"
    assert len(solution["steps"]) == 1
    assert solution["mark_split"][0]["marks"] == 2
    assert solution["confidence"] >= 0.90
    assert solution["needs_teacher_check"] is False

    second_result = final_state["results"][1]
    assert second_result["work_item_id"] == "paper_test_001:q:0001"
    assert second_result["question_number"] == "2"
    assert second_result["status"] == "completed"
