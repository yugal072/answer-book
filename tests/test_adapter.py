"""Unit test for app.generate.adapter.py"""

from copy import deepcopy

from app.generate.adapter import (
    DEFAULT_MAX_ATTEMPTS,
    paper_to_graph_state,
)
from app.models.loaders_models import Paper, Question


def make_paper() -> Paper:
    """Create a representative ingestion model for adapter tests."""
    return Paper(
        paper_id="paper-001",
        fingerprint="fingerprint-001",
        status="parsing",
        subject="Mathematics",
        class_name="Class 9",
        board="CBSE",
        total_marks=10,
        total_questions=2,
        sections=["A"],
        questions=[
            Question(
                number="Q1",
                section="A",
                text="Calculate 2 + 2.",
                marks=2,
                type="numerical",
                has_figure=False,
                page=1,
                section_title="Arithmetic",
            ),
            Question(
                number="Q1",
                section="A",
                text="Choose the correct answer.",
                marks=1,
                type="mcq",
                options=["A", "B", "C", "D"],
                has_figure=False,
                page=1,
            ),
        ],
    )


def test_adapter_preserves_paper_identity_and_metadata():
    state = paper_to_graph_state(make_paper())

    assert state["paper_id"] == "paper-001"
    assert state["fingerprint"] == "fingerprint-001"
    assert state["status"] == "parsing"

    metadata = state["paper_metadata"]
    assert metadata["subject"] == "Mathematics"
    assert metadata["class_name"] == "Class 9"
    assert metadata["board"] == "CBSE"
    assert metadata["total_marks"] == 10
    assert metadata["sections"] == ["A"]
    assert metadata["declared_total_questions"] == 2


def test_adapter_preserves_question_order_and_content():
    state = paper_to_graph_state(make_paper())

    questions = state["questions"]

    assert len(questions) == 2
    assert questions[0]["text"] == "Calculate 2 + 2."
    assert questions[1]["text"] == "Choose the correct answer."
    assert questions[0]["question_index"] == 0
    assert questions[1]["question_index"] == 1
    assert questions[0]["section_title"] == "Arithmetic"
    assert questions[1]["options"] == ["A", "B", "C", "D"]


def test_adapter_creates_distinct_ids_for_duplicate_question_numbers():
    state = paper_to_graph_state(make_paper())

    first, second = state["questions"]

    assert first["number"] == second["number"] == "Q1"
    assert first["work_item_id"] == "paper-001:q:0000"
    assert second["work_item_id"] == "paper-001:q:0001"
    assert first["work_item_id"] != second["work_item_id"]


def test_adapter_initializes_execution_fields():
    state = paper_to_graph_state(make_paper())

    for question in state["questions"]:
        assert question["selected_route"] is None
        assert question["attempts"] == 0
        assert question["max_attempts"] == DEFAULT_MAX_ATTEMPTS
        assert question["raw_solution"] is None
        assert question["verification"] is None
        assert question["confidence"] is None
        assert question["cache_status"] == "not_checked"
        assert question["solution"] is None
        assert question["error"] is None


def test_adapter_counts_actual_questions():
    paper = make_paper()
    paper.total_questions = 99  # Declared count differs from extracted count.

    state = paper_to_graph_state(paper)

    assert state["total_questions"] == 2
    assert state["paper_metadata"]["declared_total_questions"] == 99


def test_adapter_initializes_parent_progress_and_results():
    state = paper_to_graph_state(make_paper())

    assert state["current_index"] == 0
    assert state["results"] == []
    assert state["errors"] == []


def test_adapter_handles_empty_paper():
    paper = Paper(
        paper_id="empty-paper",
        fingerprint="empty-fingerprint",
        status="parsing",
    )

    state = paper_to_graph_state(paper)

    assert state["questions"] == []
    assert state["total_questions"] == 0
    assert state["current_index"] == 0
    assert state["results"] == []
    assert state["errors"] == []


def test_adapter_does_not_mutate_source_paper():
    paper = make_paper()
    original = deepcopy(paper.model_dump(mode="python"))

    paper_to_graph_state(paper)

    assert paper.model_dump(mode="python") == original


def test_adapter_is_deterministic_for_same_paper_and_order():
    paper = make_paper()

    first_state = paper_to_graph_state(paper)
    second_state = paper_to_graph_state(paper)

    first_ids = [
        item["work_item_id"] for item in first_state["questions"]
    ]
    second_ids = [
        item["work_item_id"] for item in second_state["questions"]
    ]

    assert first_ids == second_ids


def test_adapter_preserves_optional_question_values():
    paper = Paper(
        paper_id="paper-optional",
        fingerprint="fingerprint-optional",
        status="parsing",
        questions=[
            Question(
                number="Q1",
                text="Explain the concept.",
                type="short",
            )
        ],
    )

    state = paper_to_graph_state(paper)
    question = state["questions"][0]

    assert question["section"] is None
    assert question["marks"] is None
    assert question["options"] is None
    assert question["page"] is None
    assert question["choice_group"] is None
    assert question["chapter"] is None
    assert question["section_title"] is None
