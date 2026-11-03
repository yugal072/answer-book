
from app.generate.adapter import paper_to_graph_state
from app.generate.graph import graph
from app.models.loaders_models import Paper, Question


def test_graph_processes_multiple_questions():
    paper = Paper(
        paper_id="paper-test-001",
        fingerprint="fingerprint-test-001",
        status="parsing",
        subject="Mathematics",
        class_name="10",
        board="CBSE",
        questions=[
            Question(number="Q1", text="Calculate 2 + 2.", type="numerical"),
            Question(number="Q2", text="Define a prime number.", type="short"),
        ],
    )

    initial_state = paper_to_graph_state(paper)
    final_state = graph.invoke(initial_state)

    assert final_state["status"] == "ready"
    assert final_state["total_questions"] == 2
    assert len(final_state["results"]) == 2

    assert [
        result["question_index"] for result in final_state["results"]
    ] == [0, 1]

    assert all(
        result["status"] == "needs_review"
        for result in final_state["results"]
    )

    assert all(
        result["solution"]["needs_teacher_check"]
        for result in final_state["results"]
    )


def test_graph_handles_empty_paper():
    paper = Paper(
        paper_id="paper-empty-001",
        fingerprint="fingerprint-empty-001",
        status="parsing",
        questions=[],
    )

    final_state = graph.invoke(paper_to_graph_state(paper))

    assert final_state["status"] == "ready"
    assert final_state["total_questions"] == 0
    assert final_state["results"] == []


def test_graph_preserves_duplicate_question_numbers():
    paper = Paper(
        paper_id="paper-duplicate-001",
        fingerprint="fingerprint-duplicate-001",
        status="parsing",
        questions=[
            Question(number="Q1", text="First question."),
            Question(number="Q1", text="Second question."),
        ],
    )

    final_state = graph.invoke(paper_to_graph_state(paper))
    results = final_state["results"]

    assert len(results) == 2
    assert results[0]["question_number"] == "Q1"
    assert results[1]["question_number"] == "Q1"
    assert results[0]["work_item_id"] != results[1]["work_item_id"]
    assert results[0]["question_index"] == 0
    assert results[1]["question_index"] == 1
