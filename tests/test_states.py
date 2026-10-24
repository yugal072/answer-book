
from app.generate.state import (
    CommonMistakeItem,
    MarkSplitItem,
    PaperMetadata,
    PaperState,
    QuestionInput,
    QuestionResult,
    QuestionState,
    QuestionWorkItem,
    SolutionContract,
    VerificationState,
)


def test_mark_split_preserves_frozen_for_key():
    item: MarkSplitItem = {
        "marks": 2,
        "for": "Correct substitution",
    }

    assert item["marks"] == 2
    assert item["for"] == "Correct substitution"
    assert "for_" not in item


def test_solution_contract_has_expected_fields():
    solution: SolutionContract = {
        "question_number": "Q1",
        "answer": "x = 2 or x = 3",
        "steps": ["Factorize the expression", "Solve each factor"],
        "mark_split": [
            {"marks": 1, "for": "Correct factorization"},
            {"marks": 2, "for": "Correct roots"},
        ],
        "common_mistakes": [
            {
                "wrong": "Sign error while factorizing",
                "why": "It changes the roots of the equation.",
            }
        ],
        "confidence": 0.95,
        "verified_by": "symbolic_check",
        "needs_teacher_check": False,
    }

    assert solution["question_number"] == "Q1"
    assert len(solution["steps"]) == 2
    assert solution["mark_split"][0]["for"] == "Correct factorization"
    assert solution["common_mistakes"][0]["wrong"]


def test_question_input_represents_ingestion_fields():
    question: QuestionInput = {
        "number": "Q1",
        "section": "A",
        "text": "Calculate 2 + 2.",
        "marks": 2,
        "type": "numerical",
        "options": None,
        "has_figure": False,
        "page": 1,
        "choice_group": None,
        "section_title": "Arithmetic",
        "chapter": None,
    }

    assert question["text"] == "Calculate 2 + 2."
    assert question["section_title"] == "Arithmetic"
    assert question["options"] is None


def test_question_work_item_contains_execution_fields():
    work_item: QuestionWorkItem = {
        "number": "Q1",
        "section": None,
        "text": "Calculate 2 + 2.",
        "marks": 2,
        "type": "numerical",
        "options": None,
        "has_figure": False,
        "page": None,
        "choice_group": None,
        "section_title": None,
        "chapter": None,
        "work_item_id": "paper-001:q:0000",
        "question_index": 0,
        "selected_route": None,
        "attempts": 0,
        "max_attempts": 2,
        "raw_solution": None,
        "verification": None,
        "confidence": None,
        "cache_status": "not_checked",
        "solution": None,
        "error": None,
    }

    assert work_item["work_item_id"] == "paper-001:q:0000"
    assert work_item["attempts"] == 0
    assert work_item["cache_status"] == "not_checked"


def test_verification_state_can_represent_passed_check():
    verification: VerificationState = {
        "status": "passed",
        "verified_by": "symbolic_check",
        "reasons": [],
        "checks": {"arithmetic": True},
    }

    assert verification["status"] == "passed"
    assert verification["checks"]["arithmetic"] is True


def test_question_state_can_represent_one_worker():
    state: QuestionState = {
        "number": "Q1",
        "section": None,
        "text": "Calculate 2 + 2.",
        "marks": 2,
        "type": "numerical",
        "options": None,
        "has_figure": False,
        "page": None,
        "choice_group": None,
        "section_title": None,
        "chapter": None,
        "work_item_id": "paper-001:q:0000",
        "question_index": 0,
        "selected_route": "numerical",
        "attempts": 1,
        "max_attempts": 2,
        "raw_solution": {"answer": "4"},
        "verification": {"status": "passed"},
        "confidence": 0.95,
        "cache_status": "miss",
        "solution": None,
        "error": None,
    }

    assert state["question_index"] == 0
    assert state["selected_route"] == "numerical"
    assert state["raw_solution"]["answer"] == "4"


def test_question_result_associates_result_with_original_question():
    result: QuestionResult = {
        "work_item_id": "paper-001:q:0000",
        "question_index": 0,
        "question_number": "Q1",
        "solution": None,
        "status": "needs_review",
        "error": None,
    }

    assert result["work_item_id"] == "paper-001:q:0000"
    assert result["status"] == "needs_review"


def test_paper_metadata_uses_subject_key():
    metadata: PaperMetadata = {
        "subject": "Mathematics",
        "class_name": "Class 9",
        "board": "CBSE",
        "total_marks": 80,
        "sections": ["A", "B"],
        "declared_total_questions": 20,
    }

    assert metadata["subject"] == "Mathematics"
    assert "subjects" not in metadata


def test_paper_state_holds_work_items_and_results():
    state: PaperState = {
        "paper_id": "paper-001",
        "fingerprint": "fingerprint-001",
        "status": "solving",
        "paper_metadata": {
            "subject": "Mathematics",
            "class_name": "Class 9",
            "board": "CBSE",
            "total_marks": 80,
            "sections": ["A"],
            "declared_total_questions": 1,
        },
        "questions": [],
        "total_questions": 0,
        "current_index": 0,
        "results": [],
        "errors": [],
    }

    assert state["paper_id"] == "paper-001"
    assert state["questions"] == []
    assert state["results"] == []
    assert state["total_questions"] == 0


def test_common_mistake_item():
    item: CommonMistakeItem = {
        "wrong": "Dropped a negative sign",
        "why": "The sign changes the result.",
    }

    assert item["wrong"]
    assert item["why"]
