from app.verification.verifier import verify_solution


def test_verifier_accepts_correct_numerical_answer():
    question = {
        "number": "3(b)",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
    }

    solution = {
        "question_number": "3(b)",
        "answer": "4 ohm",
        "steps": [
            "Use the parallel resistance formula.",
            "Substitute the values.",
            "Calculate the result.",
        ],
        "mark_split": [],
        "common_mistakes": [],
    }

    result = verify_solution(question, solution)

    assert result["passed"] is True
    assert result["verified_by"] == "symbolic"


def test_verifier_rejects_wrong_numerical_answer():
    question = {
        "number": "3(b)",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
    }

    solution = {
        "question_number": "3(b)",
        "answer": "18 ohm",
        "steps": [
            "Add the two resistances.",
        ],
        "mark_split": [],
        "common_mistakes": [],
    }

    result = verify_solution(question, solution)

    assert result["passed"] is False
    assert result["verified_by"] == "symbolic"
def test_parallel_resistance_with_different_values():
    question = {
        "type": "numerical",
        "text": "Two resistors of 10 ohm and 10 ohm are connected in parallel.",
    }

    solution = {
        "answer": "5 ohm",
    }

    result = verify_solution(question, solution)

    assert result["passed"] is True
    assert result["verified_by"] == "symbolic"


def test_parallel_resistance_rejects_wrong_value():
    question = {
        "type": "numerical",
        "text": "Two resistors of 4 ohm and 4 ohm are connected in parallel.",
    }

    solution = {
        "answer": "8 ohm",
    }

    result = verify_solution(question, solution)

    assert result["passed"] is False
    assert result["verified_by"] == "symbolic"