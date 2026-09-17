from app.answer_generation.generator import generate_answer


def test_numerical_answer_structure():
    question = {
        "number": "3(b)",
        "section": "B",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
        "options": None,
        "has_figure": False,
        "page": 2,
        "choice_group": None,
    }

    result = generate_answer(question)

    assert result["question_number"] == "3(b)"
    assert result["answer"]
    assert "steps" in result
    assert "mark_split" in result
    assert "common_mistakes" in result
    assert "confidence" in result
    assert result["verified_by"] == "none"
    assert result["needs_teacher_check"] is True