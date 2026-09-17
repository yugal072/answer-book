from app.answer_generation.generator import generate_answer


def test_numerical_answer_structure():
    result = generate_answer(
        "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "numerical",
    )

    assert result["question"]
    assert "answer" in result
    assert "steps" in result
    assert "mark_split" in result
    assert "common_mistakes" in result
    assert "confidence" in result
    assert result["verified_by"] == "none"
    assert result["needs_teacher_check"] is True