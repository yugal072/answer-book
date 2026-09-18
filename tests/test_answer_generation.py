from app.answer_generation.generator import generate_answer
from app.llm.gemini import GeminiProvider


def test_numerical_answer_structure(monkeypatch):
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

    fake_response = """
    {
        "answer": "4 ohm",
        "steps": [
            "Use the parallel resistance formula.",
            "Substitute 6 ohm and 12 ohm.",
            "Calculate the result."
        ],
        "mark_split": [
            {
                "marks": 1,
                "for": "Correct formula"
            },
            {
                "marks": 1,
                "for": "Correct substitution"
            },
            {
                "marks": 1,
                "for": "Correct final answer"
            }
        ],
        "common_mistakes": [
            {
                "wrong": "18 ohm",
                "why": "The resistors were incorrectly treated as series."
            }
        ]
    }
    """

    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        lambda self, prompt: fake_response,
    )

    result = generate_answer(question)

    assert result["question_number"] == "3(b)"
    assert result["answer"] == "4 ohm"
    assert len(result["steps"]) == 3
    assert len(result["mark_split"]) == 3
    assert len(result["common_mistakes"]) == 1
    assert result["confidence"] == 0.0
    assert result["verified_by"] == "none"
    assert result["needs_teacher_check"] is True