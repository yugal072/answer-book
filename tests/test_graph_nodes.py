from app.graph.solve.nodes import generate_node, verify_node
from app.graph.solve.graph import build_solve_graph
from app.llm.gemini import GeminiProvider


FAKE_GEMINI_RESPONSE = """
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


def test_generate_node(monkeypatch):
    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        lambda self, prompt: FAKE_GEMINI_RESPONSE,
    )

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "retry_count": 0,
    }

    result = generate_node(state)

    assert "solution" in result
    assert result["solution"]["answer"] == "4 ohm"


def test_verify_node():
    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "solution": {
            "question_number": "3(b)",
            "answer": "4 ohm",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
        },
        "retry_count": 0,
    }

    result = verify_node(state)

    assert "verification" in result
    assert result["verification"]["passed"] is True


def test_solve_graph(monkeypatch):
    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        lambda self, prompt: FAKE_GEMINI_RESPONSE,
    )

    graph = build_solve_graph()

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "retry_count": 0,
    }

    result = graph.invoke(state)

    assert "solution" in result
    assert "verification" in result
    assert result["solution"]["answer"] == "4 ohm"
    assert result["verification"]["passed"] is True
def test_solve_graph_retries_when_verification_fails(monkeypatch):
    responses = [
        """
        {
            "answer": "18 ohm",
            "steps": ["Incorrect calculation."],
            "mark_split": [],
            "common_mistakes": []
        }
        """,
        """
        {
            "answer": "4 ohm",
            "steps": [
                "Use the parallel resistance formula.",
                "Substitute 6 ohm and 12 ohm.",
                "Calculate the result."
            ],
            "mark_split": [
                {"marks": 1, "for": "Correct formula"},
                {"marks": 1, "for": "Correct substitution"},
                {"marks": 1, "for": "Correct final answer"}
            ],
            "common_mistakes": [
                {
                    "wrong": "18 ohm",
                    "why": "The resistors were incorrectly treated as series."
                }
            ]
        }
        """
    ]

    def fake_generate(self, prompt):
        return responses.pop(0)

    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        fake_generate,
    )

    graph = build_solve_graph()

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "retry_count": 0,
    }

    result = graph.invoke(state)

    assert result["retry_count"] == 1
    assert result["solution"]["answer"] == "4 ohm"
    assert result["verification"]["passed"] is True