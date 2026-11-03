from __future__ import annotations

from app.tools.hash_utils import generate_question_signature


def test_signature_is_deterministic():
    a = generate_question_signature(
        "Q.1: Choose the correct option", 1, "CBSE", "10", options=["A", "B"]
    )
    b = generate_question_signature(
        "Choose the correct option", 1, "CBSE", "10", options=["A", "B"]
    )
    assert a == b


def test_signature_changes_when_options_or_figure_context_changes():
    base = generate_question_signature(
        "What is the answer?", 2, options=["A", "B"]
    )
    different_options = generate_question_signature(
        "What is the answer?", 2, options=["A", "C"]
    )
    figure = generate_question_signature(
        "What is the answer?", 2, options=["A", "B"], has_figure=True
    )
    assert len({base, different_options, figure}) == 3
