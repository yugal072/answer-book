
"""Dispatch individual questions to parallel solving workers."""

from __future__ import annotations

from langgraph.types import Send

from app.generate.state import PaperState


def dispatch_questions(state: PaperState) -> list[Send] | str:
    """Fan out one Send task per question.

    Intended for use as a conditional edge after prepare_paper().
    Returns the aggregate node name when there is nothing to dispatch.
    """
    if state.get("status") == "failed":
        return "aggregate_results"

    questions = state.get("questions", [])

    if not questions:
        return "aggregate_results"

    sends: list[Send] = []

    for question in questions:
        # Send payload becomes the isolated state of one worker.
        sends.append(
            Send(
                "solve_question",
                dict(question),
            )
        )

    return sends
