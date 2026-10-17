"""Dispatch isolated question workers, carrying only necessary paper context."""

from __future__ import annotations

from langgraph.types import Send

from app.generate.state import PaperState


def dispatch_questions(state: PaperState) -> list[Send] | str:
    """Evaluate paper status and dispatch parallel Send tasks for each question."""
    if state.get("status") == "failed":
        return "aggregate_results"

    questions = state.get("questions", [])
    if not questions:
        return "aggregate_results"

    metadata = state.get("paper_metadata", {})
    paper_id = state.get("paper_id", "")

    sends: list[Send] = []
    for question in questions:
        payload = dict(question)
        payload["paper_id"] = paper_id
        payload["paper_metadata"] = dict(metadata)
        sends.append(Send("solve_question", payload))

    return sends
