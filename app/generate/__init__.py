"""AnswerBook generation package.

Imports are lazy so pure contracts can be tested independently.
"""

from __future__ import annotations

from typing import Any

from app.generate.state import PaperState, QuestionState, SolutionContract


def build_graph() -> Any:
    """Lazily load and compile the LangGraph workflow."""
    from app.generate.graph import build_graph as _build_graph

    return _build_graph()


__all__ = ["QuestionState", "PaperState", "SolutionContract", "build_graph"]
