
"""Compile the initial AnswerBook question-processing graph."""

from langgraph.graph import END, START, StateGraph

from app.generate.nodes.aggregate import aggregate_results
from app.generate.nodes.dispatch import dispatch_questions
from app.generate.nodes.prepare import prepare_paper
from app.generate.nodes.solve_question import solve_question
from app.generate.state import PaperState


def build_graph():
    """Build and compile the initial paper-processing graph."""
    builder = StateGraph(PaperState)

    builder.add_node("prepare", prepare_paper)
    builder.add_node("solve_question", solve_question)
    builder.add_node("aggregate_results", aggregate_results)

    builder.add_edge(START, "prepare")

    builder.add_conditional_edges(
        "prepare",
        dispatch_questions,
        ["solve_question", "aggregate_results"],
    )

    builder.add_edge("solve_question", "aggregate_results")
    builder.add_edge("aggregate_results", END)

    return builder.compile()


# Convenient compiled graph for application code and tests.
graph = build_graph()
