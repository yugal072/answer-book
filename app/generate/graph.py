"""LangGraph Question Solve Engine: Assembles nodes, edges, and conditional routing."""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from app.generate.state import QuestionState, SolutionContract
from app.generate.nodes import (
    planner_node,
    math_solver_node,
    mcq_solver_node,
    theory_solver_node,
    normalizer_node,
)


def route_by_type(state: QuestionState) -> str:
    """Conditional routing function: directs execution based on planner's selected_route."""
    route = state.get("selected_route", "theory")
    if route in ("math", "mcq", "theory", "cached"):
        return route
    return "theory"


def build_question_solver_graph():
    """Builds and compiles the StateGraph for solving an individual question."""
    workflow = StateGraph(QuestionState)

    # 1. Register all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("math_solver", math_solver_node)
    workflow.add_node("mcq_solver", mcq_solver_node)
    workflow.add_node("theory_solver", theory_solver_node)
    workflow.add_node("normalizer", normalizer_node)

    # 2. Wire entry edge
    workflow.add_edge(START, "planner")

    # 3. Wire conditional branching from planner to specialized solvers (or cache bypass)
    workflow.add_conditional_edges(
        "planner",
        route_by_type,
        {
            "math": "math_solver",
            "mcq": "mcq_solver",
            "theory": "theory_solver",
            "cached": "normalizer",
        },
    )


    # 4. Wire solvers into the normalizer gate
    workflow.add_edge("math_solver", "normalizer")
    workflow.add_edge("mcq_solver", "normalizer")
    workflow.add_edge("theory_solver", "normalizer")

    # 5. Wire normalizer to the end
    workflow.add_edge("normalizer", END)

    # 6. Compile graph
    return workflow.compile()


# Single compiled graph instance ready for execution
question_solver_graph = build_question_solver_graph()


def solve_question(question_data: Dict[str, Any], paper_metadata: Dict[str, Any] = None) -> SolutionContract:
    """Convenience helper to run a single question through the compiled LangGraph pipeline."""
    initial_state: QuestionState = {
        "number": str(question_data.get("number", "1")),
        "section": str(question_data.get("section", "A")),
        "text": str(question_data.get("text", "")),
        "marks": int(question_data.get("marks", 1)),
        "type": str(question_data.get("type", "short")),
        "options": question_data.get("options"),
        "has_figure": bool(question_data.get("has_figure", False)),
        "page": int(question_data.get("page", 1)),
        "choice_group": question_data.get("choice_group"),
        "chapter": question_data.get("chapter"),
        "metadata": paper_metadata or {},
    }

    final_state = question_solver_graph.invoke(initial_state)
    return final_state["solution"]
