from app.answer_generation.generator import generate_answer
from app.verification.verifier import verify_solution
from app.graph.state import SolveState


def generate_node(state: SolveState) -> SolveState:
    """
    Generate a candidate solution for the current question.
    """

    question = state["question"]

    solution = generate_answer(question)

    return {
        **state,
        "solution": solution,
    }


def verify_node(state: SolveState) -> SolveState:
    """
    Verify the generated solution.
    """

    question = state["question"]
    solution = state["solution"]

    verification = verify_solution(question, solution)

    return {
        **state,
        "verification": verification,
    }


def retry_node(state: SolveState) -> SolveState:
    """
    Prepare the state for another generation attempt
    after verification fails.
    """

    retry_count = state.get("retry_count", 0)

    verification = state.get("verification", {})

    return {
        **state,
        "retry_count": retry_count + 1,
        "retry_reason": verification.get(
            "reason",
            "Previous solution failed verification.",
        ),
    }