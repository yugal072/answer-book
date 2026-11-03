
"""Convert an ingested Paper model into the initial LangGraph PaperState.

Responsibilities:
- Preserve paper metadata and question order.
- Create one uniquely identifiable work item per question.
- Initialize question-level execution and verification fields.
- Produce a graph-ready parent state.

This module performs no solving, LLM calls, database access, or I/O.
"""

from typing import Any, cast

from app.generate.state import PaperState, QuestionWorkItem
from app.models.loaders_models import Paper


# Maximum number of total solving attempts, including the first attempt.
# This is a provisional policy and can later move into application settings.
DEFAULT_MAX_ATTEMPTS = 2


def paper_to_graph_state(paper: Paper) -> PaperState:
    """Convert a validated Paper into its initial graph orchestration state.

    Args:
        paper: The validated Pydantic Paper model produced by ingestion.

    Returns:
        PaperState containing the paper metadata and all initialized
        question work items in their original ingestion order.

    Notes:
        - question_index is zero-based.
        - work_item_id is deterministic for the same paper ID and ordering.
        - total_questions counts actual questions, not the optional
          declared total_questions value from ingestion.
        - The source Paper and its Question instances are not mutated.
    """

    question_items: list[QuestionWorkItem] = []

    for index, question in enumerate(paper.questions):
        # Pydantic v2: convert the validated model into ordinary Python data.
        question_data: dict[str, Any] = question.model_dump(mode="python")

        # Include the index so repeated display numbers remain distinguishable.
        work_item_id = f"{paper.paper_id}:q:{index:04d}"

        work_item = cast(
            QuestionWorkItem,
            {
                **question_data,

                # Stable identity and original ordering
                "work_item_id": work_item_id,
                "question_index": index,

                # Routing and execution state
                "selected_route": None,
                "attempts": 0,
                "max_attempts": DEFAULT_MAX_ATTEMPTS,

                # Intermediate solution and verification state
                "raw_solution": None,
                "verification": None,
                "confidence": None,

                # Cache lookup is performed by a later graph node.
                "cache_status": "not_checked",

                # Populated by downstream solving/verification nodes.
                "solution": None,
                "error": None,
            },
        )

        question_items.append(work_item)

    paper_metadata = {
        "subject": paper.subject,
        "class_name": paper.class_name,
        "board": paper.board,
        "total_marks": paper.total_marks,
        "sections": list(paper.sections),
        "declared_total_questions": paper.total_questions,
    }

    return PaperState(
        paper_id=paper.paper_id,
        fingerprint=paper.fingerprint,
        status=paper.status,
        paper_metadata=paper_metadata,
        questions=question_items,
        total_questions=len(question_items),
        current_index=0,
        results=[],
        errors=[],
        worker_results= [],
    )
