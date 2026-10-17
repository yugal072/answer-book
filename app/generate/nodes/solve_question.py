"""Question-level LangGraph subgraph: plan -> cache -> generate -> verify -> repair -> finalize."""

from __future__ import annotations

import base64
import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.generate.prompts import (
    PLANNER_SYSTEM,
    PROMPT_VERSION,
    SOLUTION_SYSTEM,
    VERIFY_SYSTEM,
    question_context,
)
from app.generate.solver_contracts import (
    deterministic_confidence,
    validate_solution_payload,
)
from app.generate.state import QuestionResult, QuestionState, SolutionContract
from app.llm.capabilities import (
    OpenAIImageGenerationProvider,
    OpenAIVisionProvider,
)
from app.llm.client import call_groq_json
from app.tools.safe_math import safe_calculate


def _as_bool(value: Any) -> bool:
    return value is True


def plan_question(state: QuestionState) -> dict[str, Any]:
    """Plan model tier and capabilities based on question content."""
    question = dict(state)
    context = question_context(question, question.get("paper_metadata"))
    try:
        plan = call_groq_json(
            PLANNER_SYSTEM,
            context,
            model=settings.SOLUTION_PLANNER_MODEL or settings.GROQ_CHEAP_MODEL,
        )
        complexity = plan.get("complexity")
        if complexity not in {"easy", "moderate", "hard"}:
            raise ValueError("Planner returned invalid complexity.")

        tier = plan.get("model_tier")
        if tier not in {"cheap", "standard", "reasoning"}:
            tier = (
                "reasoning"
                if complexity == "hard"
                else "cheap"
                if complexity == "easy"
                else "standard"
            )

        needs_vision = _as_bool(plan.get("needs_vision"))
        needs_image_generation = _as_bool(plan.get("needs_image_generation"))
        route = {
            "complexity": complexity,
            "model_tier": tier,
            "needs_vision": needs_vision,
            "needs_image_generation": needs_image_generation,
            "needs_external_tools": plan.get("needs_external_tools", []),
            "tool_requests": plan.get("tool_requests", []),
            "verification_focus": plan.get("verification_focus", []),
            "rationale": str(plan.get("rationale", "")),
            "task_summary": str(plan.get("task_summary", "")),
            "prompt_version": PROMPT_VERSION,
        }

        tool_results: list[dict[str, Any]] = []
        requests = plan.get("tool_requests", [])
        if isinstance(requests, list):
            for request in requests[:4]:
                if not isinstance(request, dict) or request.get("name") != "calculator":
                    continue
                try:
                    tool_results.append(
                        {
                            "name": "calculator",
                            "purpose": str(request.get("purpose", "")),
                            **safe_calculate(request.get("expression", "")),
                        }
                    )
                except Exception as tool_exc:
                    tool_results.append(
                        {
                            "name": "calculator",
                            "expression": request.get("expression"),
                            "error": str(tool_exc),
                        }
                    )

        return {
            "selected_route": route,
            "tool_results": tool_results,
            "plan_error": None,
        }
    except Exception as exc:
        # A deterministic fallback plan keeps transient planner errors from routing by type.
        # Content complexity and figure references influence tier, never the Question.type field.
        text = str(question.get("text", ""))
        hard_signals = len(text) > 500 or any(
            x in text.lower()
            for x in ("prove", "derive", "justify", "evaluate", "hence")
        )
        route = {
            "complexity": "hard" if hard_signals else "moderate",
            "model_tier": "reasoning" if hard_signals else "standard",
            "needs_vision": bool(question.get("has_figure")),
            "needs_image_generation": False,
            "needs_external_tools": [],
            "tool_requests": [],
            "verification_focus": ["correctness", "assumptions", "marks"],
            "rationale": "Fallback plan after planner error; no type-based routing used.",
            "task_summary": "Solve and independently verify the supplied question.",
            "prompt_version": PROMPT_VERSION,
        }
        return {"selected_route": route, "plan_error": str(exc)}


def lookup_cache(state: QuestionState) -> dict[str, Any]:
    """Check database cache for previously verified identical solution."""
    if not settings.DATABASE_URL or not settings.SOLUTION_STORE_ENABLED:
        return {"cache_status": "miss", "cached_solution": None}
    try:
        from app.tools.hash_utils import generate_question_signature
        from app.tools.question_lookup_tool import lookup_cached_question

        marks = state.get("marks")
        signature = generate_question_signature(
            str(state.get("text", "")),
            int(marks or 0),
            str((state.get("paper_metadata") or {}).get("board") or "CBSE"),
            str((state.get("paper_metadata") or {}).get("class_name") or ""),
            options=state.get("options"),
            has_figure=bool(state.get("has_figure")),
            subject=str((state.get("paper_metadata") or {}).get("subject") or ""),
        )
        cached = lookup_cached_question(signature, str(state.get("number", "")))
        # Old lookup can return unverified fallback rows. Never accept those as cache hits.
        if (
            cached
            and cached.get("needs_teacher_check") is False
            and isinstance(cached.get("confidence"), (int, float))
            and cached["confidence"] >= settings.SOLUTION_CACHE_MIN_CONFIDENCE
            and cached.get("answer")
            and cached.get("verification_status") == "passed"
            and cached.get("model_used")
            and cached.get("prompt_version")
        ):
            return {
                "cache_status": "hit",
                "cached_solution": cached,
                "question_signature": signature,
            }
        return {
            "cache_status": "miss",
            "cached_solution": None,
            "question_signature": signature,
        }
    except Exception as exc:
        return {
            "cache_status": "error",
            "cache_error": str(exc),
            "cached_solution": None,
        }


def _select_model(state: QuestionState) -> str:
    """Select LLM model identifier based on the planned model tier."""
    route = state.get("selected_route") or {}
    tier = route.get("model_tier", "standard")
    if tier == "cheap":
        return settings.GROQ_CHEAP_MODEL
    if tier == "reasoning":
        return settings.GROQ_REASONING_MODEL
    return settings.GROQ_MODEL


def generate_solution(state: QuestionState) -> dict[str, Any]:
    """Generate solution candidate with selected model and capabilities."""
    route = state.get("selected_route") or {}
    # The current ingestion Question model exposes has_figure but not the crop/image bytes.
    # Never hallucinate figure contents when the required visual payload is unavailable.
    updates: dict[str, Any] = {}
    working_state = dict(state)

    if route.get("needs_image_generation"):
        updates["generated_diagram_base64"] = None

    if route.get("needs_vision"):
        updates["figure_interpretation"] = None
        image_bytes = state.get("figure_image_bytes")
        if not image_bytes:
            return {
                "raw_solution": None,
                "generation_error": (
                    "Question requires figure interpretation, but no figure image "
                    "bytes/crop were supplied."
                ),
                "attempts": int(state.get("attempts", 0)) + 1,
            }
        try:
            vision = OpenAIVisionProvider(
                settings.OPENAI_API_KEY, settings.OPENAI_VISION_MODEL
            )
            interpretation = vision.interpret_figure(
                image_bytes,
                "Describe only visible labels, values, geometry, and relationships. "
                "Return JSON with description and uncertainties.",
            )
            working_state["figure_interpretation"] = interpretation
            updates["figure_interpretation"] = interpretation
        except Exception as exc:
            return {
                "raw_solution": None,
                "generation_error": f"Figure interpretation failed safely: {exc}",
                "attempts": int(state.get("attempts", 0)) + 1,
            }

    context = question_context(working_state, state.get("paper_metadata"))
    repair_feedback = state.get("repair_feedback")
    user_prompt = "Question/context JSON:\n" + context
    if repair_feedback:
        user_prompt += (
            "\n\nPrevious solution failed verification. Repair only these issues:\n"
            + "\n".join(f"- {x}" for x in repair_feedback)
        )

    model_used = _select_model(state)
    try:
        payload = call_groq_json(SOLUTION_SYSTEM, user_prompt, model=model_used)
        normalized, errors = validate_solution_payload(
            payload, str(state.get("number", "?")), state.get("marks")
        )
        if errors:
            return {
                **updates,
                "raw_solution": None,
                "generation_error": "; ".join(errors),
                "attempts": int(state.get("attempts", 0)) + 1,
                "model_used": model_used,
            }

        if route.get("needs_image_generation"):
            # Image generation is opt-in via the planner; output is an artifact, not correctness evidence.
            try:
                generator = OpenAIImageGenerationProvider(
                    settings.OPENAI_API_KEY, settings.OPENAI_IMAGE_MODEL
                )
                diagram_bytes = generator.generate_diagram(
                    "Create a clean educational diagram to accompany this answer. "
                    "Do not add unsupported facts. Question: "
                    + str(state.get("text", ""))
                    + "\nAnswer: "
                    + normalized["answer"]
                )
                updates["generated_diagram_base64"] = base64.b64encode(
                    diagram_bytes
                ).decode("ascii")
            except Exception as exc:
                return {
                    **updates,
                    "raw_solution": normalized,
                    "generation_error": (
                        f"Answer generated but required diagram generation failed: {exc}"
                    ),
                    "attempts": int(state.get("attempts", 0)) + 1,
                    "model_used": model_used,
                }

        return {
            **updates,
            "raw_solution": normalized,
            "generation_error": None,
            "attempts": int(state.get("attempts", 0)) + 1,
            "model_used": model_used,
        }
    except Exception as exc:
        return {
            **updates,
            "raw_solution": None,
            "generation_error": str(exc),
            "attempts": int(state.get("attempts", 0)) + 1,
            "model_used": model_used,
        }


def verify_solution(state: QuestionState) -> dict[str, Any]:
    """Independently review solution correctness with the reasoning model."""
    solution = state.get("raw_solution")
    if not solution:
        return {
            "verification": {
                "status": "failed",
                "verified_by": "verification_node",
                "reasons": [
                    state.get("generation_error") or "No valid solution payload."
                ],
                "checks": {
                    "structure_valid": False,
                    "answer_review_passed": False,
                    "independent_check_passed": False,
                },
            },
            "confidence": 0.0,
        }

    context = question_context(dict(state), state.get("paper_metadata"))
    prompt = (
        "Question/context JSON:\n"
        + context
        + "\n\nCandidate solution JSON:\n"
        + json.dumps(solution, ensure_ascii=False)
    )

    try:
        review = call_groq_json(
            VERIFY_SYSTEM, prompt, model=settings.GROQ_REASONING_MODEL
        )
        checks = (
            review.get("checks")
            if isinstance(review.get("checks"), dict)
            else {}
        )
        passed = (
            review.get("passed") is True
            and isinstance(review.get("correctness"), (int, float))
            and review["correctness"] >= 0.8
            and checks.get("answer_review_passed") is True
            and checks.get("independent_check_passed") is True
        )

        # Force failure for unavailable visual context even if the verifier is overconfident.
        route = state.get("selected_route") or {}
        if route.get("needs_vision") and not state.get("figure_image_bytes"):
            passed = False
        if route.get("needs_image_generation") and not state.get(
            "generated_diagram_base64"
        ):
            passed = False

        verification = {
            "status": "passed" if passed else "failed",
            "verified_by": f"groq_independent_review:{settings.GROQ_REASONING_MODEL}",
            "reasons": (
                review.get("issues", [])
                if isinstance(review.get("issues", []), list)
                else [str(review.get("issues"))]
            ),
            "checks": {
                "structure_valid": True,
                "answer_review_passed": _as_bool(
                    checks.get("answer_review_passed")
                ),
                "independent_check_passed": _as_bool(
                    checks.get("independent_check_passed")
                ),
            },
        }
        confidence = deterministic_confidence(
            solution, verification, state.get("marks")
        )
        # Do not use the LLM's self-reported correctness as a calibrated probability.
        return {
            "verification": verification,
            "confidence": confidence,
            "repair_feedback": verification["reasons"],
        }
    except Exception as exc:
        return {
            "verification": {
                "status": "needs_review",
                "verified_by": "verification_error",
                "reasons": [str(exc)],
                "checks": {
                    "structure_valid": True,
                    "answer_review_passed": False,
                    "independent_check_passed": False,
                },
            },
            "confidence": 0.45,
            "repair_feedback": [f"Independent verification failed to run: {exc}"],
        }


def route_after_cache(state: QuestionState) -> str:
    """Route to finalize if cache hit, otherwise generate solution."""
    return (
        "finalize_cache"
        if state.get("cache_status") == "hit"
        else "generate_solution"
    )


def route_after_verification(state: QuestionState) -> str:
    """Route to finalize if passed or max attempts reached, else retry generation."""
    verification = state.get("verification") or {}
    if verification.get("status") == "passed":
        return "finalize_solution"

    attempts = int(state.get("attempts", 0))
    max_attempts = int(
        state.get("max_attempts", settings.SOLUTION_MAX_ATTEMPTS)
    )
    if attempts < max_attempts:
        return "generate_solution"

    return "finalize_solution"


def finalize_cache(state: QuestionState) -> dict[str, Any]:
    """Finalize question output using eligible verified cache row."""
    cached = state.get("cached_solution") or {}
    solution: SolutionContract = {
        "question_number": str(state.get("number", "?")),
        "answer": cached.get("answer", ""),
        "steps": cached.get("steps") or [],
        "mark_split": cached.get("mark_split") or [],
        "common_mistakes": cached.get("common_mistakes") or [],
        "confidence": float(cached.get("confidence", 0.0)),
        "verified_by": str(cached.get("verified_by", "verified_cache")),
        "needs_teacher_check": False,
    }
    result: QuestionResult = {
        "work_item_id": state.get("work_item_id", ""),
        "question_index": state.get("question_index", -1),
        "question_number": str(state.get("number", "?")),
        "solution": solution,
        "status": "completed",
        "error": None,
        "artifacts": [],
    }
    return {
        "solution": solution,
        "confidence": solution["confidence"],
        "verification": {
            "status": "passed",
            "verified_by": "verified_cache",
            "reasons": [],
            "checks": {"cache_eligibility_passed": True},
        },
        "question_result": result,
    }


def finalize_solution(state: QuestionState) -> dict[str, Any]:
    """Finalize candidate solution, determine review flags, and checkpoint to store."""
    raw = state.get("raw_solution")
    verification = state.get("verification") or {
        "status": "failed",
        "reasons": [state.get("generation_error") or "No solution generated."],
    }
    confidence = float(state.get("confidence", 0.0) or 0.0)
    passed = verification.get("status") == "passed" and raw is not None
    needs_teacher_check = (
        not passed or confidence < settings.SOLUTION_STORE_MIN_CONFIDENCE
    )

    solution: SolutionContract | None = None
    if raw:
        solution = {
            **raw,
            "confidence": confidence,
            "verified_by": str(verification.get("verified_by", "not_verified")),
            "needs_teacher_check": needs_teacher_check,
        }

    error = None if solution else (
        state.get("generation_error")
        or "; ".join(verification.get("reasons", []))
        or "Solution generation failed."
    )
    requires_review_without_solution = any(
        token in str(state.get("generation_error", "")).lower()
        for token in ("figure", "diagram", "vision", "ambiguous")
    )
    status = (
        "completed"
        if passed and solution
        else "needs_review"
        if solution or requires_review_without_solution
        else "failed"
    )

    if (
        solution
        and passed
        and confidence >= settings.SOLUTION_STORE_MIN_CONFIDENCE
        and settings.DATABASE_URL
        and settings.SOLUTION_STORE_ENABLED
    ):
        try:
            from app.tools.hash_utils import generate_question_signature
            from app.tools.question_save_tool import save_question_checkpoint

            metadata = state.get("paper_metadata") or {}
            signature = state.get(
                "question_signature"
            ) or generate_question_signature(
                str(state.get("text", "")),
                int(state.get("marks") or 0),
                str(metadata.get("board") or "CBSE"),
                str(metadata.get("class_name") or ""),
                options=state.get("options"),
                has_figure=bool(state.get("has_figure")),
                subject=str(metadata.get("subject") or ""),
            )
            save_question_checkpoint(
                str(state.get("paper_id", "")),
                signature,
                dict(state),
                solution,
            )
        except Exception as exc:
            # Persistence failure must not turn a generated answer into a false failure.
            error = f"Solution generated but checkpoint storage failed: {exc}"

    artifacts: list[dict[str, Any]] = []
    if state.get("generated_diagram_base64"):
        artifacts.append(
            {
                "type": "image",
                "mime_type": "image/png",
                "encoding": "base64",
                "data": state["generated_diagram_base64"],
            }
        )

    result: QuestionResult = {
        "work_item_id": state.get("work_item_id", ""),
        "question_index": state.get("question_index", -1),
        "question_number": str(state.get("number", "?")),
        "solution": solution,
        "status": status,
        "error": error,
        "artifacts": artifacts,
    }
    return {"solution": solution, "error": error, "question_result": result}


def build_question_graph():
    """Build and compile the question-level solving subgraph."""
    builder = StateGraph(QuestionState)
    builder.add_node("plan_question", plan_question)
    builder.add_node("lookup_cache", lookup_cache)
    builder.add_node("generate_solution", generate_solution)
    builder.add_node("verify_solution", verify_solution)
    builder.add_node("finalize_cache", finalize_cache)
    builder.add_node("finalize_solution", finalize_solution)

    builder.add_edge(START, "plan_question")
    builder.add_edge("plan_question", "lookup_cache")
    builder.add_conditional_edges(
        "lookup_cache",
        route_after_cache,
        ["finalize_cache", "generate_solution"],
    )
    builder.add_edge("generate_solution", "verify_solution")
    builder.add_conditional_edges(
        "verify_solution",
        route_after_verification,
        ["generate_solution", "finalize_solution"],
    )
    builder.add_edge("finalize_cache", END)
    builder.add_edge("finalize_solution", END)

    return builder.compile()


_question_graph = build_question_graph()


def solve_question(state: QuestionState) -> dict[str, Any]:
    """Outer-graph worker wrapper. Each Send executes an isolated question subgraph."""
    try:
        child = dict(state)
        child.setdefault("max_attempts", settings.SOLUTION_MAX_ATTEMPTS)
        child.setdefault("attempts", 0)
        child_state = _question_graph.invoke(child)
        result = child_state.get("question_result")
        return {"worker_results": [result] if result else []}
    except Exception as exc:
        result = {
            "work_item_id": state.get("work_item_id", ""),
            "question_index": state.get("question_index", -1),
            "question_number": str(state.get("number", "?")),
            "solution": None,
            "status": "failed",
            "error": f"Question worker crashed: {exc}",
            "artifacts": [],
        }
        return {"worker_results": [result]}
