"""Live & Simulated execution script for the AnswerBook solver pipeline.

Usage:
    # Run in simulation / mock mode (no API key required):
    python scripts/run_pipeline.py --mock

    # Run live against configured Groq API (requires GROQ_API_KEY in .env):
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from unittest.mock import patch

from app.core.config import settings
from app.generate.adapter import paper_to_graph_state
from app.generate.graph import build_graph
from app.generate.prompts import PLANNER_SYSTEM, SOLUTION_SYSTEM, VERIFY_SYSTEM
from app.models.loaders_models import Paper, Question


def mock_groq_handler(
    system_prompt: str,
    user_prompt: str,
    max_retries: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    if system_prompt == PLANNER_SYSTEM:
        return {
            "task_summary": "Solve arithmetic calculation",
            "complexity": "easy",
            "model_tier": "cheap",
            "needs_vision": False,
            "needs_image_generation": False,
            "needs_external_tools": ["calculator"],
            "tool_requests": [
                {
                    "name": "calculator",
                    "expression": "15 * 8 + 10",
                    "purpose": "evaluate arithmetic expression",
                }
            ],
            "verification_focus": ["arithmetic", "steps"],
            "rationale": "Direct arithmetic computation.",
        }

    if system_prompt == SOLUTION_SYSTEM:
        return {
            "answer": "130",
            "steps": [
                "Step 1: Compute 15 * 8 = 120.",
                "Step 2: Add 10: 120 + 10 = 130.",
            ],
            "mark_split": [
                {"marks": 1, "for": "Multiplication step"},
                {"marks": 1, "for": "Final addition and answer"},
            ],
            "common_mistakes": [
                {"wrong": "120", "why": "Forgot to add 10 in the final step"}
            ],
        }

    if system_prompt == VERIFY_SYSTEM:
        return {
            "passed": True,
            "correctness": 1.0,
            "issues": [],
            "checks": {
                "structure_valid": True,
                "answer_review_passed": True,
                "independent_check_passed": True,
            },
            "suggested_repairs": [],
        }

    raise ValueError("Unrecognized prompt in mock")


def run(mock: bool = False) -> None:
    print("=" * 70)
    mode = "SIMULATED (MOCK LLM)" if mock else "LIVE (GROQ API)"
    print(f"      ANSWERBOOK SOLVER PIPELINE EXECUTION [{mode}]")
    print("=" * 70)

    if not mock and not settings.GROQ_API_KEY:
        print(
            "\n[!] GROQ_API_KEY is not set in environment or .env file.\n"
            "    To test without an API key, run with the --mock flag:\n"
            "    python scripts/run_pipeline.py --mock\n"
        )
        sys.exit(1)

    print("\n[1/4] Constructing sample exam paper with Question models...")
    sample_paper = Paper(
        paper_id="demo_paper_2026",
        fingerprint="demo_fp_789xyz",
        status="parsing",
        subject="Mathematics",
        class_name="Class 10",
        board="CBSE",
        total_questions=2,
        total_marks=5,
        sections=["Section A", "Section B"],
        questions=[
            Question(
                number="1",
                section="Section A",
                text="Evaluate 15 * 8 + 10.",
                marks=2,
                type="numerical",
                has_figure=False,
            ),
            Question(
                number="2",
                section="Section B",
                text="Define the Pythagorean theorem and state its formula.",
                marks=3,
                type="short",
                has_figure=False,
            ),
        ],
    )
    print(
        f"      Paper ID: {sample_paper.paper_id} ({len(sample_paper.questions)} questions)"
    )

    print("\n[2/4] Converting Paper into LangGraph PaperState...")
    state = paper_to_graph_state(sample_paper)
    print(f"      Initial state total questions: {state['total_questions']}")
    print(f"      Work item IDs: {[q['work_item_id'] for q in state['questions']]}")

    print("\n[3/4] Compiling LangGraph workflow...")
    graph = build_graph()
    print("      Workflow compiled successfully: prepare -> solve_question (Send) -> aggregate_results")

    print("\n[4/4] Invoking pipeline...")
    if mock:
        with patch(
            "app.generate.nodes.solve_question.call_groq_json",
            side_effect=mock_groq_handler,
        ):
            final_output = graph.invoke(state)
    else:
        final_output = graph.invoke(state)

    print("\n" + "=" * 70)
    print(f"  EXECUTION COMPLETED: Paper Status = '{final_output.get('status')}'")
    print("=" * 70)

    for i, res in enumerate(final_output.get("results", [])):
        sol = res.get("solution") or {}
        print(f"\n--- Question #{res.get('question_number')} (Status: {res.get('status')}) ---")
        print(f"Answer:     {sol.get('answer')}")
        print(f"Confidence: {sol.get('confidence')} (Teacher Review: {sol.get('needs_teacher_check')})")
        print(f"Verified:   {sol.get('verified_by')}")
        if sol.get("steps"):
            print("Steps:")
            for s in sol["steps"]:
                print(f"  • {s}")
        if res.get("error"):
            print(f"Error:      {res.get('error')}")

    print("\nFull Output Summary JSON:")
    summary = {
        "status": final_output.get("status"),
        "total_questions": final_output.get("total_questions"),
        "errors": final_output.get("errors"),
        "results_count": len(final_output.get("results", [])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AnswerBook pipeline")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mocked LLM responses instead of calling Groq API",
    )
    args = parser.parse_args()
    run(mock=args.mock)
