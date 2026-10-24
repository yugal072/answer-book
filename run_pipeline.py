"""Live & Simulated execution script for the AnswerBook solver pipeline.

Runs an end-to-end pipeline:
1. Ingests a real exam paper (PDF/image) via app.ingestion.pipeline.ingest_file.
2. Registers paper metadata in the database for persistence.
3. Converts Paper to LangGraph PaperState and executes with full live streaming.
4. Outputs real-time streaming progress as questions are planned, solved, and verified.
5. Saves the final solution book to solutionPapers/<paper_id>.json.

Usage:
    # Run on default real paper in data/papers/ (data/papers/question_paper_455.pdf):
    python run_pipeline.py

    # Run on a specific paper with live streaming:
    python run_pipeline.py --paper data/papers/question_paper_455.pdf

    # Run on a limited number of questions (e.g. 2 questions for quick test):
    python run_pipeline.py --paper data/papers/question_paper_455.pdf --limit 2

    # Run in simulation / mock mode (no API key required):
    python run_pipeline.py --paper data/papers/question_paper_455.pdf --mock --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.config import settings
from app.generate.adapter import paper_to_graph_state
from app.generate.graph import build_graph
from app.generate.prompts import PLANNER_SYSTEM, SOLUTION_SYSTEM, VERIFY_SYSTEM
from app.ingestion.pipeline import ingest_file
from app.models.loaders_models import Paper, Question


def mock_groq_handler(
    system_prompt: str,
    user_prompt: str,
    *args,
    **kwargs,
) -> dict:
    if system_prompt == PLANNER_SYSTEM:
        return {
            "task_summary": "Solve question systematically",
            "complexity": "easy",
            "model_tier": "cheap",
            "needs_vision": False,
            "needs_image_generation": False,
            "needs_external_tools": [],
            "tool_requests": [],
            "verification_focus": ["method", "accuracy"],
            "rationale": "Standard curriculum method computation.",
        }

    if system_prompt == SOLUTION_SYSTEM:
        return {
            "answer": "Solution evaluated successfully according to curriculum standards.",
            "steps": [
                "Step 1: Parse given parameters and formulate the required relationship.",
                "Step 2: Apply the governing formula or rule and compute the final result.",
            ],
            "mark_split": [
                {"marks": 1, "for": "Formula application and calculation"},
            ],
            "common_mistakes": [
                {"wrong": "Incomplete answer", "why": "Skipped checking boundary conditions"}
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

    raise ValueError(f"Unrecognized prompt in mock: {system_prompt[:60]}...")


def find_default_paper() -> Path:
    """Finds an available PDF paper in data/papers/."""
    primary = Path("data/papers/question_paper_455.pdf")
    if primary.exists():
        return primary

    papers_dir = Path("data/papers")
    if papers_dir.exists():
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]

    return primary


def run(
    paper_path: str | Path,
    limit: Optional[int] = None,
    mock: bool = False,
) -> None:
    path = Path(paper_path)
    print("=" * 80)
    mode = "SIMULATED (MOCK LLM)" if mock else "LIVE (GROQ API)"
    print(f"      ANSWERBOOK END-TO-END PIPELINE EXECUTION [{mode}]")
    print("=" * 80)

    if not mock and not settings.GROQ_API_KEY:
        print(
            "\n[!] GROQ_API_KEY is not set in environment or .env file.\n"
            "    To test without an API key, run with the --mock flag:\n"
            f"    python run_pipeline.py --paper {path} --mock\n"
        )
        sys.exit(1)

    # 1. Ingestion of the real paper
    print(f"\n[1/4] INGESTION: Processing exam paper from '{path}'...")
    if not path.exists():
        print(f"\n[!] Error: Paper file not found at: {path.resolve()}")
        sys.exit(1)

    start_ingest = time.time()
    try:
        paper: Paper = ingest_file(path)
        ingest_duration = time.time() - start_ingest
        print(f"      Ingestion Succeeded in {ingest_duration:.2f}s!")
        print(f"      • Paper ID:        {paper.paper_id}")
        print(f"      • Fingerprint:     {paper.fingerprint}")
        print(f"      • Subject:         {paper.subject or 'Unknown'}")
        print(f"      • Class / Grade:   {paper.class_name or 'N/A'}")
        print(f"      • Board:           {paper.board or 'CBSE'}")
        print(f"      • Total Parsed:    {len(paper.questions)} questions")
        if paper.sections:
            print(f"      • Sections:        {', '.join(paper.sections)}")
    except Exception as exc:
        print(f"\n[!] Ingestion failed: {exc}")
        sys.exit(1)

    # Apply optional question limit
    if limit is not None and limit > 0:
        original_count = len(paper.questions)
        paper.questions = paper.questions[:limit]
        paper.total_questions = len(paper.questions)
        print(f"      [Filter] Sliced to first {len(paper.questions)} of {original_count} questions (--limit {limit}).")

    # 2. Database registration for foreign keys and checkpointing
    print("\n[2/4] DATABASE: Checking & recording parent paper in database...")
    if settings.DATABASE_URL and settings.SOLUTION_STORE_ENABLED:
        try:
            from app.tools.paper_cache_tool import upsert_paper_record

            upsert_paper_record(
                paper_id=paper.paper_id,
                fingerprint=paper.fingerprint,
                status="solving",
                subject=paper.subject,
                class_name=paper.class_name,
                board=paper.board or "CBSE",
                total_questions=paper.total_questions or len(paper.questions),
                total_marks=paper.total_marks or 0,
                sections=paper.sections,
                source_file=str(path),
            )
            print("      Registered parent paper record for solution persistence.")
        except Exception as db_exc:
            print(f"      [Warning] Database pre-registration note: {db_exc}")
    else:
        print("      Database storage is disabled; checkpointing to DB will be skipped.")

    # 3. Conversion to LangGraph PaperState
    print("\n[3/4] PREPARATION: Converting Paper into LangGraph PaperState...")
    state = paper_to_graph_state(paper)
    print(f"      Prepared {state['total_questions']} work items.")
    sample_ids = [q["work_item_id"] for q in state["questions"][:3]]
    if len(state["questions"]) > 3:
        sample_ids.append(f"... +{len(state['questions']) - 3} more")
    print(f"      Work item IDs: {sample_ids}")

    print("      Compiling LangGraph workflow: prepare -> solve_question (Send) -> aggregate_results")
    graph = build_graph()

    # 4. Stream graph execution
    print("\n[4/4] STREAMING PIPELINE EXECUTION (Live Stream)...")
    print("-" * 80)
    sys.stdout.flush()

    stream_start = time.time()
    final_output: dict[str, Any] = {}
    solved_results: list[dict[str, Any]] = []

    def stream_runner():
        nonlocal final_output
        stream_generator = graph.stream(state, stream_mode="updates")
        for update in stream_generator:
            for node_name, node_output in update.items():
                if node_name == "prepare":
                    status = node_output.get("status")
                    count = len(node_output.get("questions", []))
                    print(f"\n[STREAM >> prepare] Paper validated (status: '{status}'). Dispatched {count} question tasks to workers.")
                    sys.stdout.flush()

                elif node_name == "solve_question":
                    worker_results = node_output.get("worker_results", [])
                    for res in worker_results:
                        solved_results.append(res)
                        q_num = res.get("question_number", "?")
                        q_status = res.get("status", "unknown")
                        sol = res.get("solution") or {}
                        conf = sol.get("confidence")
                        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
                        teacher_flag = sol.get("needs_teacher_check", False)

                        print(f"\n[STREAM >> solve_question] Q#{q_num} | Status: {q_status.upper()} | Confidence: {conf_str} | Teacher Check: {teacher_flag}")
                        if sol.get("verified_by"):
                            print(f"  • Verified By: {sol.get('verified_by')}")
                        if sol.get("answer"):
                            print(f"  • Answer:      {sol.get('answer')}")
                        if sol.get("steps"):
                            print(f"  • Steps ({len(sol['steps'])}):")
                            for s in sol["steps"]:
                                print(f"    - {s}")
                        if sol.get("mark_split"):
                            print("  • Mark Split:")
                            for ms in sol["mark_split"]:
                                print(f"    - {ms.get('marks')} mark(s): {ms.get('for')}")
                        if sol.get("common_mistakes"):
                            print("  • Common Mistakes:")
                            for cm in sol["common_mistakes"]:
                                print(f"    - Wrong: {cm.get('wrong')} | Reason: {cm.get('why')}")
                        if res.get("error"):
                            print(f"  • [Note/Error]: {res.get('error')}")
                        sys.stdout.flush()

                elif node_name == "aggregate_results":
                    final_output = node_output
                    print(f"\n[STREAM >> aggregate_results] Aggregated {len(node_output.get('results', []))} results. Overall Status: '{node_output.get('status')}'.")
                    sys.stdout.flush()

    if mock:
        with patch(
            "app.generate.nodes.solve_question.call_json",
            side_effect=mock_groq_handler,
        ):
            stream_runner()
    else:
        stream_runner()

    stream_duration = time.time() - stream_start

    # Fallback to accumulated results if final_output wasn't captured
    if not final_output:
        final_output = {
            "status": "ready" if all(r.get("status") in {"completed", "needs_review"} for r in solved_results) else "failed",
            "total_questions": len(solved_results),
            "results": solved_results,
            "errors": [],
        }

    # 5. Persist Solution Book JSON
    output_dir = Path("solutionPapers")
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_file = output_dir / f"{paper.paper_id}.json"

    solution_payload = {
        "paper_id": paper.paper_id,
        "fingerprint": paper.fingerprint,
        "status": final_output.get("status"),
        "metadata": {
            "subject": paper.subject,
            "class": paper.class_name,
            "board": paper.board,
            "total_questions": len(final_output.get("results", [])),
            "source_file": str(path),
        },
        "total_questions": len(final_output.get("results", [])),
        "results": final_output.get("results", []),
    }

    with open(solution_file, "w", encoding="utf-8") as f:
        json.dump(solution_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"  EXECUTION COMPLETED IN {stream_duration:.2f}s: Paper Status = '{final_output.get('status')}'")
    print(f"  Solution Book Saved: {solution_file.resolve()}")
    print("=" * 80)

    summary = {
        "paper_id": paper.paper_id,
        "status": final_output.get("status"),
        "total_questions": final_output.get("total_questions"),
        "results_count": len(final_output.get("results", [])),
        "output_file": str(solution_file),
    }
    print("\nExecution Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AnswerBook End-to-End Pipeline on Real Papers")
    parser.add_argument(
        "--paper",
        type=str,
        default=None,
        help="Path to paper PDF/image in data/papers/ (default: data/papers/question_paper_455.pdf)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of questions to process (e.g. --limit 2)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mocked LLM responses instead of calling Groq API",
    )
    args = parser.parse_args()

    selected_paper = Path(args.paper) if args.paper else find_default_paper()
    run(paper_path=selected_paper, limit=args.limit, mock=args.mock)
