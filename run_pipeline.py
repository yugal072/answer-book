"""Main Pipeline Orchestrator: Connects Ingestion (Role 1) to LangGraph Solving (Role 2)
with PostgreSQL persistence, Recommendation A paper caching, and atomic checkpointing.

Usage:
    python main.py <path_to_paper> [--limit N] [--subject SUBJECT] [--class-name CLASS] [--export-json]

Examples:
    python main.py test_papers/question_paper_455.pdf --limit 3
    python main.py test_papers/question_paper_455.pdf --export-json
    python main.py test_papers/question_paper_463.pdf
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from app.ingestion.paper_loader import ingest_paper
from app.generate.graph import solve_question
from app.tools import (
    check_paper_cache,
    generate_paper_fingerprint,
    generate_question_signature,
    get_paper_resume_info,
    mark_paper_status,
    save_question_checkpoint,
    upsert_paper_record,
)


def run_pipeline(
    file_path: Path,
    limit: Optional[int] = None,
    subject: Optional[str] = None,
    class_name: Optional[str] = None,
    board: str = "CBSE",
    output_dir: Path = Path("solutionPapers"),
    export_json: bool = False,
) -> Dict[str, Any]:
    """Runs the end-to-end pipeline from raw paper file to verified Solution Book."""

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print("=" * 80)
    print("      ANSWER BOOK — END-TO-END PIPELINE (WITH DB PERSISTENCE & CACHE)")
    print("=" * 80)

    total_start_time = time.time()

    # --------------------------------------------------------------------------
    # Step 0: Pre-Flight Paper Cache Check (Recommendation A)
    # --------------------------------------------------------------------------
    raw_fingerprint = generate_paper_fingerprint(file_path)
    print(f"\n[CACHE PRE-FLIGHT] Checking PostgreSQL database for fingerprint: {raw_fingerprint[:12]}...")

    # If no limit is requested, check if the full paper is already solved in the DB
    if not limit:
        cached_paper = check_paper_cache(raw_fingerprint)
        if cached_paper:
            total_elapsed = time.time() - total_start_time
            print("=" * 80)
            print("  [CACHE HIT] Paper already fully solved in PostgreSQL! (0 LLM API calls)")
            print("=" * 80)
            print(f"  • Paper ID        : {cached_paper['paper_id']}")
            print(f"  • Fingerprint     : {cached_paper['fingerprint']}")
            print(f"  • Total Solved    : {len(cached_paper['solutions'])} questions")
            print(f"  • Status          : {cached_paper['status']}")
            print(f"  • Response Time   : {total_elapsed:.4f} seconds")

            if export_json:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_file = output_dir / f"{cached_paper['paper_id']}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(cached_paper, f, indent=2, ensure_ascii=False)
                print(f"  • Exported JSON   : {out_file.resolve()}")

            print("=" * 80)
            return cached_paper

    print("  • Cache status    : Miss / Processing required")

    # --------------------------------------------------------------------------
    # Step 1: Ingestion (Role 1)
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 1: INGESTION] Processing: {file_path.name}")
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        paper = ingest_paper(file_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        from app.ingestion.image_loader import extract_paper_from_image_file
        paper = extract_paper_from_image_file(file_path)
    else:
        print(f"[ERROR] Unsupported file format '{suffix}'. Supported: .pdf, .png, .jpg, .webp")
        sys.exit(1)

    # Resolve metadata
    resolved_subject = subject or paper.subject or "General"
    resolved_class = class_name or paper.class_name or "Class 9"
    resolved_board = board or paper.board or "CBSE"

    print(f"  • Paper ID        : {paper.paper_id}")
    print(f"  • Fingerprint     : {paper.fingerprint}")
    print(f"  • Total Questions : {paper.total_questions}")
    print(f"  • Total Marks     : {paper.total_marks}")
    print(f"  • Sections Found  : {', '.join(paper.sections) if paper.sections else 'None'}")
    print(f"  • Subject / Grade : {resolved_subject} ({resolved_class}, {resolved_board})")
    print("-" * 80)

    # Upsert paper entry in PostgreSQL with status 'solving'
    upsert_paper_record(
        paper_id=paper.paper_id,
        fingerprint=paper.fingerprint,
        status="solving",
        subject=resolved_subject,
        class_name=resolved_class,
        board=resolved_board,
        total_questions=paper.total_questions,
        total_marks=paper.total_marks,
        sections=paper.sections,
        source_file=str(file_path),
    )

    # --------------------------------------------------------------------------
    # Step 2: Crash Recovery & Solving (Role 2 — LangGraph)
    # --------------------------------------------------------------------------
    questions_to_solve = paper.questions
    if limit and limit > 0:
        questions_to_solve = questions_to_solve[:limit]
        print(f"\n[PHASE 2: SOLVING] Solving first {limit} of {len(paper.questions)} questions...")
    else:
        print(f"\n[PHASE 2: SOLVING] Solving all {len(questions_to_solve)} questions...")

    # Check for previously saved checkpoints in PostgreSQL
    existing_solutions, solved_numbers = get_paper_resume_info(paper.paper_id)
    solved_map = {sol["question_number"]: sol for sol in existing_solutions}

    if solved_numbers:
        print(f"\n[RESUME MODE] Found {len(solved_numbers)} previously saved questions in database!")
        print(f"             Pre-loaded: {sorted(list(solved_numbers))}")

    paper_metadata = {
        "paper_id": paper.paper_id,
        "fingerprint": paper.fingerprint,
        "subject": resolved_subject,
        "class": resolved_class,
        "board": resolved_board,
        "source_file": str(file_path),
    }

    solved_solutions = []

    for idx, q in enumerate(questions_to_solve, start=1):
        q_num = str(q.number)
        q_type = q.type
        marks = q.marks or 1
        prompt_snippet = q.text[:95] + "..." if len(q.text) > 95 else q.text

        # 1. Skip if already solved in a previous run (Crash Recovery)
        if q_num in solved_map:
            print(f"\n>>> [{idx}/{len(questions_to_solve)}] Q{q_num} [ALREADY SOLVED IN DB - SKIPPED]")
            solved_solutions.append(solved_map[q_num])
            continue

        print(f"\n>>> [{idx}/{len(questions_to_solve)}] Q{q_num} [{q_type.upper()}, {marks} Mark(s)]:")
        print(f"    Text: {prompt_snippet}")

        q_start = time.time()
        try:
            # Call LangGraph Question Solve Engine (checks question signature cache internally)
            solution = solve_question(q.model_dump(), paper_metadata)
            elapsed = time.time() - q_start

            # 2. Atomic Question Checkpoint: commit immediately to PostgreSQL
            sig = generate_question_signature(q.text, marks, resolved_board, resolved_class)
            save_question_checkpoint(paper.paper_id, sig, q.model_dump(), solution)

            solved_solutions.append(solution)

            print(f"    [Time: {elapsed:.2f}s | Confidence: {solution['confidence']:.2f} | Teacher Check: {solution['needs_teacher_check']}]")
            print(f"    ANSWER: {solution['answer']}")
            print(f"    STEPS ({len(solution.get('steps', []))} steps):")
            for s in solution.get("steps", [])[:3]:
                print(f"      • {s}")
            if len(solution.get("steps", [])) > 3:
                print(f"      • ... ({len(solution.get('steps', [])) - 3} more steps)")

        except Exception as e:
            print(f"    [ERROR solving Q{q_num}]: {e}")

    # --------------------------------------------------------------------------
    # Step 3: Assembly & Status Finalization
    # --------------------------------------------------------------------------
    is_fully_solved = len(solved_solutions) == len(paper.questions)
    final_status = "ready" if is_fully_solved else "partial"

    if is_fully_solved:
        mark_paper_status(paper.paper_id, "ready")

    solved_payload = {
        "paper_id": paper.paper_id,
        "fingerprint": paper.fingerprint,
        "status": final_status,
        "metadata": paper_metadata,
        "total_questions": len(solved_solutions),
        "total_marks": paper.total_marks,
        "sections": paper.sections,
        "solutions": solved_solutions,
    }

    # Optional JSON export
    if export_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{paper.paper_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(solved_payload, f, indent=2, ensure_ascii=False)
        print(f"\n  • Solution Book JSON : {output_file.resolve()}")

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("                      PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"  • Total Solved     : {len(solved_solutions)} / {len(questions_to_solve)} questions")
    print(f"  • Execution Time   : {total_elapsed:.2f} seconds")
    print(f"  • Final Status     : {final_status}")
    print(f"  • Storage Engine   : PostgreSQL (tables: 'papers', 'solved_questions')")
    print("=" * 80)

    return solved_payload


def main():
    parser = argparse.ArgumentParser(
        description="Answer Book End-to-End Pipeline: Ingest paper and generate verified Solution Book."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the PDF or image exam paper (e.g. test_papers/question_paper_455.pdf)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Limit the number of questions to solve (useful for quick tests)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional: Subject name (e.g. 'Mathematics', 'Science')",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Optional: Grade/Class (e.g. 'Class 9', 'Class 10')",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="CBSE",
        help="Optional: Board name (default: CBSE)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="solutionPapers",
        help="Optional: Directory to save the exported JSON if requested (default: solutionPapers)",
    )
    parser.add_argument(
        "--export-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional: Also export the solution book to a JSON file (default: False)",
    )

    args = parser.parse_args()

    run_pipeline(
        file_path=Path(args.file_path),
        limit=args.limit,
        subject=args.subject,
        class_name=args.class_name,
        board=args.board,
        output_dir=Path(args.output_dir),
        export_json=args.export_json,
    )


if __name__ == "__main__":
    main()
