"""Main Pipeline Orchestrator: Connects Ingestion (Role 1) to LangGraph Solving (Role 2).

Usage:
    python main.py <path_to_paper> [--limit N] [--subject SUBJECT] [--class-name CLASS]

Examples:
    python main.py test_papers/question_paper_455.pdf --limit 3
    python main.py test_papers/question_paper_463.pdf
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from app.ingestion.paper_loader import ingest_paper
from app.generate.graph import solve_question


def run_pipeline(
    file_path: Path,
    limit: int = None,
    subject: str = None,
    class_name: str = None,
    board: str = "CBSE",
    output_dir: Path = Path("solutionPapers"),
) -> Path:
    """Runs the end-to-end pipeline from raw paper file to verified Solution Book JSON."""

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print("=" * 80)
    print("      ANSWER BOOK — END-TO-END PIPELINE (INGESTION -> LANGGRAPH)")
    print("=" * 80)

    total_start_time = time.time()

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

    # --------------------------------------------------------------------------
    # Step 2: Solving & Verification (Role 2 — LangGraph)
    # --------------------------------------------------------------------------
    questions_to_solve = paper.questions
    if limit and limit > 0:
        questions_to_solve = questions_to_solve[:limit]
        print(f"\n[PHASE 2: SOLVING] Solving first {limit} of {len(paper.questions)} questions...")
    else:
        print(f"\n[PHASE 2: SOLVING] Solving all {len(questions_to_solve)} questions...")

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
        q_num = q.number
        q_type = q.type
        marks = q.marks or 1
        prompt_snippet = q.text[:95] + "..." if len(q.text) > 95 else q.text

        print(f"\n>>> [{idx}/{len(questions_to_solve)}] Q{q_num} [{q_type.upper()}, {marks} Mark(s)]:")
        print(f"    Text: {prompt_snippet}")

        q_start = time.time()
        try:
            # Call LangGraph Question Solve Engine
            solution = solve_question(q.model_dump(), paper_metadata)
            elapsed = time.time() - q_start

            solved_solutions.append(solution)

            print(f"    [Time: {elapsed:.2f}s | Confidence: {solution['confidence']:.2f} | Teacher Check: {solution['needs_teacher_check']}]")
            print(f"    ANSWER: {solution['answer']}")
            print(f"    STEPS ({len(solution.get('steps', []))} steps):")
            for s in solution.get("steps", [])[:3]:  # print up to first 3 steps
                print(f"      • {s}")
            if len(solution.get("steps", [])) > 3:
                print(f"      • ... ({len(solution.get('steps', [])) - 3} more steps)")

        except Exception as e:
            print(f"    [ERROR solving Q{q_num}]: {e}")

    # --------------------------------------------------------------------------
    # Step 3: Assembly & Output (Saving Solution Book)
    # --------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{paper.paper_id}.json"

    solved_payload = {
        "paper_id": paper.paper_id,
        "fingerprint": paper.fingerprint,
        "status": "ready",
        "metadata": paper_metadata,
        "total_questions": len(solved_solutions),
        "total_marks": paper.total_marks,
        "sections": paper.sections,
        "solutions": solved_solutions,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(solved_payload, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("                      PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"  • Total Solved     : {len(solved_solutions)} / {len(questions_to_solve)} questions")
    print(f"  • Execution Time   : {total_elapsed:.2f} seconds")
    print(f"  • Status           : ready")
    print(f"  • Solution Book    : {output_file.resolve()}")
    print("=" * 80)

    return output_file


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
        help="Optional: Directory to save the output JSON (default: solutionPapers)",
    )

    args = parser.parse_args()

    run_pipeline(
        file_path=Path(args.file_path),
        limit=args.limit,
        subject=args.subject,
        class_name=args.class_name,
        board=args.board,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
