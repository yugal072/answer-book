"""Prototype Runner: Executes static test papers through LangGraph and saves solutions."""

import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from app.generate.graph import solve_question


def run_prototype():
    data_path = Path("data/static_test_papers.json")
    if not data_path.exists():
        print(f"Error: Could not find {data_path}")
        return

    # Create output directory
    output_dir = Path("solutionPapers")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print("=" * 80)
    print("      ANSWER BOOK — PROTOTYPE QUESTION SOLVER (GROQ + LANGGRAPH)")
    print("=" * 80)

    total_start_time = time.time()

    for paper_idx, paper in enumerate(papers, start=1):
        metadata = paper.get("paper_metadata", {})
        paper_id = metadata.get("paper_id", f"paper_{paper_idx}")
        fingerprint = metadata.get("fingerprint") or paper.get("fingerprint", f"fp_{paper_id}")
        metadata["fingerprint"] = fingerprint
        subject = metadata.get("subject", "General")
        grade = metadata.get("class", "Class 9")
        questions = paper.get("questions", [])

        print(f"\n[PAPER {paper_idx}/{len(papers)}] {paper_id.upper()} — {subject} ({grade})")
        print(f"Fingerprint: {fingerprint}")
        print(f"Total Questions to solve: {len(questions)}")
        print("=" * 80)

        solved_paper_solutions = []

        for q_idx, q in enumerate(questions, start=1):
            q_num = q.get("number", str(q_idx))
            q_type = q.get("type", "short")
            marks = q.get("marks", 1)
            text = q.get("text", "")

            print(f"\n>>> Solving Q{q_num} [{q_type.upper()}, {marks} Marks] ({q_idx}/{len(questions)}):")
            print(f"    Prompt: {text[:110]}..." if len(text) > 110 else f"    Prompt: {text}")

            q_start = time.time()
            try:
                solution = solve_question(q, metadata)
                elapsed = time.time() - q_start

                solved_paper_solutions.append(solution)

                # Pretty print solution to terminal
                print(f"    [Time: {elapsed:.2f}s | Confidence: {solution['confidence']:.2f}]")
                print(f"    ANSWER: {solution['answer']}")
                print("    STEPS:")
                for s in solution.get("steps", []):
                    print(f"      • {s}")
                print("    MARK SPLIT:")
                for ms in solution.get("mark_split", []):
                    print(f"      - {ms.get('marks')} mark(s): {ms.get('for')}")
                print("    COMMON MISTAKES:")
                for cm in solution.get("common_mistakes", []):
                    print(f"      x Wrong: '{cm.get('wrong')}' -> Reason: {cm.get('why')}")

            except Exception as e:
                print(f"    [ERROR solving Q{q_num}]: {e}")

        # Save solved paper JSON
        output_file = output_dir / f"{paper_id}.json"
        solved_payload = {
            "paper_id": paper_id,
            "fingerprint": fingerprint,
            "status": "ready",
            "metadata": metadata,
            "total_questions": len(solved_paper_solutions),
            "solutions": solved_paper_solutions,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(solved_payload, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVED] Solution book saved to: {output_file}")
        print("-" * 80)

    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"ALL PAPERS PROCESSED SUCCESSFULLY IN {total_time:.2f}s!")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    run_prototype()
