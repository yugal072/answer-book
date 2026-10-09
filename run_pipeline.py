"""Backwards-compatible entry point for the end-to-end pipeline.

The implementation now lives in ``app.workflow`` (LangGraph architecture:
cache -> type-specific solve -> verify/repair -> confidence -> verified
storage -> SolvedPaper). This wrapper keeps the old command line working:

    python run_pipeline.py test_papers/question_paper_455.pdf --limit 3 --export-json

``--export-json`` is accepted for compatibility; the JSON book is now always
written to ``--output-dir`` (default ``solutionPapers/``). See docs/WORKFLOW.md.
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Answer Book end-to-end pipeline")
    ap.add_argument("file_path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--subject")
    ap.add_argument("--class-name")
    ap.add_argument("--board")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--export-json", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    if a.output_dir:
        os.environ["WORKFLOW_OUTPUT_DIR"] = a.output_dir

    from app.workflow.cli import main as cli_main

    argv = [a.file_path]
    for flag, val in (("--limit", a.limit), ("--subject", a.subject), ("--class-name", a.class_name), ("--board", a.board)):
        if val is not None:
            argv += [flag, str(val)]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
