"""Command line: ingest a paper and run the full workflow.

    python -m app.workflow.cli test_papers/question_paper_455.pdf --limit 3
    python -m app.workflow.cli paper.pdf --format pdf --out book.pdf
    python -m app.workflow.cli --resume <run_id>        # same checkpoint store

Progress is printed live from the same event stream the SSE endpoint serves.
Exit code: 0 complete, 2 needs_review/partial, 1 failed or error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from app.workflow.render import build_pdf, write_json

EXIT = {"complete": 0, "needs_review": 2, "partial": 2, "failed": 1}


def _print(ev: dict) -> None:
    d = ev["data"]
    name = ev["event"]
    if name == "run.started":
        print(f"[run {d['run_id']}] paper {d.get('paper_id')} - {d.get('total_questions', '?')} question(s)")
    elif name == "question.stage":
        extra = f" ({d['strategy']})" if d.get("strategy") else ""
        print(f"  Q{d['number']}: {d['stage']}{extra}")
    elif name == "question.completed":
        print(f"  Q{d['number']}: => {d['status'].upper()} via {d['origin']} | verified_by={d['verified_by']} | conf={d['confidence']:.2f}")
    elif name in ("run.finished", "run.error"):
        print(json.dumps(d, indent=2))


def main(argv: Optional[list] = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Answer Book: paper -> verified solution book")
    ap.add_argument("file", nargs="?", help="PDF or image of the question paper")
    ap.add_argument("--limit", type=int, default=None, help="solve only the first N questions")
    ap.add_argument("--subject")
    ap.add_argument("--class-name")
    ap.add_argument("--board")
    ap.add_argument("--format", choices=["json", "pdf"], default="json")
    ap.add_argument("--out", help="output file (default: solutionPapers/<paper_id>.json|.pdf)")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.file and not args.resume:
        ap.error("provide a file or --resume RUN_ID")

    from app.ingestion import IngestionError, ingest_file
    from app.workflow.factory import build_default_service
    from app.workflow.llm import LLMConfigError

    try:
        service = build_default_service()
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    solved = None
    events = None
    if args.resume:
        events = service.resume_stream(args.resume)
    else:
        try:
            paper = ingest_file(Path(args.file), subject=args.subject, class_name=args.class_name, board=args.board)
        except IngestionError as exc:
            print(f"ingestion failed: {exc}", file=sys.stderr)
            return 1
        events = service.stream(paper, limit=args.limit, source_file=Path(args.file).name)

    terminal = None
    for ev in events:
        if not args.quiet:
            _print(ev)
        if ev["event"] in ("run.finished", "run.error"):
            terminal = ev
    if terminal is None or terminal["event"] == "run.error":
        return 1

    paper_id = terminal["data"]["paper_id"]
    json_path = Path(service.settings.output_dir) / f"{paper_id}.json"
    solved = json.loads(json_path.read_text(encoding="utf-8"))
    if args.format == "pdf":
        out = Path(args.out or service.settings.output_dir / f"{paper_id}.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(build_pdf(solved))
        print(f"PDF written: {out}")
    elif args.out:
        print(f"JSON written: {write_json(solved, Path(args.out).parent)}")
    else:
        print(f"JSON written: {json_path}")
    return EXIT[solved["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
