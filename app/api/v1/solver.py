"""Solving endpoints, backed by the LangGraph workflow (``app.workflow``).

    POST /api/v1/process-paper?format=json|pdf|sse   upload -> ingest -> graph -> output
    POST /api/v1/solve-question                      one question through the same graph
    GET  /api/v1/solutions/{paper_id}?format=json|pdf
    POST /api/v1/runs/{run_id}/resume?format=json|sse

``/solve-question``, ``/process-paper`` and ``/solutions/{paper_id}`` keep their
paths and their contract-compatible ``solutions`` list; the response gained
explicit ``status`` / ``outcome`` / ``summary`` / ``results`` fields.
"""
import json
import logging
from typing import Any, Dict, Iterator, Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.api.v1.ingestion import process_file_ingestion
from app.workflow.adapter import AdapterError
from app.workflow.llm import LLMConfigError
from app.workflow.render import build_pdf, format_sse, safe_paper_id
from app.workflow.service import AnswerBookService, RunNotFound
from app.workflow.settings import load_settings

log = logging.getLogger(__name__)
router = APIRouter(tags=["Solver"])

OutputFormat = Literal["json", "pdf", "sse"]
_service: Optional[AnswerBookService] = None


def get_service() -> AnswerBookService:
    """Process-wide service (lazy, so importing the app never needs credentials)."""
    global _service
    if _service is None:
        from app.workflow.factory import build_default_service

        try:
            _service = build_default_service()
        except LLMConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _service


def _sse(events: Iterator[Dict[str, Any]]) -> StreamingResponse:
    def gen() -> Iterator[str]:
        seq = 0
        try:
            for ev in events:
                seq += 1
                yield format_sse(seq, ev["event"], ev["data"])
        finally:
            close = getattr(events, "close", None)
            if close:
                close()  # propagate client disconnects into the graph stream

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _respond(solved: Dict[str, Any], fmt: str) -> Response:
    if fmt == "pdf":
        return Response(
            content=build_pdf(solved), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_paper_id(solved["paper_id"])}.pdf"',
                     "X-Paper-Outcome": solved["outcome"]},
        )
    # outcome "failed" means no question could be answered: report it as an upstream failure
    return JSONResponse(content=solved, status_code=502 if solved["outcome"] == "failed" else 200)


@router.post("/solve-question")
def solve_question_endpoint(payload: Dict[str, Any]):
    """Solves a single question through the same graph (cache, verify, repair, confidence)."""
    question_data = payload.get("question")
    if not question_data or not isinstance(question_data, dict):
        raise HTTPException(status_code=400, detail="Missing 'question' in payload.")
    metadata = payload.get("metadata") or {}
    try:
        solved = get_service().solve_single(question_data, metadata)
    except (AdapterError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    r = solved.results[0]
    body = r.solution.model_dump()
    body.update(
        status=r.status, origin=r.origin,
        verification=r.verification.model_dump() if r.verification else None,
        confidence_assessment=r.confidence_assessment.model_dump() if r.confidence_assessment else None,
        cache=r.cache.model_dump(), diagnostics=[d.model_dump() for d in r.diagnostics],
    )
    return JSONResponse(content=body, status_code=502 if r.status == "failed" else 200)


@router.post("/process-paper")
def process_paper_endpoint(
    file: UploadFile = File(...),
    limit: Optional[int] = Query(None, ge=1, description="Max number of questions to solve"),
    subject: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
    format: OutputFormat = Query("json", description="json | pdf | sse"),
):
    """Ingest a PDF/image paper, run the workflow graph, and serve JSON, a PDF, or an SSE stream."""
    content = file.file.read()
    paper = process_file_ingestion(file=file, content=content, subject=subject, class_name=class_name, board=board)
    service = get_service()
    try:
        if format == "sse":
            return _sse(service.stream(paper, limit=limit, source_file=file.filename))
        solved = service.run(paper, limit=limit, source_file=file.filename)
    except AdapterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - classify, don't leak internals
        log.exception("workflow failed")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {type(exc).__name__}") from exc
    return _respond(solved.model_dump(mode="json", by_alias=True), format)


@router.get("/solutions/{paper_id}")
def get_solution_endpoint(paper_id: str, format: Literal["json", "pdf"] = Query("json")):
    """Retrieve a previously saved Solution Book (JSON or PDF)."""
    try:
        name = safe_paper_id(paper_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid paper id.")
    file_path = load_settings().output_dir / f"{name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Solution paper '{paper_id}' not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if format == "json":
        return data
    if "results" not in data or "summary" not in data:
        raise HTTPException(status_code=422, detail="This stored solution predates the workflow schema; re-run the paper to render a PDF.")
    return _respond(data, "pdf")


@router.post("/runs/{run_id}/resume")
def resume_run_endpoint(run_id: str, format: Literal["json", "sse"] = Query("json")):
    """Continue an interrupted run from its last checkpoint (same process / same checkpoint store)."""
    service = get_service()
    try:
        if format == "sse":
            return _sse(service.resume_stream(run_id))
        solved = service.resume(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail=f"No checkpointed run '{run_id}'.")
    return _respond(solved.model_dump(mode="json", by_alias=True), "json")
